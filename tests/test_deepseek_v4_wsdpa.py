# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek V4 fused prefill attention kernels."""

import copy
import math
import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models import deepseek_v4_wsdpa as wsdpa

_METAL_AVAILABLE = (
    mx.default_device() == mx.gpu and hasattr(mx, "metal") and mx.metal.is_available()
)


def _reset_wsdpa():
    wsdpa._ENABLED = True
    wsdpa._TOPK_ENABLED = True
    wsdpa._broken = False
    wsdpa._kernel = None
    wsdpa._topk_kernel = None
    wsdpa._ready.clear()
    wsdpa._topk_ready.clear()


def _max_abs(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def _inputs(heads, q_len, offset, window, pooled_len, ratio, trim=0):
    mx.random.seed(7)
    kv_len = offset + q_len - trim
    q = (mx.random.normal((1, heads, q_len, 512)) * 0.25).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, kv_len, 512)) * 0.25).astype(mx.bfloat16)
    pooled = (
        (mx.random.normal((1, pooled_len, 512)) * 0.25).astype(mx.bfloat16)
        if pooled_len
        else None
    )
    sinks = (mx.random.normal((heads,)) * 0.1).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(512)
    arrays = [q, kv, sinks]
    if pooled is not None:
        arrays.append(pooled)
    mx.eval(*arrays)
    return q, kv, pooled, sinks, scale, offset, window, ratio


def _reference_attention(
    q,
    kv,
    pooled,
    sinks,
    scale,
    offset,
    window,
    ratio,
    topk=None,
):
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.deepseek_v4 import (
        _build_window_mask,
        _compressed_visibility,
    )

    q_len = q.shape[2]
    mask = _build_window_mask(1, q_len, offset, window, kv.shape[2])
    kv_all = kv
    if pooled is not None:
        comp_mask = _compressed_visibility(1, q_len, offset, pooled.shape[1], ratio)
        if topk is not None:
            pooled_indices = mx.arange(pooled.shape[1], dtype=mx.uint32)
            selected = (topk[..., None] == pooled_indices[None, None, None, :]).any(
                axis=-2
            )[:, None, :, :]
            comp_mask = comp_mask & selected
        mask = mx.concatenate([mask, comp_mask], axis=-1)
        kv_all = mx.concatenate([kv, pooled[:, None, :, :]], axis=2)

    return scaled_dot_product_attention(
        q,
        kv_all,
        kv_all,
        cache=None,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )


class TestDeepseekV4WsdpaState(unittest.TestCase):
    def setUp(self):
        _reset_wsdpa()

    def test_source_hashed_kernel_names_change_with_source(self):
        first = wsdpa._source_hashed_name("kernel", "source-a")
        second = wsdpa._source_hashed_name("kernel", "source-b")

        self.assertEqual(first, wsdpa._source_hashed_name("kernel", "source-a"))
        self.assertNotEqual(first, second)

    def test_prefill_route_activates_per_specialization(self):
        calls = []

        def fake_kernel(**kwargs):
            calls.append(kwargs)
            heads = dict(kwargs["template"])["HEADS"]
            q_len = kwargs["output_shapes"][0][1]
            return [mx.zeros((heads, q_len, 512), dtype=mx.bfloat16)]

        with patch.object(wsdpa, "_get_kernel", return_value=fake_kernel):
            for heads in (8, 16, 32, 64):
                q = mx.zeros((1, heads, 2, 512), dtype=mx.bfloat16)
                kv = mx.zeros((1, 1, 2, 512), dtype=mx.bfloat16)
                sinks = mx.zeros((heads,), dtype=mx.bfloat16)
                out = wsdpa.wsdpa_prefill(q, kv, None, sinks, 1.0, 0, 128, 1)
                self.assertIsNotNone(out)

        self.assertEqual(
            wsdpa._ready,
            {
                (8, 512, 2, 1),
                (16, 512, 2, 1),
                (32, 512, 2, 1),
                (64, 512, 2, 1),
            },
        )
        self.assertTrue(wsdpa.wsdpa_prefill_route_active())
        self.assertEqual(
            [call["grid"][0] for call in calls],
            [(heads // 4) * 128 for heads in (8, 16, 32, 64)],
        )

    def test_dispatch_failure_disables_both_routes(self):
        q = mx.zeros((1, 8, 2, 512), dtype=mx.bfloat16)
        kv = mx.zeros((1, 1, 2, 512), dtype=mx.bfloat16)
        sinks = mx.zeros((8,), dtype=mx.bfloat16)

        with patch.object(
            wsdpa,
            "_get_kernel",
            return_value=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic dispatch failure")
            ),
        ):
            self.assertIsNone(wsdpa.wsdpa_prefill(q, kv, None, sinks, 1.0, 0, 128, 1))

        self.assertTrue(wsdpa._broken)
        self.assertFalse(wsdpa.wsdpa_prefill_route_active())
        self.assertFalse(wsdpa.wsdpa_prefill_route_active(topk=True))

    def test_first_evaluation_failure_keeps_route_inactive(self):
        q = mx.zeros((1, 8, 2, 512), dtype=mx.bfloat16)
        kv = mx.zeros((1, 1, 2, 512), dtype=mx.bfloat16)
        sinks = mx.zeros((8,), dtype=mx.bfloat16)

        with patch.object(
            wsdpa,
            "_get_kernel",
            return_value=lambda **kwargs: [mx.zeros((8, 2, 512), dtype=mx.bfloat16)],
        ):
            with patch.object(
                wsdpa.mx,
                "eval",
                side_effect=RuntimeError("synthetic evaluation failure"),
            ):
                self.assertIsNone(
                    wsdpa.wsdpa_prefill(q, kv, None, sinks, 1.0, 0, 128, 1)
                )

        self.assertTrue(wsdpa._broken)
        self.assertFalse(wsdpa._ready)

    def test_topk_route_activates_after_evaluation(self):
        q = mx.zeros((1, 8, 5, 512), dtype=mx.bfloat16)
        kv = mx.zeros((1, 1, 5, 512), dtype=mx.bfloat16)
        pooled = mx.zeros((1, 3, 512), dtype=mx.bfloat16)
        topk = mx.zeros((1, 5, 2), dtype=mx.uint32)
        sinks = mx.zeros((8,), dtype=mx.bfloat16)

        with patch.object(
            wsdpa,
            "_get_topk_kernel",
            return_value=lambda **kwargs: [mx.zeros((8, 5, 512), dtype=mx.bfloat16)],
        ):
            out = wsdpa.wsdpa_topk_prefill(q, kv, pooled, topk, sinks, 1.0, 0, 128, 4)

        self.assertIsNotNone(out)
        self.assertEqual(wsdpa._topk_ready, {(8, 512, 5, 4)})
        self.assertTrue(wsdpa.wsdpa_prefill_route_active(topk=True))

    def test_unsupported_inputs_fall_back(self):
        sinks = mx.zeros((8,), dtype=mx.bfloat16)
        valid_q = mx.zeros((1, 8, 2, 512), dtype=mx.bfloat16)
        valid_kv = mx.zeros((1, 1, 2, 512), dtype=mx.bfloat16)

        cases = [
            (mx.zeros((1, 4, 2, 512), dtype=mx.bfloat16), valid_kv, sinks, 0),
            (valid_q.astype(mx.float32), valid_kv, sinks, 0),
            (mx.zeros((2, 8, 2, 512), dtype=mx.bfloat16), valid_kv, sinks, 0),
            (mx.zeros((1, 8, 1, 512), dtype=mx.bfloat16), valid_kv, sinks, 0),
            (valid_q, valid_kv, sinks, mx.array(0, dtype=mx.int32)),
        ]

        for q, kv, case_sinks, offset in cases:
            with self.subTest(shape=q.shape, dtype=q.dtype, offset=offset):
                self.assertIsNone(
                    wsdpa.wsdpa_prefill(q, kv, None, case_sinks, 1.0, offset, 128, 1)
                )


@unittest.skipUnless(_METAL_AVAILABLE, "Metal is required")
class TestDeepseekV4WsdpaMetal(unittest.TestCase):
    def setUp(self):
        _reset_wsdpa()

    def test_prefill_matches_reference_for_all_routes_and_head_counts(self):
        for heads in (8, 16, 32, 64):
            for ratio, pooled_len in ((1, 0), (4, 3), (128, 3)):
                with self.subTest(heads=heads, ratio=ratio):
                    args = _inputs(
                        heads=heads,
                        q_len=5,
                        offset=254 if ratio == 128 else 7,
                        window=4,
                        pooled_len=pooled_len,
                        ratio=ratio,
                    )
                    out = wsdpa.wsdpa_prefill(*args)
                    ref = _reference_attention(*args)

                    self.assertIsNotNone(out)
                    mx.eval(out, ref)
                    self.assertEqual(out.shape, ref.shape)
                    self.assertLess(_max_abs(out, ref), 8e-3)

    def test_topk_prefill_matches_reference_for_all_head_counts(self):
        topk = mx.array(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 3],
                [1, 2, 3],
                [1, 2, 4],
            ],
            dtype=mx.uint32,
        )[None]

        for heads in (8, 16, 32, 64):
            with self.subTest(heads=heads):
                args = _inputs(
                    heads=heads,
                    q_len=6,
                    offset=11,
                    window=5,
                    pooled_len=5,
                    ratio=4,
                )
                q, kv, pooled, sinks, scale, offset, window, ratio = args
                out = wsdpa.wsdpa_topk_prefill(
                    q,
                    kv,
                    pooled,
                    topk,
                    sinks,
                    scale,
                    offset,
                    window,
                    ratio,
                )
                ref = _reference_attention(*args, topk=topk)

                self.assertIsNotNone(out)
                mx.eval(out, ref)
                self.assertEqual(out.shape, ref.shape)
                self.assertLess(_max_abs(out, ref), 8e-3)

    def test_prefill_matches_reference_with_trimmed_rotating_cache(self):
        args = _inputs(
            heads=16,
            q_len=8,
            offset=15,
            window=6,
            pooled_len=4,
            ratio=4,
            trim=5,
        )
        out = wsdpa.wsdpa_prefill(*args)
        ref = _reference_attention(*args)

        self.assertIsNotNone(out)
        mx.eval(out, ref)
        self.assertLess(_max_abs(out, ref), 8e-3)

    def test_v4_attention_fused_route_matches_stock_path(self):
        from mlx_lm.models import deepseek_v4

        args = deepseek_v4.ModelArgs(
            vocab_size=64,
            hidden_size=512,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=1,
            q_lora_rank=64,
            o_lora_rank=64,
            o_groups=8,
            head_dim=512,
            qk_rope_head_dim=64,
            sliding_window=8,
            compress_ratios=[4],
            index_n_heads=8,
            index_head_dim=128,
            index_topk=4,
            max_position_embeddings=256,
        )
        fused = deepseek_v4.V4Attention(args, 0)
        fused.set_dtype(mx.bfloat16)
        stock = copy.deepcopy(fused)
        x = (mx.random.normal((1, 16, args.hidden_size)) * 0.1).astype(mx.bfloat16)

        original_topk = deepseek_v4.wsdpa_topk_prefill

        def checking_topk(*call_args, **call_kwargs):
            indices = call_args[3]
            self.assertTrue(mx.all(indices[..., 1:] >= indices[..., :-1]).item())
            return original_topk(*call_args, **call_kwargs)

        with patch.object(
            deepseek_v4,
            "wsdpa_topk_prefill",
            side_effect=checking_topk,
        ):
            fused_out = fused(x, deepseek_v4.DeepseekV4Cache(args.sliding_window))
            mx.eval(fused_out)

        self.assertTrue(wsdpa.wsdpa_prefill_route_active(topk=True))
        wsdpa._broken = True
        stock_out = stock(x, deepseek_v4.DeepseekV4Cache(args.sliding_window))
        mx.eval(stock_out)

        self.assertEqual(fused_out.shape, stock_out.shape)
        self.assertLess(_max_abs(fused_out, stock_out), 2e-2)
