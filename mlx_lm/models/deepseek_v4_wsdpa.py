# SPDX-License-Identifier: Apache-2.0
"""Fused windowed and pooled prefill attention for DeepSeek V4.

Adapted from jundot/omlx PR #2568, merged as b128b23290392b4eab8df236a78308448c20642c.
The original fused kernel was contributed by @jonathan308 in c2f55836.

The stock ratio-4 prefill path selects sparse pooled rows but materializes a
full-pool mask and sends the entire growing pool through generic SDPA. These
Metal kernels visit only the visible local rows and either the visible pooled
prefix or the selected pooled rows. Unsupported inputs and setup or first-
dispatch failures fall back to the stock path.
"""

import hashlib
import logging
import os
from typing import Optional, Set, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("MLX_LM_DSV4_WSDPA", "1") == "1"
_TOPK_ENABLED = os.environ.get("MLX_LM_DSV4_WSDPA_TOPK", "1") == "1"
_SUPPORTED_HEADS = (8, 16, 32, 64)

_kernel = None
_topk_kernel = None
_broken = False
_ready: Set[Tuple[int, int, int, int]] = set()
_topk_ready: Set[Tuple[int, int, int, int]] = set()

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_SOURCE = """
    // q:      [H, L, D]  bf16 (contiguous)
    // kv:     [S, D]     bf16 (local rows)
    // pooled: [P, D]     bf16 (dummy row when P == 0)
    // sinks:  [H]        bf16
    // params: int32 [6] = {offset, window, ratio, P, S, L}
    // scalep: fp32  [1] = scale
    // out:    [H, L, D]  bf16
    const uint t    = threadgroup_position_in_grid.y;
    const uint tid  = thread_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x * 4 + (tid / 32);
    const uint lane = tid % 32;

    static_assert(D_HEAD == 512, "WSDPA loads assume head_dim 512");
    if (head >= HEADS) return;

    constant int *prm = (constant int *)&params[0];
    const int offset = prm[0];
    const int window = prm[1];
    const int ratio  = prm[2];
    const int P      = prm[3];
    const int S      = prm[4];
    const int L      = prm[5];
    const float scale = ((constant float *)&scalep[0])[0];

    constexpr uint D = D_HEAD;
    constexpr uint D4 = D / 4;
    const uint p = (uint)offset + t;

    device const bfloat4 *qh =
        (device const bfloat4 *)(q + ((uint64_t)head * L + t) * D);
    float4 qa0 = float4(qh[lane +  0]);
    float4 qa1 = float4(qh[lane + 32]);
    float4 qa2 = float4(qh[lane + 64]);
    float4 qa3 = float4(qh[lane + 96]);

    float m = -INFINITY, s = 0.0f;
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    device const bfloat4 *kvr = (device const bfloat4 *)kv;
    device const bfloat4 *pol = (device const bfloat4 *)pooled;

    const int base = offset + L - S;
    const int j0 = max(base, (int)p - window + 1) - base;
    const int j1 = (int)p - base;
    const int pvis = min((int)((p + 1) / (uint)ratio), P);
    const int n_local = j1 - j0 + 1;
    const int total = n_local + pvis;

    for (int r = 0; r < total; r++) {
        const bool is_pool = r >= n_local;
        const uint idx = is_pool ? (uint)(r - n_local) : (uint)(j0 + r);
        device const bfloat4 *row =
            (is_pool ? pol : kvr) + (uint64_t)idx * D4;
        float4 k0 = float4(row[lane + 0]);
        float4 k1 = float4(row[lane + 32]);
        float4 k2 = float4(row[lane + 64]);
        float4 k3 = float4(row[lane + 96]);
        float d = dot(qa0, k0) + dot(qa1, k1) + dot(qa2, k2) + dot(qa3, k3);
        d = simd_sum(d) * scale;
        float mn = max(m, d);
        float c = (m == -INFINITY) ? 0.0f : exp(m - mn);
        float w = exp(d - mn);
        m = mn;
        s = s * c + w;
        a0 = a0 * c + w * k0;
        a1 = a1 * c + w * k1;
        a2 = a2 * c + w * k2;
        a3 = a3 * c + w * k3;
    }

    s += exp(float(((device const bfloat *)sinks)[head]) - m);

    const float inv = (s == 0.0f) ? 0.0f : 1.0f / s;
    device bfloat4 *oh =
        (device bfloat4 *)(out + ((uint64_t)head * L + t) * D);
    oh[lane +  0] = bfloat4(a0 * inv);
    oh[lane + 32] = bfloat4(a1 * inv);
    oh[lane + 64] = bfloat4(a2 * inv);
    oh[lane + 96] = bfloat4(a3 * inv);
"""

_SOURCE_TOPK = """
    // q:      [H, L, D]  bf16 (contiguous)
    // kv:     [S, D]     bf16 (local rows)
    // pooled: [P, D]     bf16
    // topk:   [L, K]     uint32 (temporally sorted pooled indices)
    // sinks:  [H]        bf16
    // params: int32 [7] = {offset, window, ratio, P, S, L, K}
    // scalep: fp32  [1] = scale
    // out:    [H, L, D]  bf16
    const uint t    = threadgroup_position_in_grid.y;
    const uint tid  = thread_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x * 4 + (tid / 32);
    const uint lane = tid % 32;

    static_assert(D_HEAD == 512, "WSDPA loads assume head_dim 512");
    if (head >= HEADS) return;

    constant int *prm = (constant int *)&params[0];
    const int offset = prm[0];
    const int window = prm[1];
    const int ratio  = prm[2];
    const int P      = prm[3];
    const int S      = prm[4];
    const int L      = prm[5];
    const int K      = prm[6];
    const float scale = ((constant float *)&scalep[0])[0];

    constexpr uint D = D_HEAD;
    constexpr uint D4 = D / 4;
    const uint p = (uint)offset + t;

    device const bfloat4 *qh =
        (device const bfloat4 *)(q + ((uint64_t)head * L + t) * D);
    float4 qa0 = float4(qh[lane +  0]);
    float4 qa1 = float4(qh[lane + 32]);
    float4 qa2 = float4(qh[lane + 64]);
    float4 qa3 = float4(qh[lane + 96]);

    float m = -INFINITY, s = 0.0f;
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    device const bfloat4 *kvr = (device const bfloat4 *)kv;
    device const bfloat4 *pol = (device const bfloat4 *)pooled;

    const int base = offset + L - S;
    const int j0 = max(base, (int)p - window + 1) - base;
    const int j1 = (int)p - base;
    const int pvis = min((int)((p + 1) / (uint)ratio), P);
    const int n_local = j1 - j0 + 1;

    for (int r = 0; r < n_local; r++) {
        const uint idx = (uint)(j0 + r);
        device const bfloat4 *row = kvr + (uint64_t)idx * D4;
        float4 k0 = float4(row[lane + 0]);
        float4 k1 = float4(row[lane + 32]);
        float4 k2 = float4(row[lane + 64]);
        float4 k3 = float4(row[lane + 96]);
        float d = dot(qa0, k0) + dot(qa1, k1) + dot(qa2, k2) + dot(qa3, k3);
        d = simd_sum(d) * scale;
        float mn = max(m, d);
        float c = (m == -INFINITY) ? 0.0f : exp(m - mn);
        float w = exp(d - mn);
        m = mn;
        s = s * c + w;
        a0 = a0 * c + w * k0;
        a1 = a1 * c + w * k1;
        a2 = a2 * c + w * k2;
        a3 = a3 * c + w * k3;
    }

    device const uint *tk = (device const uint *)topk;
    const uint64_t tk_base = (uint64_t)t * (uint)K;
    for (int r = 0; r < K; r++) {
        const uint idx = tk[tk_base + (uint)r];
        if ((int)idx >= pvis) break;
        device const bfloat4 *row = pol + (uint64_t)idx * D4;
        float4 k0 = float4(row[lane + 0]);
        float4 k1 = float4(row[lane + 32]);
        float4 k2 = float4(row[lane + 64]);
        float4 k3 = float4(row[lane + 96]);
        float d = dot(qa0, k0) + dot(qa1, k1) + dot(qa2, k2) + dot(qa3, k3);
        d = simd_sum(d) * scale;
        float mn = max(m, d);
        float c = (m == -INFINITY) ? 0.0f : exp(m - mn);
        float w = exp(d - mn);
        m = mn;
        s = s * c + w;
        a0 = a0 * c + w * k0;
        a1 = a1 * c + w * k1;
        a2 = a2 * c + w * k2;
        a3 = a3 * c + w * k3;
    }

    s += exp(float(((device const bfloat *)sinks)[head]) - m);

    const float inv = (s == 0.0f) ? 0.0f : 1.0f / s;
    device bfloat4 *oh =
        (device bfloat4 *)(out + ((uint64_t)head * L + t) * D);
    oh[lane +  0] = bfloat4(a0 * inv);
    oh[lane + 32] = bfloat4(a1 * inv);
    oh[lane + 64] = bfloat4(a2 * inv);
    oh[lane + 96] = bfloat4(a3 * inv);
"""


def _source_hashed_name(base: str, source: str) -> str:
    digest = hashlib.sha256((_HEADER + source).encode("utf-8")).hexdigest()[:12]
    return f"{base}_{digest}"


def _route_enabled(*, topk: bool = False) -> bool:
    if _broken or not _ENABLED:
        return False
    return not topk or _TOPK_ENABLED


def wsdpa_prefill_route_active(*, topk: bool = False) -> bool:
    """Return whether any specialization of the route has evaluated."""
    if not _route_enabled(topk=topk):
        return False
    return bool(_topk_ready if topk else _ready)


def _get_kernel():
    global _kernel, _broken
    if not _route_enabled():
        return None
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if _kernel is None:
        try:
            _kernel = mx.fast.metal_kernel(
                name=_source_hashed_name("dsv4_wsdpa", _SOURCE),
                input_names=["q", "kv", "pooled", "sinks", "params", "scalep"],
                output_names=["out"],
                source=_SOURCE,
                header=_HEADER,
            )
        except Exception:
            _broken = True
            logger.warning(
                "DeepSeek V4 WSDPA setup failed; using stock SDPA",
                exc_info=True,
            )
            return None
    return _kernel


def _get_topk_kernel():
    global _topk_kernel, _broken
    if not _route_enabled(topk=True):
        return None
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if _topk_kernel is None:
        try:
            _topk_kernel = mx.fast.metal_kernel(
                name=_source_hashed_name("dsv4_wsdpa_topk", _SOURCE_TOPK),
                input_names=[
                    "q",
                    "kv",
                    "pooled",
                    "topk",
                    "sinks",
                    "params",
                    "scalep",
                ],
                output_names=["out"],
                source=_SOURCE_TOPK,
                header=_HEADER,
            )
        except Exception:
            _broken = True
            logger.warning(
                "DeepSeek V4 top-k WSDPA setup failed; using stock SDPA",
                exc_info=True,
            )
            return None
    return _topk_kernel


def _valid_common_inputs(
    q: mx.array,
    kv: mx.array,
    sinks: mx.array,
    offset: int,
    ratio: int,
) -> bool:
    return (
        not isinstance(offset, mx.array)
        and ratio in (1, 4, 128)
        and q.ndim == 4
        and q.dtype == mx.bfloat16
        and q.shape[0] == 1
        and q.shape[1] in _SUPPORTED_HEADS
        and q.shape[2] > 1
        and q.shape[3] == 512
        and kv.ndim == 4
        and kv.dtype == q.dtype
        and kv.shape[0] == 1
        and kv.shape[1] == 1
        and kv.shape[2] >= q.shape[2]
        and kv.shape[3] == q.shape[3]
        and sinks.ndim == 1
        and sinks.dtype == q.dtype
        and sinks.shape[0] == q.shape[1]
    )


def wsdpa_prefill(
    q: mx.array,
    kv: mx.array,
    pooled: Optional[mx.array],
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
) -> Optional[mx.array]:
    """Run fused window plus visible pooled-prefix prefill attention."""
    global _broken
    if not _valid_common_inputs(q, kv, sinks, offset, ratio) or window <= 0:
        return None

    if pooled is None:
        pooled_len = 0
    else:
        if (
            pooled.ndim != 3
            or pooled.dtype != q.dtype
            or pooled.shape[0] != 1
            or pooled.shape[2] != q.shape[3]
        ):
            return None
        pooled_len = pooled.shape[1]

    kernel = _get_kernel()
    if kernel is None:
        return None

    try:
        heads, q_len, head_dim = q.shape[1], q.shape[2], q.shape[3]
        specialization = (heads, head_dim, q_len, ratio)
        pooled_input = (
            mx.contiguous(pooled[0])
            if pooled_len
            else mx.zeros((1, head_dim), dtype=q.dtype)
        )
        params = mx.array(
            [offset, window, ratio, pooled_len, kv.shape[2], q_len],
            dtype=mx.int32,
        )
        output = kernel(
            inputs=[
                mx.contiguous(q[0]),
                mx.contiguous(kv[0, 0]),
                pooled_input,
                mx.contiguous(sinks),
                params,
                mx.array([scale], dtype=mx.float32),
            ],
            template=[("HEADS", heads), ("D_HEAD", head_dim)],
            grid=((heads // 4) * 128, q_len, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(heads, q_len, head_dim)],
            output_dtypes=[mx.bfloat16],
        )[0]
        if specialization not in _ready:
            mx.eval(output)
            _ready.add(specialization)
        return output[None]
    except Exception:
        _broken = True
        logger.warning(
            "DeepSeek V4 WSDPA disabled after dispatch failure; using stock SDPA",
            exc_info=True,
        )
        return None


def wsdpa_topk_prefill(
    q: mx.array,
    kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
) -> Optional[mx.array]:
    """Run fused window plus temporally sorted top-k pooled attention."""
    global _broken
    if (
        not _valid_common_inputs(q, kv, sinks, offset, ratio)
        or ratio != 4
        or window <= 0
    ):
        return None
    if (
        not _route_enabled(topk=True)
        or pooled.ndim != 3
        or pooled.dtype != q.dtype
        or pooled.shape[0] != 1
        or pooled.shape[1] == 0
        or pooled.shape[2] != q.shape[3]
        or topk.ndim != 3
        or topk.dtype != mx.uint32
        or topk.shape[0] != 1
        or topk.shape[1] != q.shape[2]
    ):
        return None

    kernel = _get_topk_kernel()
    if kernel is None:
        return None

    try:
        heads, q_len, head_dim = q.shape[1], q.shape[2], q.shape[3]
        specialization = (heads, head_dim, q_len, ratio)
        params = mx.array(
            [
                offset,
                window,
                ratio,
                pooled.shape[1],
                kv.shape[2],
                q_len,
                topk.shape[2],
            ],
            dtype=mx.int32,
        )
        output = kernel(
            inputs=[
                mx.contiguous(q[0]),
                mx.contiguous(kv[0, 0]),
                mx.contiguous(pooled[0]),
                mx.contiguous(topk[0]),
                mx.contiguous(sinks),
                params,
                mx.array([scale], dtype=mx.float32),
            ],
            template=[("HEADS", heads), ("D_HEAD", head_dim)],
            grid=((heads // 4) * 128, q_len, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(heads, q_len, head_dim)],
            output_dtypes=[mx.bfloat16],
        )[0]
        if specialization not in _topk_ready:
            mx.eval(output)
            _topk_ready.add(specialization)
        return output[None]
    except Exception:
        _broken = True
        logger.warning(
            "DeepSeek V4 top-k WSDPA disabled after dispatch failure; "
            "using stock SDPA",
            exc_info=True,
        )
        return None
