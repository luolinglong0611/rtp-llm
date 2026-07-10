"""Triton input packer for DeepGEMM ``fp4_fp4_mega_moe``."""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _ceil_ue4m3_code(x):
        x = tl.minimum(tl.maximum(x, 0.0), 448.0)

        sub_code = tl.ceil(x * 512.0).to(tl.int32)
        sub_code = tl.minimum(tl.maximum(sub_code, 1), 7)

        min_normal = 0.015625
        xn = tl.maximum(x, min_normal)
        e = tl.floor(tl.log2(xn))
        exp_code = e.to(tl.int32) + 7
        base = tl.exp2(e)
        mant = tl.ceil((xn / base - 1.0) * 8.0).to(tl.int32)
        mant = tl.maximum(mant, 0)
        overflow = mant > 7
        exp_code = exp_code + overflow.to(tl.int32)
        mant = tl.where(overflow, 0, mant)
        exp_code = tl.minimum(tl.maximum(exp_code, 1), 15)
        normal_code = (exp_code << 3) | mant
        normal_code = tl.minimum(normal_code, 126)
        return tl.where(x <= 0.013671875, sub_code, normal_code)

    @triton.jit
    def _ue4m3_code_to_float(code):
        exp = (code >> 3) & 0x0F
        mant = code & 0x07
        subnormal = mant.to(tl.float32) * 0.001953125
        normal = (1.0 + mant.to(tl.float32) * 0.125) * tl.exp2(exp.to(tl.float32) - 7.0)
        return tl.where(exp == 0, subnormal, normal)

    @triton.jit
    def _quantize_e2m1(x):
        ax = tl.minimum(tl.abs(x), 6.0)
        code = tl.zeros(ax.shape, dtype=tl.int32)
        code += (ax > 0.25).to(tl.int32)
        code += (ax > 0.75).to(tl.int32)
        code += (ax > 1.25).to(tl.int32)
        code += (ax > 1.75).to(tl.int32)
        code += (ax > 2.5).to(tl.int32)
        code += (ax > 3.5).to(tl.int32)
        code += (ax > 5.0).to(tl.int32)
        sign = ((x < 0.0) & (code != 0)).to(tl.int32)
        return code | (sign << 3)

    @triton.jit(do_not_specialize=["M"])
    def _pack_mega_moe_fp4_inputs_kernel(
        x_ptr,
        weights_ptr,
        indices_ptr,
        out_fp4_ptr,
        out_sf_ptr,
        out_weights_ptr,
        out_indices_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        x_stride_m: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m_blk = tl.program_id(0).to(tl.int64)
        pid_blk = tl.program_id(1)

        offs_m = pid_m_blk * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        row_mask = offs_m < M
        pair = tl.arange(0, 8)
        cols16 = tl.arange(0, 16)
        packed_sf = tl.zeros((BLOCK_M,), dtype=tl.int32)

        for pack_idx in tl.static_range(4):
            base_col = pid_blk * 64 + pack_idx * 16
            cols = base_col + cols16
            mask16 = row_mask[:, None] & (cols[None, :] < N)
            x16 = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + cols[None, :],
                mask=mask16,
                other=0.0,
            ).to(tl.float32)

            block_absmax = tl.maximum(tl.max(tl.abs(x16), axis=1), eps)
            sf_code = _ceil_ue4m3_code(block_absmax / 6.0)
            sf = _ue4m3_code_to_float(sf_code)
            packed_sf = packed_sf | (sf_code << (pack_idx * 8))

            even_cols = base_col + pair * 2
            odd_cols = even_cols + 1
            mask8_even = row_mask[:, None] & (even_cols[None, :] < N)
            mask8_odd = row_mask[:, None] & (odd_cols[None, :] < N)
            x_even = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + even_cols[None, :],
                mask=mask8_even,
                other=0.0,
            ).to(tl.float32)
            x_odd = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + odd_cols[None, :],
                mask=mask8_odd,
                other=0.0,
            ).to(tl.float32)
            q_even = _quantize_e2m1(x_even / sf[:, None])
            q_odd = _quantize_e2m1(x_odd / sf[:, None])
            packed = (q_even & 0x0F) | ((q_odd & 0x0F) << 4)
            out_cols = pid_blk * 32 + pack_idx * 8 + pair
            tl.store(
                out_fp4_ptr + offs_m[:, None] * out_stride_m + out_cols[None, :],
                packed,
                mask=row_mask[:, None] & (out_cols[None, :] < (N // 2)),
            )

        tl.store(out_sf_ptr + offs_m * sf_stride_m + pid_blk, packed_sf, mask=row_mask)

        if pid_blk == 0:
            router_offs = tl.arange(0, BLOCK_K)
            router_mask = row_mask[:, None] & (router_offs[None, :] < K)
            w = tl.load(
                weights_ptr + offs_m[:, None] * weights_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0.0,
            ).to(tl.float32)
            idx = tl.load(
                indices_ptr + offs_m[:, None] * indices_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0,
            ).to(tl.int64)
            tl.store(
                out_weights_ptr
                + offs_m[:, None] * out_weights_stride_m
                + router_offs[None, :],
                w,
                mask=router_mask,
            )
            tl.store(
                out_indices_ptr
                + offs_m[:, None] * out_indices_stride_m
                + router_offs[None, :],
                idx,
                mask=router_mask,
            )


def _validate_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp4: torch.Tensor,
    out_sf: torch.Tensor,
) -> tuple[int, int, int]:
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if not x.is_cuda:
        raise RuntimeError("fused MegaMoE FP4 input packer requires CUDA tensors")
    if x.dim() != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bfloat16, got {x.dtype}")
    if weights.shape != indices.shape:
        raise ValueError("weights and indices must have identical [T,topk] shape")
    if weights.dtype != torch.float32:
        raise ValueError(f"weights must be float32, got {weights.dtype}")
    if indices.dtype != torch.int64:
        raise ValueError(f"indices must be int64, got {indices.dtype}")
    T, D = x.shape
    if D % 64 != 0:
        raise ValueError(f"MegaMoE FP4 packer requires D % 64 == 0, got D={D}")
    if out_fp4.shape[1] != D // 2:
        raise ValueError(
            f"out_fp4 shape mismatch: expected second dim {D // 2}, got {out_fp4.shape}"
        )
    if out_sf.shape[1] != D // 64:
        raise ValueError(
            f"out_sf shape mismatch: expected second dim {D // 64}, got {out_sf.shape}"
        )
    return T, D, weights.shape[1]


def fused_pack_mega_moe_fp4_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp4: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    T, D, topk = _validate_inputs(x, weights, indices, out_fp4, out_sf)
    if T == 0:
        return
    block_k = triton.next_power_of_2(topk)
    block_m_env = os.environ.get("DSV4_MEGA_MOE_FP4_PACK_BLOCK_M")
    block_m = int(block_m_env) if block_m_env is not None else (8 if T >= 2048 else 2)
    if block_m not in (1, 2, 4, 8):
        raise ValueError(
            f"invalid DSV4_MEGA_MOE_FP4_PACK_BLOCK_M={block_m}; "
            "expected 1, 2, 4, or 8"
        )
    grid = (triton.cdiv(T, block_m), triton.cdiv(D, 64))
    _pack_mega_moe_fp4_inputs_kernel[grid](
        x,
        weights,
        indices,
        out_fp4,
        out_sf,
        out_weights,
        out_indices,
        T,
        D,
        topk,
        x.stride(0),
        weights.stride(0),
        indices.stride(0),
        out_fp4.stride(0),
        out_sf.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        1.0e-4,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
