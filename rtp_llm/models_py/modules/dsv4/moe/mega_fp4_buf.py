"""Symm-mem buffer cache + capability gates for ``fp4_fp4_mega_moe``."""

from __future__ import annotations

import logging
import os

import torch

_MEGA_FP4_BUF_CACHE: dict = {}
_MEGA_FP4_OUTPUT_CACHE: dict = {}

_USE_MEGA_MOE_FP4_ENV = "DSV4_USE_MEGA_MOE_FP4"


def mega_moe_fp4_requested() -> bool:
    return os.environ.get(_USE_MEGA_MOE_FP4_ENV, "0") == "1"


def estimate_mega_moe_fp4_symm_buffer_bytes(
    group_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation: str = "swiglu",
) -> int | None:
    try:
        import deep_gemm

        return int(
            deep_gemm._C.get_symm_buffer_size_for_mega_moe_fp4(
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                activation,
            )[0]
        )
    except Exception:
        return None


def _get_or_create_mega_fp4_buf(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    activation,
):
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation,
    )
    buf = _MEGA_FP4_BUF_CACHE.get(key)
    if buf is None:
        try:
            group_size = int(group.size())
        except Exception:
            group_size = 0
        estimated_bytes = (
            estimate_mega_moe_fp4_symm_buffer_bytes(
                group_size=group_size,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                activation=activation,
            )
            if group_size > 0
            else None
        )
        buf = deep_gemm.get_symm_buffer_for_mega_moe_fp4(
            group=group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            activation=activation,
        )
        actual_bytes = None
        try:
            actual_bytes = int(buf.buffer.numel() * buf.buffer.element_size())
        except Exception:
            pass
        if actual_bytes is not None and estimated_bytes is not None:
            logging.info(
                "[DSV4 MegaMoEFP4] allocated symm buffer: group_size=%d "
                "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
                "intermediate=%d actual=%.3f GiB estimated=%.3f GiB",
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                actual_bytes / (1024**3),
                estimated_bytes / (1024**3),
            )
        elif actual_bytes is not None:
            logging.info(
                "[DSV4 MegaMoEFP4] allocated symm buffer: group_size=%d "
                "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
                "intermediate=%d actual=%.3f GiB",
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                actual_bytes / (1024**3),
            )
        _MEGA_FP4_BUF_CACHE[key] = buf
    return buf


def _get_or_create_mega_fp4_output(
    capacity,
    hidden,
    dtype,
    device,
):
    key = (device, hidden, dtype)
    cached = _MEGA_FP4_OUTPUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    _MEGA_FP4_OUTPUT_CACHE[key] = cached
    return cached


def _mega_moe_fp4_unavailable_reason() -> str | None:
    try:
        import deep_gemm

        for sym in (
            "fp4_fp4_mega_moe",
            "get_symm_buffer_for_mega_moe_fp4",
            "transform_weights_for_mega_moe_fp4",
        ):
            if not hasattr(deep_gemm, sym):
                return f"deep_gemm.{sym} is missing"
    except Exception as e:
        return f"failed to import deep_gemm: {e}"
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return "torch.distributed is not initialized"
        if dist.get_world_size() <= 1:
            return f"distributed world_size={dist.get_world_size()} is not > 1"
    except Exception as e:
        return f"failed to query torch.distributed: {e}"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        return f"CUDA device capability sm{cap[0]}{cap[1]} is below SM100"
    return None


def _mega_moe_fp4_available() -> bool:
    return _mega_moe_fp4_unavailable_reason() is None


def _mega_moe_fp4_enabled() -> bool:
    return mega_moe_fp4_requested() and _mega_moe_fp4_available()


def _mega_moe_fp4_disabled_or_unavailable_reason() -> str:
    if not mega_moe_fp4_requested():
        return f"{_USE_MEGA_MOE_FP4_ENV}=1 is not set"
    return _mega_moe_fp4_unavailable_reason() or "unknown Mega MoE FP4 failure"
