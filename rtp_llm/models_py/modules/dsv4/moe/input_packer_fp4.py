"""MegaMoE FP4 input packer abstraction."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch

from .quant_layouts import _per_token_cast_to_nvfp4_packed_ue4m3
from .shared_expert import strict_fused_moe_enabled


class MegaMoeFP4InputPacker(ABC):
    """Pack routed inputs into a ``fp4_fp4_mega_moe`` symmetric buffer."""

    name: str

    @abstractmethod
    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        raise NotImplementedError


class TorchMegaMoeFP4InputPacker(MegaMoeFP4InputPacker):
    name = "torch"

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        if strict_fused_moe_enabled():
            raise RuntimeError(
                "DSV4_MOE_STRICT_FUSED=1 forbids TorchMegaMoeFP4InputPacker"
            )
        x_fp4, x_sf = _per_token_cast_to_nvfp4_packed_ue4m3(x.contiguous())
        buf.x[:tokens].copy_(x_fp4)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())


class FusedMegaMoeFP4InputPacker(MegaMoeFP4InputPacker):
    name = "fused"

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 64 == 0):
            raise RuntimeError(
                "DSV4 fused MegaMoE FP4 input packer requires CUDA bf16 input "
                f"with hidden dim divisible by 64; got device={x.device}, "
                f"dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from ._mega_fp4_input_pack_triton import fused_pack_mega_moe_fp4_inputs

        fused_pack_mega_moe_fp4_inputs(
            x,
            weights,
            indices,
            buf.x[:tokens],
            buf.x_sf[:tokens],
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
        )


def _mode() -> str:
    return os.environ.get("DSV4_MEGA_MOE_FP4_INPUT_PACKER", "fused").strip().lower()


def get_mega_moe_fp4_input_packer() -> MegaMoeFP4InputPacker:
    mode = _mode()
    if mode == "torch":
        if strict_fused_moe_enabled():
            raise RuntimeError(
                "DSV4_MOE_STRICT_FUSED=1 forbids "
                "DSV4_MEGA_MOE_FP4_INPUT_PACKER=torch"
            )
        return TorchMegaMoeFP4InputPacker()
    if mode in ("auto", "fused"):
        return FusedMegaMoeFP4InputPacker()
    raise ValueError(
        f"invalid DSV4_MEGA_MOE_FP4_INPUT_PACKER={mode!r}; " "expected auto|torch|fused"
    )
