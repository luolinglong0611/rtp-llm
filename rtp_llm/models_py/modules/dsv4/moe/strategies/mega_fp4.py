"""MegaMoEFP4Strategy: DeepGEMM ``fp4_fp4_mega_moe`` symm-mem kernel."""

from __future__ import annotations

from typing import Dict

import torch

from ...quant_layouts import NVFP4_BLOCK, prepare_nvfp4_weight_scale_for_deepgemm
from ..input_packer_fp4 import get_mega_moe_fp4_input_packer
from ..mega_fp4_buf import (
    _get_or_create_mega_fp4_buf,
    _get_or_create_mega_fp4_output,
    _mega_moe_fp4_enabled,
)
from ..warmup_sync import sync_cuda_graph_warmup_ranks
from .base import MoeCfg, register_strategy
from .mega import MegaMoEStrategy, _mega_output_capacity


@register_strategy
class MegaMoEFP4Strategy(MegaMoEStrategy):
    name = "mega_fp4"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return cfg.ep_size > 1 and _mega_moe_fp4_enabled()

    def can_use_gate_pack_static(self, gate) -> bool:
        # The existing gate-pack kernels quantize activations to FP8/UE8M0.
        # ``fp4_fp4_mega_moe`` needs packed NVFP4/UE4M3, so keep the regular
        # gate -> FP4 input-packer path.
        return False

    def setup_weights(self, layer_weights: Dict) -> None:
        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim

        st_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        st_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        st_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        st_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = st_w1_w.device
        if st_w1_s.dtype != torch.int32 or st_w3_s.dtype != torch.int32:
            raise TypeError(
                "Mega MoE FP4 requires routed w1/w3 scale tensors to be "
                f"packed UE4M3 int32; got {st_w1_s.dtype} / {st_w3_s.dtype}. "
                "Use an NVFP4 checkpoint."
            )

        w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
        s13_raw = torch.empty(
            (E, 2 * inter, D // (NVFP4_BLOCK * 4)),
            dtype=torch.int32,
            device=device,
        )
        w13[:, :inter].copy_(st_w1_w)
        s13_raw[:, :inter].copy_(st_w1_s)
        w13[:, inter:].copy_(st_w3_w)
        s13_raw[:, inter:].copy_(st_w3_s)
        del st_w1_w, st_w1_s, st_w3_w, st_w3_s
        s13_int = prepare_nvfp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        del s13_raw
        torch.cuda.empty_cache()

        st_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        st_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        if st_w2_s.dtype != torch.int32:
            raise TypeError(
                "Mega MoE FP4 requires routed w2 scale tensor to be packed "
                f"UE4M3 int32; got {st_w2_s.dtype}. Use an NVFP4 checkpoint."
            )
        w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // (NVFP4_BLOCK * 4)),
            dtype=torch.int32,
            device=device,
        )
        w2.copy_(st_w2_w)
        s2_raw.copy_(st_w2_s)
        del st_w2_w, st_w2_s
        s2_int = prepare_nvfp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        del s2_raw
        torch.cuda.empty_cache()

        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe_fp4(
            (w13, s13_int),
            (w2, s2_int),
        )
        del w13, s13_int, w2, s2_int
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf

        assert dist.is_initialized(), (
            "Mega MoE FP4 requires torch.distributed initialised; "
            "_mega_moe_fp4_available() should have gated this earlier"
        )
        group = dist.group.WORLD
        self._mega_group = group
        self._mega_buf = _get_or_create_mega_fp4_buf(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=D,
            intermediate_hidden=inter,
            activation="swiglu",
        )
        self._mega_y = _get_or_create_mega_fp4_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            D,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_moe_fp4_input_packer()
        self._maybe_warmup_jit_once()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        import deep_gemm

        T = x.size(0)
        buf = self._mega_buf
        if T > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE FP4 input tokens={T} exceeds num_max_tokens_per_rank="
                f"{buf.num_max_tokens_per_rank} (derived from max_seq_len / "
                f"max_tokens_per_rank). Raise the budget at startup."
            )
        if T > self._mega_y.size(0):
            raise RuntimeError(
                f"Mega MoE FP4 output buffer rows={self._mega_y.size(0)} is "
                f"smaller than input tokens={T}. This indicates inconsistent "
                "aligned MegaMoE FP4 buffer sizing."
            )

        self._input_packer.pack(x, weights, indices, buf, T)
        self._maybe_pre_kernel_barrier(T)
        sync_cuda_graph_warmup_ranks(
            f"dsv4.mega_moe_fp4.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:T]
        deep_gemm.fp4_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, NVFP4_BLOCK),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
            assume_all_topk_valid=False,
        )
        return y

    def forward_with_gate_pack(
        self,
        x: torch.Tensor,
        gate,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        raise RuntimeError("MegaMoE FP4 does not support FP8 gate-pack kernels")
