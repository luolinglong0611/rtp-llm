"""Compatibility re-export for DSV4 quant layout helpers."""

from rtp_llm.models_py.modules.dsv4.quant_layouts import (  # noqa: F401
    FP4_BLOCK,
    FP8_BLOCK,
    NVFP4_BLOCK,
    _per_token_cast_to_fp8_packed_ue8m0,
    _per_token_cast_to_nvfp4_packed_ue4m3,
    prepare_fp4_weight_scale_for_deepgemm,
    prepare_nvfp4_weight_scale_for_deepgemm,
)
