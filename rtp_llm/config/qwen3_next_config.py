import json
import os
from typing import Any, Dict, List

from rtp_llm.config.hybrid_kv_cache import build_hybrid_kv_cache_spec_descs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.mrope_utils import apply_mrope_section
from rtp_llm.model_factory_register import register_model_config
from rtp_llm.ops import HybridAttentionType, KVCacheSpecType


class Qwen3NextConfig:
    model_name = "Qwen3Next"

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        config_json = cls._preprocess_config_json(config_json)

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        cls._parse_basic_config(config_json, config)
        cls._parse_rope_config(config_json, config)
        cls._parse_normalization_config(config_json, config)
        cls._parse_moe_config(config_json, config)
        cls._parse_hybrid_attention_config(config_json, config)
        cls._parse_linear_attention_config(config_json, config)

        return config

    @classmethod
    def _preprocess_config_json(cls, config_json: dict) -> dict:
        return config_json

    @classmethod
    def _parse_basic_config(cls, config_json: dict, config: ModelConfig):
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
        config.attn_config.size_per_head = config_json["head_dim"]
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.max_seq_len = config_json["max_position_embeddings"]
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = config_json["rope_theta"]
        config.partial_rotary_factor = config_json["partial_rotary_factor"]
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )

    @classmethod
    def _parse_normalization_config(cls, config_json: dict, config: ModelConfig):
        config.layernorm_eps = config_json["rms_norm_eps"]
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.qk_norm = True
        config.activation_type = "SiGLU"

    @classmethod
    def _parse_moe_config(cls, config_json: Dict[str, Any], config: ModelConfig):
        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["num_experts"]
        config.moe_inter_size = config_json["moe_intermediate_size"]
        config.inter_size = config_json["shared_expert_intermediate_size"]
        config.has_moe_norm = config_json.get("norm_topk_prob", True)
        config.moe_style = 2

        moe_step = config_json.get("decoder_sparse_step", 1)
        config.moe_layer_index = [
            i for i in range(config.num_layers) if (i + 1) % moe_step == 0
        ]

    @classmethod
    def _parse_hybrid_attention_config(cls, config_json: dict, config: ModelConfig):
        attention_step = config_json["full_attention_interval"]
        config.hybrid_attention_config.enable_hybrid_attention = True
        hybrid_layer_types: List[HybridAttentionType] = []
        for i in range(config.num_layers):
            if (i + 1) % attention_step == 0:
                hybrid_layer_types.append(HybridAttentionType.NONE)
            else:
                hybrid_layer_types.append(HybridAttentionType.LINEAR)
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types

    @classmethod
    def _parse_linear_attention_config(cls, config_json: dict, config: ModelConfig):
        config.linear_attention_config.linear_conv_kernel_dim = config_json[
            "linear_conv_kernel_dim"
        ]
        config.linear_attention_config.linear_key_head_dim = config_json[
            "linear_key_head_dim"
        ]
        config.linear_attention_config.linear_num_key_heads = config_json[
            "linear_num_key_heads"
        ]
        config.linear_attention_config.linear_num_value_heads = config_json[
            "linear_num_value_heads"
        ]
        config.linear_attention_config.linear_value_head_dim = config_json[
            "linear_value_head_dim"
        ]

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        model_config.kv_cache_spec_descs = build_hybrid_kv_cache_spec_descs(
            model_config.hybrid_attention_config.hybrid_attention_types,
            KVCacheSpecType.MHA,
        )


class Qwen35MoeConfig(Qwen3NextConfig):
    model_name = "Qwen35Moe"

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        text_config_json = config_json["text_config"]

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        cls._parse_basic_config(text_config_json, config)
        cls._parse_rope_config(text_config_json, config)
        cls._parse_normalization_config(text_config_json, config)
        cls._parse_moe_config(text_config_json, config)
        cls._parse_hybrid_attention_config(text_config_json, config)
        cls._parse_linear_attention_config(text_config_json, config)
        cls._parse_mm_config(config_json, config)

        return config

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        rope_parameters = config_json["rope_parameters"]
        mrope_interleaved = rope_parameters.get("mrope_interleaved", True)
        if not mrope_interleaved:
            raise ValueError(
                "Qwen3Next requires rope_parameters.mrope_interleaved to be true"
            )
        config.attn_config.rope_config.style = 7
        config.attn_config.rope_config.base = rope_parameters["rope_theta"]
        config.partial_rotary_factor = rope_parameters["partial_rotary_factor"]
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )
        apply_mrope_section(
            config.attn_config.rope_config,
            rope_parameters["mrope_section"],
            model_name="Qwen3.5",
            interleaved=mrope_interleaved,
        )
        config.mm_model_config.mm_position_ids_style = 2

    @classmethod
    def _parse_mm_config(cls, config_json: dict, config: ModelConfig):
        config.mm_model_config.is_multimodal = True
        config.mm_model_config.mm_sep_tokens = [
            [config_json["vision_start_token_id"], config_json["vision_end_token_id"]]
        ]
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path


class Qwen35DenseConfig(Qwen35MoeConfig):
    model_name = "Qwen35Dense"

    @classmethod
    def _parse_moe_config(cls, config_json: Dict[str, Any], config: ModelConfig):
        config.inter_size = config_json["intermediate_size"]


register_model_config("qwen3_next", Qwen3NextConfig, ["Qwen3NextForCausalLM"])
register_model_config(
    "qwen35_moe", Qwen35MoeConfig, ["Qwen3_5MoeForConditionalGeneration"]
)
register_model_config(
    "qwen35_dense", Qwen35DenseConfig, ["Qwen3_5ForConditionalGeneration"]
)
