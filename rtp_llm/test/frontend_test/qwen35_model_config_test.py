import json
import os
import tempfile
import unittest

from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import _model_config_factory, _model_factory


class Qwen35ModelConfigTest(unittest.TestCase):
    def test_frontend_builds_qwen35_config_without_model_package(self):
        self.assertNotIn("qwen35_moe", _model_factory)
        self.assertIn("qwen35_moe", _model_config_factory)

        config_json = {
            "vision_start_token_id": 101,
            "vision_end_token_id": 102,
            "text_config": {
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "num_hidden_layers": 4,
                "hidden_size": 64,
                "vocab_size": 128,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-6,
                "num_experts_per_tok": 2,
                "num_experts": 8,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 64,
                "full_attention_interval": 2,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_value_head_dim": 8,
                "rope_parameters": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [2, 2, 4],
                    "mrope_interleaved": True,
                },
            },
        }

        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(
                os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8"
            ) as writer:
                json.dump(config_json, writer)

            config_cls = ModelFactory.get_model_config_cls("qwen35_moe")
            model_config = config_cls._create_config(ckpt_path)
            config_cls._post_build_model_config(model_config)

        self.assertEqual(config_cls.model_name, "Qwen35Moe")
        self.assertEqual(model_config.num_layers, 4)
        self.assertEqual(model_config.expert_num, 8)
        self.assertTrue(model_config.mm_model_config.is_multimodal)
        self.assertEqual(model_config.mm_model_config.mm_sep_tokens, [[101, 102]])
        self.assertEqual(len(model_config.kv_cache_spec_descs), 4)


if __name__ == "__main__":
    unittest.main()
