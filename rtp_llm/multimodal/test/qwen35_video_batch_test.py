"""Check video/image batching against independent real vision forwards."""

import unittest
from unittest import mock

import torch

from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Qwen35VideoBatchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)
        config = Qwen3_5MoeVisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=16,
        )
        config._attn_implementation = "eager"
        self.embedding = object.__new__(Qwen3_5MoeImageEmbedding)
        self.embedding.visual = Qwen3_5MoeVisionModel(config).eval()
        # This CPU contract test must not select the GPU-only FlashAttention
        # branch merely because a CUDA device is present on the test host.
        patcher = mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe."
            "qwen3_5_moe_vit.default_attn_impl",
            "sdpa",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def data(self, t, h, w):
        # RGB * temporal_patch_size * patch_size**2 values per patch.
        return torch.randn(t * h * w, 24), torch.tensor([[t, h, w]])

    def assert_batch_matches_serial(self, data, types):
        expected = [self.embedding.embedding(d, mm_type=t) for d, t in zip(data, types)]
        with mock.patch.object(
            self.embedding.visual, "forward", wraps=self.embedding.visual.forward
        ) as forward:
            actual = self.embedding.batched_embedding(data, types)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            features, positions = result
            ref_features, ref_positions = reference
            torch.testing.assert_close(features, ref_features, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(positions, ref_positions, atol=0, rtol=0)
            self.assertEqual(features.shape[0], positions.shape[0])

    def test_two_videos_with_different_temporal_and_spatial_extents(self):
        self.assert_batch_matches_serial(
            [self.data(3, 4, 4), self.data(2, 2, 4)],
            [MMUrlType.VIDEO, MMUrlType.VIDEO],
        )

    def test_mixed_image_and_video_keep_independent_positions(self):
        self.assert_batch_matches_serial(
            [self.data(1, 2, 4), self.data(3, 4, 2)],
            [MMUrlType.IMAGE, MMUrlType.VIDEO],
        )

    def test_images_still_batch(self):
        self.assert_batch_matches_serial(
            [self.data(1, 4, 4), self.data(1, 2, 4)],
            [MMUrlType.IMAGE, MMUrlType.IMAGE],
        )

    def test_default_single_video_uses_serial_path_without_concat(self):
        data = self.data(2, 4, 4)
        expected = self.embedding.embedding(data, mm_type=MMUrlType.VIDEO)
        with mock.patch(
            "torch.concat", side_effect=AssertionError("unexpected concat")
        ):
            actual = self.embedding.batched_embedding([data], [MMUrlType.VIDEO])
        for actual_tensor, expected_tensor in zip(actual[0], expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)

    def test_empty_batch(self):
        self.assertEqual(self.embedding.batched_embedding([], []), [])

    def test_mismatched_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            self.embedding.batched_embedding([self.data(1, 2, 2)], [])

    def check_sdpa_path(self, data, expected_calls):
        self.embedding.visual.config._attn_implementation = "sdpa"
        for block in self.embedding.visual.blocks:
            block.attn.batched_sdpa = False
        types = [MMUrlType.VIDEO] * len(data)
        reference = self.embedding.batched_embedding(data, types)
        for block in self.embedding.visual.blocks:
            block.attn.batched_sdpa = True
        interface = mock.Mock(wraps=ALL_ATTENTION_FUNCTIONS.get("sdpa"))
        registry = mock.Mock()
        registry.get.return_value = interface
        with mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe."
            "qwen3_5_moe_vit.ALL_ATTENTION_FUNCTIONS",
            registry,
        ):
            actual = self.embedding.batched_embedding(data, types)
        self.assertEqual(interface.call_count, expected_calls)
        for (features, positions), (ref_features, ref_positions) in zip(
            actual, reference
        ):
            torch.testing.assert_close(features, ref_features, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(positions, ref_positions, atol=0, rtol=0)

    def test_uniform_frames_use_one_sdpa_call_per_layer(self):
        self.check_sdpa_path([self.data(3, 4, 4), self.data(2, 4, 4)], 2)

    def test_variable_frames_keep_per_frame_sdpa_fallback(self):
        self.check_sdpa_path([self.data(3, 4, 4), self.data(2, 2, 4)], 10)


if __name__ == "__main__":
    unittest.main()
