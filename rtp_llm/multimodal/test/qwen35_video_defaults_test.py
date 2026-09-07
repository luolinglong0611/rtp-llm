"""CPU contracts for Qwen3.5 video defaults and request-level overrides."""

import io
import pickle
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins import qwen3_vl_mixin as qwen3
from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl import qwen2_5_vl_mixin as qwen25
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import qwen3_5_moe_mixin as qwen35
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Qwen35VideoDefaultsTest(unittest.TestCase):
    def resolve(self, config):
        with mock.patch.object(
            qwen25.Qwen2_5_VLImageEmbedding, "load_video", return_value="video"
        ) as loader:
            result = qwen35.Qwen3_5MoeImageEmbedding.load_video("data", config)
        self.assertEqual(result, "video")
        self.assertEqual(loader.call_args.args[0], "data")
        self.assertFalse(loader.call_args.kwargs["apply_video_pixel_budget"])
        return loader.call_args.args[1]

    def test_missing_settings_use_video_defaults_without_mutating_request(self):
        config = MMPreprocessConfig()
        original = config.to_string()
        actual = self.resolve(config)
        self.assertEqual(actual.fps, 6)
        self.assertEqual(actual.max_frames, 90)
        self.assertEqual(actual.min_pixels, 3136)
        self.assertEqual(actual.max_pixels, 4194304)
        self.assertEqual(actual.min_frames, -1)
        self.assertEqual(config.to_string(), original)

    def test_partial_override_only_replaces_supplied_default(self):
        actual = self.resolve(MMPreprocessConfig(fps=3, max_pixels=65536))
        self.assertEqual(actual.fps, 3)
        self.assertEqual(actual.max_pixels, 65536)
        self.assertEqual(actual.max_frames, 90)
        self.assertEqual(actual.min_pixels, 3136)

    def test_explicit_settings_and_unrelated_fields_are_preserved(self):
        config = MMPreprocessConfig(
            width=320,
            height=224,
            min_pixels=1024,
            max_pixels=131072,
            fps=2,
            min_frames=2,
            max_frames=16,
            crop_positions=[0.1, 0.2, 0.5, 0.6],
            mm_timeout_ms=12345,
        )
        self.assertEqual(self.resolve(config).to_string(), config.to_string())

    def test_sampling_frame_count_and_cap(self):
        config = self.resolve(MMPreprocessConfig())
        for total_frames, fps, expected in (
            (453, 30, 90),  # The first historical timeout input: 15.1 seconds.
            (3600, 30, 90),
            (30, 30, 6),
            (4, 30, 4),
            (2, 30, 2),
        ):
            with self.subTest(total_frames=total_frames):
                self.assertEqual(
                    qwen25.smart_nframes(config, total_frames, fps), expected
                )

    def test_explicit_sampling_still_overrides_defaults(self):
        config = self.resolve(MMPreprocessConfig(fps=2, max_frames=20))
        self.assertEqual(qwen25.smart_nframes(config, 453, 30), 20)

    def test_qwen25_sampling_defaults_are_unchanged(self):
        self.assertEqual(qwen25.smart_nframes(MMPreprocessConfig(), 453, 30), 30)
        self.assertEqual(qwen25.FPS_MAX_FRAMES, 768)

    def load_decoded(self, embedding_class, shape, config=None):
        video = torch.zeros(shape, dtype=torch.uint8)
        with (
            mock.patch.object(qwen25, "VideoReader", None),
            mock.patch.object(qwen25, "_load_video_with_pyav", return_value=video),
        ):
            return embedding_class.load_video(
                None, config if config is not None else MMPreprocessConfig()
            )

    def test_qwen35_pixel_range_is_not_clamped_by_qwen25_budget(self):
        shape = (2, 3, 768, 1024)
        actual = self.load_decoded(qwen35.Qwen3_5MoeImageEmbedding, shape)
        pixels = actual.shape[-2] * actual.shape[-1]
        self.assertGreater(pixels, qwen25.VIDEO_MAX_PIXELS)
        self.assertLessEqual(pixels, 4194304)
        legacy = self.load_decoded(qwen25.Qwen2_5_VLImageEmbedding, shape)
        self.assertLessEqual(
            legacy.shape[-2] * legacy.shape[-1], qwen25.VIDEO_MAX_PIXELS
        )

    def test_video_min_pixels_does_not_use_legacy_upscale_floor(self):
        actual = self.load_decoded(qwen35.Qwen3_5MoeImageEmbedding, (2, 3, 28, 28))
        self.assertEqual(tuple(actual.shape[-2:]), (56, 56))

    def test_explicit_pixel_cap_is_respected_by_loader(self):
        actual = self.load_decoded(
            qwen35.Qwen3_5MoeImageEmbedding,
            (2, 3, 256, 256),
            MMPreprocessConfig(max_pixels=16384),
        )
        self.assertLessEqual(actual.shape[-2] * actual.shape[-1], 16384)

    def test_constructor_sets_video_processor_defaults_only(self):
        processor = SimpleNamespace(
            video_processor=SimpleNamespace(do_sample_frames=True, size={}),
            image_processor=None,
        )
        image_processor = object()
        with (
            mock.patch.object(
                qwen35.AutoProcessor, "from_pretrained", return_value=processor
            ),
            mock.patch.object(
                qwen35.Qwen2VLImageProcessor,
                "from_pretrained",
                return_value=image_processor,
            ),
            mock.patch.object(
                qwen35.Qwen3_5MoeVisionConfig,
                "from_pretrained",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(qwen35.Qwen3_5MoeVisionModel, "_from_config"),
        ):
            embedding = qwen35.Qwen3_5MoeImageEmbedding(
                SimpleNamespace(config={"ckpt_path": "unused"})
            )
        self.assertFalse(embedding.mm_processor.video_processor.do_sample_frames)
        self.assertEqual(
            embedding.mm_processor.video_processor.size,
            {"longest_edge": 36864000, "shortest_edge": 2500000},
        )
        self.assertIs(embedding.mm_processor.image_processor, image_processor)

    def test_inherited_preprocess_dispatches_to_qwen35_video_loader(self):
        config = MMPreprocessConfig()
        item = MultimodalInput("unused", MMUrlType.VIDEO, torch.empty(0), config)
        processor = mock.Mock()
        processor.video_processor.return_value = {
            "pixel_values_videos": "patches",
            "video_grid_thw": "grid",
        }
        with (
            mock.patch.object(qwen3, "get_bytes_io_from_url", return_value="bytes"),
            mock.patch.object(
                qwen35.Qwen3_5MoeImageEmbedding, "load_video", return_value="video"
            ) as loader,
        ):
            result = qwen35.Qwen3_5MoeImageEmbedding.preprocess_input(
                [item], VitConfig(), processor
            )
        self.assertEqual(result, ("patches", "grid"))
        self.assertEqual(loader.call_args.args[0], "bytes")
        self.assertEqual(loader.call_args.args[1].to_string(), config.to_string())
        processor.video_processor.assert_called_once_with(
            "video", return_tensors="pt", do_resize=True
        )

    def test_image_path_does_not_apply_video_defaults(self):
        data = io.BytesIO()
        Image.new("RGB", (32, 32)).save(data, format="PNG")
        data.seek(0)
        config = MMPreprocessConfig()
        item = MultimodalInput("unused", MMUrlType.IMAGE, torch.empty(0), config)
        processor = mock.Mock()
        processor.image_processor.return_value = {
            "pixel_values": "image_patches",
            "image_grid_thw": "image_grid",
        }
        with (
            mock.patch.object(qwen3, "get_bytes_io_from_url", return_value=data),
            mock.patch.object(qwen35.Qwen3_5MoeImageEmbedding, "load_video") as loader,
        ):
            result = qwen35.Qwen3_5MoeImageEmbedding.preprocess_input(
                [item], VitConfig(), processor
            )
        self.assertEqual(result, ("image_patches", "image_grid"))
        loader.assert_not_called()
        processor.video_processor.assert_not_called()
        self.assertTrue(processor.image_processor.call_args.kwargs["do_resize"])
        self.assertEqual(config.to_string(), MMPreprocessConfig().to_string())

    def test_preprocess_remains_picklable_for_spawn_workers(self):
        callback = qwen35.Qwen3_5MoeImageEmbedding.preprocess_input
        restored = pickle.loads(pickle.dumps(callback))
        self.assertIs(restored.__self__, qwen35.Qwen3_5MoeImageEmbedding)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
