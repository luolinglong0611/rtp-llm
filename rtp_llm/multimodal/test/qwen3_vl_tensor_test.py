from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch
from transformers import Qwen2VLImageProcessor

from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.multimodal_mixins import qwen3_vl_mixin
from rtp_llm.multimodal.vit_metrics import collect_vit_preprocess_metrics
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Qwen3VLTensorTest(TestCase):
    @staticmethod
    def _mm_input(tensor, mm_type=MMUrlType.IMAGE, url=""):
        return SimpleNamespace(
            mm_type=mm_type,
            url=url,
            tensor=tensor,
            mm_preprocess_config=SimpleNamespace(
                height=1024,
                width=1024,
                min_pixels=1024,
                max_pixels=2048,
            ),
        )

    def test_cpu_contiguous_uint8_hwc_tensor_skips_fetch_and_resize(self):
        tensor = (
            torch.arange(32 * 64 * 3, dtype=torch.int32)
            .to(torch.uint8)
            .reshape(32, 64, 3)
        )
        processor = SimpleNamespace(
            image_processor=MagicMock(
                return_value={
                    "pixel_values": torch.zeros((1, 3, 32, 64)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                }
            )
        )

        with patch.object(
            qwen3_vl_mixin, "get_bytes_io_from_url"
        ) as fetch, patch.object(qwen3_vl_mixin.Image, "fromarray") as fromarray:
            with collect_vit_preprocess_metrics() as metrics:
                pixel_values, grid = (
                    qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                        [self._mm_input(tensor)],
                        SimpleNamespace(download_headers={}),
                        processor,
                    )
                )

        fetch.assert_not_called()
        fromarray.assert_not_called()
        self.assertEqual(tuple(pixel_values.shape), (1, 3, 32, 64))
        self.assertEqual(grid.tolist(), [[1, 1, 1]])
        image = processor.image_processor.call_args.args[0]
        self.assertIs(image, tensor)
        processor.image_processor.assert_called_once_with(
            tensor,
            return_tensors="pt",
            do_resize=False,
            input_data_format="channels_last",
        )

        samples = {sample.metric: sample for sample in metrics.samples}
        self.assertNotIn(GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC, samples)
        self.assertEqual(
            samples[GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC].value, 2048
        )
        self.assertEqual(
            samples[GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC].tags,
            {"model": "qwen3_vl", "mm_type": "image"},
        )

    def test_transformers_5_2_processor_accepts_hwc_torch_tensor(self):
        image_processor = Qwen2VLImageProcessor(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
        )
        tensor = torch.zeros((32, 32, 3), dtype=torch.uint8)

        result = image_processor(
            tensor,
            return_tensors="pt",
            do_resize=False,
            input_data_format="channels_last",
        )

        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 2]])
        self.assertEqual(tuple(result["pixel_values"].shape), (4, 1536))

    def test_default_image_type_also_accepts_tensor(self):
        processor = SimpleNamespace(
            image_processor=MagicMock(
                return_value={
                    "pixel_values": torch.zeros((1, 3, 32, 32)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                }
            )
        )

        qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
            [
                self._mm_input(
                    torch.zeros((32, 32, 3), dtype=torch.uint8),
                    MMUrlType.DEFAULT,
                )
            ],
            SimpleNamespace(download_headers={}),
            processor,
        )

        self.assertFalse(processor.image_processor.call_args.kwargs["do_resize"])
        self.assertEqual(
            processor.image_processor.call_args.kwargs["input_data_format"],
            "channels_last",
        )

    def test_rejects_invalid_tensor_schema(self):
        invalid_tensors = (
            torch.zeros((32, 32, 3), dtype=torch.float32),
            torch.zeros((3, 4, 5), dtype=torch.uint8),
            torch.zeros((32, 32, 3), dtype=torch.uint8).transpose(0, 1),
        )
        processor = SimpleNamespace(image_processor=MagicMock())

        for tensor in invalid_tensors:
            with self.subTest(
                shape=tensor.shape, dtype=tensor.dtype
            ), self.assertRaises(ValueError):
                qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                    [self._mm_input(tensor)],
                    SimpleNamespace(download_headers={}),
                    processor,
                )

    def test_rejects_non_aligned_or_oversized_tensor(self):
        processor = SimpleNamespace(image_processor=MagicMock())
        non_aligned = torch.zeros((33, 32, 3), dtype=torch.uint8)
        oversized = torch.as_strided(
            torch.empty(1, dtype=torch.uint8),
            size=(4800, 4928, 3),
            stride=(0, 0, 0),
        )

        with self.assertRaisesRegex(ValueError, "divisible"):
            qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                [self._mm_input(non_aligned)],
                SimpleNamespace(download_headers={}),
                processor,
            )
        with self.assertRaisesRegex(ValueError, "exceeds the maximum"):
            qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                [self._mm_input(oversized)],
                SimpleNamespace(download_headers={}),
                processor,
            )

    def test_rejects_url_and_tensor_together(self):
        with self.assertRaisesRegex(ValueError, "both url and tensor"):
            qwen3_vl_mixin.Qwen3_VLImageEmbedding.preprocess_input(
                [
                    self._mm_input(
                        torch.zeros((32, 32, 3), dtype=torch.uint8),
                        url="memory://image",
                    )
                ],
                SimpleNamespace(download_headers={}),
                SimpleNamespace(image_processor=MagicMock()),
            )


if __name__ == "__main__":
    main()
