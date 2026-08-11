import base64
import io
import logging
import tempfile
import unittest

import torch
from PIL import Image

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    TensorPB,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url, trans_mm_input


class TestMultiModalUtil(unittest.TestCase):
    def test_trans_mm_input_accepts_legacy_url_with_explicit_empty_tensor(self):
        inputs_pb = MultimodalInputsPB()
        mm_input_pb = inputs_pb.multimodal_inputs.add()
        mm_input_pb.multimodal_url = "memory://image"
        mm_input_pb.multimodal_type = 1
        mm_input_pb.multimodal_tensor.SetInParent()

        converted = trans_mm_input(inputs_pb)

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].url, "memory://image")
        self.assertEqual(converted[0].tensor.numel(), 0)

    def test_trans_mm_input_deserializes_uint8_tensor(self):
        inputs_pb = MultimodalInputsPB()
        mm_input_pb = inputs_pb.multimodal_inputs.add()
        mm_input_pb.multimodal_type = 1
        mm_input_pb.multimodal_tensor.CopyFrom(
            TensorPB(
                data_type=TensorPB.DataType.UINT8,
                shape=[2, 2, 3],
                uint8_data=bytes(range(12)),
            )
        )

        converted = trans_mm_input(inputs_pb)

        self.assertEqual(converted[0].tensor.dtype, torch.uint8)
        self.assertEqual(converted[0].tensor.shape, torch.Size([2, 2, 3]))
        self.assertEqual(converted[0].tensor.reshape(-1).tolist(), list(range(12)))

    def test_trans_mm_input_ignores_exact_legacy_url_tensor_placeholder(self):
        inputs_pb = MultimodalInputsPB()
        mm_input_pb = inputs_pb.multimodal_inputs.add()
        mm_input_pb.multimodal_url = "memory://legacy-placeholder"
        mm_input_pb.multimodal_type = 1
        mm_input_pb.multimodal_tensor.CopyFrom(
            TensorPB(
                data_type=TensorPB.DataType.FP32,
                shape=[1],
                fp32_data=b"\0\0\0\0",
            )
        )

        converted = trans_mm_input(inputs_pb)

        self.assertEqual(converted[0].url, "memory://legacy-placeholder")
        self.assertEqual(converted[0].tensor.numel(), 0)

    def test_trans_mm_input_rejects_both_or_neither_source(self):
        both_pb = MultimodalInputsPB()
        both = both_pb.multimodal_inputs.add()
        both.multimodal_url = "memory://image"
        both.multimodal_tensor.CopyFrom(
            TensorPB(
                data_type=TensorPB.DataType.UINT8,
                shape=[1],
                uint8_data=b"x",
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            trans_mm_input(both_pb)

        neither_pb = MultimodalInputsPB()
        neither_pb.multimodal_inputs.add()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            trans_mm_input(neither_pb)

    def test_get_bytes(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            temp_path = tmp_file.name

            image = Image.new("RGB", (200, 200), "white")
            image.save(temp_path, format="PNG")

            self.assertTrue(
                Image.open(get_bytes_io_from_url(temp_path)).size == image.size
            )

    def test_base64(self):
        buffer = io.BytesIO()

        image = Image.new("RGB", (200, 200), "white")
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        base64_str = "data:image/png;base64," + base64.b64encode(image_bytes).decode(
            "utf-8"
        )

        self.assertTrue(
            Image.open(get_bytes_io_from_url(base64_str)).size == image.size
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
