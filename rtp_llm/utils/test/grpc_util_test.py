import gc
import unittest

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import TensorPB
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


class GrpcUtilTest(unittest.TestCase):
    def test_uint8_noncontiguous_round_trip(self):
        tensor = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4).transpose(0, 1)

        tensor_pb = trans_from_tensor(tensor)
        restored = trans_tensor(tensor_pb)

        self.assertEqual(tensor_pb.data_type, TensorPB.DataType.UINT8)
        self.assertEqual(list(tensor_pb.shape), [3, 2, 4])
        self.assertEqual(len(tensor_pb.uint8_data), tensor.numel())
        self.assertTrue(restored.is_contiguous())
        self.assertTrue(torch.equal(restored, tensor))

    def test_serializes_into_existing_message_without_temporary_copy(self):
        destination = TensorPB(fp32_data=b"stale")

        result = trans_from_tensor(
            torch.tensor([1, 2, 3], dtype=torch.uint8), destination
        )

        self.assertIs(result, destination)
        self.assertFalse(destination.fp32_data)
        self.assertEqual(destination.uint8_data, b"\x01\x02\x03")

    def test_deserialized_tensor_owns_single_writable_buffer(self):
        tensor_pb = TensorPB(
            data_type=TensorPB.DataType.UINT8,
            shape=[2, 2],
            uint8_data=b"\x01\x02\x03\x04",
        )

        restored = trans_tensor(tensor_pb)
        del tensor_pb
        gc.collect()

        restored[0, 0] = 9
        self.assertEqual(restored.tolist(), [[9, 2], [3, 4]])

    def test_default_empty_and_scalar_are_distinct(self):
        empty = trans_tensor(TensorPB())
        scalar = trans_tensor(
            TensorPB(
                data_type=TensorPB.DataType.UINT8,
                uint8_data=b"\x07",
            )
        )

        self.assertEqual(empty.numel(), 0)
        self.assertEqual(scalar.shape, torch.Size([]))
        self.assertEqual(scalar.item(), 7)

    def test_zero_sized_shape_requires_empty_payload(self):
        restored = trans_tensor(
            TensorPB(data_type=TensorPB.DataType.UINT8, shape=[2, 0, 3])
        )
        self.assertEqual(restored.shape, torch.Size([2, 0, 3]))

    def test_rejects_negative_or_overflowing_shape(self):
        with self.assertRaisesRegex(ValueError, "negative dimension"):
            trans_tensor(
                TensorPB(
                    data_type=TensorPB.DataType.UINT8,
                    shape=[-1],
                    uint8_data=b"x",
                )
            )
        with self.assertRaisesRegex(ValueError, "overflows"):
            trans_tensor(
                TensorPB(
                    data_type=TensorPB.DataType.UINT8,
                    shape=[(1 << 63) - 1, 2],
                )
            )

    def test_rejects_short_long_or_mixed_payload(self):
        for payload in (b"x", b"xyz"):
            with self.subTest(payload_len=len(payload)), self.assertRaisesRegex(
                ValueError, "payload size mismatch"
            ):
                trans_tensor(
                    TensorPB(
                        data_type=TensorPB.DataType.UINT8,
                        shape=[2],
                        uint8_data=payload,
                    )
                )

        with self.assertRaisesRegex(ValueError, "incompatible fields"):
            trans_tensor(
                TensorPB(
                    data_type=TensorPB.DataType.UINT8,
                    shape=[1],
                    uint8_data=b"x",
                    fp32_data=b"xxxx",
                )
            )


if __name__ == "__main__":
    unittest.main()
