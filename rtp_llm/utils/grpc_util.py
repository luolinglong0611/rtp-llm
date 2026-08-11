from typing import Dict, Optional, Tuple

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import TensorPB


def trans_option(pb_object, py_object, name):
    if getattr(py_object, name):
        getattr(pb_object, name).value = getattr(py_object, name)


def trans_option_cast(pb_object, py_object, name, func):
    if getattr(py_object, name):
        getattr(pb_object, name).value = func(getattr(py_object, name))


_TENSOR_DATA_FIELDS = (
    "fp32_data",
    "int32_data",
    "fp16_data",
    "bf16_data",
    "uint8_data",
)
_MAX_TENSOR_ELEMENTS = (1 << 63) - 1


def _tensor_spec(data_type: TensorPB.DataType) -> Tuple[torch.dtype, str, int]:
    specs: Dict[int, Tuple[torch.dtype, str, int]] = {
        TensorPB.DataType.FP32: (torch.float32, "fp32_data", 4),
        TensorPB.DataType.INT32: (torch.int32, "int32_data", 4),
        TensorPB.DataType.FP16: (torch.float16, "fp16_data", 2),
        TensorPB.DataType.BF16: (torch.bfloat16, "bf16_data", 2),
        TensorPB.DataType.UINT8: (torch.uint8, "uint8_data", 1),
    }
    try:
        return specs[data_type]
    except KeyError as error:
        raise ValueError(f"unsupported TensorPB data type: {data_type}") from error


def trans_grpc_dtype(type: TensorPB.DataType):
    return _tensor_spec(type)[0]


def _validated_numel(shape, element_size: int) -> int:
    dimensions = [int(dim) for dim in shape]
    for dim in dimensions:
        if dim < 0:
            raise ValueError(
                f"TensorPB shape contains a negative dimension: {dimensions}"
            )

    if any(dim == 0 for dim in dimensions):
        return 0

    nonzero_numel = 1
    for dim in dimensions:
        if nonzero_numel > _MAX_TENSOR_ELEMENTS // dim:
            raise ValueError(
                f"TensorPB shape element count overflows int64: {dimensions}"
            )
        nonzero_numel *= dim
    numel = nonzero_numel
    if numel > _MAX_TENSOR_ELEMENTS // element_size:
        raise ValueError(f"TensorPB byte count overflows int64: {dimensions}")
    return numel


def trans_tensor(t: TensorPB):
    dtype, data_field, element_size = _tensor_spec(t.data_type)
    shape = list(t.shape)

    # An entirely default TensorPB represents an omitted tensor. A scalar is
    # still supported: it has no shape but exactly one element of payload.
    if not shape and all(not getattr(t, field) for field in _TENSOR_DATA_FIELDS):
        return torch.empty(0, dtype=dtype)

    populated_other_fields = [
        field
        for field in _TENSOR_DATA_FIELDS
        if field != data_field and getattr(t, field)
    ]
    if populated_other_fields:
        raise ValueError(
            f"TensorPB data type {t.data_type} has payload in incompatible fields: "
            f"{populated_other_fields}"
        )

    numel = _validated_numel(shape, element_size)
    data = getattr(t, data_field)
    expected_bytes = numel * element_size
    if len(data) != expected_bytes:
        raise ValueError(
            f"TensorPB payload size mismatch for shape {shape}: "
            f"expected {expected_bytes} bytes, got {len(data)}"
        )
    if numel == 0:
        return torch.empty(shape, dtype=dtype)

    # bytearray makes one owning copy of the protobuf bytes. torch keeps the
    # Python buffer alive for the lifetime of its storage, so no second clone
    # is needed (important for large OCR images).
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def trans_from_tensor(t: torch.Tensor, res: Optional[TensorPB] = None):
    res = TensorPB() if res is None else res
    res.Clear()
    if t is None:
        return res
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t)}")
    if t.numel() == 0:
        return res
    t = t.detach().cpu().contiguous()
    res.shape.extend(list(t.shape))
    if t.dtype == torch.float32:
        res.data_type = TensorPB.DataType.FP32
        res.fp32_data = t.numpy().tobytes()
    elif t.dtype == torch.int32:
        res.data_type = TensorPB.DataType.INT32
        res.int32_data = t.numpy().tobytes()
    elif t.dtype == torch.float16:
        res.data_type = TensorPB.DataType.FP16
        res.fp16_data = t.numpy().tobytes()
    elif t.dtype == torch.bfloat16:
        res.data_type = TensorPB.DataType.BF16
        res.bf16_data = t.view(torch.int16).numpy().tobytes()
    elif t.dtype == torch.uint8:
        res.data_type = TensorPB.DataType.UINT8
        res.uint8_data = t.numpy().tobytes()
    else:
        raise ValueError(f"unsupported tensor data type: {t.dtype}")
    return res
