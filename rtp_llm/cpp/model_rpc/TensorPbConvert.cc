#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm {
namespace {

struct TensorSpec {
    torch::ScalarType  dtype;
    size_t             element_size;
    const std::string* payload;
};

TensorSpec getTensorSpec(const TensorPB& tensor_pb) {
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32:
            return {torch::kFloat32, sizeof(float), &tensor_pb.fp32_data()};
        case TensorPB::INT32:
            return {torch::kInt32, sizeof(int32_t), &tensor_pb.int32_data()};
        case TensorPB::FP16:
            return {torch::kFloat16, sizeof(c10::Half), &tensor_pb.fp16_data()};
        case TensorPB::BF16:
            return {torch::kBFloat16, sizeof(c10::BFloat16), &tensor_pb.bf16_data()};
        case TensorPB::UINT8:
            return {torch::kUInt8, sizeof(uint8_t), &tensor_pb.uint8_data()};
        default:
            throw std::runtime_error("Unsupported TensorPB data type.");
    }
}

bool allPayloadsEmpty(const TensorPB& tensor_pb) {
    return tensor_pb.fp32_data().empty() && tensor_pb.int32_data().empty() && tensor_pb.fp16_data().empty()
           && tensor_pb.bf16_data().empty() && tensor_pb.uint8_data().empty();
}

void validateSelectedPayload(const TensorPB& tensor_pb) {
    const bool valid = (tensor_pb.data_type() == TensorPB::FP32 || tensor_pb.fp32_data().empty())
                       && (tensor_pb.data_type() == TensorPB::INT32 || tensor_pb.int32_data().empty())
                       && (tensor_pb.data_type() == TensorPB::FP16 || tensor_pb.fp16_data().empty())
                       && (tensor_pb.data_type() == TensorPB::BF16 || tensor_pb.bf16_data().empty())
                       && (tensor_pb.data_type() == TensorPB::UINT8 || tensor_pb.uint8_data().empty());
    if (!valid) {
        throw std::runtime_error("TensorPB contains payload data for a non-selected data type.");
    }
}

size_t validatedNumel(const std::vector<int64_t>& shape, size_t element_size) {
    for (const int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("TensorPB shape contains a negative dimension.");
        }
    }
    if (std::find(shape.begin(), shape.end(), int64_t{0}) != shape.end()) {
        return 0;
    }

    uint64_t nonzero_numel = 1;
    for (const int64_t dim : shape) {
        const auto unsigned_dim = static_cast<uint64_t>(dim);
        if (unsigned_dim > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("TensorPB dimension does not fit in size_t.");
        }
        if (nonzero_numel > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / unsigned_dim
            || nonzero_numel > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / unsigned_dim) {
            throw std::runtime_error("TensorPB shape element count overflows.");
        }
        nonzero_numel *= unsigned_dim;
    }
    if (nonzero_numel > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / element_size) {
        throw std::runtime_error("TensorPB byte count overflows.");
    }
    return static_cast<size_t>(nonzero_numel);
}

}  // namespace

torch::Tensor TensorPbConvert::pbToTorch(const TensorPB& tensor_pb) {
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    const TensorSpec     spec    = getTensorSpec(tensor_pb);
    const auto           options = torch::TensorOptions().dtype(spec.dtype).device(torch::kCPU);

    // The default protobuf value represents an omitted tensor. A scalar is
    // represented by an empty shape plus one element of payload and is handled
    // by the normal validation path below.
    if (shape.empty() && allPayloadsEmpty(tensor_pb)) {
        return torch::empty({0}, options);
    }

    validateSelectedPayload(tensor_pb);
    const size_t numel          = validatedNumel(shape, spec.element_size);
    const size_t expected_bytes = numel * spec.element_size;
    if (spec.payload->size() != expected_bytes) {
        throw std::runtime_error("TensorPB payload size does not exactly match its shape and data type.");
    }
    if (numel == 0) {
        return torch::empty(shape, options);
    }

    void* data_ptr = const_cast<char*>(spec.payload->data());
    return torch::from_blob(data_ptr, shape, options).clone();
}

void TensorPbConvert::torchToPb(TensorPB* tensor_pb, const torch::Tensor& tensor) {
    if (tensor_pb == nullptr) {
        throw std::runtime_error("TensorPB output pointer must not be null.");
    }
    tensor_pb->Clear();
    if (!tensor.defined()) {
        return;
    }

    const auto dtype = tensor.dtype().toScalarType();
    switch (dtype) {
        case torch::kFloat32:
            tensor_pb->set_data_type(TensorPB::FP32);
            break;
        case torch::kInt32:
            tensor_pb->set_data_type(TensorPB::INT32);
            break;
        case torch::kFloat16:
            tensor_pb->set_data_type(TensorPB::FP16);
            break;
        case torch::kBFloat16:
            tensor_pb->set_data_type(TensorPB::BF16);
            break;
        case torch::kUInt8:
            tensor_pb->set_data_type(TensorPB::UINT8);
            break;
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
    auto shape = tensor.sizes();
    for (auto dim : shape) {
        tensor_pb->add_shape(dim);
    }
    torch::Tensor contiguous_tensor = tensor.detach().to(torch::kCPU).contiguous();
    const auto    numel             = contiguous_tensor.numel();
    if (numel < 0) {
        throw std::runtime_error("Tensor element count must not be negative.");
    }
    const size_t element_size = contiguous_tensor.element_size();
    if (static_cast<uint64_t>(numel) > std::numeric_limits<size_t>::max() / element_size) {
        throw std::runtime_error("Tensor byte count overflows size_t.");
    }
    const size_t num_bytes = static_cast<size_t>(numel) * element_size;
    if (num_bytes == 0) {
        return;
    }
    const char* data_ptr = static_cast<const char*>(contiguous_tensor.data_ptr());

    switch (dtype) {
        case torch::kFloat32: {
            tensor_pb->set_fp32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kInt32: {
            tensor_pb->set_int32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kFloat16: {
            tensor_pb->set_fp16_data(data_ptr, num_bytes);
            break;
        }
        case torch::kBFloat16: {
            tensor_pb->set_bf16_data(data_ptr, num_bytes);
            break;
        }
        case torch::kUInt8:
            tensor_pb->set_uint8_data(data_ptr, num_bytes);
            break;
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
}

}  // namespace rtp_llm
