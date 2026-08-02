#include "infer/buffer.hpp"

#include <cstdlib>

namespace infer {

Buffer::Buffer(size_t bytes, Device device) { resize(bytes, device); }

Buffer::~Buffer() { reset(); }

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_), device_(other.device_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = other.data_;
    bytes_ = other.bytes_;
    device_ = other.device_;
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

void Buffer::reset() {
  if (!data_) return;
  if (device_ == Device::kCpu) {
    std::free(data_);
  } else {
    cudaFree(data_);
  }
  data_ = nullptr;
  bytes_ = 0;
}

void Buffer::resize(size_t bytes, Device device) {
  reset();
  device_ = device;
  bytes_ = bytes;
  if (bytes == 0) return;
  if (device == Device::kCpu) {
    INFER_CHECK(posix_memalign(&data_, 256, bytes) == 0, "CPU allocation failed");
  } else {
    INFER_CUDA_CHECK(cudaMalloc(&data_, bytes));
  }
  INFER_CHECK(data_ != nullptr, "buffer allocation returned null");
}

void* Arena::allocate(size_t bytes, size_t alignment) {
  const size_t begin = (cursor_ + alignment - 1) / alignment * alignment;
  INFER_CHECK(begin + bytes <= storage_.bytes(), "arena capacity exceeded");
  auto* base = static_cast<std::byte*>(storage_.data());
  cursor_ = begin + bytes;
  return base + begin;
}

}  // namespace infer
