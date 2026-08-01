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
#ifdef _WIN32
    _aligned_free(data_);
#else
    std::free(data_);
#endif
  } else {
#ifdef INFER_WITH_CUDA
    cudaFree(data_);
#endif
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
#ifdef _WIN32
    data_ = _aligned_malloc(bytes, 256);
#else
    INFER_CHECK(posix_memalign(&data_, 256, bytes) == 0, "CPU allocation failed");
#endif
  } else {
#ifdef INFER_WITH_CUDA
    INFER_CUDA_CHECK(cudaMalloc(&data_, bytes));
#else
    throw Error("CUDA support is not compiled");
#endif
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
