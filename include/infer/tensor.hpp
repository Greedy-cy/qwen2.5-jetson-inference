#pragma once

#include "infer/common.hpp"

namespace infer {

using Shape = std::vector<int64_t>;

inline size_t numel(const Shape& shape) {
  if (shape.empty()) return 1;
  size_t n = 1;
  for (const auto dim : shape) {
    INFER_CHECK(dim >= 0, "negative tensor dimension");
    n *= static_cast<size_t>(dim);
  }
  return n;
}

struct TensorView {
  void* data{nullptr};
  Shape shape;
  DType dtype{DType::kFloat32};
  Device device{Device::kCpu};

  size_t size() const { return numel(shape); }
  size_t bytes() const { return size() * dtype_size(dtype); }

  template <class T> T* ptr() { return static_cast<T*>(data); }
  template <class T> const T* ptr() const { return static_cast<const T*>(data); }
};

}  // namespace infer
