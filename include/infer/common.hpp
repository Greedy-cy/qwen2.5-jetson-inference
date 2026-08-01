#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef INFER_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace infer {

enum class Device { kCpu, kCuda };
enum class DType { kFloat32, kBFloat16, kInt8 };
enum class Precision { kFloat32, kW8A32, kW16A16 };

inline const char* to_string(Device v) { return v == Device::kCpu ? "cpu" : "cuda"; }
inline const char* to_string(DType v) {
  switch (v) {
    case DType::kFloat32: return "float32";
    case DType::kBFloat16: return "bfloat16";
    case DType::kInt8: return "int8";
  }
  return "unknown";
}
inline const char* to_string(Precision v) {
  switch (v) {
    case Precision::kFloat32: return "fp32";
    case Precision::kW8A32: return "w8a32";
    case Precision::kW16A16: return "w16a16";
  }
  return "unknown";
}
inline const char* archive_filename(Precision v) {
  switch (v) {
    case Precision::kFloat32: return "model.fp32.qbin";
    case Precision::kW8A32: return "model.int8.qbin";
    case Precision::kW16A16: return "model.w16a16.qbin";
  }
  return "";
}
inline size_t dtype_size(DType v) {
  switch (v) {
    case DType::kFloat32: return sizeof(float);
    case DType::kBFloat16: return sizeof(uint16_t);
    case DType::kInt8: return sizeof(int8_t);
  }
  return 0;
}

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

#define INFER_CHECK(cond, message) \
  do { if (!(cond)) throw ::infer::Error(message); } while (false)

#ifdef INFER_WITH_CUDA
inline void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    std::ostringstream os;
    os << file << ':' << line << " CUDA call " << expression << " failed: "
       << cudaGetErrorString(status);
    throw Error(os.str());
  }
}
#define INFER_CUDA_CHECK(expr) ::infer::cuda_check((expr), #expr, __FILE__, __LINE__)
#endif

class WallTimer {
 public:
  WallTimer() : start_(std::chrono::steady_clock::now()) {}
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start_).count();
  }
 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace infer
