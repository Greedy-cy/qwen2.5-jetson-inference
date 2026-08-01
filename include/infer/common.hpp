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
enum class DType { kFloat32, kInt8 };
enum class Precision { kFloat32, kInt8 };

inline const char* to_string(Device v) { return v == Device::kCpu ? "cpu" : "cuda"; }
inline const char* to_string(Precision v) { return v == Precision::kFloat32 ? "fp32" : "w8a32"; }
inline size_t dtype_size(DType v) { return v == DType::kFloat32 ? sizeof(float) : sizeof(int8_t); }

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
