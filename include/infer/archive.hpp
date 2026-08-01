#pragma once

#include "infer/tensor.hpp"

namespace infer {

struct QuantizationInfo {
  int group_size{0};
  int axis{1};
  std::string scale_tensor;
};

struct TensorRecord {
  std::string name;
  Shape shape;
  DType dtype{DType::kFloat32};
  uint64_t offset{0};
  uint64_t nbytes{0};
  std::optional<QuantizationInfo> quant;
};

class ModelArchive {
 public:
  ModelArchive() = default;
  explicit ModelArchive(const std::filesystem::path& path) { open(path); }
  ~ModelArchive();
  ModelArchive(const ModelArchive&) = delete;
  ModelArchive& operator=(const ModelArchive&) = delete;
  ModelArchive(ModelArchive&& other) noexcept;
  ModelArchive& operator=(ModelArchive&& other) noexcept;

  void open(const std::filesystem::path& path);
  void close();
  const TensorRecord& record(std::string_view name) const;
  const void* data(std::string_view name) const;
  TensorView tensor(std::string_view name) const;
  bool contains(std::string_view name) const;
  const std::vector<TensorRecord>& records() const { return records_; }
  uint64_t data_offset() const { return data_offset_; }
  size_t mapped_bytes() const { return mapped_bytes_; }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
  void* mapping_{nullptr};
  size_t mapped_bytes_{0};
  uint64_t data_offset_{0};
  std::vector<TensorRecord> records_;
  std::unordered_map<std::string, size_t> index_;
#ifdef _WIN32
  void* file_handle_{nullptr};
  void* mapping_handle_{nullptr};
#else
  int fd_{-1};
#endif
};

}  // namespace infer
