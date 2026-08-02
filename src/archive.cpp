#include "infer/archive.hpp"

#include <array>
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace infer {
namespace {

#pragma pack(push, 1)
struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t json_size;
  uint64_t data_offset;
  uint64_t reserved;
};
#pragma pack(pop)

constexpr std::array<char, 8> kMagic{'Q', 'W', 'E', 'N', 'B', 'I', 'N', '1'};

}  // namespace

ModelArchive::~ModelArchive() { close(); }

ModelArchive::ModelArchive(ModelArchive&& other) noexcept { *this = std::move(other); }

ModelArchive& ModelArchive::operator=(ModelArchive&& other) noexcept {
  if (this != &other) {
    close();
    path_ = std::move(other.path_);
    mapping_ = other.mapping_;
    mapped_bytes_ = other.mapped_bytes_;
    data_offset_ = other.data_offset_;
    records_ = std::move(other.records_);
    index_ = std::move(other.index_);
    fd_ = other.fd_;
    other.fd_ = -1;
    other.mapping_ = nullptr;
    other.mapped_bytes_ = 0;
  }
  return *this;
}

void ModelArchive::open(const std::filesystem::path& path) {
  close();
  path_ = path;
  fd_ = ::open(path.c_str(), O_RDONLY);
  INFER_CHECK(fd_ >= 0, "cannot open archive: " + path.string());
  struct stat st {};
  INFER_CHECK(fstat(fd_, &st) == 0, "cannot stat archive");
  mapped_bytes_ = static_cast<size_t>(st.st_size);
  mapping_ = mmap(nullptr, mapped_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
  INFER_CHECK(mapping_ != MAP_FAILED, "cannot mmap archive");
  madvise(mapping_, mapped_bytes_, MADV_SEQUENTIAL);
  INFER_CHECK(mapped_bytes_ >= sizeof(FileHeader), "archive is truncated");
  const auto* header = static_cast<const FileHeader*>(mapping_);
  INFER_CHECK(std::equal(kMagic.begin(), kMagic.end(), header->magic), "bad archive magic");
  INFER_CHECK(header->version == 1, "unsupported archive version");
  INFER_CHECK(sizeof(FileHeader) + header->json_size <= mapped_bytes_, "bad archive JSON size");
  INFER_CHECK(header->data_offset < mapped_bytes_, "bad archive data offset");
  data_offset_ = header->data_offset;
  const auto* json_begin = static_cast<const char*>(mapping_) + sizeof(FileHeader);
  const auto meta = nlohmann::json::parse(json_begin, json_begin + header->json_size);
  records_.clear();
  index_.clear();
  for (const auto& item : meta.at("tensors")) {
    TensorRecord rec;
    rec.name = item.at("name").get<std::string>();
    rec.shape = item.at("shape").get<Shape>();
    const auto dtype = item.at("dtype").get<std::string>();
    if (dtype == "float32") {
      rec.dtype = DType::kFloat32;
    } else if (dtype == "bfloat16") {
      rec.dtype = DType::kBFloat16;
    } else if (dtype == "int8") {
      rec.dtype = DType::kInt8;
    } else {
      throw Error("unsupported archive dtype: " + dtype);
    }
    rec.offset = item.at("offset").get<uint64_t>();
    rec.nbytes = item.at("nbytes").get<uint64_t>();
    INFER_CHECK(rec.nbytes == numel(rec.shape) * dtype_size(rec.dtype),
                "tensor size mismatch: " + rec.name);
    INFER_CHECK(data_offset_ + rec.offset + rec.nbytes <= mapped_bytes_,
                "tensor lies outside archive: " + rec.name);
    if (item.contains("quant")) {
      QuantizationInfo q;
      q.group_size = item["quant"].at("group_size").get<int>();
      q.axis = item["quant"].value("axis", 1);
      q.scale_tensor = item["quant"].at("scale_tensor").get<std::string>();
      rec.quant = q;
    }
    INFER_CHECK(index_.emplace(rec.name, records_.size()).second,
                "duplicate tensor: " + rec.name);
    records_.push_back(std::move(rec));
  }
}

void ModelArchive::close() {
  if (mapping_) {
    munmap(mapping_, mapped_bytes_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }
  mapping_ = nullptr;
  mapped_bytes_ = 0;
  data_offset_ = 0;
  records_.clear();
  index_.clear();
}

const TensorRecord& ModelArchive::record(std::string_view name) const {
  const auto it = index_.find(std::string(name));
  INFER_CHECK(it != index_.end(), "missing tensor: " + std::string(name));
  return records_[it->second];
}

const void* ModelArchive::data(std::string_view name) const {
  const auto& rec = record(name);
  return static_cast<const std::byte*>(mapping_) + data_offset_ + rec.offset;
}

TensorView ModelArchive::tensor(std::string_view name) const {
  const auto& rec = record(name);
  return TensorView{const_cast<void*>(data(name)), rec.shape, rec.dtype, Device::kCpu};
}

bool ModelArchive::contains(std::string_view name) const {
  return index_.find(std::string(name)) != index_.end();
}

}  // namespace infer
