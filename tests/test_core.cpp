#include "infer/archive.hpp"
#include "infer/buffer.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstring>
#include <fstream>

namespace {

#pragma pack(push, 1)
struct TestArchiveHeader {
  char magic[8];
  uint32_t version;
  uint32_t json_size;
  uint64_t data_offset;
  uint64_t reserved;
};
#pragma pack(pop)

class TemporaryBFloat16Archive {
 public:
  TemporaryBFloat16Archive() {
    path_ = std::filesystem::temp_directory_path() /
            ("qwen-bfloat16-" +
             std::to_string(std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count()) +
             ".qbin");
    constexpr uint64_t kDataOffset = 512;
    const nlohmann::json metadata{
        {"format", "QWENBIN1"},
        {"version", 1},
        {"precision", "w16a16"},
        {"tensors",
         {{{"name", "weight"},
           {"shape", {2}},
           {"dtype", "bfloat16"},
           {"offset", 0},
           {"nbytes", 2 * sizeof(uint16_t)}}}}};
    const std::string encoded = metadata.dump();
    if (sizeof(TestArchiveHeader) + encoded.size() > kDataOffset) {
      throw std::runtime_error("test archive metadata exceeds reserved header");
    }

    TestArchiveHeader header{};
    std::memcpy(header.magic, "QWENBIN1", sizeof(header.magic));
    header.version = 1;
    header.json_size = static_cast<uint32_t>(encoded.size());
    header.data_offset = kDataOffset;
    const uint16_t values[]{0x3F80, 0xC020};

    std::ofstream output(path_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
    output.seekp(static_cast<std::streamoff>(kDataOffset));
    output.write(reinterpret_cast<const char*>(values), sizeof(values));
    if (!output.good()) {
      throw std::runtime_error("failed to write temporary BF16 archive");
    }
  }

  ~TemporaryBFloat16Archive() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

TEST(Buffer, CpuMoveAndArenaAlignment) {
  infer::Buffer first(4096, infer::Device::kCpu);
  ASSERT_NE(first.data(), nullptr);
  infer::Buffer second(std::move(first));
  EXPECT_EQ(first.data(), nullptr);
  EXPECT_EQ(second.bytes(), 4096u);
  infer::Arena arena(1024, infer::Device::kCpu);
  auto* a = arena.allocate(7, 64);
  auto* b = arena.allocate(7, 64);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(a) % 64, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(b) % 64, 0u);
}

TEST(DType, BFloat16SizeAndNames) {
  EXPECT_EQ(infer::dtype_size(infer::DType::kFloat32), 4u);
  EXPECT_EQ(infer::dtype_size(infer::DType::kBFloat16), 2u);
  EXPECT_EQ(infer::dtype_size(infer::DType::kInt8), 1u);
  EXPECT_STREQ(infer::to_string(infer::DType::kBFloat16), "bfloat16");
  EXPECT_STREQ(infer::to_string(infer::Precision::kW16A16), "w16a16");
  EXPECT_STREQ(infer::archive_filename(infer::Precision::kW16A16),
               "model.w16a16.qbin");
}

TEST(ModelArchive, LoadsBFloat16Tensor) {
  TemporaryBFloat16Archive temporary;
  infer::ModelArchive archive(temporary.path());

  ASSERT_EQ(archive.records().size(), 1u);
  const auto& record = archive.record("weight");
  EXPECT_EQ(record.dtype, infer::DType::kBFloat16);
  EXPECT_EQ(record.shape, infer::Shape({2}));
  EXPECT_EQ(record.nbytes, 4u);

  const auto tensor = archive.tensor("weight");
  EXPECT_EQ(tensor.bytes(), 4u);
  const auto* values = tensor.ptr<uint16_t>();
  EXPECT_EQ(values[0], 0x3F80);
  EXPECT_EQ(values[1], 0xC020);
}
