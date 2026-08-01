#include "infer/qwen2.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
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

struct TestTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> values;
};

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

class TinyQwen2Model {
 public:
  TinyQwen2Model() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("qwen2-prefill-state-" + std::to_string(stamp));
    std::filesystem::create_directories(path_);
    write_config();
    write_archive();
  }

  ~TinyQwen2Model() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  void write_config() const {
    const nlohmann::json config{
        {"model_type", "qwen2"},
        {"hidden_size", 4},
        {"intermediate_size", 8},
        {"num_hidden_layers", 1},
        {"num_attention_heads", 1},
        {"num_key_value_heads", 1},
        {"vocab_size", 8},
        {"max_position_embeddings", 528},
        {"bos_token_id", 0},
        {"eos_token_id", 7},
        {"rms_norm_eps", 1e-5},
        {"rope_theta", 10000.0},
        {"tie_word_embeddings", true},
    };
    std::ofstream output(path_ / "config.json");
    output << config.dump(2) << '\n';
    if (!output.good()) throw std::runtime_error("failed to write tiny model config");
  }

  void write_archive() const {
    constexpr int kHidden = 4;
    constexpr int kIntermediate = 8;
    constexpr int kVocab = 8;
    std::vector<TestTensor> tensors;
    auto add = [&](std::string name, std::vector<int64_t> shape, float scale,
                   int seed, bool ones = false) {
      size_t count = 1;
      for (const int64_t dimension : shape) count *= static_cast<size_t>(dimension);
      std::vector<float> values(count);
      for (size_t i = 0; i < count; ++i) {
        values[i] = ones ? 1.0f
                         : scale * static_cast<float>(
                             static_cast<int>((i + static_cast<size_t>(seed)) % 13) - 6);
      }
      tensors.push_back({std::move(name), std::move(shape), std::move(values)});
    };

    add("model.embed_tokens.weight", {kVocab, kHidden}, 0.03f, 1);
    add("model.layers.0.input_layernorm.weight", {kHidden}, 0.0f, 0, true);
    add("model.layers.0.self_attn.q_proj.weight", {kHidden, kHidden}, 0.01f, 2);
    add("model.layers.0.self_attn.q_proj.bias", {kHidden}, 0.002f, 3);
    add("model.layers.0.self_attn.k_proj.weight", {kHidden, kHidden}, 0.01f, 4);
    add("model.layers.0.self_attn.k_proj.bias", {kHidden}, 0.002f, 5);
    add("model.layers.0.self_attn.v_proj.weight", {kHidden, kHidden}, 0.01f, 6);
    add("model.layers.0.self_attn.v_proj.bias", {kHidden}, 0.002f, 7);
    add("model.layers.0.self_attn.o_proj.weight", {kHidden, kHidden}, 0.01f, 8);
    add("model.layers.0.post_attention_layernorm.weight", {kHidden}, 0.0f, 0, true);
    add("model.layers.0.mlp.gate_proj.weight", {kIntermediate, kHidden}, 0.01f, 9);
    add("model.layers.0.mlp.up_proj.weight", {kIntermediate, kHidden}, 0.01f, 10);
    add("model.layers.0.mlp.down_proj.weight", {kHidden, kIntermediate}, 0.01f, 11);
    add("model.norm.weight", {kHidden}, 0.0f, 0, true);

    nlohmann::json metadata{{"tensors", nlohmann::json::array()}};
    uint64_t offset = 0;
    for (const auto& tensor : tensors) {
      const uint64_t bytes = tensor.values.size() * sizeof(float);
      metadata["tensors"].push_back({
          {"name", tensor.name},
          {"shape", tensor.shape},
          {"dtype", "float32"},
          {"offset", offset},
          {"nbytes", bytes},
      });
      offset += bytes;
    }
    const std::string metadata_text = metadata.dump();
    TestArchiveHeader header{};
    std::memcpy(header.magic, "QWENBIN1", sizeof(header.magic));
    header.version = 1;
    header.json_size = static_cast<uint32_t>(metadata_text.size());
    header.data_offset = align_up(sizeof(header) + metadata_text.size(), 256);

    std::ofstream output(path_ / "model.fp32.qbin", std::ios::binary);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(metadata_text.data(), static_cast<std::streamsize>(metadata_text.size()));
    const std::vector<char> padding(
        header.data_offset - sizeof(header) - metadata_text.size(), 0);
    output.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    for (const auto& tensor : tensors) {
      output.write(reinterpret_cast<const char*>(tensor.values.data()),
                   static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    }
    if (!output.good()) throw std::runtime_error("failed to write tiny model archive");

    output.close();

    nlohmann::json bf16_metadata{{"tensors", nlohmann::json::array()}};
    uint64_t bf16_offset = 0;
    std::vector<std::vector<uint16_t>> bf16_values;
    bf16_values.reserve(tensors.size());
    for (const auto& tensor : tensors) {
      std::vector<uint16_t> values(tensor.values.size());
      for (size_t index = 0; index < tensor.values.size(); ++index) {
        uint32_t bits = 0;
        std::memcpy(&bits, &tensor.values[index], sizeof(bits));
        bits += 0x7fffu + ((bits >> 16u) & 1u);
        values[index] = static_cast<uint16_t>(bits >> 16u);
      }
      const uint64_t bytes = values.size() * sizeof(uint16_t);
      bf16_metadata["tensors"].push_back({
          {"name", tensor.name},
          {"shape", tensor.shape},
          {"dtype", "bfloat16"},
          {"offset", bf16_offset},
          {"nbytes", bytes},
      });
      bf16_offset += bytes;
      bf16_values.push_back(std::move(values));
    }
    const std::string bf16_metadata_text = bf16_metadata.dump();
    TestArchiveHeader bf16_header{};
    std::memcpy(
        bf16_header.magic, "QWENBIN1", sizeof(bf16_header.magic));
    bf16_header.version = 1;
    bf16_header.json_size =
        static_cast<uint32_t>(bf16_metadata_text.size());
    bf16_header.data_offset =
        align_up(sizeof(bf16_header) + bf16_metadata_text.size(), 256);

    std::ofstream bf16_output(
        path_ / "model.w16a16.qbin", std::ios::binary);
    bf16_output.write(
        reinterpret_cast<const char*>(&bf16_header), sizeof(bf16_header));
    bf16_output.write(
        bf16_metadata_text.data(),
        static_cast<std::streamsize>(bf16_metadata_text.size()));
    const std::vector<char> bf16_padding(
        bf16_header.data_offset - sizeof(bf16_header) -
            bf16_metadata_text.size(),
        0);
    bf16_output.write(
        bf16_padding.data(),
        static_cast<std::streamsize>(bf16_padding.size()));
    for (const auto& values : bf16_values) {
      bf16_output.write(
          reinterpret_cast<const char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(uint16_t)));
    }
    if (!bf16_output.good()) {
      throw std::runtime_error("failed to write tiny BF16 model archive");
    }
  }

  std::filesystem::path path_;
};

infer::RuntimeOptions cpu_options(int max_sequence_length = 32) {
  infer::RuntimeOptions options;
  options.backend = infer::Device::kCpu;
  options.precision = infer::Precision::kFloat32;
  options.max_sequence_length = max_sequence_length;
  return options;
}


infer::RuntimeOptions cuda_bf16_options(
    int max_sequence_length = 32,
    infer::LinearKernel kernel = infer::LinearKernel::kCublas) {
  infer::RuntimeOptions options;
  options.backend = infer::Device::kCuda;
  options.precision = infer::Precision::kW16A16;
  options.max_sequence_length = max_sequence_length;
  options.linear_kernel = kernel;
  return options;
}

uint64_t archive_payload_bytes(const infer::ModelArchive& archive) {
  uint64_t bytes = 0;
  for (const auto& record : archive.records()) bytes += record.nbytes;
  return bytes;
}

void expect_logits_equal(const infer::Qwen2Model& lhs, const infer::Qwen2Model& rhs) {
  const auto a = lhs.last_logits_host();
  const auto b = rhs.last_logits_host();
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) EXPECT_FLOAT_EQ(a[i], b[i]);
}

double logits_cosine(const infer::Qwen2Model& lhs, const infer::Qwen2Model& rhs) {
  const auto a = lhs.last_logits_host();
  const auto b = rhs.last_logits_host();
  if (a.size() != b.size()) return 0.0;
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    norm_a += static_cast<double>(a[i]) * a[i];
    norm_b += static_cast<double>(b[i]) * b[i];
  }
  return dot / std::sqrt(norm_a * norm_b);
}

std::vector<int> prompt_tokens(size_t length) {
  std::vector<int> prompt(length);
  for (size_t i = 0; i < length; ++i) {
    prompt[i] = static_cast<int>((i * 5 + 1) % 8);
  }
  return prompt;
}

}  // namespace

TEST(Qwen2State, RejectsInvalidTransitionsAndCapacity) {
  TinyQwen2Model tiny;
  infer::Qwen2Model model(tiny.path(), cpu_options(4));

  EXPECT_THROW(model.decode_next(1), infer::Error);
  EXPECT_THROW(model.prefill({}), infer::Error);
  EXPECT_THROW(model.prefill({1, 2, 3, 4, 5}), infer::Error);
  EXPECT_THROW(model.prefill({1, 8}), infer::Error);
  EXPECT_EQ(model.position(), 0);
  EXPECT_FALSE(model.has_prefilled());

  int token = model.prefill({1});
  EXPECT_EQ(model.position(), 1);
  EXPECT_TRUE(model.has_prefilled());
  EXPECT_THROW(model.prefill({2}), infer::Error);
  for (int i = 0; i < 3; ++i) token = model.decode_next(token);
  EXPECT_EQ(model.position(), 4);
  EXPECT_THROW(model.decode_next(token), infer::Error);

  model.reset();
  EXPECT_EQ(model.position(), 0);
  EXPECT_FALSE(model.has_prefilled());
  EXPECT_NO_THROW(model.prefill({2}));
}

TEST(Qwen2State, GenerateResetsStateAndSupportsSingleTokenPrompt) {
  TinyQwen2Model tiny;
  infer::Qwen2Model model(tiny.path(), cpu_options(8));

  const auto first = model.generate({1, 2}, 5, false);
  EXPECT_EQ(model.position(), 6);
  const auto second = model.generate({1, 2}, 5, false);
  EXPECT_EQ(second.tokens, first.tokens);
  EXPECT_EQ(model.position(), 6);

  const auto no_output = model.generate({3}, 0, false);
  EXPECT_TRUE(no_output.tokens.empty());
  EXPECT_EQ(model.position(), 1);
  EXPECT_TRUE(model.has_prefilled());
  EXPECT_THROW(model.generate(std::vector<int>(9, 1), 1, false), infer::Error);
}

TEST(Qwen2MatrixizedPrefill, SupportsRequiredPromptLengths) {
  TinyQwen2Model tiny;
  constexpr std::array<int, 7> kLengths{1, 2, 31, 32, 127, 128, 512};
  for (const int length : kLengths) {
    SCOPED_TRACE("prompt_length=" + std::to_string(length));
    infer::Qwen2Model model(tiny.path(), cpu_options(512));

    const int next = model.prefill(prompt_tokens(static_cast<size_t>(length)));
    EXPECT_GE(next, 0);
    EXPECT_LT(next, model.config().vocab_size);
    EXPECT_EQ(model.position(), length);
    EXPECT_TRUE(model.has_prefilled());
    EXPECT_GT(model.prefill_workspace_bytes(), 0u);
    for (const float logit : model.last_logits_host()) {
      EXPECT_TRUE(std::isfinite(logit));
    }
  }
}

TEST(Qwen2MatrixizedPrefill, CpuDecodeStateIsDeterministicFor16Tokens) {
  TinyQwen2Model tiny;
  infer::Qwen2Model first(tiny.path(), cpu_options(32));
  infer::Qwen2Model second(tiny.path(), cpu_options(32));
  const auto prompt = prompt_tokens(8);

  int token = first.prefill(prompt);
  ASSERT_EQ(second.prefill(prompt), token);
  expect_logits_equal(first, second);
  for (int step = 0; step < 16; ++step) {
    SCOPED_TRACE("decode_step=" + std::to_string(step));
    const int first_next = first.decode_next(token);
    const int second_next = second.decode_next(token);
    ASSERT_EQ(second_next, first_next);
    expect_logits_equal(first, second);
    token = first_next;
  }
  EXPECT_EQ(first.position(), 24);
  EXPECT_EQ(second.position(), 24);
}

TEST(Qwen2MatrixizedPrefill, CudaFp32MatchesCpuAcrossRequiredLengths) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  TinyQwen2Model tiny;
  constexpr std::array<int, 7> kLengths{1, 2, 31, 32, 127, 128, 512};
  for (const int length : kLengths) {
    SCOPED_TRACE("prompt_length=" + std::to_string(length));
    auto cuda_options = cpu_options(512);
    cuda_options.backend = infer::Device::kCuda;
    infer::Qwen2Model cpu(tiny.path(), cpu_options(512));
    infer::Qwen2Model cuda(tiny.path(), cuda_options);
    const auto prompt = prompt_tokens(static_cast<size_t>(length));

    const int cpu_next = cpu.prefill(prompt);
    const int cuda_next = cuda.prefill(prompt);
    EXPECT_EQ(cuda_next, cpu_next);
    EXPECT_GE(logits_cosine(cpu, cuda), 0.999999);
    EXPECT_EQ(cpu.position(), length);
    EXPECT_EQ(cuda.position(), length);
    EXPECT_GT(cuda.prefill_workspace_bytes(), 0u);
  }
}

TEST(Qwen2MatrixizedPrefill, CudaFp32DecodeStateMatchesCpuFor16Tokens) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  TinyQwen2Model tiny;
  auto cuda_options = cpu_options(32);
  cuda_options.backend = infer::Device::kCuda;
  infer::Qwen2Model cpu(tiny.path(), cpu_options(32));
  infer::Qwen2Model cuda(tiny.path(), cuda_options);
  const auto prompt = prompt_tokens(8);

  int token = cpu.prefill(prompt);
  ASSERT_EQ(cuda.prefill(prompt), token);
  EXPECT_GE(logits_cosine(cpu, cuda), 0.999999);
  for (int step = 0; step < 16; ++step) {
    SCOPED_TRACE("decode_step=" + std::to_string(step));
    const int cpu_next = cpu.decode_next(token);
    const int cuda_next = cuda.decode_next(token);
    ASSERT_EQ(cuda_next, cpu_next);
    EXPECT_GE(logits_cosine(cpu, cuda), 0.999999);
    token = cpu_next;
  }
  EXPECT_EQ(cpu.position(), 24);
  EXPECT_EQ(cuda.position(), 24);
}

TEST(Qwen2MatrixizedPrefill, CudaCublasMatchesCustomKernel) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  TinyQwen2Model tiny;
  auto custom_options = cpu_options(64);
  custom_options.backend = infer::Device::kCuda;
  auto cublas_options = custom_options;
  cublas_options.linear_kernel = infer::LinearKernel::kCublas;
  infer::Qwen2Model custom(tiny.path(), custom_options);
  infer::Qwen2Model cublas(tiny.path(), cublas_options);
  const auto prompt = prompt_tokens(32);

  int token = custom.prefill(prompt);
  ASSERT_EQ(cublas.prefill(prompt), token);
  EXPECT_GE(logits_cosine(custom, cublas), 0.999999);
  for (int step = 0; step < 8; ++step) {
    const int custom_next = custom.decode_next(token);
    const int cublas_next = cublas.decode_next(token);
    ASSERT_EQ(cublas_next, custom_next);
    EXPECT_GE(logits_cosine(custom, cublas), 0.999999);
    token = custom_next;
  }
}

TEST(Qwen2BFloat16, PrefillMatchesFp32AcrossFixedPrompts) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP();
  }
  TinyQwen2Model tiny;
  constexpr std::array<int, 5> lengths{1, 2, 5, 8, 31};
  for (const int length : lengths) {
    SCOPED_TRACE("prompt_length=" + std::to_string(length));
    infer::Qwen2Model fp32(tiny.path(), cpu_options(32));
    infer::Qwen2Model bf16(tiny.path(), cuda_bf16_options(32));
    const auto prompt = prompt_tokens(static_cast<size_t>(length));
    const int fp32_next = fp32.prefill(prompt);
    const int bf16_next = bf16.prefill(prompt);
    EXPECT_EQ(bf16_next, fp32_next);
    EXPECT_GE(logits_cosine(fp32, bf16), 0.9999);
    EXPECT_EQ(bf16.position(), length);
    for (const float logit : bf16.last_logits_host()) {
      EXPECT_TRUE(std::isfinite(logit));
    }
  }
}

TEST(Qwen2BFloat16, PrefillKvStateContinuesDecodeFor16Tokens) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP();
  }
  TinyQwen2Model tiny;
  infer::Qwen2Model fp32(tiny.path(), cpu_options(32));
  infer::Qwen2Model bf16(tiny.path(), cuda_bf16_options(32));
  const auto prompt = prompt_tokens(8);

  int token = fp32.prefill(prompt);
  ASSERT_EQ(bf16.prefill(prompt), token);
  EXPECT_GE(logits_cosine(fp32, bf16), 0.9999);
  for (int step = 0; step < 16; ++step) {
    SCOPED_TRACE("decode_step=" + std::to_string(step));
    const int fp32_next = fp32.decode_next(token);
    const int bf16_next = bf16.decode_next(token);
    ASSERT_EQ(bf16_next, fp32_next);
    EXPECT_GE(logits_cosine(fp32, bf16), 0.9999);
    token = fp32_next;
  }
  EXPECT_EQ(bf16.position(), 24);
}

TEST(Qwen2BFloat16, CustomMatchesCublasAndUsesBf16Storage) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP();
  }
  TinyQwen2Model tiny;
  infer::Qwen2Model fp32(tiny.path(), cpu_options(64));
  infer::Qwen2Model custom(
      tiny.path(),
      cuda_bf16_options(64, infer::LinearKernel::kCustom));
  infer::Qwen2Model cublas(
      tiny.path(),
      cuda_bf16_options(64, infer::LinearKernel::kCublas));
  const auto prompt = prompt_tokens(32);

  int token = custom.prefill(prompt);
  ASSERT_EQ(cublas.prefill(prompt), token);
  EXPECT_GE(logits_cosine(custom, cublas), 0.9999);
  for (int step = 0; step < 8; ++step) {
    const int custom_next = custom.decode_next(token);
    const int cublas_next = cublas.decode_next(token);
    ASSERT_EQ(cublas_next, custom_next);
    EXPECT_GE(logits_cosine(custom, cublas), 0.9999);
    token = custom_next;
  }

  EXPECT_EQ(cublas.kv_cache_bytes() * 2, fp32.kv_cache_bytes());
  EXPECT_LE(cublas.workspace_bytes(),
            static_cast<size_t>(fp32.workspace_bytes() * 0.55));
  EXPECT_EQ(archive_payload_bytes(cublas.archive()) * 2,
            archive_payload_bytes(fp32.archive()));
}
