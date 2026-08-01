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
  }

  std::filesystem::path path_;
};

infer::RuntimeOptions cpu_options(int max_sequence_length = 32) {
  infer::RuntimeOptions options;
  options.backend = infer::Device::kCpu;
  options.precision = infer::Precision::kFloat32;
  options.max_sequence_length = max_sequence_length;
  options.prefill_mode = infer::PrefillMode::kSerial;
  return options;
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

TEST(PrefillMode, ParseAndFormat) {
  EXPECT_EQ(infer::parse_prefill_mode("auto"), infer::PrefillMode::kAuto);
  EXPECT_EQ(infer::parse_prefill_mode("serial"), infer::PrefillMode::kSerial);
  EXPECT_EQ(infer::parse_prefill_mode("batched"), infer::PrefillMode::kBatched);
  EXPECT_STREQ(infer::to_string(infer::PrefillMode::kAuto), "auto");
  EXPECT_STREQ(infer::to_string(infer::PrefillMode::kSerial), "serial");
  EXPECT_STREQ(infer::to_string(infer::PrefillMode::kBatched), "batched");
  EXPECT_THROW(infer::parse_prefill_mode("legacy"), infer::Error);
}

TEST(Qwen2State, SerialPrefillMatchesExplicitForwardAndDecode) {
  TinyQwen2Model tiny;
  infer::Qwen2Model reference(tiny.path(), cpu_options());
  infer::Qwen2Model stateful(tiny.path(), cpu_options());
  const std::vector<int> prompt{1, 2, 3};

  int reference_next = -1;
  for (size_t i = 0; i < prompt.size(); ++i) {
    reference_next = reference.forward_token(prompt[i], static_cast<int>(i));
  }
  const int stateful_next = stateful.prefill(prompt);
  EXPECT_EQ(stateful_next, reference_next);
  EXPECT_EQ(stateful.position(), 3);
  EXPECT_TRUE(stateful.has_prefilled());
  expect_logits_equal(reference, stateful);

  int token = stateful_next;
  for (int step = 0; step < 16; ++step) {
    reference_next = reference.forward_token(token, static_cast<int>(prompt.size()) + step);
    const int decoded = stateful.decode_next(token);
    EXPECT_EQ(decoded, reference_next);
    expect_logits_equal(reference, stateful);
    token = decoded;
  }
  EXPECT_EQ(stateful.position(), 19);
}

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
  auto options = cpu_options(8);
  options.prefill_mode = infer::PrefillMode::kAuto;
  infer::Qwen2Model model(tiny.path(), options);

  EXPECT_EQ(model.effective_prefill_mode(1), infer::PrefillMode::kSerial);
  EXPECT_EQ(model.effective_prefill_mode(2), infer::PrefillMode::kBatched);
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

TEST(Qwen2State, BatchedModeRunsAndAdvancesState) {
  TinyQwen2Model tiny;
  auto options = cpu_options();
  options.prefill_mode = infer::PrefillMode::kBatched;
  infer::Qwen2Model model(tiny.path(), options);

  EXPECT_EQ(model.effective_prefill_mode(2), infer::PrefillMode::kBatched);
  EXPECT_GT(model.prefill_workspace_bytes(), 0u);
  const int next = model.prefill({1, 2});
  EXPECT_GE(next, 0);
  EXPECT_LT(next, model.config().vocab_size);
  EXPECT_EQ(model.position(), 2);
  EXPECT_TRUE(model.has_prefilled());
  EXPECT_THROW(model.prefill({1}), infer::Error);
}

TEST(Qwen2BatchedPrefill, MatchesSerialAcrossRequiredLengths) {
  TinyQwen2Model tiny;
  constexpr std::array<int, 7> kLengths{1, 2, 31, 32, 127, 128, 512};
  for (const int length : kLengths) {
    SCOPED_TRACE("prompt_length=" + std::to_string(length));
    auto serial_options = cpu_options(512);
    auto batched_options = cpu_options(512);
    batched_options.prefill_mode = infer::PrefillMode::kBatched;
    infer::Qwen2Model serial(tiny.path(), serial_options);
    infer::Qwen2Model batched(tiny.path(), batched_options);
    const auto prompt = prompt_tokens(static_cast<size_t>(length));

    const int serial_next = serial.prefill(prompt);
    const int batched_next = batched.prefill(prompt);
    EXPECT_EQ(batched_next, serial_next);
    EXPECT_GE(logits_cosine(serial, batched), 0.999999);
    EXPECT_EQ(serial.position(), length);
    EXPECT_EQ(batched.position(), length);
    EXPECT_EQ(serial.prefill_workspace_bytes(), 0u);
    EXPECT_GT(batched.prefill_workspace_bytes(), 0u);
  }
}

TEST(Qwen2BatchedPrefill, DecodeStateMatchesSerialFor16Tokens) {
  TinyQwen2Model tiny;
  auto serial_options = cpu_options(32);
  auto batched_options = cpu_options(32);
  batched_options.prefill_mode = infer::PrefillMode::kBatched;
  infer::Qwen2Model serial(tiny.path(), serial_options);
  infer::Qwen2Model batched(tiny.path(), batched_options);
  const auto prompt = prompt_tokens(8);

  int token = serial.prefill(prompt);
  ASSERT_EQ(batched.prefill(prompt), token);
  EXPECT_GE(logits_cosine(serial, batched), 0.999999);
  for (int step = 0; step < 16; ++step) {
    SCOPED_TRACE("decode_step=" + std::to_string(step));
    const int serial_next = serial.decode_next(token);
    const int batched_next = batched.decode_next(token);
    ASSERT_EQ(batched_next, serial_next);
    EXPECT_GE(logits_cosine(serial, batched), 0.999999);
    token = serial_next;
  }
  EXPECT_EQ(serial.position(), 24);
  EXPECT_EQ(batched.position(), 24);
}

TEST(Qwen2BatchedPrefill, CudaAutoStaysSerialAndExplicitBatchedFails) {
  TinyQwen2Model tiny;
  auto auto_options = cpu_options(8);
  auto_options.backend = infer::Device::kCuda;
  auto_options.prefill_mode = infer::PrefillMode::kAuto;
  infer::Qwen2Model automatic(tiny.path(), auto_options);
  EXPECT_EQ(automatic.effective_prefill_mode(2), infer::PrefillMode::kSerial);
  EXPECT_EQ(automatic.prefill_workspace_bytes(), 0u);

  auto batched_options = auto_options;
  batched_options.prefill_mode = infer::PrefillMode::kBatched;
  infer::Qwen2Model batched(tiny.path(), batched_options);
  EXPECT_THROW(batched.prefill({1, 2}), infer::Error);
  EXPECT_EQ(batched.position(), 0);
  EXPECT_FALSE(batched.has_prefilled());
}
