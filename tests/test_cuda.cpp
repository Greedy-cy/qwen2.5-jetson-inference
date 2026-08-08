#include "infer/cuda_ops.hpp"
#include "infer/buffer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>


namespace {

uint16_t float_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t round = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<uint16_t>((bits + round) >> 16u);
}

float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16u;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float bf16_round(float value) { return bf16_to_float(float_to_bf16(value)); }

std::vector<uint16_t> encode_bf16(const std::vector<float>& values) {
  std::vector<uint16_t> result(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    result[i] = float_to_bf16(values[i]);
  }
  return result;
}

std::vector<float> decode_bf16(const std::vector<uint16_t>& values) {
  std::vector<float> result(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    result[i] = bf16_to_float(values[i]);
  }
  return result;
}

std::vector<float> read_bf16(const infer::Buffer& buffer, size_t count,
                             infer::cuda::Context& context) {
  std::vector<uint16_t> bits(count);
  cudaMemcpyAsync(bits.data(), buffer.data(), count * sizeof(uint16_t),
                  cudaMemcpyDeviceToHost, context.stream());
  context.synchronize();
  return decode_bf16(bits);
}

const __nv_bfloat16* bf16_data(const infer::Buffer& buffer) {
  return static_cast<const __nv_bfloat16*>(buffer.data());
}

__nv_bfloat16* bf16_data(infer::Buffer& buffer) {
  return static_cast<__nv_bfloat16*>(buffer.data());
}

void upload_bf16(infer::Buffer& buffer, const std::vector<float>& values,
                 infer::cuda::Context& context) {
  const auto bits = encode_bf16(values);
  ASSERT_EQ(cudaMemcpyAsync(buffer.data(), bits.data(),
                            bits.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice, context.stream()),
            cudaSuccess);
}

void expect_values_near(const std::vector<float>& actual,
                        const std::vector<float>& expected, float tolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
  }
}

bool cuda_available() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

}  // namespace


TEST(CudaOps, RmsNormMatchesExpected) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  infer::cuda::Context context;
  const float host_input[] = {1, 2, 3, 4};
  const float host_weight[] = {1, 1, 1, 1};
  float* input = nullptr;
  float* weight = nullptr;
  float* output = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&input), sizeof(host_input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&weight), sizeof(host_weight)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(host_input)), cudaSuccess);
  cudaMemcpyAsync(input, host_input, sizeof(host_input), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(weight, host_weight, sizeof(host_weight), cudaMemcpyHostToDevice, context.stream());
  infer::cuda::rms_norm(input, weight, output, 4, 0.0f, context.stream());
  float host_output[4]{};
  cudaMemcpyAsync(host_output, output, sizeof(host_output), cudaMemcpyDeviceToHost, context.stream());
  context.synchronize();
  const float scale = 1.0f / std::sqrt(7.5f);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(host_output[i], host_input[i] * scale, 1e-5f);
  cudaFree(input);
  cudaFree(weight);
  cudaFree(output);
}

TEST(CudaOps, Fp32Gemv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  infer::cuda::Context context;
  const float host_weight[] = {1, 2, 3, 4, -1, 0.5f, 2, -3};
  const float host_input[] = {2, -1, 0.5f, 3};
  const float host_bias[] = {0.25f, -0.5f};
  infer::Buffer weight(sizeof(host_weight), infer::Device::kCuda);
  infer::Buffer input(sizeof(host_input), infer::Device::kCuda);
  infer::Buffer bias(sizeof(host_bias), infer::Device::kCuda);
  infer::Buffer output(2 * sizeof(float), infer::Device::kCuda);
  cudaMemcpyAsync(weight.data(), host_weight, sizeof(host_weight),
                  cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(input.data(), host_input, sizeof(host_input),
                  cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(bias.data(), host_bias, sizeof(host_bias),
                  cudaMemcpyHostToDevice, context.stream());

  auto copy_output = [&]() {
    float result[2]{};
    cudaMemcpyAsync(result, output.data(), sizeof(result),
                    cudaMemcpyDeviceToHost, context.stream());
    context.synchronize();
    return std::array<float, 2>{result[0], result[1]};
  };
  infer::cuda::gemv_fp32(
      static_cast<const float*>(weight.data()),
      static_cast<const float*>(bias.data()),
      static_cast<const float*>(input.data()),
      static_cast<float*>(output.data()), 2, 4, context.stream());
  const auto fp32 = copy_output();
  EXPECT_NEAR(fp32[0], 1 * 2 + 2 * -1 + 3 * 0.5f + 4 * 3 + 0.25f, 1e-5f);
  EXPECT_NEAR(fp32[1], -1 * 2 + 0.5f * -1 + 2 * 0.5f - 3 * 3 - 0.5f,
              1e-5f);

  infer::cuda::gemv_fp32_cublas(
      context.cublas(), static_cast<const float*>(weight.data()),
      static_cast<const float*>(bias.data()),
      static_cast<const float*>(input.data()),
      static_cast<float*>(output.data()), 2, 4, context.stream());
  const auto cublas = copy_output();
  EXPECT_NEAR(cublas[0], fp32[0], 1e-5f);
  EXPECT_NEAR(cublas[1], fp32[1], 1e-5f);
}

TEST(CudaOps, TiledAndCublasFp32Gemm) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  infer::cuda::Context context;
  constexpr int tokens = 3;
  constexpr int out_features = 5;
  constexpr int in_features = 8;
  std::array<float, out_features * in_features> weight{};
  std::array<float, tokens * in_features> input{};
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  }
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i % 5) - 2.0f;
  }
  infer::Buffer d_weight(sizeof(weight), infer::Device::kCuda);
  infer::Buffer d_input(sizeof(input), infer::Device::kCuda);
  infer::Buffer d_output(
      tokens * out_features * sizeof(float), infer::Device::kCuda);
  cudaMemcpyAsync(d_weight.data(), weight.data(), sizeof(weight),
                  cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(d_input.data(), input.data(), sizeof(input),
                  cudaMemcpyHostToDevice, context.stream());
  auto read_output = [&]() {
    std::array<float, tokens * out_features> values{};
    cudaMemcpyAsync(values.data(), d_output.data(), sizeof(values),
                    cudaMemcpyDeviceToHost, context.stream());
    context.synchronize();
    return values;
  };
  infer::cuda::gemm_fp32(
      static_cast<const float*>(d_weight.data()),
      static_cast<const float*>(d_input.data()),
      static_cast<float*>(d_output.data()), tokens, out_features,
      in_features, context.stream());
  const auto fp32 = read_output();
  infer::cuda::gemm_fp32_cublas(
      context.cublas(), static_cast<const float*>(d_weight.data()),
      static_cast<const float*>(d_input.data()),
      static_cast<float*>(d_output.data()), tokens, out_features, in_features,
      context.stream());
  const auto cublas = read_output();
  for (int token = 0; token < tokens; ++token) {
    for (int row = 0; row < out_features; ++row) {
      float expected = 0.0f;
      for (int col = 0; col < in_features; ++col) {
        expected += input[token * in_features + col] *
                    weight[row * in_features + col];
      }
      EXPECT_NEAR(fp32[token * out_features + row], expected, 1e-5f);
      EXPECT_NEAR(cublas[token * out_features + row], expected, 1e-5f);
    }
  }
}

TEST(CudaBFloat16, DataMovementNormalizationAndElementwiseMatchFp32) {
  if (!cuda_available()) GTEST_SKIP();
  infer::cuda::Context context;
  constexpr int vocab = 3;
  constexpr int hidden = 6;
  constexpr int token_count = 2;
  const std::vector<float> table = {
      -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f,
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      1.0f, -1.25f, 1.5f, -1.75f, 2.0f, -2.25f};
  const std::array<int, token_count> tokens = {2, 0};
  const std::vector<float> norm_weight =
      {0.75f, 0.875f, 1.0f, 1.125f, 1.25f, 1.375f};

  infer::Buffer d_table(table.size() * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_tokens(sizeof(tokens), infer::Device::kCuda);
  infer::Buffer d_embedding(token_count * hidden * sizeof(uint16_t),
                            infer::Device::kCuda);
  infer::Buffer d_single(hidden * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_weight(hidden * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_norm(token_count * hidden * sizeof(uint16_t),
                       infer::Device::kCuda);
  upload_bf16(d_table, table, context);
  upload_bf16(d_weight, norm_weight, context);
  ASSERT_EQ(cudaMemcpyAsync(d_tokens.data(), tokens.data(), sizeof(tokens),
                            cudaMemcpyHostToDevice, context.stream()),
            cudaSuccess);

  infer::cuda::embedding_bf16(bf16_data(d_table), tokens[0],
                               bf16_data(d_single), hidden, context.stream());
  auto single = read_bf16(d_single, hidden, context);
  const auto rounded_table = decode_bf16(encode_bf16(table));
  std::vector<float> expected_single(
      rounded_table.begin() + tokens[0] * hidden,
      rounded_table.begin() + (tokens[0] + 1) * hidden);
  expect_values_near(single, expected_single, 0.0f);

  infer::cuda::embedding_batch_bf16(
      bf16_data(d_table), static_cast<const int*>(d_tokens.data()),
      bf16_data(d_embedding), token_count, hidden, context.stream());
  const auto embedded = read_bf16(d_embedding, token_count * hidden, context);
  std::vector<float> expected_embedding(token_count * hidden);
  for (int token = 0; token < token_count; ++token) {
    for (int column = 0; column < hidden; ++column) {
      expected_embedding[token * hidden + column] =
          rounded_table[tokens[token] * hidden + column];
    }
  }
  expect_values_near(embedded, expected_embedding, 0.0f);

  constexpr float epsilon = 1.0e-5f;
  infer::cuda::rms_norm_batch_bf16(
      bf16_data(d_embedding), bf16_data(d_weight), bf16_data(d_norm),
      token_count, hidden, epsilon, context.stream());
  const auto norm = read_bf16(d_norm, token_count * hidden, context);
  const auto rounded_weight = decode_bf16(encode_bf16(norm_weight));
  std::vector<float> expected_norm(token_count * hidden);
  for (int token = 0; token < token_count; ++token) {
    float sum = 0.0f;
    for (int column = 0; column < hidden; ++column) {
      const float value = expected_embedding[token * hidden + column];
      sum += value * value;
    }
    const float scale = 1.0f / std::sqrt(sum / hidden + epsilon);
    for (int column = 0; column < hidden; ++column) {
      expected_norm[token * hidden + column] =
          expected_embedding[token * hidden + column] * scale *
          rounded_weight[column];
    }
  }
  expect_values_near(norm, expected_norm, 0.016f);

  infer::cuda::rms_norm_bf16(
      bf16_data(d_single), bf16_data(d_weight), bf16_data(d_norm), hidden,
      epsilon, context.stream());
  const auto single_norm = read_bf16(d_norm, hidden, context);
  expect_values_near(
      single_norm,
      std::vector<float>(expected_norm.begin(), expected_norm.begin() + hidden),
      0.016f);

  constexpr int element_count = 7;
  const std::vector<float> x =
      {-1.0f, -0.5f, 0.0f, 0.25f, 0.75f, 1.25f, 2.0f};
  const std::vector<float> residual =
      {0.125f, -0.25f, 0.5f, 0.75f, -1.0f, 1.5f, -0.375f};
  const std::vector<float> up =
      {1.5f, -1.25f, 1.0f, 0.75f, -0.5f, 0.25f, 2.0f};
  infer::Buffer d_x(element_count * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_residual(element_count * sizeof(uint16_t),
                           infer::Device::kCuda);
  infer::Buffer d_up(element_count * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_element_output(element_count * sizeof(uint16_t),
                                 infer::Device::kCuda);
  infer::Buffer d_argmax(sizeof(int), infer::Device::kCuda);
  upload_bf16(d_x, x, context);
  upload_bf16(d_residual, residual, context);
  upload_bf16(d_up, up, context);
  const auto rounded_x = decode_bf16(encode_bf16(x));
  const auto rounded_residual = decode_bf16(encode_bf16(residual));
  const auto rounded_up = decode_bf16(encode_bf16(up));

  infer::cuda::add_inplace_bf16(bf16_data(d_x), bf16_data(d_residual),
                                 element_count, context.stream());
  const auto added = read_bf16(d_x, element_count, context);
  std::vector<float> expected_added(element_count);
  for (int i = 0; i < element_count; ++i) {
    expected_added[i] = rounded_x[i] + rounded_residual[i];
  }
  expect_values_near(added, expected_added, 0.008f);

  upload_bf16(d_x, x, context);
  infer::cuda::silu_mul_bf16(bf16_data(d_x), bf16_data(d_up),
                              bf16_data(d_element_output), element_count,
                              context.stream());
  const auto silu = read_bf16(d_element_output, element_count, context);
  std::vector<float> expected_silu(element_count);
  for (int i = 0; i < element_count; ++i) {
    expected_silu[i] =
        rounded_x[i] / (1.0f + std::exp(-rounded_x[i])) * rounded_up[i];
  }
  expect_values_near(silu, expected_silu, 0.008f);

  infer::cuda::argmax_bf16(bf16_data(d_element_output), element_count,
                            static_cast<int*>(d_argmax.data()),
                            context.stream());
  int maximum_index = -1;
  ASSERT_EQ(cudaMemcpyAsync(&maximum_index, d_argmax.data(), sizeof(int),
                            cudaMemcpyDeviceToHost, context.stream()),
            cudaSuccess);
  context.synchronize();
  EXPECT_EQ(maximum_index,
            static_cast<int>(std::max_element(silu.begin(), silu.end()) -
                             silu.begin()));
}

namespace {

void rope_reference(std::vector<float>& data, int token_count, int heads,
                    int head_dim, int start_position, float theta) {
  const int half = head_dim / 2;
  for (int token = 0; token < token_count; ++token) {
    for (int head = 0; head < heads; ++head) {
      float* values =
          data.data() + (static_cast<size_t>(token) * heads + head) * head_dim;
      for (int index = 0; index < half; ++index) {
        const float frequency =
            std::pow(theta, -2.0f * index / static_cast<float>(head_dim));
        const float angle = (start_position + token) * frequency;
        const float first = values[index];
        const float second = values[index + half];
        values[index] = first * std::cos(angle) - second * std::sin(angle);
        values[index + half] =
            second * std::cos(angle) + first * std::sin(angle);
      }
    }
  }
}

std::vector<float> causal_attention_reference(
    const std::vector<float>& q, const std::vector<float>& key_cache,
    const std::vector<float>& value_cache, int token_count, int num_heads,
    int num_kv_heads, int head_dim, int max_sequence) {
  std::vector<float> output(
      static_cast<size_t>(token_count) * num_heads * head_dim);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int token = 0; token < token_count; ++token) {
    const int sequence_length = token + 1;
    for (int head = 0; head < num_heads; ++head) {
      const int kv_head = head / (num_heads / num_kv_heads);
      const float* query =
          q.data() +
          (static_cast<size_t>(token) * num_heads + head) * head_dim;
      std::vector<float> probabilities(sequence_length);
      float maximum = -std::numeric_limits<float>::infinity();
      for (int position = 0; position < sequence_length; ++position) {
        const float* key =
            key_cache.data() +
            (static_cast<size_t>(kv_head) * max_sequence + position) *
                head_dim;
        float dot = 0.0f;
        for (int dimension = 0; dimension < head_dim; ++dimension) {
          dot += query[dimension] * key[dimension];
        }
        probabilities[position] = dot * scale;
        maximum = std::max(maximum, probabilities[position]);
      }
      float denominator = 0.0f;
      for (float& probability : probabilities) {
        probability = std::exp(probability - maximum);
        denominator += probability;
      }
      for (int dimension = 0; dimension < head_dim; ++dimension) {
        float sum = 0.0f;
        for (int position = 0; position < sequence_length; ++position) {
          const float* value =
              value_cache.data() +
              (static_cast<size_t>(kv_head) * max_sequence + position) *
                  head_dim;
          sum += probabilities[position] / denominator * value[dimension];
        }
        output[(static_cast<size_t>(token) * num_heads + head) * head_dim +
               dimension] = sum;
      }
    }
  }
  return output;
}

}  // namespace

TEST(CudaBFloat16, RopeKvAndAttentionMatchFp32) {
  if (!cuda_available()) GTEST_SKIP();
  infer::cuda::Context context;
  constexpr int token_count = 3;
  constexpr int num_heads = 4;
  constexpr int num_kv_heads = 2;
  constexpr int head_dim = 4;
  constexpr int max_sequence = 5;
  constexpr int start_position = 2;
  constexpr float theta = 10000.0f;
  constexpr int q_count = token_count * num_heads * head_dim;
  constexpr int kv_count = token_count * num_kv_heads * head_dim;
  constexpr int cache_count = num_kv_heads * max_sequence * head_dim;

  std::vector<float> q(q_count);
  std::vector<float> key(kv_count);
  std::vector<float> value(kv_count);
  for (int i = 0; i < q_count; ++i) {
    q[i] = (static_cast<int>(i % 11) - 5) * 0.08f;
  }
  for (int i = 0; i < kv_count; ++i) {
    key[i] = (static_cast<int>(i % 7) - 3) * 0.07f;
    value[i] = (static_cast<int>(i % 9) - 4) * 0.06f;
  }

  infer::Buffer d_q(q_count * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_key(kv_count * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_value(kv_count * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_key_cache(cache_count * sizeof(uint16_t),
                            infer::Device::kCuda);
  infer::Buffer d_value_cache(cache_count * sizeof(uint16_t),
                              infer::Device::kCuda);
  infer::Buffer d_output(q_count * sizeof(uint16_t), infer::Device::kCuda);
  upload_bf16(d_q, q, context);
  upload_bf16(d_key, key, context);
  upload_bf16(d_value, value, context);
  ASSERT_EQ(cudaMemsetAsync(d_key_cache.data(), 0, d_key_cache.bytes(),
                            context.stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(d_value_cache.data(), 0, d_value_cache.bytes(),
                            context.stream()),
            cudaSuccess);

  auto q_reference = decode_bf16(encode_bf16(q));
  auto key_reference = decode_bf16(encode_bf16(key));
  const auto value_reference = decode_bf16(encode_bf16(value));
  rope_reference(q_reference, token_count, num_heads, head_dim, start_position,
                 theta);
  rope_reference(key_reference, token_count, num_kv_heads, head_dim,
                 start_position, theta);

  infer::cuda::rope_batch_bf16(
      bf16_data(d_q), bf16_data(d_key), token_count, num_heads, num_kv_heads,
      head_dim, start_position, theta, context.stream());
  const auto q_after_rope = read_bf16(d_q, q_count, context);
  const auto key_after_rope = read_bf16(d_key, kv_count, context);
  expect_values_near(q_after_rope, q_reference, 0.004f);
  expect_values_near(key_after_rope, key_reference, 0.004f);

  infer::cuda::store_kv_batch_bf16(
      bf16_data(d_key), bf16_data(d_value), bf16_data(d_key_cache),
      bf16_data(d_value_cache), token_count, num_kv_heads, head_dim,
      max_sequence, context.stream());
  const auto key_cache = read_bf16(d_key_cache, cache_count, context);
  const auto value_cache = read_bf16(d_value_cache, cache_count, context);
  std::vector<float> expected_key_cache(cache_count, 0.0f);
  std::vector<float> expected_value_cache(cache_count, 0.0f);
  for (int token = 0; token < token_count; ++token) {
    for (int head = 0; head < num_kv_heads; ++head) {
      for (int dimension = 0; dimension < head_dim; ++dimension) {
        const size_t source =
            (static_cast<size_t>(token) * num_kv_heads + head) * head_dim +
            dimension;
        const size_t destination =
            (static_cast<size_t>(head) * max_sequence + token) * head_dim +
            dimension;
        expected_key_cache[destination] = key_after_rope[source];
        expected_value_cache[destination] = value_reference[source];
      }
    }
  }
  expect_values_near(key_cache, expected_key_cache, 0.0f);
  expect_values_near(value_cache, expected_value_cache, 0.0f);

  infer::cuda::attention_prefill_bf16(
      bf16_data(d_q), bf16_data(d_key_cache), bf16_data(d_value_cache),
      bf16_data(d_output), token_count, num_heads, num_kv_heads, head_dim,
      max_sequence, context.stream());
  const auto prefill_output = read_bf16(d_output, q_count, context);
  const auto expected_attention = causal_attention_reference(
      q_after_rope, key_cache, value_cache, token_count, num_heads,
      num_kv_heads, head_dim, max_sequence);
  expect_values_near(prefill_output, expected_attention, 0.004f);

  const auto* last_query =
      bf16_data(d_q) + static_cast<size_t>(token_count - 1) * num_heads *
                              head_dim;
  infer::cuda::attention_decode_bf16(
      last_query, bf16_data(d_key_cache), bf16_data(d_value_cache),
      bf16_data(d_output), num_heads, num_kv_heads, head_dim, token_count,
      max_sequence, context.stream());
  const auto decode_output =
      read_bf16(d_output, num_heads * head_dim, context);
  const std::vector<float> expected_decode(
      prefill_output.end() - num_heads * head_dim, prefill_output.end());
  expect_values_near(decode_output, expected_decode, 0.0f);

  const size_t q_row = static_cast<size_t>(num_heads) * head_dim;
  const size_t kv_row = static_cast<size_t>(num_kv_heads) * head_dim;
  const std::vector<float> single_q(
      q.begin() + (token_count - 1) * q_row, q.end());
  const std::vector<float> single_key(
      key.begin() + (token_count - 1) * kv_row, key.end());
  const std::vector<float> single_value(
      value.begin() + (token_count - 1) * kv_row, value.end());
  infer::Buffer d_single_q(q_row * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_single_key(kv_row * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_single_value(kv_row * sizeof(uint16_t),
                               infer::Device::kCuda);
  infer::Buffer d_single_key_cache(cache_count * sizeof(uint16_t),
                                   infer::Device::kCuda);
  infer::Buffer d_single_value_cache(cache_count * sizeof(uint16_t),
                                     infer::Device::kCuda);
  upload_bf16(d_single_q, single_q, context);
  upload_bf16(d_single_key, single_key, context);
  upload_bf16(d_single_value, single_value, context);
  ASSERT_EQ(cudaMemsetAsync(d_single_key_cache.data(), 0,
                            d_single_key_cache.bytes(), context.stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(d_single_value_cache.data(), 0,
                            d_single_value_cache.bytes(), context.stream()),
            cudaSuccess);

  infer::cuda::rope_bf16(
      bf16_data(d_single_q), bf16_data(d_single_key), num_heads,
      num_kv_heads, head_dim, start_position + token_count - 1, theta,
      context.stream());
  const auto single_q_after = read_bf16(d_single_q, q_row, context);
  const auto single_key_after = read_bf16(d_single_key, kv_row, context);
  expect_values_near(
      single_q_after,
      std::vector<float>(q_after_rope.end() - q_row, q_after_rope.end()),
      0.0f);
  expect_values_near(
      single_key_after,
      std::vector<float>(key_after_rope.end() - kv_row, key_after_rope.end()),
      0.0f);

  infer::cuda::store_kv_bf16(
      bf16_data(d_single_key), bf16_data(d_single_value),
      bf16_data(d_single_key_cache), bf16_data(d_single_value_cache),
      num_kv_heads, head_dim, token_count - 1, max_sequence,
      context.stream());
  const auto single_key_cache =
      read_bf16(d_single_key_cache, cache_count, context);
  const auto single_value_cache =
      read_bf16(d_single_value_cache, cache_count, context);
  for (int head = 0; head < num_kv_heads; ++head) {
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      const size_t source = static_cast<size_t>(head) * head_dim + dimension;
      const size_t destination =
          (static_cast<size_t>(head) * max_sequence + token_count - 1) *
              head_dim +
          dimension;
      EXPECT_EQ(single_key_cache[destination], single_key_after[source]);
      EXPECT_EQ(single_value_cache[destination],
                bf16_round(single_value[source]));
    }
  }
}

namespace {

std::vector<float> linear_reference(const std::vector<float>& weight,
                                    const std::vector<float>* bias,
                                    const std::vector<float>& input, int tokens,
                                    int out_features, int in_features) {
  std::vector<float> output(static_cast<size_t>(tokens) * out_features);
  for (int token = 0; token < tokens; ++token) {
    for (int row = 0; row < out_features; ++row) {
      float sum = bias ? (*bias)[row] : 0.0f;
      for (int column = 0; column < in_features; ++column) {
        sum += weight[static_cast<size_t>(row) * in_features + column] *
               input[static_cast<size_t>(token) * in_features + column];
      }
      output[static_cast<size_t>(token) * out_features + row] = sum;
    }
  }
  return output;
}

std::vector<float> w8a16_reference(
    const std::vector<int8_t>& weight, const std::vector<float>& scales,
    int group_size, const std::vector<float>* bias,
    const std::vector<float>& input, int tokens, int out_features,
    int in_features) {
  const int groups = (in_features + group_size - 1) / group_size;
  std::vector<float> output(static_cast<size_t>(tokens) * out_features);
  for (int token = 0; token < tokens; ++token) {
    for (int row = 0; row < out_features; ++row) {
      float sum = 0.0f;
      for (int column = 0; column < in_features; ++column) {
        const float dequantized =
            static_cast<float>(
                weight[static_cast<size_t>(row) * in_features + column]) *
            scales[static_cast<size_t>(row) * groups + column / group_size];
        sum = fmaf(dequantized,
                   input[static_cast<size_t>(token) * in_features + column],
                   sum);
      }
      if (bias) sum += (*bias)[row];
      output[static_cast<size_t>(token) * out_features + row] =
          bf16_round(sum);
    }
  }
  return output;
}

void check_w8a16_linear_case(infer::cuda::Context& context, int tokens,
                             int out_features, int in_features,
                             bool with_bias) {
  constexpr int group_size = 64;
  const int groups = (in_features + group_size - 1) / group_size;
  std::vector<int8_t> weight(static_cast<size_t>(out_features) * in_features);
  std::vector<float> scales(static_cast<size_t>(out_features) * groups);
  std::vector<float> input(static_cast<size_t>(tokens) * in_features);
  std::vector<float> bias(out_features);
  for (size_t index = 0; index < weight.size(); ++index) {
    weight[index] =
        static_cast<int8_t>((static_cast<int>(index * 17U % 31U)) - 15);
  }
  if (weight.size() >= 2) {
    weight[0] = -127;
    weight[1] = 127;
  }
  for (size_t index = 0; index < scales.size(); ++index) {
    scales[index] = (1 + static_cast<int>(index % 7)) / 1024.0f;
  }
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] =
        (static_cast<int>(index * 13U % 29U) - 14) / 64.0f;
  }
  for (int row = 0; row < out_features; ++row) {
    bias[row] = (row % 9 - 4) / 32.0f;
  }
  if (in_features > group_size) {
    const int row = out_features - 1;
    std::fill_n(weight.begin() + static_cast<size_t>(row) * in_features,
                group_size, static_cast<int8_t>(0));
    scales[static_cast<size_t>(row) * groups] = 1.0f;
  }

  const auto rounded_scales = decode_bf16(encode_bf16(scales));
  const auto rounded_input = decode_bf16(encode_bf16(input));
  const auto rounded_bias = decode_bf16(encode_bf16(bias));
  const auto expected = w8a16_reference(
      weight, rounded_scales, group_size,
      with_bias ? &rounded_bias : nullptr, rounded_input, tokens,
      out_features, in_features);

  infer::Buffer d_weight(weight.size() * sizeof(int8_t),
                         infer::Device::kCuda);
  infer::Buffer d_scales(scales.size() * sizeof(uint16_t),
                         infer::Device::kCuda);
  infer::Buffer d_input(input.size() * sizeof(uint16_t),
                        infer::Device::kCuda);
  infer::Buffer d_bias(bias.size() * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_gemv(static_cast<size_t>(out_features) * sizeof(uint16_t),
                       infer::Device::kCuda);
  infer::Buffer d_gemm(static_cast<size_t>(tokens) * out_features *
                           sizeof(uint16_t),
                       infer::Device::kCuda);
  ASSERT_EQ(cudaMemcpyAsync(d_weight.data(), weight.data(), d_weight.bytes(),
                            cudaMemcpyHostToDevice, context.stream()),
            cudaSuccess);
  upload_bf16(d_scales, scales, context);
  upload_bf16(d_input, input, context);
  upload_bf16(d_bias, bias, context);
  const auto* bias_pointer = with_bias ? bf16_data(d_bias) : nullptr;

  infer::cuda::gemv_w8a16(
      static_cast<const int8_t*>(d_weight.data()), bf16_data(d_scales),
      group_size, bias_pointer, bf16_data(d_input), bf16_data(d_gemv),
      out_features, in_features, context.stream());
  infer::cuda::gemm_w8a16(
      static_cast<const int8_t*>(d_weight.data()), bf16_data(d_scales),
      group_size, bias_pointer, bf16_data(d_input), bf16_data(d_gemm), tokens,
      out_features, in_features, context.stream());
  const auto gemv = read_bf16(d_gemv, out_features, context);
  const auto gemm = read_bf16(
      d_gemm, static_cast<size_t>(tokens) * out_features, context);
  float max_abs_error = 0.0f;
  for (size_t index = 0; index < gemm.size(); ++index) {
    max_abs_error =
        std::max(max_abs_error, std::abs(gemm[index] - expected[index]));
    const float tolerance =
        std::max(0.015625f, std::abs(expected[index]) * 0.01f);
    EXPECT_NEAR(gemm[index], expected[index], tolerance) << "index " << index;
  }
  std::cout << "W8A16_LINEAR tokens=" << tokens
            << " out_features=" << out_features
            << " in_features=" << in_features
            << " max_abs_error=" << max_abs_error << '\n';
  for (int row = 0; row < out_features; ++row) {
    EXPECT_EQ(gemv[row], gemm[row]) << "row " << row;
  }
}

void expect_linear_near(const std::vector<float>& actual,
                        const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    const float tolerance =
        std::max(0.004f, std::abs(expected[index]) * 0.008f);
    EXPECT_NEAR(actual[index], expected[index], tolerance) << "index " << index;
  }
}

void check_bf16_linear_case(infer::cuda::Context& context, int tokens,
                            int out_features, int in_features,
                            bool with_bias) {
  std::vector<float> weight(static_cast<size_t>(out_features) * in_features);
  std::vector<float> input(static_cast<size_t>(tokens) * in_features);
  std::vector<float> bias(out_features);
  for (size_t index = 0; index < weight.size(); ++index) {
    weight[index] =
        (static_cast<int>(index % 17) - 8) * 0.015625f;
  }
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = (static_cast<int>(index % 13) - 6) * 0.03125f;
  }
  for (int row = 0; row < out_features; ++row) {
    bias[row] = (row % 5 - 2) * 0.0625f;
  }
  const auto rounded_weight = decode_bf16(encode_bf16(weight));
  const auto rounded_input = decode_bf16(encode_bf16(input));
  const auto rounded_bias = decode_bf16(encode_bf16(bias));
  const auto expected = linear_reference(
      rounded_weight, with_bias ? &rounded_bias : nullptr, rounded_input,
      tokens, out_features, in_features);

  infer::Buffer d_weight(weight.size() * sizeof(uint16_t),
                         infer::Device::kCuda);
  infer::Buffer d_input(input.size() * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_bias(bias.size() * sizeof(uint16_t), infer::Device::kCuda);
  infer::Buffer d_custom(static_cast<size_t>(tokens) * out_features *
                             sizeof(uint16_t),
                         infer::Device::kCuda);
  infer::Buffer d_cublas(static_cast<size_t>(tokens) * out_features *
                             sizeof(uint16_t),
                         infer::Device::kCuda);
  upload_bf16(d_weight, weight, context);
  upload_bf16(d_input, input, context);
  upload_bf16(d_bias, bias, context);
  const auto* bias_pointer = with_bias ? bf16_data(d_bias) : nullptr;

  infer::cuda::gemv_bf16(
      bf16_data(d_weight), bias_pointer, bf16_data(d_input),
      bf16_data(d_custom), out_features, in_features, context.stream());
  infer::cuda::gemv_bf16_cublas(
      context.cublas(), bf16_data(d_weight), bias_pointer, bf16_data(d_input),
      bf16_data(d_cublas), out_features, in_features, context.stream());
  const auto custom_gemv = read_bf16(d_custom, out_features, context);
  const auto cublas_gemv = read_bf16(d_cublas, out_features, context);
  expect_linear_near(
      custom_gemv,
      std::vector<float>(expected.begin(), expected.begin() + out_features));
  expect_linear_near(
      cublas_gemv,
      std::vector<float>(expected.begin(), expected.begin() + out_features));
  expect_linear_near(custom_gemv, cublas_gemv);

  infer::cuda::gemm_bf16(
      bf16_data(d_weight), bias_pointer, bf16_data(d_input),
      bf16_data(d_custom), tokens, out_features, in_features,
      context.stream());
  infer::cuda::gemm_bf16_cublas(
      context.cublas(), bf16_data(d_weight), bias_pointer, bf16_data(d_input),
      bf16_data(d_cublas), tokens, out_features, in_features,
      context.stream());
  const auto custom_gemm =
      read_bf16(d_custom, static_cast<size_t>(tokens) * out_features, context);
  const auto cublas_gemm =
      read_bf16(d_cublas, static_cast<size_t>(tokens) * out_features, context);
  expect_linear_near(custom_gemm, expected);
  expect_linear_near(cublas_gemm, expected);
  expect_linear_near(custom_gemm, cublas_gemm);
}

}  // namespace

TEST(CudaBFloat16, LinearCustomAndCublasMatchFp32Reference) {
  if (!cuda_available()) GTEST_SKIP();
  infer::cuda::Context context;
  check_bf16_linear_case(context, 3, 11, 14, true);
  check_bf16_linear_case(context, 2, 5, 7, true);
  check_bf16_linear_case(context, 3, 23, 14, false);
  check_bf16_linear_case(context, 32, 64, 64, true);
  check_bf16_linear_case(context, 31, 37, 33, true);
  check_bf16_linear_case(context, 65, 70, 67, false);
}

TEST(CudaW8A16, GemvAndGemmHandleGroupBoundaryTailAndBias) {
  if (!cuda_available()) GTEST_SKIP();
  infer::cuda::Context context;
  check_w8a16_linear_case(context, 3, 5, 70, true);
  check_w8a16_linear_case(context, 2, 7, 65, false);
  EXPECT_THROW(
      infer::cuda::gemv_w8a16(nullptr, nullptr, 32, nullptr, nullptr, nullptr,
                              1, 1, context.stream()),
      infer::Error);
}

TEST(CudaW8A16, GemvAndGemmMatchDequantizedReferenceAtQwenShapes) {
  if (!cuda_available()) GTEST_SKIP();
  infer::cuda::Context context;
  check_w8a16_linear_case(context, 2, 896, 896, true);
  check_w8a16_linear_case(context, 2, 128, 896, true);
  check_w8a16_linear_case(context, 2, 4864, 896, false);
  check_w8a16_linear_case(context, 2, 896, 4864, false);
}
