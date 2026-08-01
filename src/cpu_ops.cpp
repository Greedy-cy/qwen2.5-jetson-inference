#include "infer/cpu_ops.hpp"

#include <cblas.h>

namespace infer::cpu {

void copy(const float* input, float* output, int n) {
  std::copy(input, input + n, output);
}

void add_inplace(float* x, const float* residual, int n) {
  for (int i = 0; i < n; ++i) x[i] += residual[i];
}

void rms_norm(const float* input, const float* weight, float* output,
              int n, float epsilon) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += static_cast<double>(input[i]) * input[i];
  const float scale = 1.0f / std::sqrt(static_cast<float>(sum / n) + epsilon);
  for (int i = 0; i < n; ++i) output[i] = input[i] * scale * weight[i];
}

void linear_fp32(const float* weight, const float* bias, const float* input,
                 float* output, int out_features, int in_features) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, out_features, in_features,
              1.0f, weight, in_features, input, 1, 0.0f, output, 1);
  if (bias) {
    for (int i = 0; i < out_features; ++i) output[i] += bias[i];
  }
}

void linear_int8(const int8_t* weight, const float* scales, int group_size,
                 const float* bias, const float* input, float* output,
                 int out_features, int in_features) {
  const int groups = (in_features + group_size - 1) / group_size;
  for (int row = 0; row < out_features; ++row) {
    float sum = bias ? bias[row] : 0.0f;
    for (int group = 0; group < groups; ++group) {
      const int begin = group * group_size;
      const int end = std::min(begin + group_size, in_features);
      float partial = 0.0f;
      for (int col = begin; col < end; ++col) {
        partial += static_cast<float>(weight[static_cast<size_t>(row) * in_features + col]) * input[col];
      }
      sum += partial * scales[static_cast<size_t>(row) * groups + group];
    }
    output[row] = sum;
  }
}

void rope(float* q, float* k, int num_heads, int num_kv_heads,
          int head_dim, int position, float theta) {
  const int half = head_dim / 2;
  auto rotate = [&](float* data, int heads) {
    for (int head = 0; head < heads; ++head) {
      float* values = data + static_cast<size_t>(head) * head_dim;
      for (int i = 0; i < half; ++i) {
        const float frequency = std::pow(theta, -2.0f * static_cast<float>(i) / head_dim);
        const float angle = static_cast<float>(position) * frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = values[i];
        const float second = values[i + half];
        values[i] = first * cosine - second * sine;
        values[i + half] = second * cosine + first * sine;
      }
    }
  };
  rotate(q, num_heads);
  rotate(k, num_kv_heads);
}

void attention_decode(const float* q, const float* key_cache, const float* value_cache,
                      float* output, int num_heads, int num_kv_heads,
                      int head_dim, int sequence_length, int max_sequence_length,
                      float* scores) {
  const int heads_per_kv = num_heads / num_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int head = 0; head < num_heads; ++head) {
    const int kv_head = head / heads_per_kv;
    const float* query = q + static_cast<size_t>(head) * head_dim;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int pos = 0; pos < sequence_length; ++pos) {
      const float* key = key_cache +
          (static_cast<size_t>(kv_head) * max_sequence_length + pos) * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) dot += query[d] * key[d];
      scores[pos] = dot * scale;
      maximum = std::max(maximum, scores[pos]);
    }
    float denominator = 0.0f;
    for (int pos = 0; pos < sequence_length; ++pos) {
      scores[pos] = std::exp(scores[pos] - maximum);
      denominator += scores[pos];
    }
    float* head_output = output + static_cast<size_t>(head) * head_dim;
    std::fill(head_output, head_output + head_dim, 0.0f);
    for (int pos = 0; pos < sequence_length; ++pos) {
      const float probability = scores[pos] / denominator;
      const float* value = value_cache +
          (static_cast<size_t>(kv_head) * max_sequence_length + pos) * head_dim;
      for (int d = 0; d < head_dim; ++d) head_output[d] += probability * value[d];
    }
  }
}

void silu_mul(const float* gate, const float* up, float* output, int n) {
  for (int i = 0; i < n; ++i) {
    const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
    output[i] = silu * up[i];
  }
}

int argmax(const float* values, int n) {
  return static_cast<int>(std::max_element(values, values + n) - values);
}

}  // namespace infer::cpu
