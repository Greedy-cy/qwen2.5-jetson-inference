#include "infer/cuda_ops.hpp"

#include <cfloat>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace infer::cuda {
namespace {

void cublas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) throw Error(std::string("cuBLAS failure: ") + what);
}

__global__ void embedding_kernel(const float* table, int token, float* output, int hidden) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
       i += blockDim.x * gridDim.x) {
    output[i] = table[static_cast<size_t>(token) * hidden + i];
  }
}

__global__ void rms_norm_kernel(const float* input, const float* weight, float* output,
                                int n, float epsilon) {
  __shared__ float reduction[256];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) sum += input[i] * input[i];
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(reduction[0] / n + epsilon);
  for (int i = threadIdx.x; i < n; i += blockDim.x) output[i] = input[i] * scale * weight[i];
}

__global__ void gemv_fp32_kernel(
    const float* weight, const float* bias, const float* input, float* output,
    int out_features, int in_features) {
  const int row = blockIdx.x;
  if (row >= out_features) return;
  __shared__ float reduction[256];
  float sum = 0.0f;
  for (int col = threadIdx.x; col < in_features; col += blockDim.x) {
    sum = fmaf(weight[static_cast<size_t>(row) * in_features + col],
               input[col], sum);
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = reduction[0] + (bias ? bias[row] : 0.0f);
  }
}


__global__ void gemm_tiled_kernel(const float* weight, const float* input,
                                  float* output, int tokens,
                                  int out_features, int in_features) {
  constexpr int tile = 16;
  __shared__ float input_tile[tile][tile];
  __shared__ float weight_tile[tile][tile];
  const int token = blockIdx.y * tile + threadIdx.y;
  const int row = blockIdx.x * tile + threadIdx.x;
  float sum = 0.0f;
  for (int begin = 0; begin < in_features; begin += tile) {
    const int input_col = begin + threadIdx.x;
    input_tile[threadIdx.y][threadIdx.x] =
        token < tokens && input_col < in_features
            ? input[static_cast<size_t>(token) * in_features + input_col] : 0.0f;
    const int weight_col = begin + threadIdx.y;
    weight_tile[threadIdx.x][threadIdx.y] =
        row < out_features && weight_col < in_features
            ? weight[static_cast<size_t>(row) * in_features + weight_col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < tile; ++i) {
      sum = fmaf(input_tile[threadIdx.y][i], weight_tile[threadIdx.x][i], sum);
    }
    __syncthreads();
  }
  if (token < tokens && row < out_features) {
    output[static_cast<size_t>(token) * out_features + row] = sum;
  }
}


__global__ void add_bias_kernel(float* output, const float* bias, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) output[i] += bias[i];
}

__global__ void rope_kernel(float* data, int heads, int head_dim, int position, float theta) {
  const int head = blockIdx.x;
  const int i = threadIdx.x;
  const int half = head_dim / 2;
  if (head >= heads || i >= half) return;
  float* values = data + static_cast<size_t>(head) * head_dim;
  const float frequency = powf(theta, -2.0f * i / head_dim);
  float sine;
  float cosine;
  sincosf(position * frequency, &sine, &cosine);
  const float first = values[i];
  const float second = values[i + half];
  values[i] = first * cosine - second * sine;
  values[i + half] = second * cosine + first * sine;
}

__global__ void store_kv_kernel(const float* key, const float* value,
                                float* key_cache, float* value_cache,
                                int kv_heads, int head_dim, int position, int max_sequence) {
  const int n = kv_heads * head_dim;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    const int head = i / head_dim;
    const int dim = i % head_dim;
    const size_t offset = (static_cast<size_t>(head) * max_sequence + position) * head_dim + dim;
    key_cache[offset] = key[i];
    value_cache[offset] = value[i];
  }
}

__global__ void attention_kernel(const float* q, const float* key_cache,
                                 const float* value_cache, float* output,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 int sequence_length, int max_sequence) {
  extern __shared__ float shared[];
  float* scores = shared;
  __shared__ float maximum;
  __shared__ float denominator;
  const int head = blockIdx.x;
  const int kv_head = head / (num_heads / num_kv_heads);
  const float* query = q + static_cast<size_t>(head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int pos = threadIdx.x; pos < sequence_length; pos += blockDim.x) {
    const float* key = key_cache +
        (static_cast<size_t>(kv_head) * max_sequence + pos) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot = fmaf(query[d], key[d], dot);
    scores[pos] = dot * scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -FLT_MAX;
    for (int pos = 0; pos < sequence_length; ++pos) m = fmaxf(m, scores[pos]);
    maximum = m;
  }
  __syncthreads();
  for (int pos = threadIdx.x; pos < sequence_length; pos += blockDim.x) {
    scores[pos] = expf(scores[pos] - maximum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int pos = 0; pos < sequence_length; ++pos) sum += scores[pos];
    denominator = sum;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float sum = 0.0f;
    for (int pos = 0; pos < sequence_length; ++pos) {
      const float* value = value_cache +
          (static_cast<size_t>(kv_head) * max_sequence + pos) * head_dim;
      sum = fmaf(scores[pos] / denominator, value[d], sum);
    }
    output[static_cast<size_t>(head) * head_dim + d] = sum;
  }
}

__global__ void embedding_batch_kernel(const float* table, const int* tokens,
                                       float* output, int token_count, int hidden) {
  const int n = token_count * hidden;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    const int token_index = i / hidden;
    const int column = i % hidden;
    output[i] = table[static_cast<size_t>(tokens[token_index]) * hidden + column];
  }
}

__global__ void rms_norm_batch_kernel(const float* input, const float* weight,
                                      float* output, int token_count, int hidden,
                                      float epsilon) {
  const int token = blockIdx.x;
  if (token >= token_count) return;
  __shared__ float reduction[256];
  const float* input_row = input + static_cast<size_t>(token) * hidden;
  float* output_row = output + static_cast<size_t>(token) * hidden;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    sum += input_row[i] * input_row[i];
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(reduction[0] / hidden + epsilon);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    output_row[i] = input_row[i] * scale * weight[i];
  }
}

__global__ void add_bias_batch_kernel(float* output, const float* bias,
                                      int tokens, int out_features) {
  const int n = tokens * out_features;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    output[i] += bias[i % out_features];
  }
}

__global__ void rope_batch_kernel(float* data, int token_count, int heads,
                                  int head_dim, int start_position, float theta) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int i = threadIdx.x;
  const int half = head_dim / 2;
  if (token >= token_count || head >= heads || i >= half) return;
  float* values = data +
      (static_cast<size_t>(token) * heads + head) * head_dim;
  const float frequency = powf(theta, -2.0f * i / head_dim);
  float sine;
  float cosine;
  sincosf((start_position + token) * frequency, &sine, &cosine);
  const float first = values[i];
  const float second = values[i + half];
  values[i] = first * cosine - second * sine;
  values[i + half] = second * cosine + first * sine;
}

__global__ void store_kv_batch_kernel(
    const float* key, const float* value, float* key_cache, float* value_cache,
    int token_count, int kv_heads, int head_dim, int max_sequence) {
  const int n = token_count * kv_heads * head_dim;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    const int dim = i % head_dim;
    const int token_head = i / head_dim;
    const int head = token_head % kv_heads;
    const int token = token_head / kv_heads;
    const size_t cache_offset =
        (static_cast<size_t>(head) * max_sequence + token) * head_dim + dim;
    key_cache[cache_offset] = key[i];
    value_cache[cache_offset] = value[i];
  }
}

__global__ void attention_prefill_kernel(
    const float* q, const float* key_cache, const float* value_cache,
    float* output, int token_count, int num_heads, int num_kv_heads,
    int head_dim, int max_sequence) {
  extern __shared__ float scores[];
  __shared__ float maximum;
  __shared__ float denominator;
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  if (token >= token_count || head >= num_heads) return;
  const int sequence_length = token + 1;
  const int kv_head = head / (num_heads / num_kv_heads);
  const float* query =
      q + (static_cast<size_t>(token) * num_heads + head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int pos = threadIdx.x; pos < sequence_length; pos += blockDim.x) {
    const float* key = key_cache +
        (static_cast<size_t>(kv_head) * max_sequence + pos) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot = fmaf(query[d], key[d], dot);
    scores[pos] = dot * scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -FLT_MAX;
    for (int pos = 0; pos < sequence_length; ++pos) m = fmaxf(m, scores[pos]);
    maximum = m;
  }
  __syncthreads();
  for (int pos = threadIdx.x; pos < sequence_length; pos += blockDim.x) {
    scores[pos] = expf(scores[pos] - maximum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int pos = 0; pos < sequence_length; ++pos) sum += scores[pos];
    denominator = sum;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float sum = 0.0f;
    for (int pos = 0; pos < sequence_length; ++pos) {
      const float* value = value_cache +
          (static_cast<size_t>(kv_head) * max_sequence + pos) * head_dim;
      sum = fmaf(scores[pos] / denominator, value[d], sum);
    }
    output[(static_cast<size_t>(token) * num_heads + head) * head_dim + d] = sum;
  }
}

__global__ void add_kernel(float* x, const float* residual, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) x[i] += residual[i];
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* output, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    output[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
  }
}

__global__ void argmax_kernel(const float* values, int n, int* output) {
  __shared__ float best_value[256];
  __shared__ int best_index[256];
  float value = -FLT_MAX;
  int index = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i] > value) { value = values[i]; index = i; }
  }
  best_value[threadIdx.x] = value;
  best_index[threadIdx.x] = index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride && best_value[threadIdx.x + stride] > best_value[threadIdx.x]) {
      best_value[threadIdx.x] = best_value[threadIdx.x + stride];
      best_index[threadIdx.x] = best_index[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = best_index[0];
}


using BFloat16 = __nv_bfloat16;
using BFloat16Pair = __nv_bfloat162;

__global__ void embedding_bf16_kernel(const BFloat16* table, int token,
                                      BFloat16* output, int hidden) {
  if ((hidden & 1) == 0) {
    const auto* source = reinterpret_cast<const BFloat16Pair*>(
        table + static_cast<size_t>(token) * hidden);
    auto* destination = reinterpret_cast<BFloat16Pair*>(output);
    const int pairs = hidden / 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pairs;
         i += blockDim.x * gridDim.x) {
      destination[i] = source[i];
    }
    return;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
       i += blockDim.x * gridDim.x) {
    output[i] = table[static_cast<size_t>(token) * hidden + i];
  }
}

__global__ void embedding_batch_bf16_kernel(
    const BFloat16* table, const int* tokens, BFloat16* output,
    int token_count, int hidden) {
  const int total = token_count * hidden;
  if ((hidden & 1) == 0) {
    const int pairs_per_row = hidden / 2;
    const int total_pairs = token_count * pairs_per_row;
    const auto* source = reinterpret_cast<const BFloat16Pair*>(table);
    auto* destination = reinterpret_cast<BFloat16Pair*>(output);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_pairs;
         i += blockDim.x * gridDim.x) {
      const int token_index = i / pairs_per_row;
      const int pair = i % pairs_per_row;
      destination[i] =
          source[static_cast<size_t>(tokens[token_index]) * pairs_per_row + pair];
    }
    return;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += blockDim.x * gridDim.x) {
    const int token_index = i / hidden;
    const int column = i % hidden;
    output[i] =
        table[static_cast<size_t>(tokens[token_index]) * hidden + column];
  }
}

__global__ void rms_norm_bf16_kernel(
    const BFloat16* input, const BFloat16* weight, BFloat16* output,
    int rows, int hidden, float epsilon) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float reduction[256];
  const BFloat16* input_row = input + static_cast<size_t>(row) * hidden;
  BFloat16* output_row = output + static_cast<size_t>(row) * hidden;
  float sum = 0.0f;
  if ((hidden & 1) == 0) {
    const auto* input_pairs =
        reinterpret_cast<const BFloat16Pair*>(input_row);
    const int pair_count = hidden / 2;
    for (int i = threadIdx.x; i < pair_count; i += blockDim.x) {
      const float2 values = __bfloat1622float2(input_pairs[i]);
      sum = fmaf(values.x, values.x, sum);
      sum = fmaf(values.y, values.y, sum);
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float value = __bfloat162float(input_row[i]);
      sum = fmaf(value, value, sum);
    }
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float scale = rsqrtf(reduction[0] / hidden + epsilon);
  if ((hidden & 1) == 0) {
    const auto* input_pairs =
        reinterpret_cast<const BFloat16Pair*>(input_row);
    const auto* weight_pairs =
        reinterpret_cast<const BFloat16Pair*>(weight);
    auto* output_pairs = reinterpret_cast<BFloat16Pair*>(output_row);
    const int pair_count = hidden / 2;
    for (int i = threadIdx.x; i < pair_count; i += blockDim.x) {
      const float2 values = __bfloat1622float2(input_pairs[i]);
      const float2 weights = __bfloat1622float2(weight_pairs[i]);
      output_pairs[i] = __floats2bfloat162_rn(
          values.x * weights.x * scale, values.y * weights.y * scale);
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      output_row[i] = __float2bfloat16_rn(
          __bfloat162float(input_row[i]) * __bfloat162float(weight[i]) * scale);
    }
  }
}


__global__ void rope_bf16_kernel(BFloat16* data, int token_count, int heads,
                                 int head_dim, int start_position, float theta) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int i = threadIdx.x;
  const int half = head_dim / 2;
  if (token >= token_count || head >= heads || i >= half) return;
  BFloat16* values =
      data + (static_cast<size_t>(token) * heads + head) * head_dim;
  const float frequency = powf(theta, -2.0f * i / head_dim);
  float sine;
  float cosine;
  sincosf((start_position + token) * frequency, &sine, &cosine);
  const float first = __bfloat162float(values[i]);
  const float second = __bfloat162float(values[i + half]);
  values[i] = __float2bfloat16_rn(first * cosine - second * sine);
  values[i + half] =
      __float2bfloat16_rn(second * cosine + first * sine);
}

__global__ void store_kv_bf16_kernel(
    const BFloat16* key, const BFloat16* value, BFloat16* key_cache,
    BFloat16* value_cache, int token_count, int kv_heads, int head_dim,
    int start_position, int max_sequence) {
  const int total = token_count * kv_heads * head_dim;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += blockDim.x * gridDim.x) {
    const int dim = i % head_dim;
    const int token_head = i / head_dim;
    const int head = token_head % kv_heads;
    const int token = token_head / kv_heads;
    const size_t cache_offset =
        (static_cast<size_t>(head) * max_sequence + start_position + token) *
            head_dim +
        dim;
    key_cache[cache_offset] = key[i];
    value_cache[cache_offset] = value[i];
  }
}

__global__ void attention_decode_bf16_kernel(
    const BFloat16* q, const BFloat16* key_cache,
    const BFloat16* value_cache, BFloat16* output, int num_heads,
    int num_kv_heads, int head_dim, int sequence_length, int max_sequence) {
  extern __shared__ float scores[];
  __shared__ float maximum;
  __shared__ float denominator;
  const int head = blockIdx.x;
  if (head >= num_heads) return;
  const int kv_head = head / (num_heads / num_kv_heads);
  const BFloat16* query = q + static_cast<size_t>(head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    const BFloat16* key =
        key_cache +
        (static_cast<size_t>(kv_head) * max_sequence + position) * head_dim;
    float dot = 0.0f;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      dot = fmaf(__bfloat162float(query[dimension]),
                 __bfloat162float(key[dimension]), dot);
    }
    scores[position] = dot * scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float value = -FLT_MAX;
    for (int position = 0; position < sequence_length; ++position) {
      value = fmaxf(value, scores[position]);
    }
    maximum = value;
  }
  __syncthreads();
  for (int position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    scores[position] = expf(scores[position] - maximum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int position = 0; position < sequence_length; ++position) {
      sum += scores[position];
    }
    denominator = sum;
  }
  __syncthreads();
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    float sum = 0.0f;
    for (int position = 0; position < sequence_length; ++position) {
      const BFloat16* value =
          value_cache +
          (static_cast<size_t>(kv_head) * max_sequence + position) * head_dim;
      sum = fmaf(scores[position] / denominator,
                 __bfloat162float(value[dimension]), sum);
    }
    output[static_cast<size_t>(head) * head_dim + dimension] =
        __float2bfloat16_rn(sum);
  }
}


__global__ void attention_prefill_bf16_kernel(
    const BFloat16* q, const BFloat16* key_cache,
    const BFloat16* value_cache, BFloat16* output, int token_count,
    int num_heads, int num_kv_heads, int head_dim, int max_sequence) {
  extern __shared__ float scores[];
  __shared__ float maximum;
  __shared__ float denominator;
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  if (token >= token_count || head >= num_heads) return;
  const int sequence_length = token + 1;
  const int kv_head = head / (num_heads / num_kv_heads);
  const BFloat16* query =
      q + (static_cast<size_t>(token) * num_heads + head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    const BFloat16* key =
        key_cache +
        (static_cast<size_t>(kv_head) * max_sequence + position) * head_dim;
    float dot = 0.0f;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      dot = fmaf(__bfloat162float(query[dimension]),
                 __bfloat162float(key[dimension]), dot);
    }
    scores[position] = dot * scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float value = -FLT_MAX;
    for (int position = 0; position < sequence_length; ++position) {
      value = fmaxf(value, scores[position]);
    }
    maximum = value;
  }
  __syncthreads();
  for (int position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    scores[position] = expf(scores[position] - maximum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int position = 0; position < sequence_length; ++position) {
      sum += scores[position];
    }
    denominator = sum;
  }
  __syncthreads();
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    float sum = 0.0f;
    for (int position = 0; position < sequence_length; ++position) {
      const BFloat16* value =
          value_cache +
          (static_cast<size_t>(kv_head) * max_sequence + position) * head_dim;
      sum = fmaf(scores[position] / denominator,
                 __bfloat162float(value[dimension]), sum);
    }
    output[(static_cast<size_t>(token) * num_heads + head) * head_dim +
           dimension] = __float2bfloat16_rn(sum);
  }
}

__global__ void add_bf16_kernel(BFloat16* x, const BFloat16* residual, int n) {
  const int pair_count = n / 2;
  auto* x_pairs = reinterpret_cast<BFloat16Pair*>(x);
  const auto* residual_pairs =
      reinterpret_cast<const BFloat16Pair*>(residual);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pair_count;
       i += blockDim.x * gridDim.x) {
    const float2 lhs = __bfloat1622float2(x_pairs[i]);
    const float2 rhs = __bfloat1622float2(residual_pairs[i]);
    x_pairs[i] = __floats2bfloat162_rn(lhs.x + rhs.x, lhs.y + rhs.y);
  }
  if ((n & 1) != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    x[n - 1] = __float2bfloat16_rn(
        __bfloat162float(x[n - 1]) + __bfloat162float(residual[n - 1]));
  }
}

__global__ void silu_mul_bf16_kernel(
    const BFloat16* gate, const BFloat16* up, BFloat16* output, int n) {
  const int pair_count = n / 2;
  const auto* gate_pairs = reinterpret_cast<const BFloat16Pair*>(gate);
  const auto* up_pairs = reinterpret_cast<const BFloat16Pair*>(up);
  auto* output_pairs = reinterpret_cast<BFloat16Pair*>(output);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pair_count;
       i += blockDim.x * gridDim.x) {
    const float2 gate_values = __bfloat1622float2(gate_pairs[i]);
    const float2 up_values = __bfloat1622float2(up_pairs[i]);
    const float first =
        (gate_values.x / (1.0f + expf(-gate_values.x))) * up_values.x;
    const float second =
        (gate_values.y / (1.0f + expf(-gate_values.y))) * up_values.y;
    output_pairs[i] = __floats2bfloat162_rn(first, second);
  }
  if ((n & 1) != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    const float gate_value = __bfloat162float(gate[n - 1]);
    output[n - 1] = __float2bfloat16_rn(
        (gate_value / (1.0f + expf(-gate_value))) *
        __bfloat162float(up[n - 1]));
  }
}

__global__ void argmax_bf16_kernel(
    const BFloat16* values, int n, int* output) {
  __shared__ float best_value[256];
  __shared__ int best_index[256];
  float value = -FLT_MAX;
  int index = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float candidate = __bfloat162float(values[i]);
    if (candidate > value) {
      value = candidate;
      index = i;
    }
  }
  best_value[threadIdx.x] = value;
  best_index[threadIdx.x] = index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride &&
        best_value[threadIdx.x + stride] > best_value[threadIdx.x]) {
      best_value[threadIdx.x] = best_value[threadIdx.x + stride];
      best_index[threadIdx.x] = best_index[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = best_index[0];
}


__global__ void gemv_bf16_kernel(
    const BFloat16* weight, const BFloat16* bias, const BFloat16* input,
    BFloat16* output, int out_features, int in_features) {
  const int row = blockIdx.x;
  if (row >= out_features) return;
  __shared__ float reduction[256];
  float sum = 0.0f;
  const BFloat16* row_weight =
      weight + static_cast<size_t>(row) * in_features;
  if ((in_features & 1) == 0) {
    const auto* weight_pairs =
        reinterpret_cast<const BFloat16Pair*>(row_weight);
    const auto* input_pairs =
        reinterpret_cast<const BFloat16Pair*>(input);
    const int pair_count = in_features / 2;
    for (int pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
      const float2 weights = __bfloat1622float2(weight_pairs[pair]);
      const float2 values = __bfloat1622float2(input_pairs[pair]);
      sum = fmaf(weights.x, values.x, sum);
      sum = fmaf(weights.y, values.y, sum);
    }
  } else {
    for (int column = threadIdx.x; column < in_features;
         column += blockDim.x) {
      sum = fmaf(__bfloat162float(row_weight[column]),
                 __bfloat162float(input[column]), sum);
    }
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float bias_value = bias ? __bfloat162float(bias[row]) : 0.0f;
    output[row] = __float2bfloat16_rn(reduction[0] + bias_value);
  }
}

template <int BlockM>
__global__ void gemm_bf16_wmma_kernel(
    const BFloat16* weight, const BFloat16* bias, const BFloat16* input,
    BFloat16* output, int tokens, int out_features, int in_features) {
  namespace wmma = nvcuda::wmma;
  constexpr int kWmmaM = 16;
  constexpr int kWmmaN = 16;
  constexpr int kWmmaK = 16;
  constexpr int kBlockN = 64;
  constexpr int kBlockK = 64;
  constexpr int kSharedK = kBlockK + 8;
  constexpr int kWarpRows = BlockM / kWmmaM;
  constexpr int kWarpColumns = kBlockN / kWmmaN;
  constexpr int kWarps = kWarpRows * kWarpColumns;
  static_assert(BlockM == 32);
  static_assert(kWarps * 32 == 256);

  struct OperandStorage {
    BFloat16 input[BlockM * kSharedK];
    BFloat16 weight[kBlockN * kSharedK];
  };
  union SharedStorage {
    OperandStorage operands;
    float output[BlockM * kBlockN];
  };
  __shared__ __align__(32) SharedStorage shared;
  BFloat16* input_tile = shared.operands.input;
  BFloat16* weight_tile = shared.operands.weight;

  const int warp = threadIdx.x / warpSize;
  const int warp_row = warp / kWarpColumns;
  const int warp_column = warp % kWarpColumns;
  const int block_token = blockIdx.y * BlockM;
  const int block_row = blockIdx.x * kBlockN;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int begin = 0; begin < in_features; begin += kBlockK) {
    for (int index = threadIdx.x; index < BlockM * kBlockK;
         index += blockDim.x) {
      const int local_token = index / kBlockK;
      const int local_column = index % kBlockK;
      const int token = block_token + local_token;
      const int column = begin + local_column;
      input_tile[local_token * kSharedK + local_column] =
          token < tokens && column < in_features
              ? input[static_cast<size_t>(token) * in_features + column]
              : __float2bfloat16_rn(0.0f);
    }
    for (int index = threadIdx.x; index < kBlockN * kBlockK;
         index += blockDim.x) {
      const int local_row = index / kBlockK;
      const int local_column = index % kBlockK;
      const int row = block_row + local_row;
      const int column = begin + local_column;
      weight_tile[local_row * kSharedK + local_column] =
          row < out_features && column < in_features
              ? weight[static_cast<size_t>(row) * in_features + column]
              : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int column = 0; column < kBlockK; column += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, BFloat16,
                     wmma::row_major>
          input_fragment;
      wmma::load_matrix_sync(
          input_fragment,
          input_tile + warp_row * kWmmaM * kSharedK + column, kSharedK);
      const int local_row = warp_column * kWmmaN;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, BFloat16,
                     wmma::col_major>
          weight_fragment;
      wmma::load_matrix_sync(
          weight_fragment, weight_tile + local_row * kSharedK + column,
          kSharedK);
      wmma::mma_sync(accumulator, input_fragment, weight_fragment,
                     accumulator);
    }
    __syncthreads();
  }

  const int local_row = warp_column * kWmmaN;
  wmma::store_matrix_sync(
      shared.output + warp_row * kWmmaM * kBlockN + local_row, accumulator,
      kBlockN, wmma::mem_row_major);
  __syncthreads();

  for (int index = threadIdx.x; index < BlockM * kBlockN;
       index += blockDim.x) {
    const int local_token = index / kBlockN;
    const int local_row = index % kBlockN;
    const int token = block_token + local_token;
    const int row = block_row + local_row;
    if (token < tokens && row < out_features) {
      const float bias_value = bias ? __bfloat162float(bias[row]) : 0.0f;
      output[static_cast<size_t>(token) * out_features + row] =
          __float2bfloat16_rn(shared.output[index] + bias_value);
    }
  }
}

template <int BlockM, int BlockN, int BlockK, int SharedK>
__device__ __forceinline__ void load_bf16_wmma_stage_async(
    const BFloat16* weight, const BFloat16* input, BFloat16* input_tile,
    BFloat16* weight_tile, int block_token, int block_row, int begin,
    int in_features) {
  constexpr int kValuesPerCopy = 8;
  constexpr int kCopyBytes = kValuesPerCopy * sizeof(BFloat16);
  constexpr int kInputCopies = BlockM * BlockK / kValuesPerCopy;
  constexpr int kWeightCopies = BlockN * BlockK / kValuesPerCopy;
  static_assert(kCopyBytes == 16);

  for (int copy = threadIdx.x; copy < kInputCopies; copy += blockDim.x) {
    const int local_token = copy / (BlockK / kValuesPerCopy);
    const int local_column =
        (copy % (BlockK / kValuesPerCopy)) * kValuesPerCopy;
    __pipeline_memcpy_async(
        input_tile + local_token * SharedK + local_column,
        input + static_cast<size_t>(block_token + local_token) * in_features +
            begin + local_column,
        kCopyBytes);
  }
  for (int copy = threadIdx.x; copy < kWeightCopies; copy += blockDim.x) {
    const int local_row = copy / (BlockK / kValuesPerCopy);
    const int local_column =
        (copy % (BlockK / kValuesPerCopy)) * kValuesPerCopy;
    __pipeline_memcpy_async(
        weight_tile + local_row * SharedK + local_column,
        weight + static_cast<size_t>(block_row + local_row) * in_features +
            begin + local_column,
        kCopyBytes);
  }
}

__global__ void gemm_bf16_wmma_async_kernel(
    const BFloat16* weight, const BFloat16* bias, const BFloat16* input,
    BFloat16* output, int tokens, int out_features, int in_features) {
  namespace wmma = nvcuda::wmma;
  constexpr int kWmmaM = 16;
  constexpr int kWmmaN = 16;
  constexpr int kWmmaK = 16;
  constexpr int kBlockM = 32;
  constexpr int kBlockN = 64;
  constexpr int kBlockK = 64;
  constexpr int kSharedK = kBlockK + 8;
  constexpr int kWarpColumns = kBlockN / kWmmaN;

  struct OperandStorage {
    BFloat16 input[kBlockM * kSharedK];
    BFloat16 weight[kBlockN * kSharedK];
  };
  union SharedStorage {
    OperandStorage operands[2];
    float output[kBlockM * kBlockN];
  };
  __shared__ __align__(32) SharedStorage shared;

  const int warp = threadIdx.x / warpSize;
  const int warp_row = warp / kWarpColumns;
  const int warp_column = warp % kWarpColumns;
  const int block_token = blockIdx.y * kBlockM;
  const int block_row = blockIdx.x * kBlockN;
  const int stage_count = in_features / kBlockK;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  load_bf16_wmma_stage_async<kBlockM, kBlockN, kBlockK, kSharedK>(
      weight, input, shared.operands[0].input, shared.operands[0].weight,
      block_token, block_row, 0, in_features);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  int current_buffer = 0;
  for (int stage = 0; stage < stage_count; ++stage) {
    const int next_stage = stage + 1;
    if (next_stage < stage_count) {
      const int next_buffer = current_buffer ^ 1;
      load_bf16_wmma_stage_async<kBlockM, kBlockN, kBlockK, kSharedK>(
          weight, input, shared.operands[next_buffer].input,
          shared.operands[next_buffer].weight, block_token, block_row,
          next_stage * kBlockK, in_features);
      __pipeline_commit();
    }

    const BFloat16* input_tile = shared.operands[current_buffer].input;
    const BFloat16* weight_tile = shared.operands[current_buffer].weight;
#pragma unroll
    for (int column = 0; column < kBlockK; column += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, BFloat16,
                     wmma::row_major>
          input_fragment;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, BFloat16,
                     wmma::col_major>
          weight_fragment;
      wmma::load_matrix_sync(
          input_fragment,
          input_tile + warp_row * kWmmaM * kSharedK + column, kSharedK);
      wmma::load_matrix_sync(
          weight_fragment,
          weight_tile + warp_column * kWmmaN * kSharedK + column, kSharedK);
      wmma::mma_sync(accumulator, input_fragment, weight_fragment,
                     accumulator);
    }

    if (next_stage < stage_count) {
      __pipeline_wait_prior(0);
      __syncthreads();
      current_buffer ^= 1;
    }
  }
  __syncthreads();

  const int local_row = warp_column * kWmmaN;
  wmma::store_matrix_sync(
      shared.output + warp_row * kWmmaM * kBlockN + local_row, accumulator,
      kBlockN, wmma::mem_row_major);
  __syncthreads();

  for (int index = threadIdx.x; index < kBlockM * kBlockN;
       index += blockDim.x) {
    const int local_token = index / kBlockN;
    const int local_output_row = index % kBlockN;
    const int token = block_token + local_token;
    const int row = block_row + local_output_row;
    const float bias_value = bias ? __bfloat162float(bias[row]) : 0.0f;
    output[static_cast<size_t>(token) * out_features + row] =
        __float2bfloat16_rn(shared.output[index] + bias_value);
  }
}

__global__ void gemm_bf16_wmma_async_m128n128_kernel(
    const BFloat16* weight, const BFloat16* bias, const BFloat16* input,
    BFloat16* output, int tokens, int out_features, int in_features) {
  namespace wmma = nvcuda::wmma;
  constexpr int kWmmaM = 16;
  constexpr int kWmmaN = 16;
  constexpr int kWmmaK = 16;
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 128;
  constexpr int kBlockK = 32;
  constexpr int kSharedK = kBlockK + 8;
  constexpr int kWarpColumns = kBlockN / kWmmaN;
  constexpr int kAccumulatorCount = 4;

  struct OperandStorage {
    BFloat16 input[kBlockM * kSharedK];
    BFloat16 weight[kBlockN * kSharedK];
  };
  union SharedStorage {
    OperandStorage operands[2];
    float warp_output[16][kWmmaM * kWmmaN];
  };
  __shared__ __align__(32) SharedStorage shared;

  const int warp = threadIdx.x / warpSize;
  const int warp_row_group = warp / kWarpColumns;
  const int warp_column = warp % kWarpColumns;
  const int block_token = blockIdx.y * kBlockM;
  const int block_row = blockIdx.x * kBlockN;
  const int stage_count = in_features / kBlockK;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      accumulators[kAccumulatorCount];
#pragma unroll
  for (int index = 0; index < kAccumulatorCount; ++index) {
    wmma::fill_fragment(accumulators[index], 0.0f);
  }

  load_bf16_wmma_stage_async<kBlockM, kBlockN, kBlockK, kSharedK>(
      weight, input, shared.operands[0].input, shared.operands[0].weight,
      block_token, block_row, 0, in_features);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  int current_buffer = 0;
  for (int stage = 0; stage < stage_count; ++stage) {
    const int next_stage = stage + 1;
    if (next_stage < stage_count) {
      const int next_buffer = current_buffer ^ 1;
      load_bf16_wmma_stage_async<kBlockM, kBlockN, kBlockK, kSharedK>(
          weight, input, shared.operands[next_buffer].input,
          shared.operands[next_buffer].weight, block_token, block_row,
          next_stage * kBlockK, in_features);
      __pipeline_commit();
    }

    const BFloat16* input_tile = shared.operands[current_buffer].input;
    const BFloat16* weight_tile = shared.operands[current_buffer].weight;
#pragma unroll
    for (int column = 0; column < kBlockK; column += kWmmaK) {
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, BFloat16,
                     wmma::col_major>
          weight_fragment;
      wmma::load_matrix_sync(
          weight_fragment,
          weight_tile + warp_column * kWmmaN * kSharedK + column, kSharedK);
#pragma unroll
      for (int index = 0; index < kAccumulatorCount; ++index) {
        const int tile_row = warp_row_group + index * 2;
        wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, BFloat16,
                       wmma::row_major>
            input_fragment;
        wmma::load_matrix_sync(
            input_fragment,
            input_tile + tile_row * kWmmaM * kSharedK + column, kSharedK);
        wmma::mma_sync(accumulators[index], input_fragment, weight_fragment,
                       accumulators[index]);
      }
    }

    if (next_stage < stage_count) {
      __pipeline_wait_prior(0);
      __syncthreads();
      current_buffer ^= 1;
    }
  }
  __syncthreads();

  float* warp_output = shared.warp_output[warp];
#pragma unroll
  for (int index = 0; index < kAccumulatorCount; ++index) {
    const int tile_row = warp_row_group + index * 2;
    wmma::store_matrix_sync(warp_output, accumulators[index], kWmmaN,
                            wmma::mem_row_major);
    __syncwarp();
    for (int element = threadIdx.x % warpSize;
         element < kWmmaM * kWmmaN; element += warpSize) {
      const int local_token = tile_row * kWmmaM + element / kWmmaN;
      const int local_output_row =
          warp_column * kWmmaN + element % kWmmaN;
      const int token = block_token + local_token;
      const int row = block_row + local_output_row;
      const float bias_value = bias ? __bfloat162float(bias[row]) : 0.0f;
      output[static_cast<size_t>(token) * out_features + row] =
          __float2bfloat16_rn(warp_output[element] + bias_value);
    }
    __syncwarp();
  }
}

__global__ void linear_w8a16_kernel(
    const int8_t* weight, const BFloat16* scales, int group_size,
    const BFloat16* bias, const BFloat16* input, BFloat16* output,
    int tokens, int out_features, int in_features) {
  const int row = blockIdx.x;
  const int token = blockIdx.y;
  if (row >= out_features || token >= tokens) return;

  __shared__ float reduction[256];
  const int groups = (in_features + group_size - 1) / group_size;
  const int8_t* row_weight =
      weight + static_cast<size_t>(row) * in_features;
  const BFloat16* token_input =
      input + static_cast<size_t>(token) * in_features;
  const BFloat16* row_scales =
      scales + static_cast<size_t>(row) * groups;
  const bool vector_aligned =
      (reinterpret_cast<uintptr_t>(row_weight) & 3U) == 0U &&
      (reinterpret_cast<uintptr_t>(token_input) & 3U) == 0U;

  float sum = 0.0f;
  for (int column = threadIdx.x * 4; column < in_features;
       column += blockDim.x * 4) {
    const int remaining = min(4, in_features - column);
    const int group = column / group_size;
    const bool one_group =
        (column + remaining - 1) / group_size == group;
    if (remaining == 4 && vector_aligned && one_group) {
      const char4 packed =
          *reinterpret_cast<const char4*>(row_weight + column);
      const auto* input_pairs =
          reinterpret_cast<const BFloat16Pair*>(token_input + column);
      const float2 first = __bfloat1622float2(input_pairs[0]);
      const float2 second = __bfloat1622float2(input_pairs[1]);
      const float scale = __bfloat162float(row_scales[group]);
      sum = fmaf(static_cast<float>(packed.x) * scale, first.x, sum);
      sum = fmaf(static_cast<float>(packed.y) * scale, first.y, sum);
      sum = fmaf(static_cast<float>(packed.z) * scale, second.x, sum);
      sum = fmaf(static_cast<float>(packed.w) * scale, second.y, sum);
    } else {
      for (int offset = 0; offset < remaining; ++offset) {
        const int current = column + offset;
        const float scale =
            __bfloat162float(row_scales[current / group_size]);
        sum = fmaf(static_cast<float>(row_weight[current]) * scale,
                   __bfloat162float(token_input[current]), sum);
      }
    }
  }

  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float bias_value = bias ? __bfloat162float(bias[row]) : 0.0f;
    output[static_cast<size_t>(token) * out_features + row] =
        __float2bfloat16_rn(reduction[0] + bias_value);
  }
}

__global__ void add_bias_batch_bf16_kernel(
    BFloat16* output, const BFloat16* bias, int tokens, int out_features) {
  const int total = tokens * out_features;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += blockDim.x * gridDim.x) {
    output[index] = __float2bfloat16_rn(
        __bfloat162float(output[index]) +
        __bfloat162float(bias[index % out_features]));
  }
}

int blocks_for(int n) { return std::min(65535, (n + 255) / 256); }

}  // namespace

Context::Context() {
  INFER_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  cublas_check(cublasCreate(&cublas_), "cublasCreate");
  cublas_check(cublasSetStream(cublas_, stream_), "cublasSetStream");
}

Context::~Context() {
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Context::synchronize() const { INFER_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

void embedding(const float* table, int token, float* output, int hidden_size,
               cudaStream_t stream) {
  embedding_kernel<<<blocks_for(hidden_size), 256, 0, stream>>>(table, token, output, hidden_size);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void rms_norm(const float* input, const float* weight, float* output,
              int n, float epsilon, cudaStream_t stream) {
  rms_norm_kernel<<<1, 256, 0, stream>>>(input, weight, output, n, epsilon);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void gemv_fp32(const float* weight, const float* bias, const float* input,
               float* output, int out_features, int in_features, cudaStream_t stream) {
  gemv_fp32_kernel<<<out_features, 256, 0, stream>>>(
      weight, bias, input, output, out_features, in_features);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void gemv_fp32_cublas(cublasHandle_t handle, const float* weight, const float* bias,
                      const float* input, float* output, int out_features,
                      int in_features, cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_check(cublasSgemv(handle, CUBLAS_OP_T, in_features, out_features,
                           &alpha, weight, in_features, input, 1, &beta, output, 1),
               "cublasSgemv");
  if (bias) add_bias_kernel<<<blocks_for(out_features), 256, 0, stream>>>(output, bias, out_features);
  INFER_CUDA_CHECK(cudaGetLastError());
}


void gemm_fp32(const float* weight, const float* input, float* output,
               int tokens, int out_features, int in_features, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((out_features + 15) / 16, (tokens + 15) / 16);
  gemm_tiled_kernel<<<grid, block, 0, stream>>>(
      weight, input, output, tokens, out_features, in_features);
  INFER_CUDA_CHECK(cudaGetLastError());
}


void rope(float* q, float* k, int num_heads, int num_kv_heads, int head_dim,
          int position, float theta, cudaStream_t stream) {
  rope_kernel<<<num_heads, head_dim / 2, 0, stream>>>(q, num_heads, head_dim, position, theta);
  rope_kernel<<<num_kv_heads, head_dim / 2, 0, stream>>>(k, num_kv_heads, head_dim, position, theta);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void store_kv(const float* key, const float* value, float* key_cache,
              float* value_cache, int kv_heads, int head_dim, int position,
              int max_sequence_length, cudaStream_t stream) {
  const int n = kv_heads * head_dim;
  store_kv_kernel<<<blocks_for(n), 256, 0, stream>>>(key, value, key_cache, value_cache,
                                                     kv_heads, head_dim, position,
                                                     max_sequence_length);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void attention_decode(const float* q, const float* key_cache, const float* value_cache,
                      float* output, int num_heads, int num_kv_heads, int head_dim,
                      int sequence_length, int max_sequence_length, cudaStream_t stream) {
  attention_kernel<<<num_heads, 128, sequence_length * sizeof(float), stream>>>(
      q, key_cache, value_cache, output, num_heads, num_kv_heads, head_dim,
      sequence_length, max_sequence_length);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void embedding_batch(const float* table, const int* tokens, float* output,
                     int token_count, int hidden_size, cudaStream_t stream) {
  const int n = token_count * hidden_size;
  embedding_batch_kernel<<<blocks_for(n), 256, 0, stream>>>(
      table, tokens, output, token_count, hidden_size);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void rms_norm_batch(const float* input, const float* weight, float* output,
                    int token_count, int hidden_size, float epsilon,
                    cudaStream_t stream) {
  rms_norm_batch_kernel<<<token_count, 256, 0, stream>>>(
      input, weight, output, token_count, hidden_size, epsilon);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void gemm_fp32_cublas(cublasHandle_t handle, const float* weight,
                      const float* input, float* output, int tokens,
                      int out_features, int in_features, cudaStream_t stream) {
  (void)stream;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_check(
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, tokens,
                  in_features, &alpha, weight, in_features, input, in_features,
                  &beta, output, out_features),
      "cublasSgemm");
}

void add_bias_batch(float* output, const float* bias, int tokens,
                    int out_features, cudaStream_t stream) {
  const int n = tokens * out_features;
  add_bias_batch_kernel<<<blocks_for(n), 256, 0, stream>>>(
      output, bias, tokens, out_features);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void rope_batch(float* q, float* k, int token_count, int num_heads,
                int num_kv_heads, int head_dim, int start_position,
                float theta, cudaStream_t stream) {
  const dim3 q_grid(num_heads, token_count);
  const dim3 k_grid(num_kv_heads, token_count);
  rope_batch_kernel<<<q_grid, head_dim / 2, 0, stream>>>(
      q, token_count, num_heads, head_dim, start_position, theta);
  rope_batch_kernel<<<k_grid, head_dim / 2, 0, stream>>>(
      k, token_count, num_kv_heads, head_dim, start_position, theta);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void store_kv_batch(const float* key, const float* value, float* key_cache,
                    float* value_cache, int token_count, int kv_heads,
                    int head_dim, int max_sequence_length,
                    cudaStream_t stream) {
  const int n = token_count * kv_heads * head_dim;
  store_kv_batch_kernel<<<blocks_for(n), 256, 0, stream>>>(
      key, value, key_cache, value_cache, token_count, kv_heads, head_dim,
      max_sequence_length);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void attention_prefill(const float* q, const float* key_cache,
                       const float* value_cache, float* output,
                       int token_count, int num_heads, int num_kv_heads,
                       int head_dim, int max_sequence_length,
                       cudaStream_t stream) {
  const dim3 grid(num_heads, token_count);
  attention_prefill_kernel<<<grid, 128, token_count * sizeof(float), stream>>>(
      q, key_cache, value_cache, output, token_count, num_heads, num_kv_heads,
      head_dim, max_sequence_length);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void add_inplace(float* x, const float* residual, int n, cudaStream_t stream) {
  add_kernel<<<blocks_for(n), 256, 0, stream>>>(x, residual, n);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void silu_mul(const float* gate, const float* up, float* output, int n,
              cudaStream_t stream) {
  silu_mul_kernel<<<blocks_for(n), 256, 0, stream>>>(gate, up, output, n);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void argmax(const float* values, int n, int* output, cudaStream_t stream) {
  argmax_kernel<<<1, 256, 0, stream>>>(values, n, output);
  INFER_CUDA_CHECK(cudaGetLastError());
}


void embedding_bf16(const BFloat16* table, int token, BFloat16* output,
                    int hidden_size, cudaStream_t stream) {
  embedding_bf16_kernel<<<blocks_for(hidden_size), 256, 0, stream>>>(
      table, token, output, hidden_size);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void embedding_batch_bf16(const BFloat16* table, const int* tokens,
                          BFloat16* output, int token_count, int hidden_size,
                          cudaStream_t stream) {
  const int total = token_count * hidden_size;
  embedding_batch_bf16_kernel<<<blocks_for(total), 256, 0, stream>>>(
      table, tokens, output, token_count, hidden_size);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void rms_norm_bf16(const BFloat16* input, const BFloat16* weight,
                   BFloat16* output, int n, float epsilon,
                   cudaStream_t stream) {
  rms_norm_bf16_kernel<<<1, 256, 0, stream>>>(
      input, weight, output, 1, n, epsilon);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void rms_norm_batch_bf16(const BFloat16* input, const BFloat16* weight,
                         BFloat16* output, int token_count, int hidden_size,
                         float epsilon, cudaStream_t stream) {
  rms_norm_bf16_kernel<<<token_count, 256, 0, stream>>>(
      input, weight, output, token_count, hidden_size, epsilon);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void rope_bf16(BFloat16* q, BFloat16* k, int num_heads, int num_kv_heads,
               int head_dim, int position, float theta,
               cudaStream_t stream) {
  const dim3 q_grid(num_heads, 1);
  const dim3 k_grid(num_kv_heads, 1);
  rope_bf16_kernel<<<q_grid, head_dim / 2, 0, stream>>>(
      q, 1, num_heads, head_dim, position, theta);
  rope_bf16_kernel<<<k_grid, head_dim / 2, 0, stream>>>(
      k, 1, num_kv_heads, head_dim, position, theta);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void rope_batch_bf16(BFloat16* q, BFloat16* k, int token_count,
                     int num_heads, int num_kv_heads, int head_dim,
                     int start_position, float theta, cudaStream_t stream) {
  const dim3 q_grid(num_heads, token_count);
  const dim3 k_grid(num_kv_heads, token_count);
  rope_bf16_kernel<<<q_grid, head_dim / 2, 0, stream>>>(
      q, token_count, num_heads, head_dim, start_position, theta);
  rope_bf16_kernel<<<k_grid, head_dim / 2, 0, stream>>>(
      k, token_count, num_kv_heads, head_dim, start_position, theta);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void store_kv_bf16(const BFloat16* key, const BFloat16* value,
                   BFloat16* key_cache, BFloat16* value_cache, int kv_heads,
                   int head_dim, int position, int max_sequence_length,
                   cudaStream_t stream) {
  const int total = kv_heads * head_dim;
  store_kv_bf16_kernel<<<blocks_for(total), 256, 0, stream>>>(
      key, value, key_cache, value_cache, 1, kv_heads, head_dim, position,
      max_sequence_length);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void store_kv_batch_bf16(const BFloat16* key, const BFloat16* value,
                         BFloat16* key_cache, BFloat16* value_cache,
                         int token_count, int kv_heads, int head_dim,
                         int max_sequence_length, cudaStream_t stream) {
  const int total = token_count * kv_heads * head_dim;
  store_kv_bf16_kernel<<<blocks_for(total), 256, 0, stream>>>(
      key, value, key_cache, value_cache, token_count, kv_heads, head_dim, 0,
      max_sequence_length);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void attention_decode_bf16(
    const BFloat16* q, const BFloat16* key_cache,
    const BFloat16* value_cache, BFloat16* output, int num_heads,
    int num_kv_heads, int head_dim, int sequence_length,
    int max_sequence_length, cudaStream_t stream) {
  attention_decode_bf16_kernel<<<
      num_heads, 128, sequence_length * sizeof(float), stream>>>(
      q, key_cache, value_cache, output, num_heads, num_kv_heads, head_dim,
      sequence_length, max_sequence_length);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void attention_prefill_bf16(
    const BFloat16* q, const BFloat16* key_cache,
    const BFloat16* value_cache, BFloat16* output, int token_count,
    int num_heads, int num_kv_heads, int head_dim, int max_sequence_length,
    cudaStream_t stream) {
  const dim3 grid(num_heads, token_count);
  attention_prefill_bf16_kernel<<<
      grid, 128, token_count * sizeof(float), stream>>>(
      q, key_cache, value_cache, output, token_count, num_heads, num_kv_heads,
      head_dim, max_sequence_length);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void add_inplace_bf16(BFloat16* x, const BFloat16* residual, int n,
                      cudaStream_t stream) {
  add_bf16_kernel<<<blocks_for(n), 256, 0, stream>>>(x, residual, n);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void silu_mul_bf16(const BFloat16* gate, const BFloat16* up,
                   BFloat16* output, int n, cudaStream_t stream) {
  silu_mul_bf16_kernel<<<blocks_for(n), 256, 0, stream>>>(
      gate, up, output, n);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void argmax_bf16(const BFloat16* values, int n, int* output,
                 cudaStream_t stream) {
  argmax_bf16_kernel<<<1, 256, 0, stream>>>(values, n, output);
  INFER_CUDA_CHECK(cudaGetLastError());
}


void gemv_bf16(const BFloat16* weight, const BFloat16* bias,
                const BFloat16* input, BFloat16* output, int out_features,
                int in_features, cudaStream_t stream) {
  gemv_bf16_kernel<<<out_features, 256, 0, stream>>>(
      weight, bias, input, output, out_features, in_features);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void gemv_bf16_cublas(cublasHandle_t handle, const BFloat16* weight,
                       const BFloat16* bias, const BFloat16* input,
                       BFloat16* output, int out_features, int in_features,
                       cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_check(
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, 1,
                   in_features, &alpha, weight, CUDA_R_16BF, in_features,
                   input, CUDA_R_16BF, in_features, &beta, output, CUDA_R_16BF,
                   out_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
      "cublasGemmEx BF16 GEMV");
  if (bias) {
    add_bias_batch_bf16_kernel<<<blocks_for(out_features), 256, 0, stream>>>(
        output, bias, 1, out_features);
    INFER_CUDA_CHECK(cudaGetLastError());
  }
}

void gemm_bf16(const BFloat16* weight, const BFloat16* bias,
                const BFloat16* input, BFloat16* output, int tokens,
                int out_features, int in_features, cudaStream_t stream) {
  constexpr int block_n = 64;
  constexpr int threads = 256;
  const bool aligned = out_features % block_n == 0 && in_features % 64 == 0;
  if (aligned && tokens % 128 == 0 && out_features % 128 == 0) {
    constexpr int block_m = 128;
    constexpr int large_block_n = 128;
    const dim3 grid(out_features / large_block_n, tokens / block_m);
    gemm_bf16_wmma_async_m128n128_kernel<<<grid, 512, 0, stream>>>(
        weight, bias, input, output, tokens, out_features, in_features);
  } else if (aligned && tokens % 32 == 0) {
    constexpr int block_m = 32;
    const dim3 grid(out_features / block_n, tokens / block_m);
    gemm_bf16_wmma_async_kernel<<<grid, threads, 0, stream>>>(
        weight, bias, input, output, tokens, out_features, in_features);
  } else {
    constexpr int block_m = 32;
    const dim3 grid((out_features + block_n - 1) / block_n,
                    (tokens + block_m - 1) / block_m);
    gemm_bf16_wmma_kernel<block_m><<<grid, threads, 0, stream>>>(
        weight, bias, input, output, tokens, out_features, in_features);
  }
  INFER_CUDA_CHECK(cudaGetLastError());
}

void gemv_w8a16(const int8_t* weight, const BFloat16* scales,
                 int group_size, const BFloat16* bias,
                 const BFloat16* input, BFloat16* output, int out_features,
                 int in_features, cudaStream_t stream) {
  INFER_CHECK(group_size == 64, "w8a16 group size must be 64");
  linear_w8a16_kernel<<<dim3(out_features, 1), 256, 0, stream>>>(
      weight, scales, group_size, bias, input, output, 1, out_features,
      in_features);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void gemm_w8a16(const int8_t* weight, const BFloat16* scales,
                 int group_size, const BFloat16* bias,
                 const BFloat16* input, BFloat16* output, int tokens,
                 int out_features, int in_features, cudaStream_t stream) {
  INFER_CHECK(group_size == 64, "w8a16 group size must be 64");
  linear_w8a16_kernel<<<dim3(out_features, tokens), 256, 0, stream>>>(
      weight, scales, group_size, bias, input, output, tokens, out_features,
      in_features);
  INFER_CUDA_CHECK(cudaGetLastError());
}

void gemm_bf16_cublas(cublasHandle_t handle, const BFloat16* weight,
                       const BFloat16* bias, const BFloat16* input,
                       BFloat16* output, int tokens, int out_features,
                       int in_features, cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_check(
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, tokens,
                   in_features, &alpha, weight, CUDA_R_16BF, in_features,
                   input, CUDA_R_16BF, in_features, &beta, output, CUDA_R_16BF,
                   out_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
      "cublasGemmEx BF16 GEMM");
  if (bias) {
    const int total = tokens * out_features;
    add_bias_batch_bf16_kernel<<<blocks_for(total), 256, 0, stream>>>(
        output, bias, tokens, out_features);
    INFER_CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace infer::cuda
