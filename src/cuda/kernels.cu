#include "infer/cuda_ops.hpp"

#include <cfloat>
#include <cuda_runtime.h>

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

template <class Weight, bool Quantized>
__global__ void gemv_kernel(const Weight* weight, const float* scales, int group_size,
                            const float* bias, const float* input, float* output,
                            int out_features, int in_features) {
  const int row = blockIdx.x;
  if (row >= out_features) return;
  __shared__ float reduction[256];
  float sum = 0.0f;
  for (int col = threadIdx.x; col < in_features; col += blockDim.x) {
    float w;
    if constexpr (Quantized) {
      const int groups = (in_features + group_size - 1) / group_size;
      w = static_cast<float>(weight[static_cast<size_t>(row) * in_features + col]) *
          scales[static_cast<size_t>(row) * groups + col / group_size];
    } else {
      w = weight[static_cast<size_t>(row) * in_features + col];
    }
    sum = fmaf(w, input[col], sum);
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[row] = reduction[0] + (bias ? bias[row] : 0.0f);
}

__global__ void gemv_int8x4_kernel(const int8_t* weight, const float* scales,
                                   int group_size, const float* bias,
                                   const float* input, float* output,
                                   int out_features, int in_features) {
  const int row = blockIdx.x;
  if (row >= out_features) return;
  __shared__ float reduction[256];
  const int groups = (in_features + group_size - 1) / group_size;
  const int8_t* row_weight = weight + static_cast<size_t>(row) * in_features;
  float sum = 0.0f;
  for (int col = threadIdx.x * 4; col < in_features; col += blockDim.x * 4) {
    if (col + 3 < in_features) {
      const char4 packed = *reinterpret_cast<const char4*>(row_weight + col);
      const float scale = scales[static_cast<size_t>(row) * groups + col / group_size];
      sum = fmaf(static_cast<float>(packed.x) * scale, input[col], sum);
      sum = fmaf(static_cast<float>(packed.y) * scale, input[col + 1], sum);
      sum = fmaf(static_cast<float>(packed.z) * scale, input[col + 2], sum);
      sum = fmaf(static_cast<float>(packed.w) * scale, input[col + 3], sum);
    } else {
      for (int i = col; i < in_features; ++i) {
        const float scale = scales[static_cast<size_t>(row) * groups + i / group_size];
        sum = fmaf(static_cast<float>(row_weight[i]) * scale, input[i], sum);
      }
    }
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[row] = reduction[0] + (bias ? bias[row] : 0.0f);
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

__global__ void gemm_int8x4_kernel(const int8_t* weight, const float* scales,
                                   int group_size, const float* input, float* output,
                                   int tokens, int out_features, int in_features) {
  const int row = blockIdx.x;
  const int token = blockIdx.y;
  if (row >= out_features || token >= tokens) return;
  __shared__ float reduction[256];
  const int groups = (in_features + group_size - 1) / group_size;
  const int8_t* row_weight = weight + static_cast<size_t>(row) * in_features;
  const float* token_input = input + static_cast<size_t>(token) * in_features;
  float sum = 0.0f;
  for (int col = threadIdx.x * 4; col < in_features; col += blockDim.x * 4) {
    if (col + 3 < in_features) {
      const char4 packed = *reinterpret_cast<const char4*>(row_weight + col);
      const float scale = scales[static_cast<size_t>(row) * groups + col / group_size];
      sum = fmaf(static_cast<float>(packed.x) * scale, token_input[col], sum);
      sum = fmaf(static_cast<float>(packed.y) * scale, token_input[col + 1], sum);
      sum = fmaf(static_cast<float>(packed.z) * scale, token_input[col + 2], sum);
      sum = fmaf(static_cast<float>(packed.w) * scale, token_input[col + 3], sum);
    } else {
      for (int i = col; i < in_features; ++i) {
        const float scale = scales[static_cast<size_t>(row) * groups + i / group_size];
        sum = fmaf(static_cast<float>(row_weight[i]) * scale, token_input[i], sum);
      }
    }
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[static_cast<size_t>(token) * out_features + row] = reduction[0];
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
  gemv_kernel<float, false><<<out_features, 256, 0, stream>>>(
      weight, nullptr, 0, bias, input, output, out_features, in_features);
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

void gemv_int8(const int8_t* weight, const float* scales, int group_size,
               const float* bias, const float* input, float* output,
               int out_features, int in_features, cudaStream_t stream) {
  gemv_int8x4_kernel<<<out_features, 256, 0, stream>>>(
      weight, scales, group_size, bias, input, output, out_features, in_features);
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

void gemm_int8(const int8_t* weight, const float* scales, int group_size,
               const float* input, float* output, int tokens,
               int out_features, int in_features, cudaStream_t stream) {
  const dim3 grid(out_features, tokens);
  gemm_int8x4_kernel<<<grid, 256, 0, stream>>>(
      weight, scales, group_size, input, output, tokens, out_features, in_features);
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

}  // namespace infer::cuda
