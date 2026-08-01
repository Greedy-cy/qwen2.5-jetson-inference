#pragma once

#include "infer/common.hpp"

#include <cublas_v2.h>

namespace infer::cuda {

class Context {
 public:
  Context();
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  cudaStream_t stream() const { return stream_; }
  cublasHandle_t cublas() const { return cublas_; }
  void synchronize() const;
 private:
  cudaStream_t stream_{nullptr};
  cublasHandle_t cublas_{nullptr};
};

void embedding(const float* table, int token, float* output, int hidden_size,
               cudaStream_t stream);
void rms_norm(const float* input, const float* weight, float* output,
              int n, float epsilon, cudaStream_t stream);
void gemv_fp32(const float* weight, const float* bias, const float* input,
               float* output, int out_features, int in_features, cudaStream_t stream);
void gemv_fp32_cublas(cublasHandle_t handle, const float* weight, const float* bias,
                      const float* input, float* output, int out_features,
                      int in_features, cudaStream_t stream);
void gemv_int8(const int8_t* weight, const float* scales, int group_size,
               const float* bias, const float* input, float* output,
               int out_features, int in_features, cudaStream_t stream);
void gemm_fp32(const float* weight, const float* input, float* output,
               int tokens, int out_features, int in_features, cudaStream_t stream);
void gemm_int8(const int8_t* weight, const float* scales, int group_size,
               const float* input, float* output, int tokens,
               int out_features, int in_features, cudaStream_t stream);
void rope(float* q, float* k, int num_heads, int num_kv_heads, int head_dim,
          int position, float theta, cudaStream_t stream);
void store_kv(const float* key, const float* value, float* key_cache,
              float* value_cache, int kv_heads, int head_dim, int position,
              int max_sequence_length, cudaStream_t stream);
void attention_decode(const float* q, const float* key_cache, const float* value_cache,
                      float* output, int num_heads, int num_kv_heads, int head_dim,
                      int sequence_length, int max_sequence_length, cudaStream_t stream);
void add_inplace(float* x, const float* residual, int n, cudaStream_t stream);
void silu_mul(const float* gate, const float* up, float* output, int n,
              cudaStream_t stream);
void argmax(const float* values, int n, int* output, cudaStream_t stream);

}  // namespace infer::cuda
