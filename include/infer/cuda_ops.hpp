#pragma once

#include "infer/common.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>

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
void embedding_batch(const float* table, const int* tokens, float* output,
                     int token_count, int hidden_size, cudaStream_t stream);
void rms_norm_batch(const float* input, const float* weight, float* output,
                    int token_count, int hidden_size, float epsilon,
                    cudaStream_t stream);
void gemm_fp32_cublas(cublasHandle_t handle, const float* weight,
                      const float* input, float* output, int tokens,
                      int out_features, int in_features, cudaStream_t stream);
void add_bias_batch(float* output, const float* bias, int tokens,
                    int out_features, cudaStream_t stream);
void rope_batch(float* q, float* k, int token_count, int num_heads,
                int num_kv_heads, int head_dim, int start_position,
                float theta, cudaStream_t stream);
void store_kv_batch(const float* key, const float* value, float* key_cache,
                    float* value_cache, int token_count, int kv_heads,
                    int head_dim, int max_sequence_length,
                    cudaStream_t stream);
void attention_prefill(const float* q, const float* key_cache,
                       const float* value_cache, float* output,
                       int token_count, int num_heads, int num_kv_heads,
                       int head_dim, int max_sequence_length,
                       cudaStream_t stream);
void add_inplace(float* x, const float* residual, int n, cudaStream_t stream);
void silu_mul(const float* gate, const float* up, float* output, int n,
              cudaStream_t stream);
void argmax(const float* values, int n, int* output, cudaStream_t stream);



void gemv_bf16(const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                const __nv_bfloat16* input, __nv_bfloat16* output,
                int out_features, int in_features, cudaStream_t stream);
void gemv_bf16_cublas(cublasHandle_t handle, const __nv_bfloat16* weight,
                       const __nv_bfloat16* bias,
                       const __nv_bfloat16* input, __nv_bfloat16* output,
                       int out_features, int in_features,
                       cudaStream_t stream);
void gemm_bf16(const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                const __nv_bfloat16* input, __nv_bfloat16* output,
                int tokens, int out_features, int in_features,
                cudaStream_t stream);
void gemm_bf16_cublas(cublasHandle_t handle, const __nv_bfloat16* weight,
                       const __nv_bfloat16* bias,
                       const __nv_bfloat16* input, __nv_bfloat16* output,
                       int tokens, int out_features, int in_features,
                       cudaStream_t stream);

void gemv_w8a16(const int8_t* weight, const __nv_bfloat16* scales,
                 int group_size, const __nv_bfloat16* bias,
                 const __nv_bfloat16* input, __nv_bfloat16* output,
                 int out_features, int in_features, cudaStream_t stream);
void gemm_w8a16(const int8_t* weight, const __nv_bfloat16* scales,
                 int group_size, const __nv_bfloat16* bias,
                 const __nv_bfloat16* input, __nv_bfloat16* output,
                 int tokens, int out_features, int in_features,
                 cudaStream_t stream);

void embedding_bf16(const __nv_bfloat16* table, int token,
                    __nv_bfloat16* output, int hidden_size,
                    cudaStream_t stream);
void embedding_batch_bf16(const __nv_bfloat16* table, const int* tokens,
                          __nv_bfloat16* output, int token_count,
                          int hidden_size, cudaStream_t stream);
void rms_norm_bf16(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                   __nv_bfloat16* output, int n, float epsilon,
                   cudaStream_t stream);
void rms_norm_batch_bf16(const __nv_bfloat16* input,
                         const __nv_bfloat16* weight,
                         __nv_bfloat16* output, int token_count,
                         int hidden_size, float epsilon, cudaStream_t stream);
void rope_bf16(__nv_bfloat16* q, __nv_bfloat16* k, int num_heads,
               int num_kv_heads, int head_dim, int position, float theta,
               cudaStream_t stream);
void rope_batch_bf16(__nv_bfloat16* q, __nv_bfloat16* k, int token_count,
                     int num_heads, int num_kv_heads, int head_dim,
                     int start_position, float theta, cudaStream_t stream);
void store_kv_bf16(const __nv_bfloat16* key, const __nv_bfloat16* value,
                   __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
                   int kv_heads, int head_dim, int position,
                   int max_sequence_length, cudaStream_t stream);
void store_kv_batch_bf16(const __nv_bfloat16* key,
                         const __nv_bfloat16* value,
                         __nv_bfloat16* key_cache,
                         __nv_bfloat16* value_cache, int token_count,
                         int kv_heads, int head_dim, int max_sequence_length,
                         cudaStream_t stream);
void attention_decode_bf16(const __nv_bfloat16* q,
                           const __nv_bfloat16* key_cache,
                           const __nv_bfloat16* value_cache,
                           __nv_bfloat16* output, int num_heads,
                           int num_kv_heads, int head_dim, int sequence_length,
                           int max_sequence_length, cudaStream_t stream);
void attention_prefill_bf16(const __nv_bfloat16* q,
                            const __nv_bfloat16* key_cache,
                            const __nv_bfloat16* value_cache,
                            __nv_bfloat16* output, int token_count,
                            int num_heads, int num_kv_heads, int head_dim,
                            int max_sequence_length, cudaStream_t stream);
void add_inplace_bf16(__nv_bfloat16* x, const __nv_bfloat16* residual,
                      int n, cudaStream_t stream);
void silu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* output, int n, cudaStream_t stream);
void argmax_bf16(const __nv_bfloat16* values, int n, int* output,
                 cudaStream_t stream);

}  // namespace infer::cuda
