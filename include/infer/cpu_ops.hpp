#pragma once

#include "infer/common.hpp"

namespace infer::cpu {

void copy(const float* input, float* output, int n);
void embedding_batch(const float* embedding, const int* tokens, float* output,
                     int token_count, int hidden_size);
void add_inplace(float* x, const float* residual, int n);
void rms_norm(const float* input, const float* weight, float* output,
              int n, float epsilon);
void rms_norm_batch(const float* input, const float* weight, float* output,
                    int token_count, int hidden_size, float epsilon);
void linear_fp32(const float* weight, const float* bias, const float* input,
                 float* output, int out_features, int in_features);
void linear_fp32_batch(const float* weight, const float* bias, const float* input,
                       float* output, int token_count, int out_features,
                       int in_features);
void linear_int8(const int8_t* weight, const float* scales, int group_size,
                 const float* bias, const float* input, float* output,
                 int out_features, int in_features);
void rope(float* q, float* k, int num_heads, int num_kv_heads,
          int head_dim, int position, float theta);
void rope_batch(float* q, float* k, int token_count, int num_heads,
                int num_kv_heads, int head_dim, int start_position, float theta);
void store_kv_batch(const float* k, const float* v, float* key_cache,
                    float* value_cache, int token_count, int num_kv_heads,
                    int head_dim, int max_sequence_length);
void attention_decode(const float* q, const float* key_cache, const float* value_cache,
                      float* output, int num_heads, int num_kv_heads,
                      int head_dim, int sequence_length, int max_sequence_length,
                      float* score_scratch);
void attention_prefill(const float* q, const float* key_cache,
                       const float* value_cache, float* output, int token_count,
                       int num_heads, int num_kv_heads, int head_dim,
                       int max_sequence_length, float* score_scratch);
void silu_mul(const float* gate, const float* up, float* output, int n);
int argmax(const float* values, int n);

}  // namespace infer::cpu
