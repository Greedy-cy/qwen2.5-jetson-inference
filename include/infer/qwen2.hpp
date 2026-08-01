#pragma once

#include "infer/archive.hpp"
#include "infer/buffer.hpp"
#include "infer/cuda_ops.hpp"
#include "infer/model_config.hpp"

namespace infer {

enum class LinearKernel { kCustom, kCublas };

inline const char* to_string(LinearKernel value) {
  return value == LinearKernel::kCustom ? "custom" : "cublas";
}

struct RuntimeOptions {
  Device backend{Device::kCuda};
  Precision precision{Precision::kFloat32};
  int max_sequence_length{2048};
  LinearKernel linear_kernel{LinearKernel::kCustom};
};

struct GenerationStats {
  int prompt_tokens{0};
  int generated_tokens{0};
  double ttft_ms{0.0};
  double decode_ms{0.0};
  double total_ms{0.0};
  double decode_tokens_per_second() const {
    return decode_ms > 0.0 && generated_tokens > 1
               ? (generated_tokens - 1) * 1000.0 / decode_ms
               : 0.0;
  }
};

struct GenerationResult {
  std::vector<int> tokens;
  GenerationStats stats;
};

class Qwen2Model {
 public:
  Qwen2Model(const std::filesystem::path& model_directory, RuntimeOptions options);
  ~Qwen2Model();
  Qwen2Model(const Qwen2Model&) = delete;
  Qwen2Model& operator=(const Qwen2Model&) = delete;

  GenerationResult generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                            bool stop_on_eos = true);
  int prefill(const std::vector<int>& prompt_tokens);
  int decode_next(int token);
  std::vector<float> last_logits_host() const;
  void reset();

  const ModelConfig& config() const { return config_; }
  const ModelArchive& archive() const { return archive_; }
  const RuntimeOptions& options() const { return options_; }
  int position() const { return position_; }
  bool has_prefilled() const { return has_prefilled_; }
  size_t workspace_bytes() const {
    return workspace_.bytes() + prefill_workspace_.bytes() + prefill_tokens_.bytes();
  }
  size_t prefill_workspace_bytes() const {
    return prefill_workspace_.bytes() + prefill_tokens_.bytes();
  }
  size_t kv_cache_bytes() const { return kv_cache_.bytes(); }
  size_t device_weight_bytes() const { return device_weights_.bytes(); }

 private:
  void initialize_weights();
  void initialize_workspace();
  void linear(std::string_view weight_name, std::string_view bias_name,
              const float* input, float* output, int out_features,
              int in_features, bool prefer_cublas = false);
  void linear_cuda_bf16(std::string_view weight_name,
                         std::string_view bias_name,
                         const __nv_bfloat16* input, __nv_bfloat16* output,
                         int out_features, int in_features);
  void linear_cpu_fp32_batch(std::string_view weight_name,
                             std::string_view bias_name, const float* input,
                             float* output, int token_count, int out_features,
                             int in_features);
  void linear_cuda_batch(std::string_view weight_name,
                         std::string_view bias_name, const float* input,
                         float* output, int token_count, int out_features,
                         int in_features);
  void linear_cuda_bf16_batch(std::string_view weight_name,
                              std::string_view bias_name,
                              const __nv_bfloat16* input,
                              __nv_bfloat16* output, int token_count,
                              int out_features, int in_features);
  const void* weight(std::string_view name) const;
  const float* float_weight(std::string_view name) const;
  const float* optional_float_weight(std::string_view name) const;
  const __nv_bfloat16* bf16_weight(std::string_view name) const;
  const __nv_bfloat16* optional_bf16_weight(std::string_view name) const;
  float* layer_key_cache(int layer);
  float* layer_value_cache(int layer);
  __nv_bfloat16* bf16_layer_key_cache(int layer);
  __nv_bfloat16* bf16_layer_value_cache(int layer);
  int decode_token(int token, int position);
  int decode_token_cpu(int token, int position);
  int prefill_cpu_fp32(const std::vector<int>& prompt_tokens);
  int prefill_cuda(const std::vector<int>& prompt_tokens);
  int prefill_cuda_bf16(const std::vector<int>& prompt_tokens);
  int decode_token_cuda(int token, int position);
  int decode_token_cuda_bf16(int token, int position);
  void synchronize() const;

  RuntimeOptions options_;
  ModelConfig config_;
  ModelArchive archive_;
  std::unordered_map<std::string, const void*> weights_;
  Buffer device_weights_;
  Buffer workspace_;
  Buffer prefill_workspace_;
  Buffer prefill_tokens_;
  Buffer kv_cache_;
  Buffer argmax_buffer_;
  std::unique_ptr<cuda::Context> cuda_context_;

  float* x_{nullptr};
  float* norm_{nullptr};
  float* q_{nullptr};
  float* k_{nullptr};
  float* v_{nullptr};
  float* attention_{nullptr};
  float* hidden_tmp_{nullptr};
  float* gate_{nullptr};
  float* up_{nullptr};
  float* mlp_{nullptr};
  float* logits_{nullptr};
  float* attention_scores_{nullptr};
  float* prefill_x_{nullptr};
  float* prefill_norm_{nullptr};
  float* prefill_q_{nullptr};
  float* prefill_k_{nullptr};
  float* prefill_v_{nullptr};
  float* prefill_attention_{nullptr};
  float* prefill_hidden_tmp_{nullptr};
  float* prefill_gate_{nullptr};
  float* prefill_up_{nullptr};
  float* prefill_mlp_{nullptr};
  __nv_bfloat16* bf16_x_{nullptr};
  __nv_bfloat16* bf16_norm_{nullptr};
  __nv_bfloat16* bf16_q_{nullptr};
  __nv_bfloat16* bf16_k_{nullptr};
  __nv_bfloat16* bf16_v_{nullptr};
  __nv_bfloat16* bf16_attention_{nullptr};
  __nv_bfloat16* bf16_hidden_tmp_{nullptr};
  __nv_bfloat16* bf16_gate_{nullptr};
  __nv_bfloat16* bf16_up_{nullptr};
  __nv_bfloat16* bf16_mlp_{nullptr};
  __nv_bfloat16* bf16_logits_{nullptr};
  __nv_bfloat16* bf16_prefill_x_{nullptr};
  __nv_bfloat16* bf16_prefill_norm_{nullptr};
  __nv_bfloat16* bf16_prefill_q_{nullptr};
  __nv_bfloat16* bf16_prefill_k_{nullptr};
  __nv_bfloat16* bf16_prefill_v_{nullptr};
  __nv_bfloat16* bf16_prefill_attention_{nullptr};
  __nv_bfloat16* bf16_prefill_hidden_tmp_{nullptr};
  __nv_bfloat16* bf16_prefill_gate_{nullptr};
  __nv_bfloat16* bf16_prefill_up_{nullptr};
  __nv_bfloat16* bf16_prefill_mlp_{nullptr};
  int position_{0};
  bool has_prefilled_{false};
};

}  // namespace infer
