#include "infer/qwen2.hpp"

#include "infer/cpu_ops.hpp"

#ifdef INFER_WITH_NVTX
#include <nvToolsExt.h>
#endif

namespace infer {
namespace {

size_t align_up(size_t value, size_t alignment = 256) {
  return (value + alignment - 1) / alignment * alignment;
}

class ProfileRange {
 public:
  explicit ProfileRange(const char* name) {
#ifdef INFER_WITH_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~ProfileRange() {
#ifdef INFER_WITH_NVTX
    nvtxRangePop();
#endif
  }
};

std::string layer_prefix(int layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

}  // namespace

Qwen2Model::Qwen2Model(const std::filesystem::path& model_directory,
                       RuntimeOptions options)
    : options_(options),
      config_(ModelConfig::load(model_directory / "config.json")),
      archive_(model_directory /
               (options.precision == Precision::kFloat32 ? "model.fp32.qbin"
                                                          : "model.int8.qbin")) {
  INFER_CHECK(options_.max_sequence_length > 0, "max sequence length must be positive");
  INFER_CHECK(options_.max_sequence_length <= config_.max_position_embeddings,
              "max sequence length exceeds model limit");
  initialize_weights();
  initialize_workspace();
}

Qwen2Model::~Qwen2Model() = default;

void Qwen2Model::initialize_weights() {
  if (options_.backend == Device::kCpu) {
    for (const auto& rec : archive_.records()) weights_[rec.name] = archive_.data(rec.name);
    return;
  }
  cuda_context_ = std::make_unique<cuda::Context>();
  size_t total = 0;
  std::unordered_map<std::string, size_t> offsets;
  for (const auto& rec : archive_.records()) {
    total = align_up(total);
    offsets[rec.name] = total;
    total += rec.nbytes;
  }
  device_weights_.resize(total, Device::kCuda);
  auto* base = static_cast<std::byte*>(device_weights_.data());
  for (const auto& rec : archive_.records()) {
    void* destination = base + offsets.at(rec.name);
    INFER_CUDA_CHECK(cudaMemcpyAsync(destination, archive_.data(rec.name), rec.nbytes,
                                     cudaMemcpyHostToDevice, cuda_context_->stream()));
    weights_[rec.name] = destination;
  }
  cuda_context_->synchronize();
}

void Qwen2Model::initialize_workspace() {
  const int hidden = config_.hidden_size;
  const int kv = config_.kv_dim();
  const int intermediate = config_.intermediate_size;
  const int vocab = config_.vocab_size;
  const size_t float_count =
      static_cast<size_t>(hidden) * 5 + static_cast<size_t>(kv) * 2 +
      static_cast<size_t>(intermediate) * 3 + vocab + options_.max_sequence_length;
  workspace_.resize(align_up(float_count * sizeof(float)), options_.backend);
  auto* base = static_cast<float*>(workspace_.data());
  size_t cursor = 0;
  auto take = [&](int n) {
    float* result = base + cursor;
    cursor += static_cast<size_t>(n);
    return result;
  };
  x_ = take(hidden);
  norm_ = take(hidden);
  q_ = take(hidden);
  k_ = take(kv);
  v_ = take(kv);
  attention_ = take(hidden);
  hidden_tmp_ = take(hidden);
  gate_ = take(intermediate);
  up_ = take(intermediate);
  mlp_ = take(intermediate);
  logits_ = take(vocab);
  attention_scores_ = take(options_.max_sequence_length);

  const size_t per_layer = static_cast<size_t>(config_.num_kv_heads) *
                           options_.max_sequence_length * config_.head_dim();
  kv_cache_.resize(2 * config_.num_layers * per_layer * sizeof(float), options_.backend);
  if (options_.backend == Device::kCuda) argmax_buffer_.resize(sizeof(int), Device::kCuda);
}

const void* Qwen2Model::weight(std::string_view name) const {
  const auto it = weights_.find(std::string(name));
  INFER_CHECK(it != weights_.end(), "missing runtime weight: " + std::string(name));
  return it->second;
}

const float* Qwen2Model::float_weight(std::string_view name) const {
  INFER_CHECK(archive_.record(name).dtype == DType::kFloat32,
              "expected float32 weight: " + std::string(name));
  return static_cast<const float*>(weight(name));
}

const float* Qwen2Model::optional_float_weight(std::string_view name) const {
  if (name.empty() || !archive_.contains(name)) return nullptr;
  return float_weight(name);
}

void Qwen2Model::linear(std::string_view weight_name, std::string_view bias_name,
                        const float* input, float* output, int out_features,
                        int in_features, bool prefer_cublas) {
  const auto& rec = archive_.record(weight_name);
  const float* bias = optional_float_weight(bias_name);
  if (options_.backend == Device::kCpu) {
    if (rec.dtype == DType::kFloat32) {
      cpu::linear_fp32(static_cast<const float*>(weight(weight_name)), bias, input, output,
                       out_features, in_features);
    } else {
      INFER_CHECK(rec.quant.has_value(), "int8 tensor lacks quantization metadata");
      cpu::linear_int8(static_cast<const int8_t*>(weight(weight_name)),
                       float_weight(rec.quant->scale_tensor), rec.quant->group_size,
                       bias, input, output, out_features, in_features);
    }
    return;
  }
  if (rec.dtype == DType::kFloat32) {
    if (prefer_cublas || options_.use_cublas_gemv) {
      cuda::gemv_fp32_cublas(cuda_context_->cublas(),
                             static_cast<const float*>(weight(weight_name)), bias,
                             input, output, out_features, in_features,
                             cuda_context_->stream());
    } else {
      cuda::gemv_fp32(static_cast<const float*>(weight(weight_name)), bias, input, output,
                      out_features, in_features, cuda_context_->stream());
    }
  } else {
    INFER_CHECK(rec.quant.has_value(), "int8 tensor lacks quantization metadata");
    cuda::gemv_int8(static_cast<const int8_t*>(weight(weight_name)),
                    float_weight(rec.quant->scale_tensor), rec.quant->group_size,
                    bias, input, output, out_features, in_features,
                    cuda_context_->stream());
  }
}

float* Qwen2Model::layer_key_cache(int layer) {
  const size_t per_layer = static_cast<size_t>(config_.num_kv_heads) *
                           options_.max_sequence_length * config_.head_dim();
  return static_cast<float*>(kv_cache_.data()) + static_cast<size_t>(layer) * per_layer;
}

float* Qwen2Model::layer_value_cache(int layer) {
  const size_t per_layer = static_cast<size_t>(config_.num_kv_heads) *
                           options_.max_sequence_length * config_.head_dim();
  return static_cast<float*>(kv_cache_.data()) +
         static_cast<size_t>(config_.num_layers + layer) * per_layer;
}

int Qwen2Model::forward_cpu(int token, int position) {
  const int hidden = config_.hidden_size;
  const int kv_dim = config_.kv_dim();
  const int head_dim = config_.head_dim();
  const auto* embedding = float_weight("model.embed_tokens.weight");
  std::copy(embedding + static_cast<size_t>(token) * hidden,
            embedding + static_cast<size_t>(token + 1) * hidden, x_);
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    const auto prefix = layer_prefix(layer);
    cpu::rms_norm(x_, float_weight(prefix + "input_layernorm.weight"), norm_, hidden,
                  config_.rms_norm_eps);
    linear(prefix + "self_attn.q_proj.weight", prefix + "self_attn.q_proj.bias",
           norm_, q_, hidden, hidden);
    linear(prefix + "self_attn.k_proj.weight", prefix + "self_attn.k_proj.bias",
           norm_, k_, kv_dim, hidden);
    linear(prefix + "self_attn.v_proj.weight", prefix + "self_attn.v_proj.bias",
           norm_, v_, kv_dim, hidden);
    cpu::rope(q_, k_, config_.num_heads, config_.num_kv_heads, head_dim,
              position, config_.rope_theta);
    float* key_cache = layer_key_cache(layer);
    float* value_cache = layer_value_cache(layer);
    for (int head = 0; head < config_.num_kv_heads; ++head) {
      std::copy(k_ + head * head_dim, k_ + (head + 1) * head_dim,
                key_cache + (static_cast<size_t>(head) * options_.max_sequence_length + position) * head_dim);
      std::copy(v_ + head * head_dim, v_ + (head + 1) * head_dim,
                value_cache + (static_cast<size_t>(head) * options_.max_sequence_length + position) * head_dim);
    }
    cpu::attention_decode(q_, key_cache, value_cache, attention_, config_.num_heads,
                          config_.num_kv_heads, head_dim, position + 1,
                          options_.max_sequence_length, attention_scores_);
    linear(prefix + "self_attn.o_proj.weight", {}, attention_, hidden_tmp_, hidden, hidden);
    cpu::add_inplace(x_, hidden_tmp_, hidden);
    cpu::rms_norm(x_, float_weight(prefix + "post_attention_layernorm.weight"), norm_,
                  hidden, config_.rms_norm_eps);
    linear(prefix + "mlp.gate_proj.weight", {}, norm_, gate_,
           config_.intermediate_size, hidden);
    linear(prefix + "mlp.up_proj.weight", {}, norm_, up_,
           config_.intermediate_size, hidden);
    cpu::silu_mul(gate_, up_, mlp_, config_.intermediate_size);
    linear(prefix + "mlp.down_proj.weight", {}, mlp_, hidden_tmp_,
           hidden, config_.intermediate_size);
    cpu::add_inplace(x_, hidden_tmp_, hidden);
  }
  cpu::rms_norm(x_, float_weight("model.norm.weight"), norm_, hidden,
                config_.rms_norm_eps);
  const std::string lm_head = archive_.contains("lm_head.weight")
                                  ? "lm_head.weight" : "model.embed_tokens.weight";
  linear(lm_head, {}, norm_, logits_, config_.vocab_size, hidden);
  return cpu::argmax(logits_, config_.vocab_size);
}

int Qwen2Model::forward_cuda(int token, int position) {
  const int hidden = config_.hidden_size;
  const int kv_dim = config_.kv_dim();
  const int head_dim = config_.head_dim();
  auto stream = cuda_context_->stream();
  cuda::embedding(float_weight("model.embed_tokens.weight"), token, x_, hidden, stream);
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    const auto prefix = layer_prefix(layer);
    cuda::rms_norm(x_, float_weight(prefix + "input_layernorm.weight"), norm_, hidden,
                   config_.rms_norm_eps, stream);
    linear(prefix + "self_attn.q_proj.weight", prefix + "self_attn.q_proj.bias",
           norm_, q_, hidden, hidden);
    linear(prefix + "self_attn.k_proj.weight", prefix + "self_attn.k_proj.bias",
           norm_, k_, kv_dim, hidden);
    linear(prefix + "self_attn.v_proj.weight", prefix + "self_attn.v_proj.bias",
           norm_, v_, kv_dim, hidden);
    cuda::rope(q_, k_, config_.num_heads, config_.num_kv_heads, head_dim,
               position, config_.rope_theta, stream);
    float* key_cache = layer_key_cache(layer);
    float* value_cache = layer_value_cache(layer);
    cuda::store_kv(k_, v_, key_cache, value_cache, config_.num_kv_heads,
                   head_dim, position, options_.max_sequence_length, stream);
    cuda::attention_decode(q_, key_cache, value_cache, attention_, config_.num_heads,
                           config_.num_kv_heads, head_dim, position + 1,
                           options_.max_sequence_length, stream);
    linear(prefix + "self_attn.o_proj.weight", {}, attention_, hidden_tmp_, hidden, hidden);
    cuda::add_inplace(x_, hidden_tmp_, hidden, stream);
    cuda::rms_norm(x_, float_weight(prefix + "post_attention_layernorm.weight"), norm_,
                   hidden, config_.rms_norm_eps, stream);
    linear(prefix + "mlp.gate_proj.weight", {}, norm_, gate_,
           config_.intermediate_size, hidden);
    linear(prefix + "mlp.up_proj.weight", {}, norm_, up_,
           config_.intermediate_size, hidden);
    cuda::silu_mul(gate_, up_, mlp_, config_.intermediate_size, stream);
    linear(prefix + "mlp.down_proj.weight", {}, mlp_, hidden_tmp_,
           hidden, config_.intermediate_size);
    cuda::add_inplace(x_, hidden_tmp_, hidden, stream);
  }
  cuda::rms_norm(x_, float_weight("model.norm.weight"), norm_, hidden,
                 config_.rms_norm_eps, stream);
  const std::string lm_head = archive_.contains("lm_head.weight")
                                  ? "lm_head.weight" : "model.embed_tokens.weight";
  linear(lm_head, {}, norm_, logits_, config_.vocab_size, hidden, true);
  cuda::argmax(logits_, config_.vocab_size, static_cast<int*>(argmax_buffer_.data()), stream);
  int result = 0;
  INFER_CUDA_CHECK(cudaMemcpyAsync(&result, argmax_buffer_.data(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
  cuda_context_->synchronize();
  return result;
}

int Qwen2Model::forward_token(int token, int position) {
  INFER_CHECK(token >= 0 && token < config_.vocab_size, "input token out of range");
  INFER_CHECK(position >= 0 && position < options_.max_sequence_length,
              "KV cache capacity exceeded");
  ProfileRange range("qwen2.forward_token");
  return options_.backend == Device::kCpu ? forward_cpu(token, position)
                                          : forward_cuda(token, position);
}

std::vector<float> Qwen2Model::last_logits_host() const {
  std::vector<float> result(static_cast<size_t>(config_.vocab_size));
  if (options_.backend == Device::kCpu) {
    std::copy(logits_, logits_ + config_.vocab_size, result.begin());
  } else {
    INFER_CUDA_CHECK(cudaMemcpyAsync(result.data(), logits_, result.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, cuda_context_->stream()));
    cuda_context_->synchronize();
  }
  return result;
}

void Qwen2Model::synchronize() const {
  if (cuda_context_) cuda_context_->synchronize();
}

void Qwen2Model::reset() {
  position_ = 0;
  has_prefilled_ = false;
}

PrefillMode Qwen2Model::effective_prefill_mode(size_t prompt_tokens) const {
  INFER_CHECK(prompt_tokens > 0, "prompt must contain at least one token");
  // Checkpoint 5 only splits the state machine. Batched kernels arrive later.
  if (options_.prefill_mode == PrefillMode::kAuto) return PrefillMode::kSerial;
  return options_.prefill_mode;
}

int Qwen2Model::prefill(const std::vector<int>& prompt_tokens) {
  INFER_CHECK(!prompt_tokens.empty(), "prompt must contain at least one token");
  INFER_CHECK(position_ == 0 && !has_prefilled_,
              "prefill requires a fresh model state; call reset first");
  INFER_CHECK(prompt_tokens.size() <= static_cast<size_t>(options_.max_sequence_length),
              "prompt exceeds KV cache capacity");
  for (const int token : prompt_tokens) {
    INFER_CHECK(token >= 0 && token < config_.vocab_size, "input token out of range");
  }

  const PrefillMode mode = effective_prefill_mode(prompt_tokens.size());
  INFER_CHECK(mode == PrefillMode::kSerial,
              "batched prefill is not implemented in checkpoint 5");

  ProfileRange range("qwen2.prefill");
  int next = 0;
  for (const int token : prompt_tokens) {
    next = forward_token(token, position_);
    ++position_;
  }
  has_prefilled_ = true;
  return next;
}

int Qwen2Model::decode_next(int token) {
  INFER_CHECK(has_prefilled_, "decode_next requires a successful prefill");
  INFER_CHECK(position_ < options_.max_sequence_length, "KV cache capacity exceeded");
  ProfileRange range("qwen2.decode_next");
  const int next = forward_token(token, position_);
  ++position_;
  return next;
}

GenerationResult Qwen2Model::generate(const std::vector<int>& prompt_tokens,
                                      int max_new_tokens, bool stop_on_eos) {
  INFER_CHECK(!prompt_tokens.empty(), "prompt must contain at least one token");
  INFER_CHECK(max_new_tokens >= 0, "max_new_tokens must be non-negative");
  INFER_CHECK(static_cast<int>(prompt_tokens.size()) + std::max(0, max_new_tokens - 1) <=
                  options_.max_sequence_length,
              "prompt and generation exceed KV cache capacity");
  reset();
  GenerationResult result;
  result.stats.prompt_tokens = static_cast<int>(prompt_tokens.size());
  WallTimer total;
  WallTimer prefill_timer;
  int next = prefill(prompt_tokens);
  synchronize();
  result.stats.ttft_ms = prefill_timer.elapsed_ms();
  if (max_new_tokens > 0) result.tokens.push_back(next);
  WallTimer decode;
  {
    ProfileRange range("qwen2.decode");
    while (static_cast<int>(result.tokens.size()) < max_new_tokens &&
           (!stop_on_eos || result.tokens.back() != config_.eos_token_id)) {
      next = decode_next(result.tokens.back());
      result.tokens.push_back(next);
    }
  }
  synchronize();
  result.stats.decode_ms = decode.elapsed_ms();
  result.stats.total_ms = total.elapsed_ms();
  result.stats.generated_tokens = static_cast<int>(result.tokens.size());
  return result;
}

}  // namespace infer
