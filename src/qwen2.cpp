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
      archive_(model_directory / archive_filename(options.precision)) {
  INFER_CHECK(options_.max_sequence_length > 0, "max sequence length must be positive");
  INFER_CHECK(options_.max_sequence_length <= config_.max_position_embeddings,
              "max sequence length exceeds model limit");
  INFER_CHECK(options_.backend != Device::kCpu ||
                  options_.precision == Precision::kFloat32,
              "CPU backend currently supports FP32 only");
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
  const bool use_bf16 = options_.precision == Precision::kW16A16;
  const size_t element_size =
      use_bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
  const size_t element_count =
      static_cast<size_t>(hidden) * 5 + static_cast<size_t>(kv) * 2 +
      static_cast<size_t>(intermediate) * 3 + vocab +
      options_.max_sequence_length;
  workspace_.resize(align_up(element_count * element_size), options_.backend);

  if (use_bf16) {
    auto* base = static_cast<__nv_bfloat16*>(workspace_.data());
    size_t cursor = 0;
    auto take = [&](int count) {
      __nv_bfloat16* result = base + cursor;
      cursor += static_cast<size_t>(count);
      return result;
    };
    bf16_x_ = take(hidden);
    bf16_norm_ = take(hidden);
    bf16_q_ = take(hidden);
    bf16_k_ = take(kv);
    bf16_v_ = take(kv);
    bf16_attention_ = take(hidden);
    bf16_hidden_tmp_ = take(hidden);
    bf16_gate_ = take(intermediate);
    bf16_up_ = take(intermediate);
    bf16_mlp_ = take(intermediate);
    bf16_logits_ = take(vocab);
  } else {
    auto* base = static_cast<float*>(workspace_.data());
    size_t cursor = 0;
    auto take = [&](int count) {
      float* result = base + cursor;
      cursor += static_cast<size_t>(count);
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
  }

  const size_t row_width =
      static_cast<size_t>(hidden) * 5 + static_cast<size_t>(kv) * 2 +
      static_cast<size_t>(intermediate) * 3;
  const size_t prefill_element_count =
      static_cast<size_t>(options_.max_sequence_length) * row_width;
  prefill_workspace_.resize(
      align_up(prefill_element_count * element_size), options_.backend);
  if (use_bf16) {
    auto* base = static_cast<__nv_bfloat16*>(prefill_workspace_.data());
    size_t cursor = 0;
    auto take = [&](int width) {
      __nv_bfloat16* result = base + cursor;
      cursor += static_cast<size_t>(options_.max_sequence_length) * width;
      return result;
    };
    bf16_prefill_x_ = take(hidden);
    bf16_prefill_norm_ = take(hidden);
    bf16_prefill_q_ = take(hidden);
    bf16_prefill_k_ = take(kv);
    bf16_prefill_v_ = take(kv);
    bf16_prefill_attention_ = take(hidden);
    bf16_prefill_hidden_tmp_ = take(hidden);
    bf16_prefill_gate_ = take(intermediate);
    bf16_prefill_up_ = take(intermediate);
    bf16_prefill_mlp_ = take(intermediate);
  } else {
    auto* base = static_cast<float*>(prefill_workspace_.data());
    size_t cursor = 0;
    auto take = [&](int width) {
      float* result = base + cursor;
      cursor += static_cast<size_t>(options_.max_sequence_length) * width;
      return result;
    };
    prefill_x_ = take(hidden);
    prefill_norm_ = take(hidden);
    prefill_q_ = take(hidden);
    prefill_k_ = take(kv);
    prefill_v_ = take(kv);
    prefill_attention_ = take(hidden);
    prefill_hidden_tmp_ = take(hidden);
    prefill_gate_ = take(intermediate);
    prefill_up_ = take(intermediate);
    prefill_mlp_ = take(intermediate);
  }
  if (options_.backend == Device::kCuda) {
    prefill_tokens_.resize(
        options_.max_sequence_length * sizeof(int), Device::kCuda);
  }

  const size_t per_layer = static_cast<size_t>(config_.num_kv_heads) *
                           options_.max_sequence_length * config_.head_dim();
  kv_cache_.resize(
      2 * config_.num_layers * per_layer * element_size, options_.backend);
  if (options_.backend == Device::kCuda) {
    argmax_buffer_.resize(sizeof(int), Device::kCuda);
  }
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


const __nv_bfloat16* Qwen2Model::bf16_weight(std::string_view name) const {
  INFER_CHECK(archive_.record(name).dtype == DType::kBFloat16,
              "expected bfloat16 weight: " + std::string(name));
  return static_cast<const __nv_bfloat16*>(weight(name));
}

const __nv_bfloat16* Qwen2Model::optional_bf16_weight(
    std::string_view name) const {
  if (name.empty() || !archive_.contains(name)) return nullptr;
  return bf16_weight(name);
}

void Qwen2Model::linear_cuda_bf16(
    std::string_view weight_name, std::string_view bias_name,
    const __nv_bfloat16* input, __nv_bfloat16* output, int out_features,
    int in_features) {
  INFER_CHECK(options_.backend == Device::kCuda &&
                  options_.precision == Precision::kW16A16,
              "BF16 linear requires CUDA W16A16");
  const auto* weight_pointer = bf16_weight(weight_name);
  const auto* bias = optional_bf16_weight(bias_name);
  if (options_.linear_kernel == LinearKernel::kCublas) {
    cuda::gemv_bf16_cublas(
        cuda_context_->cublas(), weight_pointer, bias, input, output,
        out_features, in_features, cuda_context_->stream());
  } else {
    cuda::gemv_bf16(weight_pointer, bias, input, output, out_features,
                    in_features, cuda_context_->stream());
  }
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
    if (prefer_cublas || options_.linear_kernel == LinearKernel::kCublas) {
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


void Qwen2Model::linear_cuda_bf16_batch(
    std::string_view weight_name, std::string_view bias_name,
    const __nv_bfloat16* input, __nv_bfloat16* output, int token_count,
    int out_features, int in_features) {
  INFER_CHECK(options_.backend == Device::kCuda &&
                  options_.precision == Precision::kW16A16,
              "batched BF16 linear requires CUDA W16A16");
  const auto* weight_pointer = bf16_weight(weight_name);
  const auto* bias = optional_bf16_weight(bias_name);
  if (options_.linear_kernel == LinearKernel::kCublas) {
    cuda::gemm_bf16_cublas(
        cuda_context_->cublas(), weight_pointer, bias, input, output,
        token_count, out_features, in_features, cuda_context_->stream());
  } else {
    cuda::gemm_bf16(weight_pointer, bias, input, output, token_count,
                    out_features, in_features, cuda_context_->stream());
  }
}

void Qwen2Model::linear_cpu_fp32_batch(
    std::string_view weight_name, std::string_view bias_name,
    const float* input, float* output, int token_count, int out_features,
    int in_features) {
  INFER_CHECK(options_.backend == Device::kCpu &&
                  options_.precision == Precision::kFloat32,
              "batched FP32 linear requires CPU FP32");
  INFER_CHECK(archive_.record(weight_name).dtype == DType::kFloat32,
              "batched FP32 linear requires float32 weights");
  cpu::linear_fp32_batch(float_weight(weight_name),
                         optional_float_weight(bias_name), input, output,
                         token_count, out_features, in_features);
}

void Qwen2Model::linear_cuda_batch(
    std::string_view weight_name, std::string_view bias_name,
    const float* input, float* output, int token_count, int out_features,
    int in_features) {
  INFER_CHECK(options_.backend == Device::kCuda,
              "batched CUDA linear requires CUDA backend");
  const auto& rec = archive_.record(weight_name);
  if (rec.dtype == DType::kFloat32) {
    if (options_.linear_kernel == LinearKernel::kCublas) {
      cuda::gemm_fp32_cublas(
          cuda_context_->cublas(),
          static_cast<const float*>(weight(weight_name)), input, output,
          token_count, out_features, in_features, cuda_context_->stream());
    } else {
      cuda::gemm_fp32(static_cast<const float*>(weight(weight_name)), input,
                      output, token_count, out_features, in_features,
                      cuda_context_->stream());
    }
  } else {
    INFER_CHECK(rec.quant.has_value(), "int8 tensor lacks quantization metadata");
    cuda::gemm_int8(
        static_cast<const int8_t*>(weight(weight_name)),
        float_weight(rec.quant->scale_tensor), rec.quant->group_size, input,
        output, token_count, out_features, in_features,
        cuda_context_->stream());
  }
  const float* bias = optional_float_weight(bias_name);
  if (bias) {
    cuda::add_bias_batch(output, bias, token_count, out_features,
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


__nv_bfloat16* Qwen2Model::bf16_layer_key_cache(int layer) {
  const size_t per_layer = static_cast<size_t>(config_.num_kv_heads) *
                           options_.max_sequence_length * config_.head_dim();
  return static_cast<__nv_bfloat16*>(kv_cache_.data()) +
         static_cast<size_t>(layer) * per_layer;
}

__nv_bfloat16* Qwen2Model::bf16_layer_value_cache(int layer) {
  const size_t per_layer = static_cast<size_t>(config_.num_kv_heads) *
                           options_.max_sequence_length * config_.head_dim();
  return static_cast<__nv_bfloat16*>(kv_cache_.data()) +
         static_cast<size_t>(config_.num_layers + layer) * per_layer;
}

int Qwen2Model::decode_token_cpu(int token, int position) {
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

int Qwen2Model::prefill_cpu_fp32(const std::vector<int>& prompt_tokens) {
  INFER_CHECK(options_.backend == Device::kCpu &&
                  options_.precision == Precision::kFloat32,
              "matrixized prefill currently supports CPU FP32 only");
  INFER_CHECK(prefill_workspace_.data() != nullptr,
              "matrixized prefill workspace was not allocated");
  const int token_count = static_cast<int>(prompt_tokens.size());
  const int hidden = config_.hidden_size;
  const int kv_dim = config_.kv_dim();
  const int head_dim = config_.head_dim();
  const int intermediate = config_.intermediate_size;

  cpu::embedding_batch(float_weight("model.embed_tokens.weight"),
                       prompt_tokens.data(), prefill_x_, token_count, hidden);
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    const auto prefix = layer_prefix(layer);
    cpu::rms_norm_batch(
        prefill_x_, float_weight(prefix + "input_layernorm.weight"),
        prefill_norm_, token_count, hidden, config_.rms_norm_eps);
    linear_cpu_fp32_batch(
        prefix + "self_attn.q_proj.weight", prefix + "self_attn.q_proj.bias",
        prefill_norm_, prefill_q_, token_count, hidden, hidden);
    linear_cpu_fp32_batch(
        prefix + "self_attn.k_proj.weight", prefix + "self_attn.k_proj.bias",
        prefill_norm_, prefill_k_, token_count, kv_dim, hidden);
    linear_cpu_fp32_batch(
        prefix + "self_attn.v_proj.weight", prefix + "self_attn.v_proj.bias",
        prefill_norm_, prefill_v_, token_count, kv_dim, hidden);
    cpu::rope_batch(prefill_q_, prefill_k_, token_count, config_.num_heads,
                    config_.num_kv_heads, head_dim, 0, config_.rope_theta);
    float* key_cache = layer_key_cache(layer);
    float* value_cache = layer_value_cache(layer);
    cpu::store_kv_batch(prefill_k_, prefill_v_, key_cache, value_cache,
                        token_count, config_.num_kv_heads, head_dim,
                        options_.max_sequence_length);
    cpu::attention_prefill(
        prefill_q_, key_cache, value_cache, prefill_attention_, token_count,
        config_.num_heads, config_.num_kv_heads, head_dim,
        options_.max_sequence_length, attention_scores_);
    linear_cpu_fp32_batch(prefix + "self_attn.o_proj.weight", {},
                          prefill_attention_, prefill_hidden_tmp_, token_count,
                          hidden, hidden);
    cpu::add_inplace(prefill_x_, prefill_hidden_tmp_, token_count * hidden);
    cpu::rms_norm_batch(
        prefill_x_, float_weight(prefix + "post_attention_layernorm.weight"),
        prefill_norm_, token_count, hidden, config_.rms_norm_eps);
    linear_cpu_fp32_batch(prefix + "mlp.gate_proj.weight", {}, prefill_norm_,
                          prefill_gate_, token_count, intermediate, hidden);
    linear_cpu_fp32_batch(prefix + "mlp.up_proj.weight", {}, prefill_norm_,
                          prefill_up_, token_count, intermediate, hidden);
    cpu::silu_mul(prefill_gate_, prefill_up_, prefill_mlp_,
                  token_count * intermediate);
    linear_cpu_fp32_batch(prefix + "mlp.down_proj.weight", {}, prefill_mlp_,
                          prefill_hidden_tmp_, token_count, hidden,
                          intermediate);
    cpu::add_inplace(prefill_x_, prefill_hidden_tmp_, token_count * hidden);
  }

  const float* last_hidden =
      prefill_x_ + static_cast<size_t>(token_count - 1) * hidden;
  cpu::rms_norm(last_hidden, float_weight("model.norm.weight"), norm_, hidden,
                config_.rms_norm_eps);
  const std::string lm_head = archive_.contains("lm_head.weight")
                                  ? "lm_head.weight"
                                  : "model.embed_tokens.weight";
  linear(lm_head, {}, norm_, logits_, config_.vocab_size, hidden);
  return cpu::argmax(logits_, config_.vocab_size);
}

int Qwen2Model::prefill_cuda_bf16(
    const std::vector<int>& prompt_tokens) {
  INFER_CHECK(options_.backend == Device::kCuda &&
                  options_.precision == Precision::kW16A16,
              "BF16 prefill requires CUDA W16A16");
  INFER_CHECK(prefill_workspace_.data() != nullptr &&
                  prefill_tokens_.data() != nullptr,
              "BF16 prefill workspace was not allocated");
  const int token_count = static_cast<int>(prompt_tokens.size());
  const int hidden = config_.hidden_size;
  const int kv_dim = config_.kv_dim();
  const int head_dim = config_.head_dim();
  const int intermediate = config_.intermediate_size;
  auto stream = cuda_context_->stream();

  INFER_CUDA_CHECK(cudaMemcpyAsync(
      prefill_tokens_.data(), prompt_tokens.data(),
      static_cast<size_t>(token_count) * sizeof(int), cudaMemcpyHostToDevice,
      stream));
  cuda::embedding_batch_bf16(
      bf16_weight("model.embed_tokens.weight"),
      static_cast<const int*>(prefill_tokens_.data()), bf16_prefill_x_,
      token_count, hidden, stream);
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    const auto prefix = layer_prefix(layer);
    cuda::rms_norm_batch_bf16(
        bf16_prefill_x_,
        bf16_weight(prefix + "input_layernorm.weight"),
        bf16_prefill_norm_, token_count, hidden, config_.rms_norm_eps, stream);
    linear_cuda_bf16_batch(
        prefix + "self_attn.q_proj.weight",
        prefix + "self_attn.q_proj.bias", bf16_prefill_norm_,
        bf16_prefill_q_, token_count, hidden, hidden);
    linear_cuda_bf16_batch(
        prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.k_proj.bias", bf16_prefill_norm_,
        bf16_prefill_k_, token_count, kv_dim, hidden);
    linear_cuda_bf16_batch(
        prefix + "self_attn.v_proj.weight",
        prefix + "self_attn.v_proj.bias", bf16_prefill_norm_,
        bf16_prefill_v_, token_count, kv_dim, hidden);
    cuda::rope_batch_bf16(
        bf16_prefill_q_, bf16_prefill_k_, token_count, config_.num_heads,
        config_.num_kv_heads, head_dim, 0, config_.rope_theta, stream);
    auto* key_cache = bf16_layer_key_cache(layer);
    auto* value_cache = bf16_layer_value_cache(layer);
    cuda::store_kv_batch_bf16(
        bf16_prefill_k_, bf16_prefill_v_, key_cache, value_cache, token_count,
        config_.num_kv_heads, head_dim, options_.max_sequence_length, stream);
    cuda::attention_prefill_bf16(
        bf16_prefill_q_, key_cache, value_cache, bf16_prefill_attention_,
        token_count, config_.num_heads, config_.num_kv_heads, head_dim,
        options_.max_sequence_length, stream);
    linear_cuda_bf16_batch(
        prefix + "self_attn.o_proj.weight", {}, bf16_prefill_attention_,
        bf16_prefill_hidden_tmp_, token_count, hidden, hidden);
    cuda::add_inplace_bf16(
        bf16_prefill_x_, bf16_prefill_hidden_tmp_, token_count * hidden,
        stream);
    cuda::rms_norm_batch_bf16(
        bf16_prefill_x_,
        bf16_weight(prefix + "post_attention_layernorm.weight"),
        bf16_prefill_norm_, token_count, hidden, config_.rms_norm_eps, stream);
    linear_cuda_bf16_batch(
        prefix + "mlp.gate_proj.weight", {}, bf16_prefill_norm_,
        bf16_prefill_gate_, token_count, intermediate, hidden);
    linear_cuda_bf16_batch(
        prefix + "mlp.up_proj.weight", {}, bf16_prefill_norm_,
        bf16_prefill_up_, token_count, intermediate, hidden);
    cuda::silu_mul_bf16(
        bf16_prefill_gate_, bf16_prefill_up_, bf16_prefill_mlp_,
        token_count * intermediate, stream);
    linear_cuda_bf16_batch(
        prefix + "mlp.down_proj.weight", {}, bf16_prefill_mlp_,
        bf16_prefill_hidden_tmp_, token_count, hidden, intermediate);
    cuda::add_inplace_bf16(
        bf16_prefill_x_, bf16_prefill_hidden_tmp_, token_count * hidden,
        stream);
  }

  const auto* last_hidden =
      bf16_prefill_x_ + static_cast<size_t>(token_count - 1) * hidden;
  cuda::rms_norm_bf16(
      last_hidden, bf16_weight("model.norm.weight"), bf16_norm_, hidden,
      config_.rms_norm_eps, stream);
  const std::string lm_head = archive_.contains("lm_head.weight")
                                  ? "lm_head.weight"
                                  : "model.embed_tokens.weight";
  linear_cuda_bf16(
      lm_head, {}, bf16_norm_, bf16_logits_, config_.vocab_size, hidden);
  cuda::argmax_bf16(
      bf16_logits_, config_.vocab_size,
      static_cast<int*>(argmax_buffer_.data()), stream);
  int result = 0;
  INFER_CUDA_CHECK(cudaMemcpyAsync(
      &result, argmax_buffer_.data(), sizeof(int), cudaMemcpyDeviceToHost,
      stream));
  cuda_context_->synchronize();
  return result;
}

int Qwen2Model::prefill_cuda(const std::vector<int>& prompt_tokens) {
  if (options_.precision == Precision::kW16A16) {
    return prefill_cuda_bf16(prompt_tokens);
  }
  INFER_CHECK(options_.backend == Device::kCuda,
              "CUDA matrixized prefill requires CUDA backend");
  INFER_CHECK(prefill_workspace_.data() != nullptr &&
                  prefill_tokens_.data() != nullptr,
              "CUDA matrixized prefill workspace was not allocated");
  const int token_count = static_cast<int>(prompt_tokens.size());
  const int hidden = config_.hidden_size;
  const int kv_dim = config_.kv_dim();
  const int head_dim = config_.head_dim();
  const int intermediate = config_.intermediate_size;
  auto stream = cuda_context_->stream();

  INFER_CUDA_CHECK(cudaMemcpyAsync(
      prefill_tokens_.data(), prompt_tokens.data(),
      static_cast<size_t>(token_count) * sizeof(int), cudaMemcpyHostToDevice,
      stream));
  cuda::embedding_batch(
      float_weight("model.embed_tokens.weight"),
      static_cast<const int*>(prefill_tokens_.data()), prefill_x_, token_count,
      hidden, stream);
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    const auto prefix = layer_prefix(layer);
    cuda::rms_norm_batch(
        prefill_x_, float_weight(prefix + "input_layernorm.weight"),
        prefill_norm_, token_count, hidden, config_.rms_norm_eps, stream);
    linear_cuda_batch(
        prefix + "self_attn.q_proj.weight", prefix + "self_attn.q_proj.bias",
        prefill_norm_, prefill_q_, token_count, hidden, hidden);
    linear_cuda_batch(
        prefix + "self_attn.k_proj.weight", prefix + "self_attn.k_proj.bias",
        prefill_norm_, prefill_k_, token_count, kv_dim, hidden);
    linear_cuda_batch(
        prefix + "self_attn.v_proj.weight", prefix + "self_attn.v_proj.bias",
        prefill_norm_, prefill_v_, token_count, kv_dim, hidden);
    cuda::rope_batch(
        prefill_q_, prefill_k_, token_count, config_.num_heads,
        config_.num_kv_heads, head_dim, 0, config_.rope_theta, stream);
    float* key_cache = layer_key_cache(layer);
    float* value_cache = layer_value_cache(layer);
    cuda::store_kv_batch(
        prefill_k_, prefill_v_, key_cache, value_cache, token_count,
        config_.num_kv_heads, head_dim, options_.max_sequence_length, stream);
    cuda::attention_prefill(
        prefill_q_, key_cache, value_cache, prefill_attention_, token_count,
        config_.num_heads, config_.num_kv_heads, head_dim,
        options_.max_sequence_length, stream);
    linear_cuda_batch(prefix + "self_attn.o_proj.weight", {},
                      prefill_attention_, prefill_hidden_tmp_, token_count,
                      hidden, hidden);
    cuda::add_inplace(prefill_x_, prefill_hidden_tmp_, token_count * hidden,
                      stream);
    cuda::rms_norm_batch(
        prefill_x_, float_weight(prefix + "post_attention_layernorm.weight"),
        prefill_norm_, token_count, hidden, config_.rms_norm_eps, stream);
    linear_cuda_batch(prefix + "mlp.gate_proj.weight", {}, prefill_norm_,
                      prefill_gate_, token_count, intermediate, hidden);
    linear_cuda_batch(prefix + "mlp.up_proj.weight", {}, prefill_norm_,
                      prefill_up_, token_count, intermediate, hidden);
    cuda::silu_mul(prefill_gate_, prefill_up_, prefill_mlp_,
                   token_count * intermediate, stream);
    linear_cuda_batch(prefix + "mlp.down_proj.weight", {}, prefill_mlp_,
                      prefill_hidden_tmp_, token_count, hidden, intermediate);
    cuda::add_inplace(prefill_x_, prefill_hidden_tmp_, token_count * hidden,
                      stream);
  }

  const float* last_hidden =
      prefill_x_ + static_cast<size_t>(token_count - 1) * hidden;
  cuda::rms_norm(last_hidden, float_weight("model.norm.weight"), norm_, hidden,
                 config_.rms_norm_eps, stream);
  const std::string lm_head = archive_.contains("lm_head.weight")
                                  ? "lm_head.weight"
                                  : "model.embed_tokens.weight";
  linear(lm_head, {}, norm_, logits_, config_.vocab_size, hidden, true);
  cuda::argmax(logits_, config_.vocab_size,
               static_cast<int*>(argmax_buffer_.data()), stream);
  int result = 0;
  INFER_CUDA_CHECK(cudaMemcpyAsync(&result, argmax_buffer_.data(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
  cuda_context_->synchronize();
  return result;
}

int Qwen2Model::decode_token_cuda_bf16(int token, int position) {
  INFER_CHECK(options_.backend == Device::kCuda &&
                  options_.precision == Precision::kW16A16,
              "BF16 decode requires CUDA W16A16");
  const int hidden = config_.hidden_size;
  const int kv_dim = config_.kv_dim();
  const int head_dim = config_.head_dim();
  auto stream = cuda_context_->stream();

  cuda::embedding_bf16(
      bf16_weight("model.embed_tokens.weight"), token, bf16_x_, hidden,
      stream);
  for (int layer = 0; layer < config_.num_layers; ++layer) {
    const auto prefix = layer_prefix(layer);
    cuda::rms_norm_bf16(
        bf16_x_, bf16_weight(prefix + "input_layernorm.weight"), bf16_norm_,
        hidden, config_.rms_norm_eps, stream);
    linear_cuda_bf16(
        prefix + "self_attn.q_proj.weight",
        prefix + "self_attn.q_proj.bias", bf16_norm_, bf16_q_, hidden,
        hidden);
    linear_cuda_bf16(
        prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.k_proj.bias", bf16_norm_, bf16_k_, kv_dim,
        hidden);
    linear_cuda_bf16(
        prefix + "self_attn.v_proj.weight",
        prefix + "self_attn.v_proj.bias", bf16_norm_, bf16_v_, kv_dim,
        hidden);
    cuda::rope_bf16(
        bf16_q_, bf16_k_, config_.num_heads, config_.num_kv_heads, head_dim,
        position, config_.rope_theta, stream);
    auto* key_cache = bf16_layer_key_cache(layer);
    auto* value_cache = bf16_layer_value_cache(layer);
    cuda::store_kv_bf16(
        bf16_k_, bf16_v_, key_cache, value_cache, config_.num_kv_heads,
        head_dim, position, options_.max_sequence_length, stream);
    cuda::attention_decode_bf16(
        bf16_q_, key_cache, value_cache, bf16_attention_, config_.num_heads,
        config_.num_kv_heads, head_dim, position + 1,
        options_.max_sequence_length, stream);
    linear_cuda_bf16(
        prefix + "self_attn.o_proj.weight", {}, bf16_attention_,
        bf16_hidden_tmp_, hidden, hidden);
    cuda::add_inplace_bf16(bf16_x_, bf16_hidden_tmp_, hidden, stream);
    cuda::rms_norm_bf16(
        bf16_x_, bf16_weight(prefix + "post_attention_layernorm.weight"),
        bf16_norm_, hidden, config_.rms_norm_eps, stream);
    linear_cuda_bf16(
        prefix + "mlp.gate_proj.weight", {}, bf16_norm_, bf16_gate_,
        config_.intermediate_size, hidden);
    linear_cuda_bf16(
        prefix + "mlp.up_proj.weight", {}, bf16_norm_, bf16_up_,
        config_.intermediate_size, hidden);
    cuda::silu_mul_bf16(
        bf16_gate_, bf16_up_, bf16_mlp_, config_.intermediate_size, stream);
    linear_cuda_bf16(
        prefix + "mlp.down_proj.weight", {}, bf16_mlp_, bf16_hidden_tmp_,
        hidden, config_.intermediate_size);
    cuda::add_inplace_bf16(bf16_x_, bf16_hidden_tmp_, hidden, stream);
  }

  cuda::rms_norm_bf16(
      bf16_x_, bf16_weight("model.norm.weight"), bf16_norm_, hidden,
      config_.rms_norm_eps, stream);
  const std::string lm_head = archive_.contains("lm_head.weight")
                                  ? "lm_head.weight"
                                  : "model.embed_tokens.weight";
  linear_cuda_bf16(
      lm_head, {}, bf16_norm_, bf16_logits_, config_.vocab_size, hidden);
  cuda::argmax_bf16(
      bf16_logits_, config_.vocab_size,
      static_cast<int*>(argmax_buffer_.data()), stream);
  int result = 0;
  INFER_CUDA_CHECK(cudaMemcpyAsync(
      &result, argmax_buffer_.data(), sizeof(int), cudaMemcpyDeviceToHost,
      stream));
  cuda_context_->synchronize();
  return result;
}

int Qwen2Model::decode_token_cuda(int token, int position) {
  if (options_.precision == Precision::kW16A16) {
    return decode_token_cuda_bf16(token, position);
  }
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

int Qwen2Model::decode_token(int token, int position) {
  INFER_CHECK(token >= 0 && token < config_.vocab_size, "input token out of range");
  INFER_CHECK(position >= 0 && position < options_.max_sequence_length,
              "KV cache capacity exceeded");
  ProfileRange range("qwen2.decode_token");
  return options_.backend == Device::kCpu ? decode_token_cpu(token, position)
                                          : decode_token_cuda(token, position);
}

std::vector<float> Qwen2Model::last_logits_host() const {
  std::vector<float> result(static_cast<size_t>(config_.vocab_size));
  if (options_.backend == Device::kCpu) {
    std::copy(logits_, logits_ + config_.vocab_size, result.begin());
  } else if (options_.precision == Precision::kW16A16) {
    std::vector<uint16_t> bits(result.size());
    INFER_CUDA_CHECK(cudaMemcpyAsync(
        bits.data(), bf16_logits_, bits.size() * sizeof(uint16_t),
        cudaMemcpyDeviceToHost, cuda_context_->stream()));
    cuda_context_->synchronize();
    for (size_t index = 0; index < bits.size(); ++index) {
      const uint32_t fp32_bits = static_cast<uint32_t>(bits[index]) << 16u;
      std::memcpy(&result[index], &fp32_bits, sizeof(float));
    }
  } else {
    INFER_CUDA_CHECK(cudaMemcpyAsync(
        result.data(), logits_, result.size() * sizeof(float),
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

int Qwen2Model::prefill(const std::vector<int>& prompt_tokens) {
  INFER_CHECK(!prompt_tokens.empty(), "prompt must contain at least one token");
  INFER_CHECK(position_ == 0 && !has_prefilled_,
              "prefill requires a fresh model state; call reset first");
  INFER_CHECK(prompt_tokens.size() <= static_cast<size_t>(options_.max_sequence_length),
              "prompt exceeds KV cache capacity");
  for (const int token : prompt_tokens) {
    INFER_CHECK(token >= 0 && token < config_.vocab_size, "input token out of range");
  }

  ProfileRange range("qwen2.prefill");
  ProfileRange implementation("qwen2.prefill.matrixized");
  const int next = options_.backend == Device::kCpu
                       ? prefill_cpu_fp32(prompt_tokens)
                       : prefill_cuda(prompt_tokens);
  position_ = static_cast<int>(prompt_tokens.size());
  has_prefilled_ = true;
  return next;
}

int Qwen2Model::decode_next(int token) {
  INFER_CHECK(has_prefilled_, "decode_next requires a successful prefill");
  INFER_CHECK(position_ < options_.max_sequence_length, "KV cache capacity exceeded");
  ProfileRange range("qwen2.decode_next");
  const int next = decode_token(token, position_);
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
