#include "infer/model_config.hpp"

#include <nlohmann/json.hpp>

namespace infer {

int ModelConfig::head_dim() const {
  INFER_CHECK(num_heads > 0 && hidden_size % num_heads == 0,
              "hidden_size must be divisible by num_attention_heads");
  return hidden_size / num_heads;
}

int ModelConfig::kv_dim() const { return num_kv_heads * head_dim(); }

ModelConfig ModelConfig::load(const std::filesystem::path& path) {
  std::ifstream input(path);
  INFER_CHECK(input.good(), "cannot open model config: " + path.string());
  nlohmann::json j;
  input >> j;
  INFER_CHECK(j.value("model_type", std::string{}) == "qwen2",
              "only model_type=qwen2 is supported");
  ModelConfig c;
  c.hidden_size = j.at("hidden_size").get<int>();
  c.intermediate_size = j.at("intermediate_size").get<int>();
  c.num_layers = j.at("num_hidden_layers").get<int>();
  c.num_heads = j.at("num_attention_heads").get<int>();
  c.num_kv_heads = j.at("num_key_value_heads").get<int>();
  c.vocab_size = j.at("vocab_size").get<int>();
  c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
  c.bos_token_id = j.value("bos_token_id", -1);
  c.eos_token_id = j.value("eos_token_id", -1);
  c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
  c.rope_theta = j.value("rope_theta", 1e6f);
  c.tie_word_embeddings = j.value("tie_word_embeddings", true);
  c.validate();
  return c;
}

void ModelConfig::validate() const {
  INFER_CHECK(hidden_size > 0 && intermediate_size > 0 && num_layers > 0,
              "invalid model dimensions");
  INFER_CHECK(num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0,
              "invalid GQA head counts");
  INFER_CHECK(hidden_size % num_heads == 0, "invalid attention head dimension");
  INFER_CHECK(vocab_size > 0, "invalid vocabulary size");
}

}  // namespace infer
