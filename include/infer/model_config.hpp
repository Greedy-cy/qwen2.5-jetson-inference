#pragma once

#include "infer/common.hpp"

namespace infer {

struct ModelConfig {
  int hidden_size{0};
  int intermediate_size{0};
  int num_layers{0};
  int num_heads{0};
  int num_kv_heads{0};
  int vocab_size{0};
  int max_position_embeddings{0};
  int bos_token_id{-1};
  int eos_token_id{-1};
  float rms_norm_eps{1e-6f};
  float rope_theta{1e6f};
  bool tie_word_embeddings{true};

  int head_dim() const;
  int kv_dim() const;
  static ModelConfig load(const std::filesystem::path& path);
  void validate() const;
};

}  // namespace infer
