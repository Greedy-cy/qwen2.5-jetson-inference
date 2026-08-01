#pragma once

#include "infer/common.hpp"

namespace infer {

class QwenTokenizer {
 public:
  struct Codepoint {
    uint32_t value;
    std::string bytes;
  };

  explicit QwenTokenizer(const std::filesystem::path& tokenizer_json);

  std::vector<int> encode(std::string_view text, bool allow_special = true) const;
  std::string decode(const std::vector<int>& ids, bool skip_special = false) const;
  std::string apply_chat_template(std::string_view user_prompt,
                                  std::string_view system_prompt =
                                      "You are a helpful assistant.") const;
  int token_to_id(std::string_view token) const;
  size_t vocab_size() const { return id_to_token_.size(); }

 private:
  static std::vector<Codepoint> utf8_split(std::string_view text);
  static std::string utf8_encode(uint32_t cp);
  static bool is_letter(uint32_t cp);
  static bool is_number(uint32_t cp);
  static bool is_space(uint32_t cp);
  static bool is_newline(uint32_t cp);

  std::vector<std::string> pretokenize(std::string_view text) const;
  std::vector<int> encode_regular(std::string_view text) const;
  std::vector<std::string> bpe(std::string token) const;
  std::string byte_encode(std::string_view text) const;
  std::string byte_decode(std::string_view text) const;
  static std::string pair_key(std::string_view a, std::string_view b);

  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_rank_;
  std::unordered_map<std::string, int> special_to_id_;
  std::unordered_map<int, std::string> id_to_special_;
  std::array<std::string, 256> byte_encoder_{};
  std::unordered_map<uint32_t, uint8_t> byte_decoder_;
};

}  // namespace infer
