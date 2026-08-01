#include "infer/tokenizer.hpp"

#include <array>
#include <cctype>
#include <nlohmann/json.hpp>

namespace infer {
namespace {

bool starts_with_case_insensitive(const std::vector<QwenTokenizer::Codepoint>& cps,
                                  size_t position, std::string_view suffix) {
  if (position + suffix.size() > cps.size()) return false;
  for (size_t i = 0; i < suffix.size(); ++i) {
    if (cps[position + i].value > 127) return false;
    const auto got = static_cast<char>(std::tolower(static_cast<unsigned char>(cps[position + i].value)));
    if (got != suffix[i]) return false;
  }
  return true;
}

}  // namespace

std::string QwenTokenizer::utf8_encode(uint32_t cp) {
  std::string out;
  if (cp <= 0x7f) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7ff) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp <= 0xffff) {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
  return out;
}

std::vector<QwenTokenizer::Codepoint> QwenTokenizer::utf8_split(std::string_view text) {
  std::vector<Codepoint> result;
  for (size_t i = 0; i < text.size();) {
    const auto first = static_cast<uint8_t>(text[i]);
    size_t length = 1;
    uint32_t cp = first;
    if ((first & 0xe0) == 0xc0) { length = 2; cp = first & 0x1f; }
    else if ((first & 0xf0) == 0xe0) { length = 3; cp = first & 0x0f; }
    else if ((first & 0xf8) == 0xf0) { length = 4; cp = first & 0x07; }
    if (i + length > text.size()) { length = 1; cp = 0xfffd; }
    bool valid = true;
    for (size_t j = 1; j < length; ++j) {
      const auto b = static_cast<uint8_t>(text[i + j]);
      if ((b & 0xc0) != 0x80) { valid = false; break; }
      cp = (cp << 6) | (b & 0x3f);
    }
    if (!valid) { length = 1; cp = 0xfffd; }
    result.push_back({cp, std::string(text.substr(i, length))});
    i += length;
  }
  return result;
}

QwenTokenizer::QwenTokenizer(const std::filesystem::path& tokenizer_json) {
  std::ifstream input(tokenizer_json);
  INFER_CHECK(input.good(), "cannot open tokenizer: " + tokenizer_json.string());
  nlohmann::json root;
  input >> root;

  std::vector<int> base_bytes;
  for (int i = 33; i <= 126; ++i) base_bytes.push_back(i);
  for (int i = 161; i <= 172; ++i) base_bytes.push_back(i);
  for (int i = 174; i <= 255; ++i) base_bytes.push_back(i);
  std::vector<int> codepoints = base_bytes;
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(base_bytes.begin(), base_bytes.end(), b) == base_bytes.end()) {
      base_bytes.push_back(b);
      codepoints.push_back(256 + extra++);
    }
  }
  for (size_t i = 0; i < base_bytes.size(); ++i) {
    byte_encoder_[base_bytes[i]] = utf8_encode(static_cast<uint32_t>(codepoints[i]));
    byte_decoder_[static_cast<uint32_t>(codepoints[i])] = static_cast<uint8_t>(base_bytes[i]);
  }

  int max_id = -1;
  for (auto it = root.at("model").at("vocab").begin();
       it != root.at("model").at("vocab").end(); ++it) {
    const int id = it.value().get<int>();
    vocab_[it.key()] = id;
    max_id = std::max(max_id, id);
  }
  for (const auto& item : root.value("added_tokens", nlohmann::json::array())) {
    const int id = item.at("id").get<int>();
    const std::string content = item.at("content").get<std::string>();
    vocab_[content] = id;
    special_to_id_[content] = id;
    id_to_special_[id] = content;
    max_id = std::max(max_id, id);
  }
  id_to_token_.resize(static_cast<size_t>(max_id + 1));
  for (const auto& [token, id] : vocab_) id_to_token_.at(static_cast<size_t>(id)) = token;

  int rank = 0;
  for (const auto& merge : root.at("model").at("merges")) {
    std::string left;
    std::string right;
    if (merge.is_array()) {
      left = merge.at(0).get<std::string>();
      right = merge.at(1).get<std::string>();
    } else {
      const auto line = merge.get<std::string>();
      const auto split = line.find(' ');
      INFER_CHECK(split != std::string::npos, "invalid tokenizer merge");
      left = line.substr(0, split);
      right = line.substr(split + 1);
    }
    merge_rank_.emplace(pair_key(left, right), rank++);
  }
}

bool QwenTokenizer::is_letter(uint32_t cp) {
  if (cp < 128) return std::isalpha(static_cast<unsigned char>(cp)) != 0;
  if ((cp >= 0x3000 && cp <= 0x303f) || (cp >= 0x1f000 && cp <= 0x1faff)) return false;
  return !is_space(cp) && !is_number(cp);
}

bool QwenTokenizer::is_number(uint32_t cp) {
  return (cp >= '0' && cp <= '9') || (cp >= 0xff10 && cp <= 0xff19);
}

bool QwenTokenizer::is_space(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n' || cp == '\f' || cp == '\v' ||
         cp == 0x00a0 || cp == 0x3000;
}

bool QwenTokenizer::is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

std::vector<std::string> QwenTokenizer::pretokenize(std::string_view text) const {
  const auto cps = utf8_split(text);
  std::vector<std::string> pieces;
  size_t i = 0;
  const std::array<std::string_view, 7> contractions{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  while (i < cps.size()) {
    size_t begin = i;
    bool matched = false;
    if (cps[i].value == '\'') {
      for (const auto suffix : contractions) {
        if (starts_with_case_insensitive(cps, i, suffix)) {
          i += suffix.size();
          matched = true;
          break;
        }
      }
    }
    if (!matched && is_number(cps[i].value)) {
      i += 1;
      while (i < cps.size() && i - begin < 3 && is_number(cps[i].value)) ++i;
      matched = true;
    }
    if (!matched) {
      const bool prefix_for_letter = cps[i].value == ' ' && i + 1 < cps.size() && is_letter(cps[i + 1].value);
      if (is_letter(cps[i].value) || prefix_for_letter) {
        if (prefix_for_letter) ++i;
        while (i < cps.size() && is_letter(cps[i].value)) ++i;
        matched = true;
      }
    }
    if (!matched) {
      const bool prefix_for_symbol = cps[i].value == ' ' && i + 1 < cps.size() &&
          !is_space(cps[i + 1].value) && !is_letter(cps[i + 1].value) && !is_number(cps[i + 1].value);
      const bool symbol = !is_space(cps[i].value) && !is_letter(cps[i].value) && !is_number(cps[i].value);
      if (symbol || prefix_for_symbol) {
        if (prefix_for_symbol) ++i;
        while (i < cps.size() && !is_space(cps[i].value) &&
               !is_letter(cps[i].value) && !is_number(cps[i].value)) ++i;
        while (i < cps.size() && is_newline(cps[i].value)) ++i;
        matched = true;
      }
    }
    if (!matched && is_space(cps[i].value)) {
      const size_t whitespace_begin = i;
      while (i < cps.size() && is_space(cps[i].value)) ++i;
      // The Qwen regex uses `\\s+(?!\\S)` before the final `\\s+` branch.
      // When a whitespace run is followed by text it backtracks one codepoint,
      // allowing that last space to become the optional prefix of the next word.
      if (i < cps.size() && i - whitespace_begin > 1 &&
          !is_newline(cps[i - 1].value)) {
        --i;
      }
      matched = true;
    }
    if (!matched) ++i;
    std::string piece;
    for (size_t j = begin; j < i; ++j) piece += cps[j].bytes;
    pieces.push_back(std::move(piece));
  }
  return pieces;
}

std::string QwenTokenizer::byte_encode(std::string_view text) const {
  std::string result;
  for (const unsigned char b : text) result += byte_encoder_[b];
  return result;
}

std::string QwenTokenizer::byte_decode(std::string_view text) const {
  std::string result;
  for (const auto& cp : utf8_split(text)) {
    const auto it = byte_decoder_.find(cp.value);
    if (it != byte_decoder_.end()) result.push_back(static_cast<char>(it->second));
    else result += cp.bytes;
  }
  return result;
}

std::string QwenTokenizer::pair_key(std::string_view a, std::string_view b) {
  std::string key(a);
  key.push_back('\x1f');
  key.append(b);
  return key;
}

std::vector<std::string> QwenTokenizer::bpe(std::string token) const {
  std::vector<std::string> words;
  for (const auto& cp : utf8_split(token)) words.push_back(cp.bytes);
  if (words.size() < 2) return words;
  while (true) {
    int best_rank = std::numeric_limits<int>::max();
    std::string best_left;
    std::string best_right;
    for (size_t i = 0; i + 1 < words.size(); ++i) {
      const auto it = merge_rank_.find(pair_key(words[i], words[i + 1]));
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_left = words[i];
        best_right = words[i + 1];
      }
    }
    if (best_rank == std::numeric_limits<int>::max()) break;
    std::vector<std::string> merged;
    for (size_t i = 0; i < words.size();) {
      if (i + 1 < words.size() && words[i] == best_left && words[i + 1] == best_right) {
        merged.push_back(words[i] + words[i + 1]);
        i += 2;
      } else {
        merged.push_back(words[i++]);
      }
    }
    words.swap(merged);
  }
  return words;
}

std::vector<int> QwenTokenizer::encode_regular(std::string_view text) const {
  std::vector<int> ids;
  for (const auto& piece : pretokenize(text)) {
    for (const auto& token : bpe(byte_encode(piece))) {
      const auto it = vocab_.find(token);
      INFER_CHECK(it != vocab_.end(), "tokenizer produced token absent from vocabulary");
      ids.push_back(it->second);
    }
  }
  return ids;
}

std::vector<int> QwenTokenizer::encode(std::string_view text, bool allow_special) const {
  if (!allow_special || special_to_id_.empty()) return encode_regular(text);
  std::vector<int> result;
  size_t cursor = 0;
  while (cursor < text.size()) {
    size_t next = std::string_view::npos;
    const std::pair<const std::string, int>* found = nullptr;
    for (const auto& item : special_to_id_) {
      const auto pos = text.find(item.first, cursor);
      if (pos != std::string_view::npos && (next == std::string_view::npos || pos < next)) {
        next = pos;
        found = &item;
      }
    }
    if (!found) {
      auto tail = encode_regular(text.substr(cursor));
      result.insert(result.end(), tail.begin(), tail.end());
      break;
    }
    if (next > cursor) {
      auto prefix = encode_regular(text.substr(cursor, next - cursor));
      result.insert(result.end(), prefix.begin(), prefix.end());
    }
    result.push_back(found->second);
    cursor = next + found->first.size();
  }
  return result;
}

std::string QwenTokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string encoded;
  for (const int id : ids) {
    const auto special = id_to_special_.find(id);
    if (special != id_to_special_.end()) {
      if (!skip_special) encoded += special->second;
      continue;
    }
    INFER_CHECK(id >= 0 && static_cast<size_t>(id) < id_to_token_.size(), "token id out of range");
    encoded += byte_decode(id_to_token_[static_cast<size_t>(id)]);
  }
  return encoded;
}

std::string QwenTokenizer::apply_chat_template(std::string_view user_prompt,
                                               std::string_view system_prompt) const {
  std::string text;
  text += "<|im_start|>system\n";
  text += system_prompt;
  text += "<|im_end|>\n<|im_start|>user\n";
  text += user_prompt;
  text += "<|im_end|>\n<|im_start|>assistant\n";
  return text;
}

int QwenTokenizer::token_to_id(std::string_view token) const {
  const auto it = vocab_.find(std::string(token));
  return it == vocab_.end() ? -1 : it->second;
}

}  // namespace infer
