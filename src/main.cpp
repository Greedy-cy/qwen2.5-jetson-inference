#include "infer/qwen2.hpp"
#include "infer/tokenizer.hpp"

#include <cstdlib>
#include <iomanip>
#include <nlohmann/json.hpp>

#include <sys/resource.h>

namespace {

using infer::Device;
using infer::LinearKernel;
using infer::Precision;

class Arguments {
 public:
  Arguments(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) values_.emplace_back(argv[i]);
  }
  std::string command() const { return values_.empty() ? "help" : values_[0]; }
  bool has(std::string_view key) const {
    return std::find(values_.begin(), values_.end(), key) != values_.end();
  }
  std::string get(std::string_view key, std::string fallback = {}) const {
    for (size_t i = 0; i + 1 < values_.size(); ++i) {
      if (values_[i] == key) return values_[i + 1];
    }
    return fallback;
  }
  int get_int(std::string_view key, int fallback) const {
    const auto text = get(key);
    return text.empty() ? fallback : std::stoi(text);
  }
 private:
  std::vector<std::string> values_;
};

void usage() {
  std::cout <<
      "Qwen2.5 C++/CUDA inference\n\n"
      "Usage:\n"
      "  llm_infer generate --model DIR (--prompt TEXT | --token-ids CSV) [--backend cpu|cuda]\n"
      "                     [--precision fp32|w16a16|w8a16] [--max-new-tokens 128]\n"
      "                     [--max-seq-len 2048] [--system TEXT] [--raw]\n"
      "                     [--linear-kernel custom|cublas]\n"
      "  llm_infer benchmark --model DIR (--prompt TEXT | --token-ids CSV)\n"
      "                      [--warmup 5] [--repeat 20] [--json FILE]\n"
      "                      [--telemetry-markers] [other generation options]\n";
}

Device parse_device(const std::string& text) {
  if (text == "cpu") return Device::kCpu;
  if (text == "cuda") return Device::kCuda;
  throw infer::Error("backend must be cpu or cuda");
}

LinearKernel parse_linear_kernel(const std::string& text) {
  if (text == "custom") return LinearKernel::kCustom;
  if (text == "cublas") return LinearKernel::kCublas;
  throw infer::Error("linear kernel must be custom or cublas");
}

Precision parse_precision(const std::string& text) {
  if (text == "fp32") return Precision::kFloat32;
  if (text == "w16a16") return Precision::kW16A16;
  if (text == "w8a16") return Precision::kW8A16;
  throw infer::Error("precision must be fp32, w16a16, or w8a16");
}

infer::RuntimeOptions runtime_options(const Arguments& args) {
  infer::RuntimeOptions options;
  options.backend = parse_device(args.get("--backend", "cuda"));
  options.precision = parse_precision(args.get("--precision", "fp32"));
  options.max_sequence_length = args.get_int("--max-seq-len", 2048);
  const std::string linear_kernel = args.get("--linear-kernel");
  if (!linear_kernel.empty()) {
    options.linear_kernel = parse_linear_kernel(linear_kernel);
  } else if (options.precision == Precision::kW16A16) {
    options.linear_kernel = LinearKernel::kCustom;
  }
  return options;
}

std::vector<int> make_prompt(const Arguments& args, const infer::QwenTokenizer& tokenizer) {
  const std::string prompt = args.get("--prompt");
  if (prompt.empty()) throw infer::Error("--prompt is required");
  const std::string formatted = args.has("--raw")
      ? prompt
      : tokenizer.apply_chat_template(prompt,
            args.get("--system", "You are a helpful assistant."));
  return tokenizer.encode(formatted, true);
}

std::vector<int> parse_token_ids(const std::string& text) {
  std::vector<int> ids;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) ids.push_back(std::stoi(item));
  }
  if (ids.empty()) throw infer::Error("--token-ids must contain at least one id");
  return ids;
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(std::ceil(p * values.size())) - 1;
  return values[std::min(index, values.size() - 1)];
}

nlohmann::json summarize(const std::vector<double>& values) {
  INFER_CHECK(!values.empty(), "cannot summarize empty samples");
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double variance = 0.0;
  for (const double value : values) {
    const double delta = value - mean;
    variance += delta * delta;
  }
  variance /= values.size();
  const double stddev = std::sqrt(variance);
  return {
      {"mean", mean},
      {"p05", percentile(values, 0.05)},
      {"p50", percentile(values, 0.50)},
      {"p95", percentile(values, 0.95)},
      {"min", *std::min_element(values.begin(), values.end())},
      {"max", *std::max_element(values.begin(), values.end())},
      {"stddev", stddev},
      {"cv_percent", mean != 0.0 ? stddev * 100.0 / std::abs(mean) : 0.0},
  };
}

long peak_rss_kib() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) return usage.ru_maxrss;
  return 0;
}

nlohmann::json memory_size(size_t bytes) {
  constexpr double kBytesPerMib = 1024.0 * 1024.0;
  const double mib = static_cast<double>(bytes) / kBytesPerMib;
  const double rounded_mib = std::round(mib * 100.0) / 100.0;
  return {{"bytes", bytes}, {"mib", rounded_mib}};
}

std::string environment_value(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' ? value : "library_default";
}



void generate(const Arguments& args) {
  const std::filesystem::path directory = args.get("--model");
  if (directory.empty()) throw infer::Error("--model is required");
  infer::QwenTokenizer tokenizer(directory / "tokenizer.json");
  const auto token_ids = args.get("--token-ids");
  if (!token_ids.empty() && args.has("--prompt")) {
    throw infer::Error("--prompt and --token-ids are mutually exclusive");
  }
  const auto prompt = token_ids.empty() ? make_prompt(args, tokenizer)
                                        : parse_token_ids(token_ids);
  infer::Qwen2Model model(directory, runtime_options(args));
  const auto result = model.generate(prompt, args.get_int("--max-new-tokens", 128));
  std::cout << tokenizer.decode(result.tokens, true) << "\n\n"
            << "[prompt=" << result.stats.prompt_tokens
            << ", generated=" << result.stats.generated_tokens
            << ", ttft_ms=" << std::fixed << std::setprecision(2) << result.stats.ttft_ms
            << ", decode_tok/s=" << result.stats.decode_tokens_per_second()
            << ", total_ms=" << result.stats.total_ms << "]\n";
  if (args.has("--show-token-ids")) {
    std::cout << nlohmann::json(result.tokens).dump() << '\n';
  }
}


void benchmark(const Arguments& args) {
  const std::filesystem::path directory = args.get("--model");
  if (directory.empty()) throw infer::Error("--model is required");
  infer::QwenTokenizer tokenizer(directory / "tokenizer.json");
  const auto token_ids = args.get("--token-ids");
  if (!token_ids.empty() && args.has("--prompt")) {
    throw infer::Error("--prompt and --token-ids are mutually exclusive");
  }
  const auto prompt = token_ids.empty() ? make_prompt(args, tokenizer)
                                        : parse_token_ids(token_ids);
  const auto options = runtime_options(args);
  infer::Qwen2Model model(directory, options);
  const int new_tokens = args.get_int("--max-new-tokens", 128);
  const int warmup = args.get_int("--warmup", 5);
  const int repeat = args.get_int("--repeat", 20);
  if (warmup < 0 || repeat <= 0 || new_tokens <= 0) {
    throw infer::Error("warmup must be >= 0, repeat and max-new-tokens must be > 0");
  }
  for (int i = 0; i < warmup; ++i) model.generate(prompt, new_tokens, false);

  if (args.has("--telemetry-markers")) {
    std::cerr << "BENCHMARK_MEASURE_BEGIN\n" << std::flush;
  }
  infer::WallTimer measurement;
  std::vector<double> totals;
  std::vector<double> ttfts;
  std::vector<double> decode_times;
  std::vector<double> tpots;
  std::vector<double> throughputs;
  std::vector<double> prefill_throughputs;
  nlohmann::json samples = nlohmann::json::array();
  for (int i = 0; i < repeat; ++i) {
    const auto result = model.generate(prompt, new_tokens, false);
    INFER_CHECK(result.stats.generated_tokens == new_tokens,
                "benchmark must generate exactly max-new-tokens");
    const double tpot = result.stats.generated_tokens > 1
                            ? result.stats.decode_ms / (result.stats.generated_tokens - 1)
                            : 0.0;
    const double prefill_throughput = result.stats.ttft_ms > 0.0
                                          ? prompt.size() * 1000.0 / result.stats.ttft_ms
                                          : 0.0;
    totals.push_back(result.stats.total_ms);
    ttfts.push_back(result.stats.ttft_ms);
    decode_times.push_back(result.stats.decode_ms);
    tpots.push_back(tpot);
    throughputs.push_back(result.stats.decode_tokens_per_second());
    prefill_throughputs.push_back(prefill_throughput);
    samples.push_back({
        {"iteration", i},
        {"ttft_ms", result.stats.ttft_ms},
        {"decode_ms", result.stats.decode_ms},
        {"total_ms", result.stats.total_ms},
        {"tpot_ms", tpot},
        {"prefill_tokens_per_second", prefill_throughput},
        {"decode_tokens_per_second", result.stats.decode_tokens_per_second()},
        {"generated_tokens", result.stats.generated_tokens},
    });
  }
  const double measurement_ms = measurement.elapsed_ms();
  if (args.has("--telemetry-markers")) {
    std::cerr << "BENCHMARK_MEASURE_END\n" << std::flush;
  }
  const long peak_rss = peak_rss_kib();
  const size_t peak_rss_bytes =
      peak_rss > 0 ? static_cast<size_t>(peak_rss) * 1024ULL : 0;
  nlohmann::json report{
      {"schema_version", 2},
      {"configuration",
       {{"backend", infer::to_string(options.backend)},
        {"precision", infer::to_string(options.precision)},
        {"batch_size", 1},
        {"decoding", "greedy_argmax"},
        {"early_stop", false},
        {"prefill_implementation", "matrixized"},
        {"max_sequence_length", options.max_sequence_length},
        {"linear_kernel", infer::to_string(options.linear_kernel)},
        {"use_cublas_gemv", options.linear_kernel == LinearKernel::kCublas},
        {"warmup", warmup},
        {"repeat", repeat},
        {"threading",
         {{"openblas_num_threads_env", environment_value("OPENBLAS_NUM_THREADS")},
          {"omp_num_threads_env", environment_value("OMP_NUM_THREADS")}}}}},
      {"tokens",
       {{"prompt", prompt.size()},
        {"max_new", new_tokens},
        {"generated_per_iteration", new_tokens},
        {"prompt_ids", prompt}}},
      {"measurement", {{"wall_time_ms", measurement_ms}}},
      {"statistics",
       {{"ttft_ms", summarize(ttfts)},
        {"decode_ms", summarize(decode_times)},
        {"total_ms", summarize(totals)},
        {"tpot_ms", summarize(tpots)},
        {"prefill_tokens_per_second", summarize(prefill_throughputs)},
        {"decode_tokens_per_second", summarize(throughputs)}}},
      {"samples", samples},
      {"memory",
       {{"peak_rss", memory_size(peak_rss_bytes)},
        {"model_archive", memory_size(model.archive().mapped_bytes())},
        {"device_weights", memory_size(model.device_weight_bytes())},
        {"workspace", memory_size(model.workspace_bytes())},
        {"kv_cache", memory_size(model.kv_cache_bytes())}}}};
  std::cout << report.dump(2) << '\n';
  const auto json_path = args.get("--json");
  if (!json_path.empty()) {
    const auto parent = std::filesystem::path(json_path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream output(json_path);
    output << report.dump(2) << '\n';
    INFER_CHECK(output.good(), "failed to write benchmark JSON");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments args(argc, argv);
    if (args.command() == "generate") generate(args);
    else if (args.command() == "benchmark") benchmark(args);
    else { usage(); return args.command() == "help" ? 0 : 1; }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
