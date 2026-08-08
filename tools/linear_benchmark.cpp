#include "infer/archive.hpp"
#include "infer/buffer.hpp"
#include "infer/cuda_ops.hpp"

#include <array>
#include <functional>
#include <nlohmann/json.hpp>

namespace {

struct Arguments {
  std::filesystem::path archive;
  std::filesystem::path json_path;
  std::string precision{"w16a16"};
  int tokens{32};
  int warmup{5};
  int repeat{20};
};

struct LinearCase {
  const char* name;
  const char* weight;
  const char* bias;
  bool benchmark_gemm;
};

constexpr std::array<LinearCase, 8> kCases = {{
    {"q_proj", "model.layers.0.self_attn.q_proj.weight",
     "model.layers.0.self_attn.q_proj.bias", true},
    {"k_proj", "model.layers.0.self_attn.k_proj.weight",
     "model.layers.0.self_attn.k_proj.bias", true},
    {"v_proj", "model.layers.0.self_attn.v_proj.weight",
     "model.layers.0.self_attn.v_proj.bias", true},
    {"o_proj", "model.layers.0.self_attn.o_proj.weight", "", true},
    {"gate_proj", "model.layers.0.mlp.gate_proj.weight", "", true},
    {"up_proj", "model.layers.0.mlp.up_proj.weight", "", true},
    {"down_proj", "model.layers.0.mlp.down_proj.weight", "", true},
    {"tied_lm_head", "model.embed_tokens.weight", "", false},
}};

uint16_t float_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<uint16_t>(bits >> 16u);
}

float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16u;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto take = [&]() {
      INFER_CHECK(index + 1 < argc, "missing value after " + argument);
      return std::string(argv[++index]);
    };
    if (argument == "--archive") {
      result.archive = take();
    } else if (argument == "--precision") {
      result.precision = take();
    } else if (argument == "--json") {
      result.json_path = take();
    } else if (argument == "--tokens") {
      result.tokens = std::stoi(take());
    } else if (argument == "--warmup") {
      result.warmup = std::stoi(take());
    } else if (argument == "--repeat") {
      result.repeat = std::stoi(take());
    } else {
      throw infer::Error("unknown argument: " + argument);
    }
  }
  INFER_CHECK(!result.archive.empty(), "--archive is required");
  INFER_CHECK(result.precision == "w16a16" || result.precision == "w8a16",
              "--precision must be w16a16 or w8a16");
  INFER_CHECK(result.tokens > 0 && result.warmup >= 0 && result.repeat > 0,
              "tokens/repeat must be positive and warmup non-negative");
  return result;
}

std::vector<double> measure(infer::cuda::Context& context, int warmup,
                            int repeat, const std::function<void()>& operation) {
  for (int index = 0; index < warmup; ++index) operation();
  context.synchronize();
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  INFER_CUDA_CHECK(cudaEventCreate(&start));
  INFER_CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> samples;
  samples.reserve(repeat);
  for (int index = 0; index < repeat; ++index) {
    INFER_CUDA_CHECK(cudaEventRecord(start, context.stream()));
    operation();
    INFER_CUDA_CHECK(cudaEventRecord(stop, context.stream()));
    INFER_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    INFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    samples.push_back(elapsed);
  }
  INFER_CUDA_CHECK(cudaEventDestroy(start));
  INFER_CUDA_CHECK(cudaEventDestroy(stop));
  return samples;
}

nlohmann::json summarize(std::vector<double> values) {
  INFER_CHECK(!values.empty(), "cannot summarize empty samples");
  std::sort(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  auto percentile = [&](double fraction) {
    const size_t index = static_cast<size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
  };
  return {{"mean_ms", sum / values.size()},
          {"p50_ms", percentile(0.50)},
          {"p95_ms", percentile(0.95)},
          {"min_ms", values.front()},
          {"max_ms", values.back()}};
}

std::vector<uint16_t> make_input(int tokens, int in_features) {
  std::vector<uint16_t> result(static_cast<size_t>(tokens) * in_features);
  for (size_t index = 0; index < result.size(); ++index) {
    const float value =
        (static_cast<int>((index * 17 + 3) % 31) - 15) * 0.00390625f;
    result[index] = float_to_bf16(value);
  }
  return result;
}

double max_magnitude(const std::vector<uint16_t>& values) {
  double maximum = 0.0;
  for (uint16_t value : values) {
    maximum =
        std::max(maximum, static_cast<double>(std::abs(bf16_to_float(value))));
  }
  return maximum;
}

double max_difference(const std::vector<uint16_t>& first,
                      const std::vector<uint16_t>& second) {
  INFER_CHECK(first.size() == second.size(), "output size mismatch");
  double maximum = 0.0;
  for (size_t index = 0; index < first.size(); ++index) {
    maximum = std::max(
        maximum, static_cast<double>(
                     std::abs(bf16_to_float(first[index]) -
                              bf16_to_float(second[index]))));
  }
  return maximum;
}

double sampled_reference_error(
    const uint16_t* weight, const uint16_t* bias,
    const std::vector<uint16_t>& input, const std::vector<uint16_t>& output,
    int tokens, int out_features, int in_features) {
  const std::array<int, 3> rows = {0, out_features / 2, out_features - 1};
  const std::array<int, 2> token_indices = {0, tokens - 1};
  double maximum = 0.0;
  for (int token : token_indices) {
    for (int row : rows) {
      float expected = bias ? bf16_to_float(bias[row]) : 0.0f;
      for (int column = 0; column < in_features; ++column) {
        expected = std::fma(
            bf16_to_float(
                weight[static_cast<size_t>(row) * in_features + column]),
            bf16_to_float(
                input[static_cast<size_t>(token) * in_features + column]),
            expected);
      }
      const float actual =
          bf16_to_float(output[static_cast<size_t>(token) * out_features + row]);
      maximum = std::max(
          maximum, static_cast<double>(std::abs(actual - expected)));
    }
  }
  return maximum;
}

double sampled_dequant_reference_error(
    const int8_t* weight, const uint16_t* scales, int group_size,
    const uint16_t* bias, const std::vector<uint16_t>& input,
    const std::vector<uint16_t>& output, int tokens, int out_features,
    int in_features) {
  const int groups = (in_features + group_size - 1) / group_size;
  const std::array<int, 3> rows = {0, out_features / 2, out_features - 1};
  const std::array<int, 2> token_indices = {0, tokens - 1};
  double maximum = 0.0;
  for (int token : token_indices) {
    for (int row : rows) {
      float expected = 0.0f;
      for (int column = 0; column < in_features; ++column) {
        const float dequantized =
            static_cast<float>(
                weight[static_cast<size_t>(row) * in_features + column]) *
            bf16_to_float(scales[static_cast<size_t>(row) * groups +
                                 column / group_size]);
        expected = std::fma(
            dequantized,
            bf16_to_float(
                input[static_cast<size_t>(token) * in_features + column]),
            expected);
      }
      if (bias) expected += bf16_to_float(bias[row]);
      const float actual =
          bf16_to_float(output[static_cast<size_t>(token) * out_features + row]);
      maximum = std::max(
          maximum, static_cast<double>(std::abs(actual - expected)));
    }
  }
  return maximum;
}

std::vector<uint16_t> read_output(const infer::Buffer& buffer, size_t count,
                                  infer::cuda::Context& context) {
  std::vector<uint16_t> result(count);
  INFER_CUDA_CHECK(cudaMemcpyAsync(result.data(), buffer.data(),
                                    count * sizeof(uint16_t),
                                    cudaMemcpyDeviceToHost, context.stream()));
  context.synchronize();
  return result;
}

nlohmann::json run_case(const LinearCase& linear_case,
                        const infer::ModelArchive& archive,
                        infer::cuda::Context& context, const Arguments& args) {
  const auto& record = archive.record(linear_case.weight);
  const bool quantized = record.dtype == infer::DType::kInt8;
  INFER_CHECK(record.dtype == infer::DType::kBFloat16 || quantized,
              std::string(linear_case.weight) + " is not BF16/INT8");
  INFER_CHECK(record.shape.size() == 2, "linear weight must be rank 2");
  INFER_CHECK(!quantized || args.precision == "w8a16",
              "INT8 weight requires --precision w8a16");
  const int out_features = static_cast<int>(record.shape[0]);
  const int in_features = static_cast<int>(record.shape[1]);
  const int group_size =
      quantized ? record.quant->group_size : 0;
  const int groups =
      quantized ? (in_features + group_size - 1) / group_size : 0;

  const auto* host_weight =
      static_cast<const uint8_t*>(archive.data(linear_case.weight));
  const uint16_t* host_bf16_weight = nullptr;
  const int8_t* host_int8_weight = nullptr;
  const uint16_t* host_scales = nullptr;
  if (quantized) {
    host_int8_weight = reinterpret_cast<const int8_t*>(host_weight);
    host_scales = static_cast<const uint16_t*>(
        archive.data(record.quant->scale_tensor));
  } else {
    host_bf16_weight = reinterpret_cast<const uint16_t*>(host_weight);
  }
  const uint16_t* host_bias = nullptr;
  const infer::TensorRecord* bias_record = nullptr;
  if (linear_case.bias && linear_case.bias[0] != '\0') {
    bias_record = &archive.record(linear_case.bias);
    INFER_CHECK(bias_record->dtype == infer::DType::kBFloat16,
                "linear bias is not BF16");
    host_bias = static_cast<const uint16_t*>(archive.data(linear_case.bias));
  }

  infer::Buffer device_weight(record.nbytes, infer::Device::kCuda);
  infer::Buffer device_scales;
  if (quantized) {
    device_scales.resize(archive.record(record.quant->scale_tensor).nbytes,
                         infer::Device::kCuda);
  }
  infer::Buffer device_bias;
  if (bias_record) device_bias.resize(bias_record->nbytes, infer::Device::kCuda);
  const auto input = make_input(args.tokens, in_features);
  infer::Buffer device_input(input.size() * sizeof(uint16_t),
                             infer::Device::kCuda);
  const size_t output_count =
      static_cast<size_t>(args.tokens) * out_features;
  infer::Buffer first_output(output_count * sizeof(uint16_t),
                             infer::Device::kCuda);
  infer::Buffer second_output(output_count * sizeof(uint16_t),
                              infer::Device::kCuda);
  infer::Buffer argmax_first(sizeof(int), infer::Device::kCuda);
  infer::Buffer argmax_second(sizeof(int), infer::Device::kCuda);
  const size_t lm_head_blocks = (static_cast<size_t>(out_features) + 7) / 8;
  infer::Buffer lm_head_max_value(lm_head_blocks * sizeof(float),
                                  infer::Device::kCuda);
  infer::Buffer lm_head_max_index(lm_head_blocks * sizeof(int),
                                  infer::Device::kCuda);
  INFER_CUDA_CHECK(cudaMemcpyAsync(device_weight.data(), host_weight,
                                    record.nbytes, cudaMemcpyHostToDevice,
                                    context.stream()));
  if (quantized) {
    INFER_CUDA_CHECK(cudaMemcpyAsync(
        device_scales.data(), host_scales, device_scales.bytes(),
        cudaMemcpyHostToDevice, context.stream()));
  }
  if (bias_record) {
    INFER_CUDA_CHECK(cudaMemcpyAsync(device_bias.data(), host_bias,
                                      bias_record->nbytes,
                                      cudaMemcpyHostToDevice,
                                      context.stream()));
  }
  INFER_CUDA_CHECK(cudaMemcpyAsync(device_input.data(), input.data(),
                                    input.size() * sizeof(uint16_t),
                                    cudaMemcpyHostToDevice, context.stream()));
  context.synchronize();

  const auto* weight_bf16 =
      static_cast<const __nv_bfloat16*>(device_weight.data());
  const auto* weight_int8 =
      static_cast<const int8_t*>(device_weight.data());
  const auto* scales_bf16 =
      quantized ? static_cast<const __nv_bfloat16*>(device_scales.data())
                : nullptr;
  const auto* bias = bias_record
                         ? static_cast<const __nv_bfloat16*>(device_bias.data())
                         : nullptr;
  const auto* device_input_bf16 =
      static_cast<const __nv_bfloat16*>(device_input.data());
  auto* first = static_cast<__nv_bfloat16*>(first_output.data());
  auto* second = static_cast<__nv_bfloat16*>(second_output.data());

  const auto first_input = std::vector<uint16_t>(
      input.begin(), input.begin() + in_features);
  nlohmann::json result{
      {"name", linear_case.name},
      {"tensor", linear_case.weight},
      {"dtype", quantized ? "int8" : "bfloat16"},
      {"group_size", group_size},
      {"out_features", out_features},
      {"in_features", in_features},
      {"bias", bias_record != nullptr},
  };

  const bool is_lm_head = std::string(linear_case.name) == "tied_lm_head";
  if (is_lm_head) {
    auto* lm_max_value = static_cast<float*>(lm_head_max_value.data());
    auto* lm_max_index = static_cast<int*>(lm_head_max_index.data());
    auto* first_argmax = static_cast<int*>(argmax_first.data());
    auto* second_argmax = static_cast<int*>(argmax_second.data());
    infer::cuda::lm_head_bf16(
        weight_bf16, nullptr, device_input_bf16, first, lm_max_value,
        lm_max_index, first_argmax, out_features, in_features,
        context.stream());
    infer::cuda::gemv_bf16_cublas(
        context.cublas(), weight_bf16, nullptr, device_input_bf16, second,
        out_features, in_features, context.stream());
    infer::cuda::argmax_bf16(second, out_features, second_argmax,
                             context.stream());
    const auto custom_logits = read_output(first_output, out_features, context);
    const auto cublas_logits =
        read_output(second_output, out_features, context);
    int custom_index = -1;
    int cublas_index = -1;
    INFER_CUDA_CHECK(cudaMemcpyAsync(
        &custom_index, argmax_first.data(), sizeof(int),
        cudaMemcpyDeviceToHost, context.stream()));
    INFER_CUDA_CHECK(cudaMemcpyAsync(
        &cublas_index, argmax_second.data(), sizeof(int),
        cudaMemcpyDeviceToHost, context.stream()));
    context.synchronize();
    const double peak =
        std::max(max_magnitude(custom_logits), max_magnitude(cublas_logits));
    result["gemv"] = {
        {"custom_fused",
         summarize(measure(context, args.warmup, args.repeat, [&] {
           infer::cuda::lm_head_bf16(
               weight_bf16, nullptr, device_input_bf16, first, lm_max_value,
               lm_max_index, first_argmax, out_features, in_features,
               context.stream());
         }))},
        {"cublas_with_argmax",
         summarize(measure(context, args.warmup, args.repeat, [&] {
           infer::cuda::gemv_bf16_cublas(
               context.cublas(), weight_bf16, nullptr, device_input_bf16,
               second, out_features, in_features, context.stream());
           infer::cuda::argmax_bf16(second, out_features, second_argmax,
                                    context.stream());
         }))},
        {"output_peak_abs", peak},
        {"max_custom_vs_cublas", max_difference(custom_logits, cublas_logits)},
        {"max_custom_vs_cublas_relative_to_peak",
         peak > 0.0 ? max_difference(custom_logits, cublas_logits) / peak
                    : 0.0},
        {"custom_argmax", custom_index},
        {"cublas_argmax", cublas_index},
        {"argmax_match", custom_index == cublas_index},
    };
    return result;
  }

  const std::function<void(__nv_bfloat16*)> optimized_gemv =
      quantized ? std::function<void(__nv_bfloat16*)>(
                      [&](__nv_bfloat16* out) {
                        infer::cuda::gemv_w8a16(
                            weight_int8, scales_bf16, group_size, bias,
                            device_input_bf16, out, out_features,
                            in_features, context.stream());
                      })
                : std::function<void(__nv_bfloat16*)>(
                      [&](__nv_bfloat16* out) {
                        infer::cuda::gemv_bf16(
                            weight_bf16, bias, device_input_bf16, out,
                            out_features, in_features, context.stream());
                      });
  optimized_gemv(first);
  infer::cuda::gemv_bf16_cublas(
      context.cublas(), weight_bf16, bias, device_input_bf16, second,
      out_features, in_features, context.stream());
  const auto optimized_gemv_out = read_output(first_output, out_features, context);
  const auto cublas_gemv_out = read_output(second_output, out_features, context);
  const double gemv_peak =
      std::max(max_magnitude(optimized_gemv_out),
               max_magnitude(cublas_gemv_out));
  const double gemv_optimized_reference = quantized
      ? sampled_dequant_reference_error(
            host_int8_weight, host_scales, group_size, host_bias, first_input,
            optimized_gemv_out, 1, out_features, in_features)
      : sampled_reference_error(host_bf16_weight, host_bias, first_input,
                                optimized_gemv_out, 1, out_features,
                                in_features);
  nlohmann::json gemv_section{
      {"custom",
       summarize(measure(context, args.warmup, args.repeat, [&] {
         optimized_gemv(first);
       }))},
      {"cublas",
       summarize(measure(context, args.warmup, args.repeat, [&] {
         infer::cuda::gemv_bf16_cublas(
             context.cublas(), weight_bf16, bias, device_input_bf16, second,
             out_features, in_features, context.stream());
       }))},
      {"output_peak_abs", gemv_peak},
      {"max_custom_vs_cublas",
       max_difference(optimized_gemv_out, cublas_gemv_out)},
      {"max_custom_vs_cublas_relative_to_peak",
       gemv_peak > 0.0
           ? max_difference(optimized_gemv_out, cublas_gemv_out) / gemv_peak
           : 0.0},
      {"sampled_max_vs_reference", gemv_optimized_reference},
      {"sampled_max_vs_reference_relative_to_peak",
       gemv_peak > 0.0 ? gemv_optimized_reference / gemv_peak : 0.0},
  };
  if (quantized) {
    gemv_section["reference"] = "explicit_dequant_fp32";
  }
  result["gemv"] = gemv_section;

  if (linear_case.benchmark_gemm) {
    const std::function<void(__nv_bfloat16*)> optimized_gemm =
        quantized ? std::function<void(__nv_bfloat16*)>(
                        [&](__nv_bfloat16* out) {
                          infer::cuda::gemm_w8a16(
                              weight_int8, scales_bf16, group_size, bias,
                              device_input_bf16, out, args.tokens,
                              out_features, in_features, context.stream());
                        })
                  : std::function<void(__nv_bfloat16*)>(
                        [&](__nv_bfloat16* out) {
                          infer::cuda::gemm_bf16(
                              weight_bf16, bias, device_input_bf16, out,
                              args.tokens, out_features, in_features,
                              context.stream());
                        });
    optimized_gemm(first);
    infer::cuda::gemm_bf16_cublas(
        context.cublas(), weight_bf16, bias, device_input_bf16, second,
        args.tokens, out_features, in_features, context.stream());
    const auto optimized_gemm_out =
        read_output(first_output, output_count, context);
    const auto cublas_gemm_out =
        read_output(second_output, output_count, context);
    const double gemm_peak =
        std::max(max_magnitude(optimized_gemm_out),
                 max_magnitude(cublas_gemm_out));
    const double gemm_optimized_reference = quantized
        ? sampled_dequant_reference_error(
              host_int8_weight, host_scales, group_size, host_bias, input,
              optimized_gemm_out, args.tokens, out_features, in_features)
        : sampled_reference_error(host_bf16_weight, host_bias, input,
                                  optimized_gemm_out, args.tokens,
                                  out_features, in_features);
    nlohmann::json gemm_section{
        {"tokens", args.tokens},
        {"custom",
         summarize(measure(context, args.warmup, args.repeat, [&] {
           optimized_gemm(first);
         }))},
        {"cublas",
         summarize(measure(context, args.warmup, args.repeat, [&] {
           infer::cuda::gemm_bf16_cublas(
               context.cublas(), weight_bf16, bias, device_input_bf16, second,
               args.tokens, out_features, in_features, context.stream());
         }))},
        {"output_peak_abs", gemm_peak},
        {"max_custom_vs_cublas",
         max_difference(optimized_gemm_out, cublas_gemm_out)},
        {"max_custom_vs_cublas_relative_to_peak",
         gemm_peak > 0.0
             ? max_difference(optimized_gemm_out, cublas_gemm_out) / gemm_peak
             : 0.0},
        {"sampled_max_vs_reference", gemm_optimized_reference},
        {"sampled_max_vs_reference_relative_to_peak",
         gemm_peak > 0.0 ? gemm_optimized_reference / gemm_peak : 0.0},
        {"first_row_matches_gemv",
         [&]() {
           for (int row = 0; row < out_features; ++row) {
             if (optimized_gemm_out[static_cast<size_t>(row)] !=
                 optimized_gemv_out[static_cast<size_t>(row)]) {
               return false;
             }
           }
           return true;
         }()},
    };
    if (quantized) {
      gemm_section["reference"] = "explicit_dequant_fp32";
    }
    result["gemm"] = gemm_section;
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    infer::ModelArchive archive(arguments.archive);
    infer::cuda::Context context;
    nlohmann::json report{
        {"schema_version", 2},
        {"benchmark", "linear_benchmark"},
        {"precision", arguments.precision},
        {"archive", arguments.archive.string()},
        {"tokens", arguments.tokens},
        {"warmup", arguments.warmup},
        {"repeat", arguments.repeat},
        {"cases", nlohmann::json::array()}};
    for (const auto& linear_case : kCases) {
      std::cerr << "benchmarking " << linear_case.name << "\n";
      report["cases"].push_back(
          run_case(linear_case, archive, context, arguments));
    }
    const std::string text = report.dump(2) + "\n";
    if (!arguments.json_path.empty()) {
      std::ofstream output(arguments.json_path);
      INFER_CHECK(output.good(), "failed to open JSON output");
      output << text;
    }
    std::cout << text;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
