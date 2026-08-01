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
  INFER_CHECK(record.dtype == infer::DType::kBFloat16,
              std::string(linear_case.weight) + " is not BF16");
  INFER_CHECK(record.shape.size() == 2, "linear weight must be rank 2");
  const int out_features = static_cast<int>(record.shape[0]);
  const int in_features = static_cast<int>(record.shape[1]);
  const auto* host_weight =
      static_cast<const uint16_t*>(archive.data(linear_case.weight));
  const uint16_t* host_bias = nullptr;
  const infer::TensorRecord* bias_record = nullptr;
  if (linear_case.bias && linear_case.bias[0] != '\0') {
    bias_record = &archive.record(linear_case.bias);
    INFER_CHECK(bias_record->dtype == infer::DType::kBFloat16,
                "linear bias is not BF16");
    host_bias = static_cast<const uint16_t*>(archive.data(linear_case.bias));
  }

  infer::Buffer device_weight(record.nbytes, infer::Device::kCuda);
  infer::Buffer device_bias;
  if (bias_record) device_bias.resize(bias_record->nbytes, infer::Device::kCuda);
  const auto input = make_input(args.tokens, in_features);
  infer::Buffer device_input(input.size() * sizeof(uint16_t),
                             infer::Device::kCuda);
  const size_t output_count =
      static_cast<size_t>(args.tokens) * out_features;
  infer::Buffer custom_output(output_count * sizeof(uint16_t),
                              infer::Device::kCuda);
  infer::Buffer cublas_output(output_count * sizeof(uint16_t),
                              infer::Device::kCuda);
  INFER_CUDA_CHECK(cudaMemcpyAsync(device_weight.data(), host_weight,
                                    record.nbytes, cudaMemcpyHostToDevice,
                                    context.stream()));
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

  const auto* weight =
      static_cast<const __nv_bfloat16*>(device_weight.data());
  const auto* bias = bias_record
                         ? static_cast<const __nv_bfloat16*>(device_bias.data())
                         : nullptr;
  const auto* device_input_bf16 =
      static_cast<const __nv_bfloat16*>(device_input.data());
  auto* custom = static_cast<__nv_bfloat16*>(custom_output.data());
  auto* cublas = static_cast<__nv_bfloat16*>(cublas_output.data());

  infer::cuda::gemv_bf16(weight, bias, device_input_bf16, custom,
                          out_features, in_features, context.stream());
  infer::cuda::gemv_bf16_cublas(
      context.cublas(), weight, bias, device_input_bf16, cublas, out_features,
      in_features, context.stream());
  const auto custom_gemv = read_output(custom_output, out_features, context);
  const auto cublas_gemv = read_output(cublas_output, out_features, context);
  const std::vector<uint16_t> first_input(input.begin(),
                                          input.begin() + in_features);
  const double custom_gemv_reference = sampled_reference_error(
      host_weight, host_bias, first_input, custom_gemv, 1, out_features,
      in_features);
  const double cublas_gemv_reference = sampled_reference_error(
      host_weight, host_bias, first_input, cublas_gemv, 1, out_features,
      in_features);

  const double gemv_difference =
      max_difference(custom_gemv, cublas_gemv);
  const double gemv_peak =
      std::max(max_magnitude(custom_gemv), max_magnitude(cublas_gemv));
  nlohmann::json result{
      {"name", linear_case.name},
      {"tensor", linear_case.weight},
      {"out_features", out_features},
      {"in_features", in_features},
      {"bias", bias_record != nullptr},
      {"gemv",
       {{"custom",
         summarize(measure(context, args.warmup, args.repeat, [&] {
           infer::cuda::gemv_bf16(weight, bias, device_input_bf16, custom,
                                   out_features, in_features,
                                   context.stream());
         }))},
        {"cublas",
         summarize(measure(context, args.warmup, args.repeat, [&] {
           infer::cuda::gemv_bf16_cublas(
               context.cublas(), weight, bias, device_input_bf16, cublas,
               out_features, in_features, context.stream());
         }))},
        {"output_peak_abs", gemv_peak},
        {"max_custom_vs_cublas", gemv_difference},
        {"max_custom_vs_cublas_relative_to_peak",
         gemv_peak > 0.0 ? gemv_difference / gemv_peak : 0.0},
        {"sampled_max_custom_vs_fp32", custom_gemv_reference},
        {"sampled_max_custom_vs_fp32_relative_to_peak",
         gemv_peak > 0.0 ? custom_gemv_reference / gemv_peak : 0.0},
        {"sampled_max_cublas_vs_fp32", cublas_gemv_reference},
        {"sampled_max_cublas_vs_fp32_relative_to_peak",
         gemv_peak > 0.0 ? cublas_gemv_reference / gemv_peak : 0.0}}}};

  if (linear_case.benchmark_gemm) {
    infer::cuda::gemm_bf16(weight, bias, device_input_bf16, custom, args.tokens,
                            out_features, in_features, context.stream());
    infer::cuda::gemm_bf16_cublas(
        context.cublas(), weight, bias, device_input_bf16, cublas, args.tokens,
        out_features, in_features, context.stream());
    const auto custom_gemm = read_output(custom_output, output_count, context);
    const auto cublas_gemm = read_output(cublas_output, output_count, context);
    const double gemm_difference =
        max_difference(custom_gemm, cublas_gemm);
    const double gemm_peak =
        std::max(max_magnitude(custom_gemm), max_magnitude(cublas_gemm));
    const double custom_reference = sampled_reference_error(
        host_weight, host_bias, input, custom_gemm, args.tokens, out_features,
        in_features);
    const double cublas_reference = sampled_reference_error(
        host_weight, host_bias, input, cublas_gemm, args.tokens, out_features,
        in_features);
    result["gemm"] = {
        {"tokens", args.tokens},
        {"custom",
         summarize(measure(context, args.warmup, args.repeat, [&] {
           infer::cuda::gemm_bf16(
               weight, bias, device_input_bf16, custom, args.tokens,
               out_features, in_features, context.stream());
         }))},
        {"cublas",
         summarize(measure(context, args.warmup, args.repeat, [&] {
           infer::cuda::gemm_bf16_cublas(
               context.cublas(), weight, bias, device_input_bf16, cublas,
               args.tokens, out_features, in_features, context.stream());
         }))},
        {"output_peak_abs", gemm_peak},
        {"max_custom_vs_cublas", gemm_difference},
        {"max_custom_vs_cublas_relative_to_peak",
         gemm_peak > 0.0 ? gemm_difference / gemm_peak : 0.0},
        {"sampled_max_custom_vs_fp32", custom_reference},
        {"sampled_max_custom_vs_fp32_relative_to_peak",
         gemm_peak > 0.0 ? custom_reference / gemm_peak : 0.0},
        {"sampled_max_cublas_vs_fp32", cublas_reference},
        {"sampled_max_cublas_vs_fp32_relative_to_peak",
         gemm_peak > 0.0 ? cublas_reference / gemm_peak : 0.0}};
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    infer::ModelArchive archive(arguments.archive);
    infer::cuda::Context context;
    const std::array<LinearCase, 8> cases = {{
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
    nlohmann::json report{
        {"schema_version", 1},
        {"benchmark", "cuda_bf16_linear"},
        {"archive", arguments.archive.string()},
        {"tokens", arguments.tokens},
        {"warmup", arguments.warmup},
        {"repeat", arguments.repeat},
        {"cases", nlohmann::json::array()}};
    for (const auto& linear_case : cases) {
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
