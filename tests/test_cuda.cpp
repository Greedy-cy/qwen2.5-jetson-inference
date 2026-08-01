#include "infer/cuda_ops.hpp"
#include "infer/buffer.hpp"

#include <array>
#include <gtest/gtest.h>

TEST(CudaOps, RmsNormMatchesExpected) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  infer::cuda::Context context;
  const float host_input[] = {1, 2, 3, 4};
  const float host_weight[] = {1, 1, 1, 1};
  float* input = nullptr;
  float* weight = nullptr;
  float* output = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&input), sizeof(host_input)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&weight), sizeof(host_weight)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(host_input)), cudaSuccess);
  cudaMemcpyAsync(input, host_input, sizeof(host_input), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(weight, host_weight, sizeof(host_weight), cudaMemcpyHostToDevice, context.stream());
  infer::cuda::rms_norm(input, weight, output, 4, 0.0f, context.stream());
  float host_output[4]{};
  cudaMemcpyAsync(host_output, output, sizeof(host_output), cudaMemcpyDeviceToHost, context.stream());
  context.synchronize();
  const float scale = 1.0f / std::sqrt(7.5f);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(host_output[i], host_input[i] * scale, 1e-5f);
  cudaFree(input);
  cudaFree(weight);
  cudaFree(output);
}

TEST(CudaOps, Fp32AndInt8Gemv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  infer::cuda::Context context;
  const float host_weight[] = {1, 2, 3, 4, -1, 0.5f, 2, -3};
  const int8_t host_qweight[] = {1, 2, 3, 4, -2, 1, 4, -6};
  const float host_scales[] = {1, 1, 0.5f, 0.5f};
  const float host_input[] = {2, -1, 0.5f, 3};
  const float host_bias[] = {0.25f, -0.5f};
  infer::Buffer weight(sizeof(host_weight), infer::Device::kCuda);
  infer::Buffer qweight(sizeof(host_qweight), infer::Device::kCuda);
  infer::Buffer scales(sizeof(host_scales), infer::Device::kCuda);
  infer::Buffer input(sizeof(host_input), infer::Device::kCuda);
  infer::Buffer bias(sizeof(host_bias), infer::Device::kCuda);
  infer::Buffer output(2 * sizeof(float), infer::Device::kCuda);
  cudaMemcpyAsync(weight.data(), host_weight, sizeof(host_weight), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(qweight.data(), host_qweight, sizeof(host_qweight), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(scales.data(), host_scales, sizeof(host_scales), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(input.data(), host_input, sizeof(host_input), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(bias.data(), host_bias, sizeof(host_bias), cudaMemcpyHostToDevice, context.stream());

  auto copy_output = [&]() {
    float result[2]{};
    cudaMemcpyAsync(result, output.data(), sizeof(result), cudaMemcpyDeviceToHost, context.stream());
    context.synchronize();
    return std::array<float, 2>{result[0], result[1]};
  };
  infer::cuda::gemv_fp32(static_cast<const float*>(weight.data()),
                          static_cast<const float*>(bias.data()),
                          static_cast<const float*>(input.data()),
                          static_cast<float*>(output.data()), 2, 4, context.stream());
  const auto fp32 = copy_output();
  EXPECT_NEAR(fp32[0], 1 * 2 + 2 * -1 + 3 * 0.5f + 4 * 3 + 0.25f, 1e-5f);
  EXPECT_NEAR(fp32[1], -1 * 2 + 0.5f * -1 + 2 * 0.5f - 3 * 3 - 0.5f, 1e-5f);

  infer::cuda::gemv_fp32_cublas(context.cublas(), static_cast<const float*>(weight.data()),
                                 static_cast<const float*>(bias.data()),
                                 static_cast<const float*>(input.data()),
                                 static_cast<float*>(output.data()), 2, 4, context.stream());
  const auto cublas = copy_output();
  EXPECT_NEAR(cublas[0], fp32[0], 1e-5f);
  EXPECT_NEAR(cublas[1], fp32[1], 1e-5f);

  infer::cuda::gemv_int8(static_cast<const int8_t*>(qweight.data()),
                          static_cast<const float*>(scales.data()), 2,
                          static_cast<const float*>(bias.data()),
                          static_cast<const float*>(input.data()),
                          static_cast<float*>(output.data()), 2, 4, context.stream());
  const auto int8 = copy_output();
  EXPECT_NEAR(int8[0], fp32[0], 1e-5f);
  EXPECT_NEAR(int8[1], fp32[1], 1e-5f);
}

TEST(CudaOps, TiledFp32AndBatchedInt8Gemm) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  infer::cuda::Context context;
  constexpr int tokens = 3;
  constexpr int out_features = 5;
  constexpr int in_features = 8;
  std::array<int8_t, out_features * in_features> qweight{};
  std::array<float, out_features * in_features> weight{};
  std::array<float, tokens * in_features> input{};
  std::array<float, out_features * 2> scales{};
  for (size_t i = 0; i < qweight.size(); ++i) {
    qweight[i] = static_cast<int8_t>(static_cast<int>(i % 7) - 3);
    weight[i] = static_cast<float>(qweight[i]);
  }
  for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 5) - 2.0f;
  scales.fill(1.0f);
  infer::Buffer d_weight(sizeof(weight), infer::Device::kCuda);
  infer::Buffer d_qweight(sizeof(qweight), infer::Device::kCuda);
  infer::Buffer d_scales(sizeof(scales), infer::Device::kCuda);
  infer::Buffer d_input(sizeof(input), infer::Device::kCuda);
  infer::Buffer d_output(tokens * out_features * sizeof(float), infer::Device::kCuda);
  cudaMemcpyAsync(d_weight.data(), weight.data(), sizeof(weight), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(d_qweight.data(), qweight.data(), sizeof(qweight), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(d_scales.data(), scales.data(), sizeof(scales), cudaMemcpyHostToDevice, context.stream());
  cudaMemcpyAsync(d_input.data(), input.data(), sizeof(input), cudaMemcpyHostToDevice, context.stream());
  auto read_output = [&]() {
    std::array<float, tokens * out_features> values{};
    cudaMemcpyAsync(values.data(), d_output.data(), sizeof(values), cudaMemcpyDeviceToHost, context.stream());
    context.synchronize();
    return values;
  };
  infer::cuda::gemm_fp32(static_cast<const float*>(d_weight.data()),
                          static_cast<const float*>(d_input.data()),
                          static_cast<float*>(d_output.data()), tokens, out_features,
                          in_features, context.stream());
  const auto fp32 = read_output();
  infer::cuda::gemm_int8(static_cast<const int8_t*>(d_qweight.data()),
                          static_cast<const float*>(d_scales.data()), 4,
                          static_cast<const float*>(d_input.data()),
                          static_cast<float*>(d_output.data()), tokens, out_features,
                          in_features, context.stream());
  const auto int8 = read_output();
  for (int token = 0; token < tokens; ++token) {
    for (int row = 0; row < out_features; ++row) {
      float expected = 0.0f;
      for (int col = 0; col < in_features; ++col) {
        expected += input[token * in_features + col] * weight[row * in_features + col];
      }
      EXPECT_NEAR(fp32[token * out_features + row], expected, 1e-5f);
      EXPECT_NEAR(int8[token * out_features + row], expected, 1e-5f);
    }
  }
}
