#include "infer/cpu_ops.hpp"

#include <gtest/gtest.h>

TEST(CpuOps, RmsNorm) {
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float output[4]{};
  infer::cpu::rms_norm(input, weight, output, 4, 0.0f);
  const float scale = 1.0f / std::sqrt(7.5f);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(output[i], input[i] * scale, 1e-6f);
}

TEST(CpuOps, Int8LinearMatchesDequantized) {
  const int8_t weight[] = {1, -2, 3, -4, 5, 6, -7, 8};
  const float scales[] = {0.25f, 0.5f, 0.1f, 0.2f};
  const float input[] = {1, 2, 3, 4};
  float output[2]{};
  infer::cpu::linear_int8(weight, scales, 2, nullptr, input, output, 2, 4);
  EXPECT_NEAR(output[0], (1 * 1 - 2 * 2) * 0.25f + (3 * 3 - 4 * 4) * 0.5f, 1e-6f);
  EXPECT_NEAR(output[1], (5 * 1 + 6 * 2) * 0.1f + (-7 * 3 + 8 * 4) * 0.2f, 1e-6f);
}

TEST(CpuOps, RopePositionZeroIsIdentity) {
  float q[] = {1, 2, 3, 4};
  float k[] = {5, 6, 7, 8};
  infer::cpu::rope(q, k, 1, 1, 4, 0, 10000.0f);
  EXPECT_FLOAT_EQ(q[0], 1);
  EXPECT_FLOAT_EQ(q[3], 4);
  EXPECT_FLOAT_EQ(k[0], 5);
  EXPECT_FLOAT_EQ(k[3], 8);
}
