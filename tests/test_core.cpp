#include "infer/buffer.hpp"
#include "infer/model_config.hpp"

#include <gtest/gtest.h>

TEST(Buffer, CpuMoveAndArenaAlignment) {
  infer::Buffer first(4096, infer::Device::kCpu);
  ASSERT_NE(first.data(), nullptr);
  infer::Buffer second(std::move(first));
  EXPECT_EQ(first.data(), nullptr);
  EXPECT_EQ(second.bytes(), 4096u);
  infer::Arena arena(1024, infer::Device::kCpu);
  auto* a = arena.allocate(7, 64);
  auto* b = arena.allocate(7, 64);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(a) % 64, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(b) % 64, 0u);
}
