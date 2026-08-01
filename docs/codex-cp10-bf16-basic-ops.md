# Codex Checkpoint 10：CUDA BF16 基础算子

## 结论

Checkpoint 10 已完成。CUDA A16 路径按项目约定使用 **BF16**，不是 FP16。本步实现并验证了后续 W16A16 decode 与 batched prefill 共用的基础算子，但尚未实现 BF16 Linear GEMV/GEMM，也尚未接入端到端 Qwen2 推理。

- 实现 commit：`889d2742910e664ec86ffc0702bbf6b7b24f3ecf`
- 目标设备：Jetson Orin Nano Super 8GB（aarch64）
- 构建：Release
- CUDA 编译器：12.6（`Build cuda_12.6.r12.6/compiler.34714021_0`）
- 完整回归：21/21 通过
- BF16 专项测试：2/2 通过

## 本步实现范围

### 数据与状态算子

- 单 token BF16 embedding
- batched BF16 embedding
- 单 token BF16 RMSNorm
- batched BF16 RMSNorm
- 单 token BF16 RoPE
- batched BF16 RoPE（支持绝对 `start_position`）
- 单 token BF16 KV Cache 写入
- batched BF16 KV Cache 写入
- BF16 decode attention
- BF16 causal prefill attention
- BF16 residual add
- BF16 SwiGLU 的 SiLU×up
- BF16 argmax

### 数值策略

| 环节 | 存储/输出 | 内部计算 |
|---|---|---|
| embedding、KV Cache | BF16 | BF16 搬运 |
| RMSNorm | BF16 | FP32 平方和、归约和 scale |
| RoPE | BF16 | FP32 sin/cos 与旋转计算 |
| attention | BF16 | FP32 dot、max、softmax、分母和加权求和 |
| residual、SwiGLU | BF16 | BF162 读取，FP32 运算后 RNE 转回 BF16 |
| argmax | BF16 | FP32 比较，输出 INT32 index |

在维度允许时使用 `__nv_bfloat162` 成对读写；奇数长度保留标量尾部路径。所有 FP32→BF16 写回均采用 round-to-nearest-even。

## 正确性测试

测试不是用另一套 CUDA kernel 互相比较，而是在 CPU 上独立计算 FP32 reference。输入先按真实数据路径舍入成 BF16，再进行 reference 计算，从而把“输入量化误差”和“算子实现误差”区分开。

### 测试 1：数据搬运、归一化与逐元素算子

`CudaBFloat16.DataMovementNormalizationAndElementwiseMatchFp32`

覆盖：

- 单 token/batched embedding：要求 BF16 bit 对应的数值完全一致。
- 单 token/batched RMSNorm：对独立 FP32 reference，绝对误差门槛 `0.016`。
- residual add：使用 7 个元素覆盖 BF162 主循环和标量 tail，门槛 `0.008`。
- SwiGLU：同样覆盖奇数 tail，门槛 `0.008`。
- argmax：要求 index 完全一致。

### 测试 2：RoPE、KV Cache 与 attention

`CudaBFloat16.RopeKvAndAttentionMatchFp32`

测试形状为 3 tokens、4 query heads、2 KV heads、head_dim 4，并设置非零绝对起始位置：

- batched RoPE 对 FP32 reference：门槛 `0.004`。
- 单 token RoPE 与 batch 对应行：完全一致。
- batched KV 写入：缓存布局和未写区域逐元素检查。
- 单 token KV 写入：指定 position 的缓存布局逐元素检查。
- causal prefill attention 对 FP32 reference：门槛 `0.004`。
- decode attention 与 prefill 最后一行：完全一致。

这组测试同时验证了 GQA head 映射和 causal 边界：第 `t` 个 query 只能访问 `[0,t]`。

## 验证命令与结果

```bash
cmake --build build -j2
./build/infer_tests --gtest_filter='CudaBFloat16.*'
ctest --test-dir build --output-on-failure
```

结果：

- BF16 专项：2 tests passed。
- 全量回归：21 tests passed，0 failed。
- `git diff --check`：通过。

## 本步没有声称的内容

- 没有实现 W16A16/BF16 Linear GEMV。
- 没有实现 W16A16/BF16 Linear GEMM。
- 没有把 BF16 workspace、weights、KV Cache 接入 Qwen2 端到端状态机。
- 没有做性能 benchmark，因此本报告不包含速度提升结论。

这些内容属于后续 checkpoint。Checkpoint 10 的作用是先冻结一套可独立验证的 BF16 基础算子层，避免在端到端接通后把数值问题与 Linear kernel 问题混在一起。
