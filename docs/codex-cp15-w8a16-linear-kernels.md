# Codex Checkpoint 15：CUDA W8A16 GEMV/GEMM

## 结论

Checkpoint 15 已完成。项目新增了 CUDA W8A16 Linear 基础算子：

- Decode 使用 `gemv_w8a16()`。
- Matrixized Prefill 使用 `gemm_w8a16()`。
- weight：INT8。
- group scale：BF16。
- input、bias、output：BF16。
- dot product 与 block reduction：FP32。
- group size：固定为 64，其他值在 kernel launch 前显式报错。

实现 commit：`4ea14822b450d4deb38f47275c5f7fc66ab7c231`。

本 checkpoint 只验证独立 Linear kernel，不接入 Qwen2 端到端推理，因此不报告 Prefill/Decode token/s。端到端正确性属于 CP16，正式吞吐率属于 CP17。

## Kernel 数据流

每个 CUDA block 对应一个 `(token, output row)`：

```text
INT8 weight ─┐
BF16 scale ──┼─> on-the-fly dequantize ─┐
BF16 input ──┘                          ├─> FP32 FMA/reduction
BF16 bias ──────────────────────────────┘
                                             │
                                             └─> round-to-nearest-even BF16 output
```

GEMV 使用 grid `(out_features, 1)`；GEMM 使用二维 grid `(out_features, tokens)`。两条 API 复用同一计算核心，避免 Decode 与 Prefill 出现不同的量化语义。

## 向量化与边界

真实 Qwen2.5 维度的输入宽度均满足 4-byte 对齐，主路径使用：

- `char4` 一次读取 4 个 INT8 weights。
- 两个 `__nv_bfloat162` 一次读取 4 个 BF16 activations。
- 同一 group 内的 4 个 weights 复用一个 BF16 scale。
- 每个线程以 4 个元素为单位遍历 K。

安全路径覆盖：

- `K` 不是 4 的倍数时的尾部元素。
- 由于 `row * K` 导致 weight 行首未按 4-byte 对齐。
- token input 行首未按 4-byte 对齐。
- 一个 4-element chunk 跨越 group 边界。

上述情况逐元素读取 weight、activation 和对应 group scale，不会静默丢弃尾部。

## 显式反量化 Reference

测试 reference 完全按照归档/runtime 的 dtype 计算：

1. INT8 weight 转为 FP32。
2. 读取已经 BF16 舍入的 group scale。
3. 得到显式反量化 FP32 weight。
4. 读取已经 BF16 舍入的 activation。
5. FP32 FMA 累加。
6. 累加完成后加入 BF16 bias。
7. 最终结果舍入为 BF16。

kernel 输出与 reference 比较，同时验证 GEMV 输出等于 GEMM 第一个 token 的输出。

## 测试矩阵

| Tokens | Out features | In features K | Bias | 覆盖目的 | Max abs error |
|---:|---:|---:|:---:|---|---:|
| 3 | 5 | 70 | 是 | group 边界、非 4 整除、未对齐行 | 0 |
| 2 | 7 | 65 | 否 | 单元素尾 group、无 bias | 0 |
| 2 | 896 | 896 | 是 | Q Projection 真实维度 | 0 |
| 2 | 128 | 896 | 是 | K/V Projection 真实维度 | 0 |
| 2 | 4864 | 896 | 否 | gate/up Projection 真实维度 | 0 |
| 2 | 896 | 4864 | 否 | down Projection 真实维度 | 0 |

测试数据还覆盖：

- INT8 极值 `-127` 与 `127`。
- 全零 group 与 scale=1。
- 多个不同 BF16 group scales。
- BF16 input 和 bias。
- group-size 32 被显式拒绝。

六组测试在 Jetson Orin Nano Super 上得到的 BF16 输出与显式反量化 reference 最大绝对差值均为 0。

## 自动化测试

新增：

- `CudaW8A16.GemvAndGemmHandleGroupBoundaryTailAndBias`
- `CudaW8A16.GemvAndGemmMatchDequantizedReferenceAtQwenShapes`

Release 全量结果：

```text
28 tests passed, 0 failed
```

专项测试运行时间约 `0.15 s`；这个时间包含测试数据准备、显存分配、H2D/D2H 和同步，不是 kernel benchmark，不应解释为模型吞吐率。

目标设备没有安装 `compute-sanitizer` 或旧版 `cuda-memcheck`，因此本步没有声称完成 sanitizer 检查。边界正确性由精确尺寸分配、非整除 K、未对齐行和输出对照测试覆盖。

## API

```cpp
void gemv_w8a16(
    const int8_t* weight,
    const __nv_bfloat16* scales,
    int group_size,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int out_features,
    int in_features,
    cudaStream_t stream);

void gemm_w8a16(
    const int8_t* weight,
    const __nv_bfloat16* scales,
    int group_size,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int tokens,
    int out_features,
    int in_features,
    cudaStream_t stream);
```

## 本步没有做什么

- 没有增加 `Precision::kW8A16` runtime 路径。
- 没有把真实 `model.w8a16.qbin` 上传到 Qwen2Model。
- 没有接入 Prefill、Decode、KV Cache 或 LM Head。
- 没有进行 token/s、功耗或 Nsight 正式性能测量。
- 没有预先判断 W8A16 一定快于 W16A16。

这些工作分别留给 CP16 和 CP17。本 checkpoint 到此停止。
