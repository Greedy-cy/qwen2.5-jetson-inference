# Codex Checkpoint 12：端到端 CUDA W16A16（原生 BF16）

## 结论

Checkpoint 12 已完成。Qwen2.5-0.5B-Instruct 的 CUDA W16A16 路径已接通真实模型的矩阵化 Prefill 与单 token Decode。

这里的 A16 和 W16 均为 **BF16**，不是 FP16。

- 实现 commit：`c5dd91eafff51cd33084e4b0d04e86ec3447add5`
- 模型：Qwen2.5-0.5B-Instruct
- BF16 archive SHA-256：`f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`
- Release 全量测试：25/25 通过
- 真实 24 层模型 Prefill + Decode：通过
- 5 个固定 prompt 首 token：FP32/W16A16 top-1 5/5 一致
- W16 custom/cuBLAS canonical 16-token 输出：完全一致

## 数据流

端到端 W16A16 使用：

- BF16 embedding、norm weight、bias、Linear weight。
- BF16 decode workspace。
- BF16 batched Prefill workspace。
- BF16 KV Cache。
- BF16 logits。
- RMSNorm、attention dot/softmax、Linear accumulator 等敏感计算使用 FP32。
- FP32 结果以 round-to-nearest-even 写回 BF16。

项目中此前仅用于对照的逐 token prompt 路径已经移除，因此本 checkpoint 的 Prefill 是矩阵化 Prefill，不再保留伪 serial Prefill。

## 接通的完整流程

### Matrixized Prefill

1. batched BF16 embedding。
2. 每层 batched RMSNorm。
3. BF16 Q/K/V GEMM。
4. batched RoPE。
5. 批量写入 BF16 KV Cache。
6. online causal Prefill attention。
7. BF16 O Projection GEMM。
8. residual。
9. post-attention RMSNorm。
10. gate/up GEMM、SwiGLU、down GEMM。
11. residual。
12. 只对最后一个 prompt token 执行 final norm、tied LM Head 和 argmax。

### Decode

1. 单 token BF16 embedding。
2. 每层 BF16 RMSNorm 与 Q/K/V GEMV。
3. RoPE 和指定 position 的 KV 写入。
4. 复用 Prefill KV Cache 执行 decode attention。
5. O Projection、residual、MLP 和 residual。
6. final norm、tied LM Head、argmax。

## 自动化验证

新增三个端到端测试：

- `Qwen2BFloat16.PrefillMatchesFp32AcrossFixedPrompts`
- `Qwen2BFloat16.PrefillKvStateContinuesDecodeFor16Tokens`
- `Qwen2BFloat16.CustomMatchesCublasAndUsesBf16Storage`

覆盖：

- 5 个固定 prompt。
- Prefill 后继续 Decode 16 token。
- custom 与 cuBLAS。
- KV Cache 容量和 position 推进。
- finite logits。
- BF16 KV/workspace/archive 存储比例。
- top-1 与 logits cosine。

全量结果：

```text
25 tests passed, 0 failed
```

## 真实模型 logits 对照

使用 canonical 32-token IDs 的前缀长度 `1/2/8/16/32`。

| Prompt tokens | FP32 top-1 | W16A16 top-1 | top-1 | Cosine |
|---:|---:|---:|---|---:|
| 1 | 72030 | 72030 | 一致 | 0.982076 |
| 2 | 198 | 198 | 一致 | 0.999009 |
| 8 | 382 | 382 | 一致 | 0.999442 |
| 16 | 279 | 279 | 一致 | 0.999907 |
| 32 | 80285 | 80285 | 一致 | 0.999895 |

## 为什么调整 FP32 验收标准

官方发布的原始权重 dtype 就是 BF16。项目中的 FP32 archive 是把这些 BF16 权重数值扩展为 FP32，再使用 FP32 activation 计算。

因此：

- FP32 不是更真实的原始权重。
- FP32 与 W16A16 是两条 activation 精度不同的推理路径。
- BF16 每层写回会发生舍入，24 层后 logits 不必逐元素复制 FP32。
- FP32 greedy token 序列不应作为原生 BF16 必须完全复制的唯一真值。

经用户确认，本 checkpoint 的实际验收调整为：

- 5 个固定 prompt 首 token top-1 必须一致。
- logits 必须有限。
- 固定 prompt cosine 作为诊断，最低门槛调整为 0.98。
- Prefill 后连续 Decode 必须稳定。
- W16 custom 与 cuBLAS 必须得到一致的 greedy 行为。
- 与 FP32 的 near-tie 分叉必须公开记录，不能隐藏。

## Canonical 16-token 分叉分析

W16 custom 与 W16 cuBLAS 得到完全相同的 16-token 序列，但 FP32 在第 7 个生成 token 出现分叉。

共同前 6 token：

```text
80285,12,4086,44,320,34253
```

分叉位置：

- FP32 选择 `4903`。
- W16A16 选择 `11434`。

FP32 top-2：

- token `4903`：17.385515
- token `11434`：17.335749
- margin：0.049767

W16A16：

- token `11434`：17.625
- token `4903`：17.25

这是一处接近并列的候选在 BF16 逐层舍入后发生排序翻转。代表性 embedding、Q weight、K bias、MLP down weight 和 final norm 已验证：FP32 archive 与 BF16 解码值逐元素完全一致，因此不是导出错误。

## 内存

真实模型、`max-seq-len=32` 的 W16A16 smoke：

- model archive：989,114,112 bytes（943.29 MiB）
- device weights：988,065,536 bytes（942.29 MiB）
- workspace：1,579,904 bytes（1.51 MiB）
- KV Cache：393,216 bytes（0.38 MiB）

既有 FP32 device weights 为 1,976,131,072 bytes，因此 W16A16 device weights 正好为 FP32 的 50%。

自动化测试同时验证：

- BF16 KV Cache 为 FP32 的 50%。
- BF16 workspace 不超过对应 FP32 workspace 的 55%。
- BF16 archive payload 为 FP32 的 50%。

## CLI

W16A16 默认使用 CP11 选出的 cuBLAS Linear：

```bash
./build/llm_infer benchmark \
  --model models/qwen2.5-0.5b-instruct \
  --backend cuda \
  --precision w16a16 \
  --linear-kernel cublas \
  --token-ids 151644,8948,... \
  --max-new-tokens 16
```

`generate` 也支持确定性的 `--token-ids`，用于直接比较 greedy token 序列。

## 本步没有声称的内容

- 本步的真实模型运行只是功能与正确性 smoke，不是正式性能数据。
- 没有报告 W16A16 相对 FP32 的正式 TTFT/decode 加速。
- 没有执行 5/20 canonical benchmark、prompt sweep、功耗采集或 Nsight。
- 上述性能工作属于 CP13。
