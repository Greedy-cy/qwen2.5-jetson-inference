# CODEX-CP06｜CPU FP32 矩阵化 Prefill 实现报告

## 结论

Checkpoint 6 已实现 CPU FP32 的完整矩阵化 Prefill。prompt 不再逐 token 调用完整
`forward_token()`，而是以 `[tokens, hidden]` 数据流逐层处理；Linear 使用
OpenBLAS `cblas_sgemm`，最终 RMSNorm、LM Head 和 argmax 只处理最后一个 prompt token。

serial 路径仍然保留，作为正确性参考和后续性能 A/B 基线。本阶段没有修改 CUDA
FP32/W8A32 路径，也没有进入 BF16 A16 实现。

## 数据布局与逐层计算

令 prompt 长度为 `T`、hidden size 为 `H`、KV width 为 `K`、MLP intermediate
size 为 `I`：

- hidden、norm、Q、attention、residual workspace：`[T,H]`。
- K/V：`[T,K]`。
- gate/up/SwiGLU：`[T,I]`。
- 权重仍保持归档中的 `[out_features,in_features]` row-major 布局。
- Batched Linear 计算 `Y[T,O] = X[T,I] × W[O,I]^T`。

每层执行：

```text
batched RMSNorm
  -> Q/K/V SGEMM
  -> batched absolute-position RoPE
  -> write all prompt K/V into this layer's KV Cache
  -> per-query causal online softmax over KV[0..t]
  -> O projection SGEMM + residual
  -> batched RMSNorm
  -> gate/up SGEMM + SwiGLU
  -> down projection SGEMM + residual
```

全部层结束后，只读取最后一行 hidden，执行最终 RMSNorm、一次 LM Head GEMV 和一次
argmax。与旧 serial 路径相比，prompt 中间 token 不再重复执行 LM Head。

## Causal attention 与 workspace

实现没有分配 `[T,T]` attention matrix。每个 query 只访问 `[0,t]` 的 K/V，并复用
一条长度为 `max-seq-len` 的 score scratch 进行 softmax。

CPU FP32 的 batched/auto 模型在构造时按 `max-seq-len` 一次性预分配 Prefill
workspace：

```text
max_seq_len × (5H + 2K + 3I) × sizeof(float)
```

运行期间没有逐 token workspace 分配。显式 serial 模型不分配这块额外 workspace；
超过配置容量会在计算前报错。

## 模式选择

- `--prefill-mode serial`：始终使用原逐 token 参考路径。
- `--prefill-mode batched`：当前仅支持 CPU FP32。
- `--prefill-mode auto`：
  - CPU FP32 且 `T >= 2`：batched。
  - 单 token、CUDA FP32/W8A32 或其他尚未支持的路径：serial。
- CUDA 上显式请求 batched 会报错，不会静默回退。

benchmark JSON 继续分别记录请求的 `prefill_mode` 和实际
`effective_prefill_mode`。

## 正确性验证

Release 构建后的 17 个测试全部通过：

- `cblas_sgemm` batch Linear 与逐行 `cblas_sgemv` 对照通过，覆盖 bias 和非方形矩阵。
- prompt 长度覆盖 `1/2/31/32/127/128/512`。
- 各长度 batched 与 serial 的最终 top-1 一致，logits cosine 不低于 `0.999999`。
- Batched Prefill 后继续 Decode 16 token，生成 token 与 serial 完全一致，逐步 logits
  cosine 不低于 `0.999999`。
- 验证了 position、reset、容量、单 token auto 路由和 CUDA 路由隔离。

正式 `Qwen2.5-0.5B-Instruct` 验证：

- canonical 32-token prompt 的 top-1：serial 与 batched 均为 `80285`。
- 完整 151,936 维 logits cosine：`0.9999999999986753`。
- logits 最大绝对误差：`2.4318695e-5`。
- 5-token prompt 后继续 greedy Decode 16 token：两条路径的 16 个 token 完全一致。

## 非正式性能冒烟

以下数据只用于确认 GEMM 路径没有性能倒退，不能替代 Checkpoint 8 的正式 5/20
benchmark：

- canonical 32-token prompt、2-token output。
- CPU 0-5、OpenBLAS/OMP 6 线程。
- `max-seq-len=64`、warmup=0、repeat=1。
- 未执行正式锁频、温度稳定和 CV 检查。

| 模式 | TTFT | Prefill tok/s | 总 workspace |
|---|---:|---:|---:|
| serial | 2449.53 ms | 13.06 | 0.65 MiB |
| batched | 405.89 ms | 78.84 | 5.37 MiB |

单次冒烟中 TTFT 加速为 `6.03x`，额外 workspace 为 `4.72 MiB`。这些数字不得写入
最终简历或正式性能表；正式收益将在 Checkpoint 8 按 prompt sweep 和 5/20 协议测量。

## 下一步边界

下一 checkpoint 是 CUDA FP32/W8A32 矩阵化 Prefill：接入已有 CUDA GEMM，并实现
CUDA batch embedding、RMSNorm、RoPE、KV 写入和 causal attention。本阶段到此停止。

后续 A16 路线继续使用 BF16 权重/激活，不使用 FP16。
