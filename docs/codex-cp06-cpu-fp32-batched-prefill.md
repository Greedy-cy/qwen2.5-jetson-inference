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

## 正式性能 A/B

这里的 serial 指旧框架的 prompt 路径：32 个 prompt token 逐个执行完整
`forward_token()`，包括 32 次 LM Head。batched 指新的矩阵化 Prefill：按层对 32 行
hidden 执行 GEMM，只对最后一行执行一次 LM Head。

两条路径在同一 commit、同一模型和同一设备上使用以下协议：

- canonical 32-token prompt、128-token output。
- `max-seq-len=256`、5 次 warmup、20 次正式测量。
- CPU 0-5，OpenBLAS/OMP 固定 6 线程。
- MAXN_SUPER，runner 临时锁频并在退出时恢复；测试后 governor 为 `schedutil`。
- Decode 统计后续 127 次 `decode_next()`，不把 TTFT 混入 Decode tok/s。

| 指标 | 旧 serial | 新 batched | 变化 |
|---|---:|---:|---:|
| TTFT mean | 2300.36 ms | 355.46 ms | `6.47x`，降低 84.55% |
| Prefill tok/s mean | 13.91 | 90.02 | `6.47x` |
| Decode tok/s mean | 13.483 | 13.456 | -0.20% |
| 32+128 总延迟 mean | 11719.80 ms | 9793.72 ms | `1.197x`，降低 16.43% |
| 能耗/generated token | 1.205 J | 1.008 J | 降低 16.39% |
| mean VDD_IN | 13.166 W | 13.170 W | +0.03% |
| 总 workspace | 0.65 MiB | 19.53 MiB | +18.875 MiB |
| peak RSS | 2054.97 MiB | 2070.84 MiB | +15.87 MiB |

serial/batched 的 TTFT CV 分别为 `0.154%` 和 `0.388%`；Decode CV 分别为
`0.104%` 和 `0.056%`，均明显低于 3% 门槛。Decode 吞吐只变化 -0.20%，证明本次
端到端收益来自 Prefill，而不是 Decode 路径变化。

TTFT 提升 6.47 倍不等于完整 32+128 生成提升 6.47 倍：长输出仍由 127 次串行 Decode
主导，因此总延迟改善为 16.43%。prompt 越长、输出越短，Prefill 优化对请求总延迟的
占比越高。

审核摘要位于
`benchmarks/results/codex-cp06-cpu-fp32-prefill-ab-summary.json`，原始结果位于忽略的
`results/codex-cp06-cpu-fp32-{serial,batched}-5x20.json`。Checkpoint 8 仍会补充
`1/32/128/512` prompt sweep。

## 下一步边界

下一 checkpoint 是 CUDA FP32/W8A32 矩阵化 Prefill：接入已有 CUDA GEMM，并实现
CUDA batch embedding、RMSNorm、RoPE、KV 写入和 causal attention。本阶段到此停止。

后续 A16 路线继续使用 BF16 权重/激活，不使用 FP16。
