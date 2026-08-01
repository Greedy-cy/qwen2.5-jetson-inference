# CODEX-CP07：CUDA FP32/W8A32 矩阵化 Prefill

## 范围

本 checkpoint 将 Checkpoint 6 的完整矩阵化 Prefill 数据流接到 CUDA FP32 和现有 W8A32 路径。Decode 保持原单 token 路径不变；后续 A16 仍按已确认约定使用 BF16，本步骤未引入 FP16/BF16。

## 实现

- prompt token IDs 只进行一次 H2D 拷贝，device token buffer 按 `max-seq-len` 预分配。
- 新增 CUDA 批量 embedding、RMSNorm、RoPE、KV Cache 写入和 causal attention。
- causal attention 每个 query 只访问 `[0,t]`，在线 softmax，不分配完整 `[tokens,tokens]` 矩阵。
- Q/K/V/O 与 MLP 使用 GEMM；bias 按 token 广播。
- FP32 同时支持自研 tiled GEMM 和 `--cublas` 的 cuBLAS SGEMM。
- W8A32 使用现有 group-wise INT8 weight + FP32 activation GEMM。
- 整个 prompt 只对最后一行 hidden 运行最终 RMSNorm、LM Head 和 argmax。
- `auto` 在 CUDA prompt 长度大于等于 2 时选择 batched；显式 serial 路径继续作为 reference。
- CLI 正式名称改为 `w8a32`，旧 `int8` 仍是兼容输入别名。

## 正确性

Release 全量测试：18/18 通过。

- tiny FP32 CUDA prompt 长度 `1/2/31/32/127/128/512`：serial/batched top-1 一致，最终 logits cosine 不低于 `0.999999`。
- tiny FP32 CUDA：Prefill 后继续 Decode 16 token，token 与每步 logits 均满足阈值。
- 真实 Qwen2.5-0.5B FP32，32-token prompt：
  - custom batched 对 serial：cosine 约 `1.0`，top-1 均为 64，max abs `2.7895e-4`。
  - cuBLAS batched 对 serial：cosine约 `1.0`，top-1 均为 64，max abs `3.2902e-5`。
- 真实 W8A32，32-token prompt：serial/batched logits 完全一致，top-1 均为 64。
- 真实 FP32 的 16-token greedy 序列一致；真实 W8A32 chat prompt 在 EOS 前 10 个 token 完全一致。

## 32-token 非正式冒烟性能

工作负载：32-token token IDs，`max-seq-len=64`，只生成首 token，1 次 warmup、3 次测量。未锁频、未采集功耗，因此只用于验证实现，不作为最终性能结论。

| 路径 | TTFT mean | Prefill tok/s | TTFT CV | Workspace |
|---|---:|---:|---:|---:|
| CUDA FP32 serial | 1395.368 ms | 22.933 | 0.13% | 0.65 MiB |
| CUDA FP32 batched custom | 498.377 ms | 64.208 | 0.00% | 5.37 MiB |
| CUDA FP32 batched cuBLAS | 48.869 ms | 658.181 | 7.34% | 5.37 MiB |
| CUDA W8A32 serial | 960.324 ms | 33.322 | 0.05% | 0.65 MiB |
| CUDA W8A32 batched | 723.281 ms | 44.243 | 0.03% | 5.37 MiB |

短测加速比：

- FP32 custom batched / serial：`2.80x`。
- FP32 cuBLAS batched / serial：`28.55x`。
- FP32 cuBLAS / custom batched：`10.20x`。
- W8A32 batched / serial：`1.33x`。
- batched workspace 增量约 `4.72 MiB`（本次 `max-seq-len=64`）。

W8A32 batched 收益较小，是因为当前 INT8 GEMM 为每个 token/output row 启动一个 256-thread reduction block，尚未像成熟 GEMM 那样复用 activation 和 scale。该结果保留给 Checkpoint 8 的正式 profiling；本 checkpoint 不跨步优化。

## 数据

忽略目录中的原始 JSON：

- `results/codex-cp07-cuda-fp32-serial-smoke.json`
- `results/codex-cp07-cuda-fp32-custom-batched-smoke.json`
- `results/codex-cp07-cuda-fp32-cublas-batched-smoke.json`
- `results/codex-cp07-cuda-w8a32-serial-smoke.json`
- `results/codex-cp07-cuda-w8a32-batched-smoke.json`

受版本控制的审核摘要：

- `benchmarks/results/codex-cp07-cuda-batched-prefill-smoke-summary.json`

下一步 Checkpoint 8 将锁定 MAXN_SUPER/jetson_clocks，对 prompt 长度 `1/32/128/512` 做 serial/batched 正式评测、功耗采集与 Nsight 验证。
