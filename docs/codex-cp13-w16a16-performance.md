# Codex Checkpoint 13：CUDA W16A16（原生 BF16）正式性能报告

## 结论

Checkpoint 13 已完成。在 Jetson Orin Nano Super 8GB、MAXN_SUPER、CUDA 12.6 上，W16A16（BF16）相对当前 CUDA FP32 矩阵化 Prefill 基线同时改善了 Prefill、Decode、内存和能效。

最重要的绝对性能数据如下，而不是只报告倍数：

- W16A16 32-token Prefill：`1893.93 token/s`，TTFT `16.896 ms`。
- W16A16 Decode：`67.59 token/s`，TPOT `14.796 ms/token`。
- FP32 32-token Prefill：`790.91 token/s`，TTFT `40.460 ms`。
- FP32 Decode：`40.97 token/s`，TPOT `24.407 ms/token`。
- W16A16 device weights：`988,065,536 bytes`，恰好是 FP32 的 50%。
- W16A16 canonical 能耗：`0.2619 J/generated token`；FP32 为 `0.4459 J/generated token`。

性能门槛通过：W16A16 Decode 快于 FP32，32/128/512-token Prefill 均快于 FP32，正式 Decode CV 远低于 3%。

这里的 W16 和 A16 均为 **BF16**，不是 FP16。

## 固定测试协议

- 设备：Jetson Orin Nano Super 8GB。
- 功耗模式：MAXN_SUPER。
- CUDA：12.6。
- 模型：Qwen2.5-0.5B-Instruct。
- 代码 commit：`b628aceedaeddb86625489968b36c707ea00efb6`。
- W16A16 archive SHA-256：`f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`。
- FP32 archive SHA-256：`a58918c8fcbe0332c3a09d233f14425d1ecf32facca658b92203ff89d81c0b63`。
- 后端：CUDA，cuBLAS Linear。
- Prefill：矩阵化 Prefill；项目已按此前决定移除伪 serial Prefill。
- 解码：batch=1、greedy argmax、固定 token IDs、不开 early stop。
- 正式协议：5 次 warmup、20 次 measurement。
- canonical workload：32-token prompt、128 generated tokens、`max-seq-len=256`。
- prompt sweep：prompt 长度 1/32/128/512、1 generated token、`max-seq-len=512`。
- telemetry：100 ms 采样。
- 时钟：外层 ClockGuard 锁定 `jetson_clocks`，退出时通过 trap 恢复；所有临时 state 均已确认删除。

`max-new-tokens=1` 的 sweep 只用于测量 Prefill，因此没有有意义的 Decode token/s；canonical 32→128 才用于正式 Decode 测量。

## Canonical 32→128 绝对性能

| 指标 | CUDA FP32 | CUDA W16A16 BF16 | W16A16 改善 |
|---|---:|---:|---:|
| Prefill throughput | 790.91 token/s | 1893.93 token/s | 2.395x |
| TTFT | 40.460 ms | 16.896 ms | 降低 58.24% |
| Decode throughput | 40.97 token/s | 67.59 token/s | 1.650x |
| TPOT | 24.407 ms/token | 14.796 ms/token | 降低 39.38% |
| Total latency | 3140.174 ms | 1895.964 ms | 降低 39.62% |
| Decode CV | 0.0188% | 0.0104% | 均通过 ≤3% |

均值之外的 W16A16 Decode p50 为 `67.585 token/s`；FP32 p50 为 `40.972 token/s`。W16A16 Prefill p50 为 `1893.66 token/s`，p95 为 `1894.90 token/s`。

## Prompt sweep：Prefill 吞吐率与 TTFT

| Prompt tokens | FP32 Prefill token/s | W16A16 Prefill token/s | FP32 TTFT | W16A16 TTFT | Prefill speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 41.73 | 69.84 | 23.963 ms | 14.319 ms | 1.673x |
| 32 | 792.38 | 1895.33 | 40.385 ms | 16.884 ms | 2.392x |
| 128 | 1097.83 | 2773.58 | 116.593 ms | 46.150 ms | 2.526x |
| 512 | 694.13 | 1124.21 | 737.609 ms | 455.431 ms | 1.620x |

四组 W16A16 TTFT CV 分别为 `0.0649% / 0.0493% / 0.0292% / 0.0164%`；FP32 分别为 `0.0378% / 0.0294% / 0.0175% / 0.0358%`。

吞吐率从 1 token 到 128 tokens 上升，是因为更大的矩阵提高了 GEMM 并行利用率；到 512 tokens 时下降，是因为 causal attention 的计算量随 prompt 长度近似二次增长。这不是 Decode 变慢，也不是 Prefill 退化，而是 Prefill 的形状收益与 attention 复杂度共同作用。

## 内存

Canonical 使用 `max-seq-len=256`：

| 指标 | CUDA FP32 | CUDA W16A16 BF16 | W16A16/FP32 |
|---|---:|---:|---:|
| Model archive | 1,977,179,648 B | 989,114,112 B | 50.03% |
| Device weights | 1,976,131,072 B | 988,065,536 B | 50.00% |
| KV Cache | 6,291,456 B | 3,145,728 B | 50.00% |
| Workspace | 20,478,976 B | 10,240,000 B | 50.00% |
| Peak RSS | 4131.14 MiB | 2349.23 MiB | 56.87% |

W16A16 peak RSS 比 FP32 降低 43.13%，为后续更长上下文或更大 workspace 留出更多设备内存余量。

## 功耗与能效

Canonical measurement window 内：

| 指标 | CUDA FP32 | CUDA W16A16 BF16 | 变化 |
|---|---:|---:|---:|
| 平均 VDD_IN | 18.202 W | 17.709 W | 降低 2.71% |
| Measurement energy | 1141.378 J | 670.337 J | 降低 41.27% |
| Energy/generated token | 0.4459 J | 0.2619 J | 降低 41.27% |
| 最高 GPU 温度 | 69.0 °C | 65.25 °C | -3.75 °C |

Canonical 的 telemetry 分别包含 FP32 599 个样本、W16A16 362 个样本，因此用它作为正式功耗和能效结论。Prompt sweep 中短 prompt 的采样窗口过短，功耗仅作为原始诊断数据保留，不作为主要结论。

## Nsight Systems 证据

短 trace：32-token prompt、8 generated tokens、0 warmup、1 measurement。这个 trace 用于结构分析，不用于正式吞吐率，因为首次 cuBLAS 初始化和 profiler 开销会放大时间。

文件：`profiles/codex-cp13-cuda-w16a16-p32-short.nsys-rep`。

### Prefill

- `qwen2.prefill.matrixized` 内 GPU kernels 合计 `16.872 ms`，460 个实例。
- 两类 BF16 GEMM kernels 合计 `12.791 ms`、169 个实例，占 Prefill GPU kernel 时间约 75.8%。
- causal Prefill attention：`2.014 ms`、24 个实例。
- argmax：`0.427 ms`、1 个实例，证明 LM Head/argmax 只针对最后一个 prompt token。

### Decode

- `qwen2.decode` 内 GPU kernels 合计 `98.072 ms`，对应首 token 之后的 7 个 Decode steps。
- cuBLAS 选择的 BF16 GEMM/GEMV 微内核合计 `85.428 ms`，占 Decode GPU kernel 时间约 87.1%，线性层仍是 Decode 主瓶颈。
- Decode attention 合计 `2.540 ms`。
- argmax 合计 `2.999 ms`。

### 分配与传输

- trace 中 9 次 `cudaMalloc` 全部发生在模型初始化阶段。
- `qwen2.prefill` 内 `cudaMalloc=0`；`qwen2.decode` 内 `cudaMalloc=0`。
- Prefill 只有一次 128-byte H2D prompt IDs 和一次 4-byte D2H argmax。
- 7 个 Decode steps 的 H2D 为 0；仅有每步一次 4-byte D2H argmax，共 28 bytes。

因此运行期没有逐 token `cudaMalloc`，也没有逐 token H2D token 搬运。

## 验收

- W16A16 Decode `67.59 token/s` > FP32 `40.97 token/s`：通过。
- W16A16 在 32/128/512 prompt 下 Prefill 均快于 FP32：通过。
- W16A16 canonical Decode CV `0.0104%` ≤3%：通过。
- W16A16 device weights 为 FP32 的 50%，≤55%：通过。
- Nsight 证明 Prefill 使用 BF16 GEMM、LM Head/argmax 只执行一次：通过。
- Prefill/Decode 区间无 `cudaMalloc`、Decode 无 H2D：通过。
- Release 回归测试：25/25 通过。

## 归档

审核后的轻量汇总：

- `benchmarks/results/codex-cp13-w16a16-performance-summary.json`

忽略版本控制的原始结果：

- `results/codex-cp13-w16a16-canonical-5x20.json`
- `results/codex-cp13-fp32-canonical-5x20.json`
- `results/codex-cp13-w16a16-p1-5x20.json`
- `results/codex-cp13-w16a16-p32-5x20.json`
- `results/codex-cp13-w16a16-p128-5x20.json`
- `results/codex-cp13-w16a16-p512-5x20.json`
- `results/codex-cp13-fp32-p1-5x20.json`
- `results/codex-cp13-fp32-p32-5x20.json`
- `results/codex-cp13-fp32-p128-5x20.json`
- `results/codex-cp13-fp32-p512-5x20.json`

本 checkpoint 到此停止。下一 checkpoint 是 W8A16 导出，不在本次提交中启动。
