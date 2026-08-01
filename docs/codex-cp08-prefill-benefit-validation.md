# CODEX-CP08：完整 Prefill 收益验证

## 结论

Checkpoint 8 通过。CPU FP32、CUDA FP32 custom/cuBLAS 和 CUDA W8A32 在 prompt 长度 32、128、512 下，矩阵化 Prefill 均明显快于 serial reference；28 组正式结果的 TTFT CV 全部小于 3%。

本 checkpoint 只验证 Prefill，没有进入 BF16/W16A16。后续 A16 仍使用 BF16。

## 正式测试协议

- 设备：Jetson Orin Nano Super 8GB，MAXN_SUPER，CUDA 12.6。
- commit：`d667fb113619ef133dea7f54085216799cf1ae72`。
- 模型：Qwen2.5-0.5B-Instruct。
- prompt 长度：`1/32/128/512`，由 canonical 32-token IDs 循环扩展。
- `max-seq-len=512`，只生成首 token，使测量窗口隔离完整 Prefill。
- 每组 5 次 warmup、20 次正式测量。
- 每组使用 `jetson_clocks` 锁频，退出路径恢复原状态。
- CPU：OpenBLAS/OMP 固定 6 线程，`taskset -c 0-5`。
- 遥测：100 ms tegrastats，记录 VDD_IN、RAM、GPU 频率、温度和能量。
- 测试后确认 governor 恢复为 `schedutil`、GPU 306 MHz idle、EMC `FreqOverride=0`、风扇恢复动态控制。

## TTFT 与 Prefill 吞吐

### CPU FP32

| Prompt | Serial TTFT | Batched TTFT | 加速比 | Serial tok/s | Batched tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 71.411 ms | 129.087 ms | 0.55x | 14.01 | 7.75 |
| 32 | 2309.908 ms | 355.420 ms | 6.50x | 13.85 | 90.04 |
| 128 | 9388.482 ms | 1133.086 ms | 8.29x | 13.63 | 112.97 |
| 512 | 40361.326 ms | 6647.901 ms | 6.07x | 12.69 | 77.02 |

### CUDA FP32

| Prompt | Serial TTFT | Custom batched | cuBLAS batched | Custom 加速 | cuBLAS 加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 43.263 ms | 254.250 ms | 24.000 ms | 0.17x | 1.80x |
| 32 | 1390.069 ms | 494.785 ms | 40.382 ms | 2.81x | 34.42x |
| 128 | 5582.747 ms | 1950.669 ms | 116.783 ms | 2.86x | 47.80x |
| 512 | 22761.997 ms | 8044.329 ms | 738.301 ms | 2.83x | 30.83x |

cuBLAS batched 的 Prefill 吞吐分别为 `41.67/792.43/1096.05/693.48 tok/s`。自研 tiled GEMM 已证明数据流正确且优于 serial，但与 cuBLAS 仍有约一个数量级差距，后续优化应基于 profiling，不应把自研 kernel 的存在等同于高性能。

### CUDA W8A32

| Prompt | Serial TTFT | Batched TTFT | 加速比 | Serial tok/s | Batched tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.803 ms | 30.320 ms | 0.98x | 33.55 | 32.98 |
| 32 | 958.802 ms | 700.469 ms | 1.37x | 33.37 | 45.68 |
| 128 | 3859.673 ms | 2794.245 ms | 1.38x | 33.16 | 45.81 |
| 512 | 15871.716 ms | 11447.732 ms | 1.39x | 32.26 | 44.73 |

W8A32 batched 收益稳定但较小。Nsight 显示 `gemm_int8x4_kernel` 占 batched Prefill GPU 时间的绝大部分；当前实现为每个 token/output row 使用一个 reduction block，缺少成熟 GEMM 的数据复用。这是后续 W8A16 kernel 设计必须避免的问题。

## 为什么 auto 在单 token 时使用 serial

显式 batched 在 prompt=1 时：

- CPU FP32 为 serial 的 0.55x。
- CUDA FP32 custom 为 serial 的 0.17x。
- CUDA W8A32 为 serial 的 0.98x。

因此 `auto` 的“长度 1 走 serial，长度至少 2 走 batched”策略符合当前实现。cuBLAS FP32 在长度 1 仍较快，但不能代表 CPU、custom 和 W8A32 的共同策略。

## 功耗、能耗与内存

下表为 prompt 32/128/512 的 batched 相对 serial 每请求能耗降低：

| 路径 | P32 | P128 | P512 |
|---|---:|---:|---:|
| CPU FP32 | 85.42% | 89.00% | 86.69% |
| CUDA FP32 custom | 75.24% | 75.69% | 75.41% |
| CUDA FP32 cuBLAS | 97.32% | 97.56% | 96.77% |
| CUDA W8A32 | 28.71% | 29.34% | 29.52% |

cuBLAS 在 P32/P128 的瞬时功耗从 serial 的约 `14.8 W` 上升到约 `16.7/17.6 W`，但完成时间大幅缩短，因此总能耗反而降低约 97%。瞬时功耗不能替代能耗分析。

统一 `max-seq-len=512` 时：

- serial workspace：`0.66 MiB`。
- batched workspace：`38.41 MiB`。
- 增量：`37.75 MiB`。
- 全部正式配置最高温度：`75.25 °C`。
- 正式 TTFT 最大 CV：`2.05%`。

## Nsight Systems 验证

短 trace 使用 32-token prompt 和 2-token 输出；以下统计只取 `qwen2.prefill` NVTX 范围，trace 时长受 profiler 和首次库加载影响，不作为正式性能数字。

| 指标 | Serial | Batched custom/W8A32 | Batched cuBLAS |
|---|---:|---:|---:|
| 各层 Linear | 5376 次 GEMV | 168 次 GEMM | 168 次 SGEMM |
| LM Head GEMV | 32 | 1 | 1 |
| Argmax | 32 | 1 | 1 |
| Attention launch | 768 | 24 | 24 |
| GPU kernel launch | 12416 | 460 | 556 |
| `cudaLaunchKernel` API 时间 | 156–159 ms | 4.71–5.20 ms | 13.94 ms |
| Prefill 内 `cudaMalloc/cudaFree` | 0/0 | 0/0 | 0/0 |
| Prompt H2D | 0 | 1 次 / 128 bytes | 1 次 / 128 bytes |
| Argmax D2H | 32 次 / 128 bytes | 1 次 / 4 bytes | 1 次 / 4 bytes |

cuBLAS 一次 GEMM 可能对应多个内部 kernel，因此其 GPU kernel launch 总数高于 custom，但仍远低于 serial。

KV Cache 衔接由 Checkpoint 7 的 required-length logits test 和 Prefill 后 16-token Decode test 验证；本次 Nsight trace 额外包含一次 `decode_next`，确认 batched Prefill 后进入单-token Decode 时间线。

## 验收

- prompt 32/128/512 的 CPU FP32 batched：通过。
- prompt 32/128/512 的 CUDA FP32 custom/cuBLAS batched：通过。
- prompt 32/128/512 的 CUDA W8A32 batched：通过。
- 全部 28 组 TTFT CV ≤3%：通过。
- 无逐 token `cudaMalloc` 或 prompt H2D：通过。
- LM Head 从 32 次降为 1 次：通过。
- KV Cache 与后续 Decode 正确衔接：通过。

## 数据归档

原始正式 JSON 位于忽略目录：

- `results/codex-cp08-*-5x20.json`，共 28 份，约 11 MiB。
- 每份 JSON 内含 commit、模型 SHA-256、完整命令、20 个 sample 和全部 tegrastats 样本。

Nsight 原始文件位于忽略目录：

- `profiles/codex-cp08-cuda-*.nsys-rep`，共 5 份。
- 对应 SQLite 与 `results/codex-cp08-*-nsys.json` 同样不纳入 Git。

审核摘要：

- `benchmarks/results/codex-cp08-prefill-sweep-summary.json`。

下一步在用户确认后进入 Checkpoint 9：增加 BF16 归档和类型系统。计划中的文件名仍沿用 `model.w16a16.qbin`，但 A16 的实际数据类型为 BF16。
