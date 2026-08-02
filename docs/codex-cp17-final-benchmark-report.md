# CODEX-CP17：最终集中性能测试报告

## 结论摘要

本检查点在 Jetson Orin Nano Super 8GB 上，使用同一提交、同一模型、固定时钟和
相同 5/20 协议，完成 CPU FP32、CUDA FP32、CUDA W16A16（BF16）和 CUDA W8A16
四条最终路径的正式测量。

- 综合性能最优路径是 CUDA W16A16（BF16）：32-token Prefill 为
  `1901.34 tok/s`，Decode 为 `67.61 tok/s`。
- CUDA FP32 Decode 比 CPU FP32 快 `1.70x`，但当前自研 FP32 Prefill 在该模型上
  只有 `64.66 tok/s`，低于 CPU FP32 的 `90.23 tok/s`。
- W8A16 device weights 仅 `611.71 MiB`，比 W16A16 再下降 `35.08%`；但它的
  Prefill 只有 `42.16 tok/s`，Decode 只有 `33.57 tok/s`，没有达到快于 W16A16
  的性能目标。
- Nsight Systems 显示 W8A16 的 `linear_w8a16_kernel` 占 GPU kernel 时间
  `94.3%`，下一步应优化 W8A16 Linear，而不是 attention 或 argmax。

CP17 的数据采集目标已完成，但 W8A16 性能验收门槛未通过。该负面结果作为真实
工程结论保留，不进入未经验证的“W8A16 更快”简历表述。

## 测试身份

- 数据提交：`f7d44b948b93c5637fc6d06f533d963311427614`
- 模型：`Qwen2.5-0.5B-Instruct`
- 设备：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super，8GB
- Jetson Linux：R36.4.7
- CUDA：12.6，SM87
- 功耗模式：MAXN_SUPER
- 时钟：每组测试前由 `jetson_clocks` 锁定，`finally`/trap 恢复
- Prefill：仅矩阵化 Prefill，无 serial/伪 Prefill 路径

模型归档 SHA-256：

| Archive | SHA-256 |
|---|---|
| FP32 | `a58918c8fcbe0332c3a09d233f14425d1ecf32facca658b92203ff89d81c0b63` |
| W16A16 BF16 | `f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6` |
| W8A16 | `730d0ca9a60b55a63534c8edce19353d8f55fda07e6e5fc04ec88470bcf57814` |

## 正式协议

Canonical workload：

- batch size：1
- 固定 32-token prompt
- 128-token greedy 输出
- `max-seq-len=256`
- 5 次 warmup，20 次正式测量
- tegrastats 100 ms 采样
- CPU：OpenBLAS/OMP 6 threads，`taskset -c 0-5`
- CUDA W16A16/W8A16：cuBLAS linear 路径

Prompt sweep：

- prompt 长度：`1/32/128/512`
- 128/512 prompt 由 canonical 32-token 序列重复 4/16 次
- 每轮只生成首 token，用于隔离 TTFT/Prefill
- `max-seq-len=512`
- 每个点同样采用 5 次 warmup、20 次正式测量

runner 在计算归档 SHA-256 后调用 `POSIX_FADV_DONTNEED`，避免哈希读取产生的
page cache 与 CUDA device weights 在 Jetson 统一内存中竞争。该步骤位于计时窗口外。

## Canonical 端到端结果

| Path | TTFT mean | TTFT p50/p95 | Prefill tok/s | Decode tok/s | TPOT | Total latency |
|---|---:|---:|---:|---:|---:|---:|
| CPU FP32 | 354.67 ms | 353.93 / 358.76 ms | 90.23 | 13.42 | 74.50 ms | 9816.21 ms |
| CUDA FP32 | 494.87 ms | 494.87 / 494.96 ms | 64.66 | 22.86 | 43.74 ms | 6049.34 ms |
| CUDA W16A16 BF16 | 16.83 ms | 16.83 / 16.85 ms | 1901.34 | 67.61 | 14.79 ms | 1895.22 ms |
| CUDA W8A16 | 758.96 ms | 758.96 / 759.04 ms | 42.16 | 33.57 | 29.79 ms | 4541.76 ms |

稳定性：

| Path | TTFT CV | Decode tok/s CV | 结果 |
|---|---:|---:|---|
| CPU FP32 | 0.552% | 0.225% | 通过 |
| CUDA FP32 | 0.012% | 0.012% | 通过 |
| CUDA W16A16 BF16 | 0.058% | 0.009% | 通过 |
| CUDA W8A16 | 0.010% | 0.012% | 通过 |

全部 Decode CV 均远低于 `3%` 门槛。

## 内存、功耗和能耗

| Path | Archive | Device weights | Peak RSS | Workspace / KV | Mean / max power | Energy/output token |
|---|---:|---:|---:|---:|---:|---:|
| CPU FP32 | 1885.59 MiB | 0 | 2077.94 MiB | 19.53 / 6.00 MiB | 13.11 / 13.22 W | 1.005 J |
| CUDA FP32 | 1885.59 MiB | 1884.59 MiB | 3922.09 MiB | 19.53 / 6.00 MiB | 14.46 / 14.85 W | 0.682 J |
| CUDA W16A16 BF16 | 943.29 MiB | 942.29 MiB | 2351.48 MiB | 9.77 / 3.00 MiB | 18.07 / 18.27 W | 0.267 J |
| CUDA W8A16 | 612.71 MiB | 611.71 MiB | 1672.14 MiB | 9.77 / 3.00 MiB | 14.73 / 14.83 W | 0.522 J |

W16A16 功率最高，但因为完成得快，单位输出 token 能耗最低。W8A16 比 CUDA FP32
节能 `23.50%`，但仍比 W16A16 多消耗约 `95%` 能量/token。

## Prompt sweep

TTFT mean：

| Prompt | CPU FP32 | CUDA FP32 | W16A16 BF16 | W8A16 |
|---:|---:|---:|---:|---:|
| 1 | 130.59 ms | 254.26 ms | 14.32 ms | 29.29 ms |
| 32 | 361.91 ms | 494.70 ms | 16.82 ms | 758.72 ms |
| 128 | 1136.51 ms | 1950.67 ms | 46.06 ms | 3036.16 ms |
| 512 | 6666.62 ms | 8043.98 ms | 455.48 ms | 12424.90 ms |

Prefill throughput：

| Prompt | CPU FP32 | CUDA FP32 | W16A16 BF16 | W8A16 |
|---:|---:|---:|---:|---:|
| 1 | 7.66 tok/s | 3.93 tok/s | 69.84 tok/s | 34.14 tok/s |
| 32 | 88.42 tok/s | 64.69 tok/s | 1902.66 tok/s | 42.18 tok/s |
| 128 | 112.63 tok/s | 65.62 tok/s | 2779.02 tok/s | 42.16 tok/s |
| 512 | 76.80 tok/s | 63.65 tok/s | 1124.09 tok/s | 41.21 tok/s |

所有 sweep TTFT CV 均不超过 `2.20%`。短 p1 W16A16 测量窗口只有约 0.29 秒，
因此其 tegrastats 样本较少；TTFT 使用程序内部计时，不受这一限制。Sweep 的
energy/generated-token 被完整 Prefill 成本主导，不与 canonical Decode 能耗横向比较。

## 关键倍率

- CUDA FP32 相对 CPU FP32：Decode `1.70x`，总延迟 `1.62x`；但 32-token
  Prefill 只有 CPU 的 `0.717x`。
- W16A16 相对 CUDA FP32：Prefill `29.40x`，Decode `2.96x`，TTFT 降低
  `96.60%`，单位输出 token 能耗降低 `60.84%`。
- W8A16 相对 CUDA FP32：Decode `1.47x`，但 Prefill 只有 `0.652x`；device
  weights 减少 `67.54%`，单位输出 token 能耗降低 `23.50%`。
- W8A16 相对 W16A16：Decode 只有 `0.497x`，Prefill 只有 `0.0222x`，TTFT
  高 `45.09x`；device weights 则进一步减少 `35.08%`。

## Nsight Systems 结论

Trace workload：32-token prompt、16-token 输出、1 次 warmup、3 次测量；trace 包含
初始化，所以 CUDA API 汇总不作为端到端性能数字，只用 GPU kernel 与 NVTX 定位瓶颈。

| Path | 主要 GPU kernel | GPU kernel time | NVTX Prefill median | NVTX Decode token median |
|---|---|---:|---:|---:|
| CUDA FP32 | custom `gemv_fp32_kernel` | 47.8% | 494.91 ms | 44.85 ms |
| CUDA FP32 | custom `gemm_tiled_kernel` | 41.9% | 同上 | 同上 |
| W16A16 BF16 | Ampere BF16 GEMM kernels | 82.9% | 17.18 ms | 15.05 ms |
| W8A16 | custom `linear_w8a16_kernel` | 94.3% | 795.47 ms | 30.90 ms |

三份 trace 全程只有 9 次 `cudaMalloc`，均为模型初始化阶段固定分配，没有逐 token
`cudaMalloc`。每个 Prefill 只出现一次对应 NVTX range，后续 Decode 使用独立
`decode_next/decode_token` range，证明矩阵化 Prefill 与 KV Cache 衔接保持成立。

原始 trace 位于被 Git 忽略的：

- `results/codex-cp17-nsys-cuda-fp32.nsys-rep`
- `results/codex-cp17-nsys-cuda-w16a16-bf16.nsys-rep`
- `results/codex-cp17-nsys-cuda-w8a16.nsys-rep`

## 验收结论与下一步

通过：

- 4 条路径 canonical 数据完整，所有 CV 门槛通过。
- 16 个 prompt sweep 点完整，全部使用矩阵化 Prefill。
- W16A16 Decode 快于 CUDA FP32，且 W16A16 Prefill 在所有长度显著更快。
- W8A16 模型、device weights、Peak RSS 均进一步下降。

未通过：

- W8A16 Decode 没有快于 W16A16。
- W8A16 Prefill 在 32/128/512 tokens 下显著慢于 W16A16，甚至慢于当前 CUDA FP32。

因此 CP17 测量工作完成，但 W8A16 性能目标失败。CP18 应选择 Nsight 总耗时最高的
`linear_w8a16_kernel`，先分别优化 Prefill GEMM 与 Decode GEMV 的反量化、向量化加载、
scale 复用和 reduction；在该 kernel 改善前，不继续包装“W8A16 更快”的简历结论。
