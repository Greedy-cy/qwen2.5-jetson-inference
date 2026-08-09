# Jetson 最终性能与 Profiling 报告

## 1. 结论

在 Jetson Orin Nano Super 8GB 上，项目完成 CPU FP32 reference、CUDA BF16 cuBLAS、
CUDA BF16 custom 和 CUDA W8A16 custom 四条最终路径的统一测试。

- BF16 custom：1647.29 Prefill tok/s、68.19 Decode tok/s；Decode 为 cuBLAS 的
  100.72%。
- W8A16 custom：1193.53 Prefill tok/s、80.25 Decode tok/s；Decode 相对 BF16
  custom 提高 17.68%。
- W8A16 device weights 从 942.29 MiB 降至 611.71 MiB（-35.08%），canonical
  energy/token 从 0.2679 J 降至 0.2167 J（-19.09%）。
- W8A16 Prefill 没有快于 BF16，瓶颈是 fused dequantize GEMM；因此只把 W8 定位为
  Decode、权重体积和能效优化。

## 2. 测试条件

| 项目 | 配置 |
|---|---|
| 设备 | Jetson Orin Nano Super 8GB，SM87 |
| 系统 | Jetson Linux R36.4.7，aarch64 |
| CUDA | 12.6，nvcc 12.6.68 |
| 功耗模式 | MAXN_SUPER |
| 模型 | Qwen2.5-0.5B-Instruct |
| CPU | OpenBLAS/OMP 6 threads，`taskset -c 0-5` |
| 构建 | CMake Release，独立 clean build |
| 锁频 | GPU 1020 MHz、EMC 3199 MHz，并固定 CPU；结束后恢复 |
| 遥测 | tegrastats 100 ms |

正式性能数据绑定提交 `fddc85f86d586e646746e3949516fb582c4966c0`，
`git_dirty=false`。最终精简提交 `d44a0af23e26075b1d716d3eeecf10966f72ad75`
只删除公开范围外路径、收敛 CLI 并增加非法组合测试，没有改变 BF16/W8 kernel；该提交
在独立 Release 目录通过 29/29 CTest 和 CLI smoke。

模型归档 SHA-256：

| 精度 | SHA-256 |
|---|---|
| FP32 | `a58918c8fcbe0332c3a09d233f14425d1ecf32facca658b92203ff89d81c0b63` |
| BF16 | `f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6` |
| W8A16 | `730d0ca9a60b55a63534c8edce19353d8f55fda07e6e5fc04ec88470bcf57814` |

统一协议：

| 测试 | Workload | Warmup / Repeat |
|---|---|---:|
| Canonical | 32 prompt + 128 output，max-seq-len 256 | 5 / 20 |
| Prefill sweep | 1/32/128/512 prompt + 1 output，max-seq-len 640 | 5 / 20 |
| Decode context sweep | 32/128/512 prompt + 128 output，max-seq-len 640 | 5 / 20 |
| Linear microbenchmark | 七个 Transformer Linear + tied LM Head | 20 / 100 |
| Nsight Systems | 32 prompt + 16 output | 1 / 3 |

Prompt 使用固定 token IDs。TTFT 包含矩阵化 Prefill、最后一行 LM Head 和首 token
选择；TPOT/Decode 只统计之后的自回归 token。

## 3. Canonical 端到端性能

| 路径 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | TPOT (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|
| CPU FP32 reference | 353.36 | 90.56 | 13.51 | 74.00 | 9751.09 |
| CUDA BF16 cuBLAS | 16.79 | 1906.25 | 67.70 | 14.77 | 1892.62 |
| CUDA BF16 custom | 19.43 | 1647.29 | 68.19 | 14.66 | 1881.82 |
| CUDA W8A16 custom | 26.81 | 1193.53 | 80.25 | 12.46 | 1609.43 |

| 路径 | TTFT p50 / p95 (ms) | Decode p50 / p95 (tok/s) | Decode CV |
|---|---:|---:|---:|
| CPU FP32 | 353.00 / 355.64 | 13.52 / 13.53 | 0.162% |
| BF16 cuBLAS | 16.79 / 16.80 | 67.71 / 67.72 | 0.025% |
| BF16 custom | 19.42 / 19.49 | 68.19 / 68.22 | 0.034% |
| W8A16 custom | 26.81 / 26.87 | 80.25 / 80.29 | 0.041% |

全部正式 Decode 测试的最大 CV 为 0.218%，低于 3% 重跑门槛。

## 4. Prefill prompt-length sweep

单元格为 `Prefill tok/s (TTFT ms)`。

| 路径 | p1 | p32 | p128 | p512 |
|---|---:|---:|---:|---:|
| CPU FP32 | 7.96 (125.71) | 90.32 (354.32) | 113.86 (1124.21) | 77.08 (6642.54) |
| BF16 cuBLAS | 69.78 (14.33) | 1904.72 (16.80) | 2778.62 (46.07) | 1123.96 (455.53) |
| BF16 custom | 14.24 (70.24) | 1652.13 (19.37) | 2226.00 (57.50) | 1047.50 (488.78) |
| W8A16 custom | 33.67 (29.70) | 1194.15 (26.80) | 1884.44 (67.92) | 984.68 (519.96) |

BF16 custom 相对 cuBLAS 在 p32/p128/p512 为 86.74%/80.11%/93.20%；W8A16
相对 BF16 custom 为 72.28%/84.66%/94.00%。p1 主要反映固定 launch、LM Head
和小矩阵利用率，不代表批量 Prefill 稳态性能。

对“哪种 Prefill 尺寸最合适”的解释：若看绝对吞吐，当前 BF16/W8 都在 p128 最高；
p512 因 causal attention 工作量按序列长度增长而回落。若看 custom 对 baseline 的效率，
p512 更接近 cuBLAS，但 TTFT 已显著增加。交互式应用更适合 p32--p128；p512 是长上下文
覆盖与效率验证，不是最低 TTFT 点。

## 5. Decode context sweep

单元格为 `Decode tok/s (TPOT ms)`。

| 路径 | context 32 | context 128 | context 512 |
|---|---:|---:|---:|
| CPU FP32 | 13.55 (73.82) | 13.04 (76.68) | 11.51 (86.86) |
| BF16 cuBLAS | 67.66 (14.78) | 65.71 (15.22) | 59.08 (16.93) |
| BF16 custom | 68.10 (14.68) | 66.19 (15.11) | 59.43 (16.83) |
| W8A16 custom | 80.29 (12.46) | 77.53 (12.90) | 68.58 (14.58) |

W8A16 在 context 32/128/512 下相对 BF16 custom 快 17.9%/17.1%/15.4%。上下文
增长会增加 attention/KV 访问，但量化 GEMV 的权重带宽优势保持稳定。

## 6. 内存、功耗与能效

| 路径 | Archive (MiB) | Device weights (MiB) | Workspace (MiB) | KV Cache (MiB) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| CPU FP32 | 1885.59 | 0.00 | 19.53 | 6.00 | 2071.80 |
| BF16 cuBLAS | 943.29 | 942.29 | 9.91 | 3.00 | 2353.86 |
| BF16 custom | 943.29 | 942.29 | 9.91 | 3.00 | 2163.20 |
| W8A16 custom | 612.71 | 611.71 | 9.91 | 3.00 | 1496.17 |

| 路径 | Mean power (W) | Max power (W) | Energy/generated token (J) |
|---|---:|---:|---:|
| CPU FP32 | 13.234 | 13.343 | 1.0077 |
| BF16 cuBLAS | 17.828 | 18.026 | 0.2629 |
| BF16 custom | 18.273 | 18.387 | 0.2679 |
| W8A16 custom | 17.298 | 17.411 | 0.2167 |

GPU 的瞬时功率高于 CPU，但完成时间短，因此 energy/token 更低。W8A16 同时减少
权重流量和总延迟，是四条路径中能效最高的一条。

## 7. Linear 微基准与原因分析

下表为一层七个 Transformer Linear 的 mean kernel time 合计；speedup 大于 1 表示
custom 更快。

| 精度 | Tokens | GEMV custom/ref (ms) | GEMV speedup | GEMM custom/ref (ms) | GEMM speedup |
|---|---:|---:|---:|---:|---:|
| BF16 | 1 | 0.3779 / 0.4453 | 1.18x | 2.3261 / 0.4449 | 0.19x |
| BF16 | 32 | 0.3743 / 0.4537 | 1.21x | 0.5049 / 0.4431 | 0.88x |
| BF16 | 128 | 0.3754 / 0.4478 | 1.19x | 1.0169 / 0.5899 | 0.58x |
| BF16 | 512 | 0.3777 / 0.4477 | 1.19x | 3.2691 / 2.0695 | 0.63x |
| W8A16 | 1 | 0.2896 / 0.4448 | 1.54x | 1.0414 / 0.4455 | 0.43x |
| W8A16 | 32 | 0.2890 / 0.4476 | 1.55x | 0.7986 / 0.4429 | 0.55x |
| W8A16 | 128 | 0.2887 / 0.4452 | 1.54x | 1.4600 / 0.5961 | 0.41x |
| W8A16 | 512 | 0.2892 / 0.4466 | 1.54x | 4.5743 / 2.0410 | 0.45x |

W8 GEMV 达到 1.54--1.55x，解释了 Decode 加速；W8 fused dequant GEMM 只有
reference 的 0.41--0.55x，解释了 Prefill 下降。专用 tied LM Head 在 p32 为
3.2811 ms，cuBLAS + argmax 为 3.1745 ms，达到 96.75%。

## 8. Nsight Systems

| 路径 | Prefill NVTX median (ms) | Decode NVTX median (ms) | CUDA kernel 时间主要构成 |
|---|---:|---:|---|
| BF16 cuBLAS | 17.173 | 226.231 | BF16 GEMM family 72.1% |
| BF16 custom | 19.837 | 223.879 | GEMV 60.6%，LM Head 25.2%，GEMM 5.2% |
| W8A16 custom | 27.186 | 191.183 | W8 GEMV 52.4%，LM Head 28.3%，W8 GEMM 9.4% |

W8 的 Decode NVTX 最短。W8 主干变快后，共享 BF16 tied LM Head 占比升至 28.3%；
Prefill 下一候选瓶颈是 W8 fused dequant GEMM，而非已达到 1.54x 的 GEMV。

## 9. 正确性与限制

- BF16 custom 对 cuBLAS：5 个固定 prompt top-1 为 5/5；canonical 16-token greedy
  完全一致；fused LM Head cosine 0.999999999，argmax 一致；
- W8A16 对 BF16：5 个固定 prompt 首 token top-1 为 5/5；KV Cache 连续 Decode
  测试通过；无完整 BF16 反量化权重副本；
- W8 最大相对输出误差为 0.006993，小于 0.007 门槛；
- W8 canonical 16-token greedy 为 14/16。两个边界 token 差异来自在线反量化到 BF16
  后的舍入误差经 24 层传播，不能声称逐 token 与 BF16 完全一致。

审核后的机器可读摘要位于
`benchmarks/results/final-jetson-performance.json`。原始逐轮 JSON 和 Nsight trace 保留在
Git 忽略的 `results/`，不进入最终展示树。
