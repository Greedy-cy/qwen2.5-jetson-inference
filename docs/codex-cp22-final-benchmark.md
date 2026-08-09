# CODEX-CP22-FINAL-BENCHMARK

## 结论

CP22 在同一个 clean commit 上完成 CPU FP32 reference、CUDA BF16 cuBLAS、CUDA
BF16 custom 与 CUDA W8A16 custom 四条路径的集中测试。Release clean build 与
32/32 CTest 通过，40 份正式 JSON 均已归档；所有端到端数据绑定提交
`fddc85f86d586e646746e3949516fb582c4966c0`，且 `git_dirty=false`。

Canonical workload（32-token prompt + 128-token output）主要结果：

- BF16 custom 达到 cuBLAS 的 `86.42%` Prefill 吞吐，并以 `68.19 tok/s` 达到
  cuBLAS Decode 的 `100.72%`。
- W8A16 custom 达到 `80.25 decode tok/s`，相对 BF16 custom 提升 `17.68%`；
  TPOT 从 `14.66 ms` 降至 `12.46 ms`。
- W8A16 device weights 为 `611.71 MiB`，相对 BF16 的 `942.29 MiB` 下降
  `35.08%`；canonical energy/token 从 `0.2679 J` 降至 `0.2167 J`，下降
  `19.09%`。
- W8A16 Prefill 并未快于 BF16：canonical 为 `1193.53 tok/s`，是 BF16 custom
  的 `72.45%`。因此项目应把 W8A16 定位为 Decode、权重体积与能效优化，而不能笼统
  声称所有阶段均加速。

## 测试条件与可复现性

| 项目 | 配置 |
|---|---|
| 设备 | Jetson Orin Nano Super 8GB，SM87 |
| 系统 | Jetson Linux R36.4.7，aarch64 |
| CUDA | 12.6，nvcc 12.6.68 |
| 功耗模式 | MAXN_SUPER |
| 模型 | Qwen2.5-0.5B-Instruct |
| CPU | OpenBLAS/OMP 6 threads，`taskset -c 0-5` |
| 构建 | CMake Release，独立目录 `/tmp/codex-cp22-build-20260809` |
| 锁频 | 测量期间 GPU 1020 MHz、EMC 3199 MHz，并固定 CPU；结束后已恢复 |
| 遥测 | tegrastats 100 ms |

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

Prompt 全部使用固定 token IDs；正式结果不依赖 tokenizer 或自然语言 prompt 的差异。
本阶段不把 CUDA FP32 纳入正式比较。

## Canonical 端到端性能

TTFT 包含矩阵化 Prefill、最后一行 LM Head 与首 token 选择；TPOT 只描述后续
Decode token。总延迟为一次 32+128 完整生成的平均值。

| 路径 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | TPOT (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|
| CPU FP32 reference | 353.36 | 90.56 | 13.51 | 74.00 | 9751.09 |
| CUDA BF16 cuBLAS | 16.79 | 1906.25 | 67.70 | 14.77 | 1892.62 |
| CUDA BF16 custom | 19.43 | 1647.29 | 68.19 | 14.66 | 1881.82 |
| CUDA W8A16 custom | 26.81 | 1193.53 | 80.25 | 12.46 | 1609.43 |

稳定性与分位数：

| 路径 | TTFT p50 / p95 (ms) | Decode p50 / p95 (tok/s) | Decode CV |
|---|---:|---:|---:|
| CPU FP32 reference | 353.00 / 355.64 | 13.52 / 13.53 | 0.162% |
| CUDA BF16 cuBLAS | 16.79 / 16.80 | 67.71 / 67.72 | 0.025% |
| CUDA BF16 custom | 19.42 / 19.49 | 68.19 / 68.22 | 0.034% |
| CUDA W8A16 custom | 26.81 / 26.87 | 80.25 / 80.29 | 0.041% |

全部正式 Decode 测试中的最大 CV 为 `0.218%`，显著低于 `3%` 重跑门槛。

## Prefill prompt-length sweep

单元格格式为 `Prefill tok/s (TTFT ms)`。

| 路径 | p1 | p32 | p128 | p512 |
|---|---:|---:|---:|---:|
| CPU FP32 | 7.96 (125.71) | 90.32 (354.32) | 113.86 (1124.21) | 77.08 (6642.54) |
| BF16 cuBLAS | 69.78 (14.33) | 1904.72 (16.80) | 2778.62 (46.07) | 1123.96 (455.53) |
| BF16 custom | 14.24 (70.24) | 1652.13 (19.37) | 2226.00 (57.50) | 1047.50 (488.78) |
| W8A16 custom | 33.67 (29.70) | 1194.15 (26.80) | 1884.44 (67.92) | 984.68 (519.96) |

BF16 custom 相对 cuBLAS 在 p32/p128/p512 分别达到 `86.74% / 80.11% /
93.20%`；W8A16 相对 BF16 custom 分别达到 `72.28% / 84.66% / 94.00%`。
两者均通过既定的长 prompt 门槛。p1 主要受固定 launch、LM Head 和小矩阵利用率
影响，不代表批量 Prefill 的稳态能力。

## Decode context sweep

单元格格式为 `Decode tok/s (TPOT ms)`。

| 路径 | context 32 | context 128 | context 512 |
|---|---:|---:|---:|
| CPU FP32 | 13.55 (73.82) | 13.04 (76.68) | 11.51 (86.86) |
| BF16 cuBLAS | 67.66 (14.78) | 65.71 (15.22) | 59.08 (16.93) |
| BF16 custom | 68.10 (14.68) | 66.19 (15.11) | 59.43 (16.83) |
| W8A16 custom | 80.29 (12.46) | 77.53 (12.90) | 68.58 (14.58) |

W8A16 在 context 32/128/512 下相对 BF16 custom 分别快约 `17.9% / 17.1% /
15.4%`。上下文增长后各路径都会因 attention 和 KV 访问增加而下降，但 W8A16 的
Decode 优势保持稳定。

## 内存、功耗与能效

| 路径 | Archive (MiB) | Device weights (MiB) | Workspace (MiB) | KV Cache (MiB) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| CPU FP32 | 1885.59 | 0.00 | 19.53 | 6.00 | 2071.80 |
| BF16 cuBLAS | 943.29 | 942.29 | 9.91 | 3.00 | 2353.86 |
| BF16 custom | 943.29 | 942.29 | 9.91 | 3.00 | 2163.20 |
| W8A16 custom | 612.71 | 611.71 | 9.91 | 3.00 | 1496.17 |

| 路径 | Mean power (W) | Max power (W) | Energy / generated token (J) |
|---|---:|---:|---:|
| CPU FP32 | 13.234 | 13.343 | 1.0077 |
| BF16 cuBLAS | 17.828 | 18.026 | 0.2629 |
| BF16 custom | 18.273 | 18.387 | 0.2679 |
| W8A16 custom | 17.298 | 17.411 | 0.2167 |

功率本身不能代表能效：GPU 路径瞬时功率更高，但完成工作更快，因此每生成 token
的能量显著更低。W8A16 同时降低权重流量和总延迟，得到四条路径中最低的
energy/token。

## Linear 微基准

下表为一层七个 Transformer Linear 的 mean kernel time 合计。W16 reference 为
cuBLAS；W8 reference 为按归档 BF16 scale 显式反量化后的 reference。`speedup` 大于
1 表示 custom 更快。

| 精度 | Tokens | GEMV custom / ref (ms) | GEMV speedup | GEMM custom / ref (ms) | GEMM speedup |
|---|---:|---:|---:|---:|---:|
| BF16 | 1 | 0.3779 / 0.4453 | 1.18x | 2.3261 / 0.4449 | 0.19x |
| BF16 | 32 | 0.3743 / 0.4537 | 1.21x | 0.5049 / 0.4431 | 0.88x |
| BF16 | 128 | 0.3754 / 0.4478 | 1.19x | 1.0169 / 0.5899 | 0.58x |
| BF16 | 512 | 0.3777 / 0.4477 | 1.19x | 3.2691 / 2.0695 | 0.63x |
| W8A16 | 1 | 0.2896 / 0.4448 | 1.54x | 1.0414 / 0.4455 | 0.43x |
| W8A16 | 32 | 0.2890 / 0.4476 | 1.55x | 0.7986 / 0.4429 | 0.55x |
| W8A16 | 128 | 0.2887 / 0.4452 | 1.54x | 1.4600 / 0.5961 | 0.41x |
| W8A16 | 512 | 0.2892 / 0.4466 | 1.54x | 4.5743 / 2.0410 | 0.45x |

该结果解释了端到端现象：W8 GEMV 确实减少 Decode 的权重带宽压力，但 fused
dequantize + Tensor Core GEMM 仍慢于 BF16 reference，所以 W8 Prefill 没有超过
BF16。W8 当前最值得继续优化的是 GEMM，而不是已经达到 1.54--1.55x 的 GEMV。

专用 tied LM Head 在 p32 微基准中为 `3.2811 ms`，cuBLAS + argmax 为
`3.1745 ms`，custom 达到 cuBLAS 的约 `96.75%`；argmax 一致，并继续写出完整
151936 维 BF16 logits。CP22 微基准的 W8 最大相对误差为 `0.006993 < 0.007`。

## Nsight Systems

短序列 trace 使用 32 prompt + 16 output、1 warmup + 3 repeat。NVTX 表中 Decode
是一次请求内 15 个 `decode_next()` 的总区间。

| 路径 | Prefill NVTX median (ms) | Decode NVTX median (ms) | CUDA kernel 时间主要构成 |
|---|---:|---:|---|
| BF16 cuBLAS | 17.173 | 226.231 | 最大 BF16 GEMM kernel family 72.1% |
| BF16 custom | 19.837 | 223.879 | GEMV 60.6%，fused LM Head+reduce 25.2%，GEMM 5.2% |
| W8A16 custom | 27.186 | 191.183 | W8 GEMV 52.4%，fused LM Head+reduce 28.3%，W8 GEMM 9.4% |

Nsight 与端到端结果一致：W8 的 Decode NVTX 区间最短；量化 GEMV 已优于 BF16，
但共享的 BF16 tied LM Head 因 Decode 主干缩短而占到 W8 GPU kernel 时间的
`28.3%`。Prefill 中 W8 GEMM 的占比和微基准耗时都表明 fused dequant GEMM 仍是
下一阶段的明确候选瓶颈。

原始 trace：

- `results/codex-cp22-nsys-cuda-w16a16-cublas.nsys-rep`
- `results/codex-cp22-nsys-cuda-w16a16-custom.nsys-rep`
- `results/codex-cp22-nsys-cuda-w8a16-custom.nsys-rep`

## 正确性状态

CP22 是集中性能测试，不重复制造新的正确性口径；它继承同一代码提交之前 CP19--
CP21 的验证结论：

- Release CTest：32/32 通过。
- BF16 custom 对 cuBLAS：5 个固定 prompt top-1 为 5/5；canonical 16-token greedy
  完全一致；同输入 fused LM Head cosine `0.999999999`，argmax 一致。
- W8A16 对 BF16：5 个固定 prompt top-1 为 5/5；KV Cache 连续 Decode 测试通过；
  archive/device workspace 中没有完整 BF16 反量化权重副本。
- W8A16 canonical 16-token greedy 为 14/16 一致。两个边界 token 的差异来自在线
  反量化到 BF16 后的舍入误差经 24 层传播。该行为已接受并保留记录，最终简历不得
  声称 W8A16 逐 token 与 BF16 完全一致。

## 验收与最终定位

| 项目 | 结果 | 状态 |
|---|---:|---|
| BF16 custom canonical Prefill vs cuBLAS | 86.42% | 通过 |
| BF16 custom canonical Decode vs cuBLAS | 100.72% | 通过 |
| BF16 custom p128 Prefill vs cuBLAS | 80.11% | 通过 |
| BF16 custom p512 Prefill vs cuBLAS | 93.20% | 通过 |
| W8 Decode vs BF16 custom | 1.177x | 通过 |
| W8 p32/p128/p512 Prefill vs BF16 custom | 72.28% / 84.66% / 94.00% | 通过 |
| W8 device weights vs BF16 | 64.92% | 通过 |
| W8 energy/token vs BF16 custom | 80.91% | 通过 |
| 最大 Decode CV | 0.218% | 通过 |
| W8 canonical greedy | 14/16 | 已知限制，未达到原 16/16 门槛 |

适合简历的真实叙事是：自研 shape-aware BF16 GEMV 和 fused LM Head 使 Decode 达到
并略超 cuBLAS 基线；group-size 64 W8A16 在线反量化将 Decode 提升至 80.25 tok/s，
同时将 device weights 降至 611.71 MiB。W8 Prefill 的 fused dequant GEMM 仍落后于
BF16，因此只陈述实际的阶段性数据，不使用模糊的“全链路量化加速”表述。

## 数据位置

- 原始端到端与微基准：`results/codex-cp22-*.json`（Git 忽略）
- Nsight trace/SQLite/CSV：`results/codex-cp22-nsys-*`（Git 忽略）
- 审核后的机器可读摘要：
  `benchmarks/results/codex-cp22-final-summary.json`

报告完成后，测试设备已恢复为 GPU `306--1020 MHz` 动态范围、EMC `2133 MHz`、
EMC lock `0`；未遗留锁频状态。
