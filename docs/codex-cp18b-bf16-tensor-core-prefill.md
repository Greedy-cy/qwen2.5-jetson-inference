# CODEX-CP18B：BF16 Tensor Core Prefill GEMM 优化

## 结论摘要

本检查点只优化 CUDA W16A16（实际为 BF16）Prefill GEMM，不修改 Decode GEMV、
Attention、KV Cache 或 W8A16 路径。自研 GEMM 已由朴素 CUDA Core FMA 改为 SM87
Tensor Core WMMA，并加入 16-byte async copy、双缓冲和真实 Qwen 形状分流。

- Canonical 32-token Prefill 从 `64.37 tok/s` 提升到 `1219.24 tok/s`，TTFT 从
  `497.15 ms` 降至 `26.25 ms`，均为 `18.94x` 改善。
- 一层七个 Transformer GEMM 的 P50 合计相对 CP18A 分别加速
  `40.24x/77.90x/97.53x`（M=`32/128/512`）。
- M=32 达到 cuBLAS 吞吐的 `87.22%`，通过 80% stretch goal；M=128/512 达到
  `57.57%/58.65%`，未达到 70% 阶段门槛。
- Prompt sweep 的端到端 Prefill 为 `1218.46/1992.68/1033.94 tok/s`
  （prompt=`32/128/512`），分别达到 cuBLAS 的 `64.00%/71.71%/91.99%`。
- 28/28 测试通过；最终微基准中 custom 与 cuBLAS 最大差异相对输出峰值不超过
  `0.006579`，与优化前 BF16 数值边界一致。

因此 CP18B 的实现、正确性、正式测量和瓶颈定位已完成，但长 prompt 纯 GEMM 的 70%
性能门槛未通过。不得把当前结果描述为“所有形状均接近 cuBLAS”。

## 测试身份与边界

- 测量提交：`c9cca8814eb0deb0d413e445242c6091ee34b4e1`
- 前置 Tensor Core 提交：`a689b9f264539ae62d4224e285862717864339da`
- 测量时工作树：clean
- 模型：`Qwen2.5-0.5B-Instruct`
- 精度：BF16 weight/activation/KV/logits，Tensor Core accumulator 为 FP32
- 模型 SHA-256：`f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`
- 设备：Jetson Orin Nano Super 8GB，Jetson Linux R36.4.7，CUDA 12.6，SM87
- 功耗模式：MAXN_SUPER
- 时钟：每组测试临时启用 `jetson_clocks`，trap 恢复原状态
- Prefill：工程当前唯一的矩阵化 Prefill 基线，不包含逐 token 伪 Prefill

CP18A 的 custom/cublas 差异只在线性层实现。本检查点保持这条 A/B 边界不变。

## 实现内容

### 32-token 对齐路径

- block tile：`M32 x N64 x K64`，256 threads，8 warps。
- WMMA BF16 输入和权重、FP32 accumulator。
- 每 warp 负责一个 16x16 输出 tile。
- 使用 16-byte `__pipeline_memcpy_async` 将 global memory 搬到 shared memory。
- 双缓冲覆盖下一 K stage 的加载，并融合 bias 与 BF16 output conversion。

### 128/512-token 对齐路径

- block tile：`M128 x N128 x K32`，512 threads，16 warps。
- 每 warp 持有 4 个 accumulator，提高 input/weight 在 M/N 方向的复用。
- 每 warp 使用独立的 16x16 临时输出区，避免为整个 128x128 FP32 tile 保留 shared
  memory。
- 仅在 M/N/K 满足真实 Qwen 对齐条件时启用；其他形状使用支持边界处理的 WMMA
  fallback，不静默调用 cuBLAS。

SASS 检查确认生成了 BF16 `HMMA.16816.F32.BF16` 和 16-byte async-copy
`LDGSTS.E.BYPASS.128`。

### 对已有 GEMM 工程的使用

参考了目标设备 `/home/greedy/AI_infra/cuda/operator/gemm` 中 FP32 SGEMM 的
cp.async、双缓冲和 tile 复用方法。该实现的数据类型和计算核心是 FP32 SIMT，未直接复制
为 BF16 kernel；本项目复用的是 pipeline 设计原则，并以 WMMA/Tensor Core 重新实现。

### 被拒绝的局部实验

- `M128 x N64 x K32`：p128/p512 仅达到 cuBLAS 的约 `55%/57%`。
- shared stride 由 40 增至 48：shared memory 上升、occupancy 下降，p128/p512 退化到
  `55.13%/54.34%`。
- K stage 从 32 降至 16：同步次数增加，p128/p512 退化到 `41.61%/49.27%`。

最终保留实测最优的 `M128 x N128 x K32`、shared stride 40。

## 正式测量协议

真实形状微基准覆盖 Q/K/V/O、gate/up/down 与 tied LM Head；本检查点的 GEMM 合计只统计
每层七个 Transformer Linear。每个 case 同进程测 custom 与 cuBLAS，20 次 warmup、
100 次 CUDA Event 测量。

端到端 canonical：固定 32-token prompt、128-token greedy 输出、`max-seq-len=256`、
5 次 warmup、20 次测量、tegrastats 100 ms。Prompt sweep 使用 prompt=`1/32/128/512`、
输出 1 token、`max-seq-len=512`、5/20 协议。

Nsight Systems：32-token prompt、16-token 输出、1 次 warmup、3 次测量；只使用 GPU
kernel summary 与 NVTX 阶段中位数定位瓶颈。

## 真实 Qwen 形状微基准

### 一层七个 Transformer GEMM 的 P50 合计

| Tokens | CP18A custom | CP18B custom | cuBLAS | CP18B 加速 | cuBLAS attainment |
|---:|---:|---:|---:|---:|---:|
| 32 | 20.1516 ms | 0.5008 ms | 0.4368 ms | 40.24x | 87.22% |
| 128 | 79.6883 ms | 1.0230 ms | 0.5889 ms | 77.90x | 57.57% |
| 512 | 317.8669 ms | 3.2591 ms | 1.9114 ms | 97.53x | 58.65% |

### M=32 分层 P50

| Linear | custom | cuBLAS | custom attainment |
|---|---:|---:|---:|
| q_proj | 0.0317 ms | 0.0327 ms | 103.13% |
| k_proj | 0.0207 ms | 0.0235 ms | 113.47% |
| v_proj | 0.0207 ms | 0.0234 ms | 113.14% |
| o_proj | 0.0327 ms | 0.0262 ms | 80.12% |
| gate_proj | 0.1308 ms | 0.1118 ms | 85.45% |
| up_proj | 0.1328 ms | 0.1131 ms | 85.16% |
| down_proj | 0.1313 ms | 0.1061 ms | 80.77% |

短 prompt 的 Transformer GEMM 已接近 cuBLAS；长 prompt 的剩余差距主要在 MLP
形状和 shared-memory 访问效率。

## Canonical 端到端结果

| Linear path | TTFT mean | Prefill tok/s | Decode tok/s | Decode CV | Total latency |
|---|---:|---:|---:|---:|---:|
| custom | 26.25 ms | 1219.24 | 27.75 | 0.0113% | 4602.66 ms |
| cuBLAS | 16.84 ms | 1900.26 | 67.54 | 0.0083% | 1897.22 ms |

相对 CP18A custom：Prefill/TTFT 改善 `18.94x`，总延迟只改善 `1.10x`。原因是 128-token
输出的绝大多数时间仍由未优化 Decode GEMV 占据。custom 平均功率为 `13.70 W`，单位
生成 token 能耗由 CP18A 的 `0.527 J/token` 降至 `0.492 J/token`；cuBLAS 为
`0.268 J/token`。

Transformer GEMM 微基准达到 cuBLAS 的 87.22%，但端到端 32-token Prefill 只有
64.16%。主要差额来自 Prefill 末尾仍需执行一次 tied LM Head，而该层走尚未优化的
custom GEMV；它不是本检查点的 GEMM 优化对象。

## Prompt sweep

| Prompt | custom TTFT | cuBLAS TTFT | custom Prefill | cuBLAS Prefill | attainment | 对 CP18A custom |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77.03 ms | 14.34 ms | 12.98 tok/s | 69.75 tok/s | 18.61% | 3.35x |
| 32 | 26.26 ms | 16.81 ms | 1218.46 tok/s | 1903.88 tok/s | 64.00% | 18.93x |
| 128 | 64.24 ms | 46.06 ms | 1992.68 tok/s | 2778.85 tok/s | 71.71% | 30.40x |
| 512 | 495.20 ms | 455.55 ms | 1033.94 tok/s | 1123.92 tok/s | 91.99% | 16.25x |

p512 端到端接近 cuBLAS 并不表示纯 GEMM 已达到 92%；该长度下 causal attention 在 TTFT
中的占比显著增加，稀释了 Linear 差距。所有 TTFT CV 均低于 `0.07%`。

## Nsight 结论

| Path | GPU kernel time 主项 | 占比 | NVTX Prefill median | NVTX Decode token median |
|---|---|---:|---:|---:|
| custom | `gemv_bf16_kernel` | 93.1% | 27.14 ms | 37.65 ms |
| custom | `gemm_bf16_wmma_async_kernel` | 2.1% | 同上 | 同上 |
| cuBLAS | 两类 BF16 Tensor Core GEMM | 82.9% | 17.20 ms | 15.04 ms |

在 canonical 风格短序列 trace 中，新 Prefill GEMM 已不再是 custom 的主瓶颈；Decode
GEMV（包含 tied LM Head）占 GPU kernel 时间 93.1%。因此下一阶段若以简历中的
batch=1 生成性能为目标，应先优化 Decode GEMV/LM Head，而不是继续微调短 prompt GEMM。

长 prompt MLP 的定点 Nsight Compute 显示：global memory throughput `63.51%`、
SM throughput `47.38%`、achieved occupancy `61.53%`，并报告约 `32%` 的 excessive
shared-memory wavefront。若后续单独追求长 prompt 纯 GEMM，应实现 shared layout/swizzle
并重新验证 p128/p512；该工作不与 Decode 优化混在同一 checkpoint。

## 正确性与验收

- 28/28 CTest 通过。
- 对齐 fast path、非整齐 tail、bias 和真实 Qwen 维度测试通过。
- 微基准最大 custom/cuBLAS 相对峰值差异：p32 `0.004032`，p128/p512
  `0.006579`；没有数值退化。
- p32 cuBLAS attainment：`87.22%`，通过 70% target 与 80% stretch goal。
- p128/p512 attainment：`57.57%/58.65%`，未通过 70% target。
- canonical Decode CV：custom `0.0113%`、cuBLAS `0.0083%`，通过 3% 稳定性门槛。

CP18B 在“实现、正确性、数据和瓶颈定位”上完成，在“所有真实 shape 均达到 cuBLAS 70%”
上部分未通过。按 checkpoint 约定在此停止，不进入 W8A16。

## 原始数据

Git 忽略目录：

- `results/codex-cp18b-bf16-linear-p{32,128,512}.json`
- `results/codex-cp18b-canonical-w16a16-{custom,cublas}.json`
- `results/codex-cp18b-sweep-w16a16-{custom,cublas}-p{1,32,128,512}.json`
- `results/codex-cp18b-nsys-w16a16-{custom,cublas}.nsys-rep`
- `results/codex-cp18b-nsys-w16a16-{custom,cublas}_{cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum}.txt`

审核后的机器可读摘要：

- `benchmarks/results/codex-cp18b-bf16-tensor-core-prefill-summary.json`
