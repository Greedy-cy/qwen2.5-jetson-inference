# CODEX-CP21：Prefill GEMM 攻坚（bf16 m128n128 BlockK=64 深化）

## 结论摘要

本检查点针对 CP19 冻结的 `gemm_bf16_wmma_async_m128n128_kernel`（Prefill
GEMM 与 cuBLAS 的差距项）做结构化攻坚：8 个变体逐一实现并测量，最终把
BlockK 从 32 加深到 64 并加 `__launch_bounds__(512, 2)` 收紧寄存器到 64，
获得本轮唯一稳定的正向收益。

- GEMM 微基准（7-linear p50 合计）：p128 `1.0244 → 0.9904 ms`（cuBLAS 的
  `57.6% → 59.6%`），p512 `3.2661 → 3.1779 ms`（`60.5% → 63.1%`）。
- 端到端 Prefill sweep（决定性的门槛）：p128 `80.2% → 81.6%` cuBLAS
  （≥80% 门槛通过），p512 `92.2% → 92.9%`（≥90% 门槛通过）。
- Canonical 无回退：Prefill `1658.8 tok/s`（cuBLAS 的 87.5%）、Decode
  `67.75 tok/s`、总延迟仍低于 cuBLAS。
- NCU 定位（用户执行）：kernel 为 latency-bound（每发指令 15.75 cycle，
  无任何流水线饱和：memory 45%、compute 44%、L2 hit 77%）；cp.async
  staging 的 shared store 存在 4.1-way bank conflict（约 26% 潜在收益，
  但 swizzle 与 `wmma::load_matrix_sync` 的固定布局不兼容）。
- 工程结论：在 GA10B 上，WMMA + cp.async 路径经 8 个结构变体后已接近其
  实际下限；剩余 ~37% 差距来自 cuBLAS 的 swizzle+ldmatrix 定制路径，按
  CP19 的"端到端优先"决策不再复刻。端到端门槛（CP19 已过、CP21 再提升）
  是这套后端真正的性能判定标准。

## 测试身份与边界

- 测量提交：`f5fcf34`（GEMM BlockK=64 深化）
- 测量时工作树：clean
- 模型：`Qwen2.5-0.5B-Instruct`，W16A16 archive SHA-256
  `f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`
- 设备：Jetson Orin Nano Super 8GB，CUDA 12.6，SM87，GPU 锁定 1020 MHz
- 协议：GEMM 微基准 20/100 CUDA Event P50；e2e sweep 5/20；canonical
  5/20（32 prompt + 128 output）
- 对照：cuBLAS `cublasGemmEx`（BF16），同一 `--linear-kernel` 开关
- 本次只改 bf16 m128n128 路径；m32n64（p32）、W8 路径、GEMV、LM Head
  均未改动

## 实现内容

`gemm_bf16_wmma_async_m128n128_kernel`（BlockM=128, BlockN=128, 512
threads, 16 warps, 每 warp 4 个 accumulator tile）：

- `BlockK 32 → 64`（SharedK 64→72）：stage 数减半（28→14），每 stage 每
  warp 的 mma 工作翻倍（18→36 条指令），缩短每 stage 中等待 cp.async
  到达的时间占比。动态 shared 2×36,864 = 73,728B（1 block/SM）。
- `__launch_bounds__(512, 2)`：寄存器 66→64（3 个 spill），使 2 blocks/SM
  在寄存器维度上可行。
- 流水结构保持：每 iteration 顶部为下一 stage 发起 cp.async（2-buffer
  issue-ahead，`wait_prior(1)`），等待前一 stage 的 group 到达后 mma，
  iteration 末尾 `__syncthreads` 保护缓冲区复用。

### 实测未获收益的结构变体（全部测量，均回退）

| 变体 | p128 | p512 | 结论 |
|---|---:|---:|---|
| 4-buffer 深度流水（80KB，1 block/SM） | 1.0972 ms | 3.4808 ms | 1 block/SM 的串行等待无法被隐藏 |
| 2-buffer issue-ahead（BlockK=32） | 1.0778 ms | 3.4116 ms | 单 iteration 余量不足 |
| m32n64 tiling 用于 p128/p512 | 1.6355 ms | 6.9550 ms | 每 warp 单 accumulator，fragment 装载爆炸 |
| SharedK=80（BlockK=64） | 1.0675 ms | 3.4578 ms | store 冲突模式未改善 |

## NCU 定位（用户执行，`--set full`，p512 GEMM）

- Duration `206.98 us`（单次 GEMM launch，28 blocks）
- Memory Throughput `45.10%`、Compute `43.66%`、L2 Cache `42.24%` ——
  无任何管道饱和，latency-bound。
- L2 Hit Rate `77.03%`（权重/输入跨 block 复用良好）。
- Warp Cycles Per Issued Instruction `15.75`；Active Warps/Scheduler
  `7.05`（achieved occupancy 58.66%，2 blocks/SM）。
- Shared stores `4.1-way` bank conflict（占 store wavefront 50.54%，
  估算 ~26% 提速空间）：cp.async 16B store 在 144B 行距下的固有冲突；
  修复需要 XOR swizzle，与 `wmma::load_matrix_sync` 固定布局不兼容。

## 端到端结果（决定性门槛，5/20）

| Prompt | custom | cuBLAS | attainment | CP19 attainment |
|---:|---:|---:|---:|---:|
| 128 | 2262.8 tok/s | 2774.4 tok/s | **81.6%** | 80.2% |
| 512 | 1036.4 tok/s | 1115.5 tok/s | **92.9%** | 92.2% |

## Canonical（32+128，5/20，无回退确认）

| 指标 | custom | cuBLAS |
|---|---:|---:|
| Prefill | 1658.78 tok/s | 1896.65 tok/s（87.5%） |
| TTFT | 19.29 ms | 16.87 ms |
| Decode | 67.75 tok/s | 67.00 tok/s |
| Total latency | 1893.9 ms | 1912.3 ms |

## GEMM 微基准（20/100，七个 Linear 合计 P50）

| Tokens | custom | cuBLAS | attainment | CP19 |
|---:|---:|---:|---:|---:|
| 128 | 0.9904 ms | 0.6021 ms | 60.6% | 57.6% |
| 512 | 3.1824 ms | 2.0097 ms | 63.1% | 60.5% |

## 验收门槛

| 门槛 | 要求 | 实测 | 结果 |
|---|---:|---:|---|
| p128 端到端 Prefill | ≥ 80% cuBLAS | 81.6% | 通过（CP19 80.2% →） |
| p512 端到端 Prefill | ≥ 90% cuBLAS | 92.9% | 通过（CP19 92.2% →） |
| Canonical 无回退 | Prefill ≥ 87% / Decode ≥ 67 tok/s | 87.5% / 67.75 | 通过 |
| 测试 | 全量 | 32/32 | 通过 |
| GEMM 微基准（信息项） | — | 60.6% / 63.1% | 记录：latency-bound，已到 WMMA 路径下限 |

## 原始数据

- `results/codex-cp21-sweep-w16a16-{custom,cublas}-p{128,512}.json`
- `results/codex-cp21-canonical-w16a16-{custom,cublas}.json`
- `results/codex-cp21-ncu-m128n128.txt`
- 微基准：`/tmp/opencode/cp21-final-p{128,512}.json`（及 v2–vA 系列变体数据）

机器可读摘要：`benchmarks/results/codex-cp21-prefill-gemm-summary.json`
