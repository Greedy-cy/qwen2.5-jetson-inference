# CODEX-CP19：shape-aware Decode GEMV 与融合 tied LM Head argmax

## 结论摘要

本检查点收口自研 BF16 后端的 Decode 侧：把每个 `(row)` 一个 256-thread block 的朴素
GEMV 重写为按真实 Qwen 形状自动调度的 shape-aware GEMV，并新增融合 argmax 的 tied
LM Head kernel。不改动 Prefill GEMM、Attention 或 W8A16 路径。

- Canonical custom Decode 从 `27.75 tok/s` 提升到 `67.92 tok/s`（`2.45x`），并反超
  cuBLAS 基线（`66.84 tok/s`，`101.6%`），远超 85% 门槛（约 `57.4 tok/s`）。
- 一层七个 Transformer Decode GEMV 加权 P50 合计 `0.3855 ms` vs cuBLAS `0.4465 ms`
  （attainment `115.8%`），通过 85% 门槛与 90% stretch goal。
- 融合 tied LM Head `3.2797 ms` vs cuBLAS GEMV+argmax `3.1933 ms`（`97.37%`），
  通过 80% 门槛；同真实输入下 fused 与 cuBLAS 仅差 1 个 BF16 ULP（50/151936 元素），
  cosine `0.999999999`，argmax 一致。
- Canonical custom Prefill `1656.25 tok/s`（cuBLAS 的 `87.6%`，原 `64.0%`），TTFT
  `19.32 ms`；总延迟 `1889 ms`，低于 cuBLAS `1917 ms`。能量 `0.274 J/token`，低于
  cuBLAS `0.281 J/token`。
- Prompt sweep：p128 端到端 Prefill 达 cuBLAS `80.2%`（原 `71.71%`），p512 `92.2%`
  （≥90%）。据此冻结 `M128N128K32` GEMM，不再为纯 GEMM 指标扩展复杂度。
- W16A16 默认 Linear kernel 由 cuBLAS 切换为 custom（commit `35a6250`）；cuBLAS 保留
  为显式基线。默认路径实测 `67.82 tok/s`，与 custom 一致。
- 30/30 CTest 通过；5 个固定 prompt top-1 5/5；canonical 16-token greedy 一致；
  Decode CV `0.2293%`（≤3%）。

所有 CP19 验收门槛通过，按约定在此收口，进入 CP20 前等待确认。

## 测试身份与边界

- 测量提交：`ed62bb3ebddedfbbd76005ee26b7915d31f7714f1`
- 默认 kernel 切换提交：`35a6250680f0a88e4540914b0ba430003d7cb314`
- 锁频脚本提交：`febee9c28e97427b7f72492b00d6b4a9ecb7a3e0c`
- 测量时工作树：clean
- 模型：`Qwen2.5-0.5B-Instruct`，archive SHA-256
  `f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`
- 精度：BF16 weight/activation/KV/logits，FP32 accumulation
- 设备：Jetson Orin Nano Super 8GB，Jetson Linux R36.4.7，CUDA 12.6，SM87
- 功耗模式：MAXN_SUPER；GPU 通过 `scripts/lock_clocks.sh` 锁定 1020 MHz，CPU 1728 MHz。
  由于无免密 sudo，runner 的 `jetson_clocks` store/restore 不可用；测量前后 GPU 频率
  均确认处于 1020 MHz 锁定状态。
- Prefill：矩阵化 Prefill 唯一基线；Decode：单 token autoregressive。

CP18B 的 custom/cuBLAS A/B 边界保持不变（仅 Linear kernel 实现不同）。本检查点不改变
W8A16 语义：其 tied LM Head 为 BF16，与 W16 custom 共用同一融合实现。

## 实现内容

### Decode GEMV（`gemv_bf16` 重写）

- `K <= 1024`（Q/K/V/O/gate/up，K=896）：一个 warp 计算一个输出 row，每 block
  8 warps / 8 rows。
- `K > 1024`（down，K=4864）：4 个 warp 协作一个 row，每 block 2 rows。
- BF16 weight/input 使用 16-byte 对齐向量（8 元素）加载；input 协作载入 shared
  memory，供同 block 的多个输出 row 复用。
- FP32 FMA accumulation + warp shuffle reduction；移除原每个 row 的 256-thread
  shared reduction 和多轮 `__syncthreads()`（K<=1024 路径无需跨 warp 同步）。
- bias 融合到 BF16 写回前。
- 非 8 对齐 K、指针非 16-byte 对齐、小 shape 使用保留的标量 tail kernel。

### 融合 tied LM Head + argmax（`lm_head_bf16`）

- `151936 x 896` 专用 GEMV，8 rows/block、1 warp/row，与 GEMV 同构。
- 第一阶段同时写出完整 BF16 logits 和每 block 的 `(max_value, max_index)`；
  max 基于写回后的 BF16 值计算，保证与对 logits 做 argmax 严格一致。
- 第二阶段（1 block/256 threads）归约 block maxima，tie 固定选择最小 token index。
- 使用预分配 workspace（`ceil(vocab/8)` 组 float+int），无逐 token `cudaMalloc`。
- W16 custom 与 W8 custom 的 BF16 tied LM Head 共用该实现；cuBLAS 对照继续
  `cublasGemmEx + argmax_bf16`。
- 顺带修复 `argmax_kernel`/`argmax_bf16_kernel` 树归约的 tie 行为：相等值时取
  最小 index（此前跨 thread 相等值时可能取到较大 index）。

### 微基准工具统一

`tools/benchmark_bf16_linear.cpp` 重命名为 `tools/linear_benchmark.cpp`（target
`linear_benchmark`），CLI 统一为：

```text
--archive DIR/model.w16a16.qbin|model.w8a16.qbin
--precision w16a16|w8a16  --tokens N  --warmup N  --repeat N  --json FILE
```

- W16：输出 custom/cuBLAS（LM Head 输出 fused vs cublas+argmax，含 argmax 一致性）。
- W8：输出 optimized（`gemv_w8a16`/`gemm_w8a16`）与显式反量化 FP32 reference 的
  sampled 误差；tied LM Head 按 BF16 处理。

### workspace 对齐

`initialize_workspace` 的 BF16 take 偏移按 8 元素（16 字节）对齐，保证 fast path 的
向量加载合法；dispatch 同时做运行时指针对齐检查。

## 正确性

- 30/30 CTest 通过，新增覆盖：
  - shape-aware GEMV 真实 Qwen 形状：`896x896`（bias）、`128x896`、`4864x896`、
    `896x4864`、`1024x1024`、`897x896`、`3x1024`、`512x70`（tail）。
  - fused LM Head：logits 与 cublas 逐元素一致（容差内），argmax 与自身 logits 的
    host argmax 一致，构造重复最大值（2.0 权重 + 100 主导输入）验证 tie→最小 index。
- 真实模型对照（custom vs cuBLAS）：
  - 5 个固定 prompt 首 token top-1 全部一致（5/5）。
  - 端到端首 token logits cosine 最低 `0.999871`。差异来源是 Prefill GEMM 的
    custom/cuBLAS 累加路径经 24 层传导，而非 LM Head kernel：同一真实 norm 输入下
    fused 与 cuBLAS+argmax 最大差恰为 1 个 BF16 ULP（`0.03125`，50/151936 元素），
    cosine `0.999999999`，argmax 一致。
  - canonical 32-token prompt 16-token greedy 输出完全一致。
  - `last_logits_host()` 在融合路径下仍返回完整 151936 维 BF16 logits（全有限）。

## 真实 Qwen 形状微基准（20 warmup / 100 repeat，CUDA Event P50）

### 一层七个 Decode GEMV 加权合计（每 token）

| 实现 | 合计 P50 | vs cuBLAS | 门槛 |
|---:|---:|---:|---|
| custom | 0.3855 ms | 115.8% | ≥85%（stretch 90%）通过 |
| cuBLAS | 0.4465 ms | — | — |

分层 P50：

| Linear | custom | cuBLAS | attainment |
|---|---:|---:|---:|
| q_proj | 0.0204 ms | 0.0365 ms | 179.2% |
| k_proj | 0.0101 ms | 0.0180 ms | 177.7% |
| v_proj | 0.0099 ms | 0.0180 ms | 181.1% |
| o_proj | 0.0200 ms | 0.0296 ms | 148.3% |
| gate_proj | 0.1106 ms | 0.1135 ms | 102.7% |
| up_proj | 0.1090 ms | 0.1124 ms | 103.1% |
| down_proj | 0.1055 ms | 0.1185 ms | 112.4% |

### tied LM Head（含 argmax）

| 实现 | P50 | attainment |
|---:|---:|---:|
| custom fused | 3.2797 ms | 97.37% |
| cuBLAS GEMV+argmax | 3.1933 ms | — |

### Prefill GEMM（未改动，与 CP18B 一致）

| Tokens | custom P50 合计 | cuBLAS P50 合计 | attainment |
|---:|---:|---:|---:|
| 32 | 0.5033 ms | 0.4442 ms | 88.27% |
| 128 | 1.0244 ms | 0.5902 ms | 57.61% |
| 512 | 3.2661 ms | 1.9749 ms | 60.46% |

## Canonical 端到端（32 prompt + 128 output，5/20）

| Linear path | Prefill tok/s | TTFT | Decode tok/s | Decode CV | Total latency | J/token |
|---|---:|---:|---:|---:|---:|---:|
| custom | 1656.25 | 19.32 ms | 67.92 | 0.2293% | 1889.3 ms | 0.274 |
| cuBLAS | 1891.21 | 16.92 ms | 66.84 | 0.5727% | 1917.0 ms | 0.281 |

相对 CP18B custom：Prefill `1.36x`，Decode `2.45x`，总延迟 `2.44x`（4602.7 →
1889.3 ms）。custom 的 TTFT 提升来自 Prefill 末尾融合 LM Head 的加速。

## Prompt sweep（1 token 输出，5/20）

| Prompt | custom Prefill | cuBLAS Prefill | attainment | CP18B attainment |
|---:|---:|---:|---:|---:|
| 1 | 14.2 tok/s | 69.1 tok/s | 20.6% | 18.6% |
| 32 | 1658.7 tok/s | 1899.7 tok/s | 87.3% | 64.0% |
| 128 | 2228.2 tok/s | 2778.4 tok/s | 80.2% | 71.71% |
| 512 | 1034.8 tok/s | 1122.3 tok/s | 92.2% | 91.99% |

p1 的差距来自 M=1 的 GEMM 形状与 LM Head 固定开销，不在本检查点范围。p128 达到 80%
门槛，p512 保持 90% 以上 → W16 长 prompt 条件分支按“端到端优先”标准收口：
冻结 `M128N128K32` GEMM，不追加 shared swizzle。

## Decode context sweep（128 token 输出，5/20）

| Prompt | custom Decode | cuBLAS Decode | attainment |
|---:|---:|---:|---:|
| 32 | 67.71 tok/s | 66.95 tok/s | 101.1% |
| 128 | 65.79 tok/s | 64.91 tok/s | 101.4% |
| 512 | 59.11 tok/s | 58.54 tok/s | 101.0% |

所有 context 长度下 custom Decode 均不低于 cuBLAS，CV 全部 ≤ 0.3%。

## Nsight Systems（32 prompt + 16 output）

| Path | GPU kernel time 主项 | 占比 | NVTX Decode token median |
|---|---|---:|---:|
| custom | `gemv_bf16_fast_kernel<8,1>` | 45.9% | 15.052 ms |
| custom | `lm_head_bf16_kernel` | 24.6% | 同上 |
| custom | `gemv_bf16_fast_kernel<2,4>` | 14.8% | 同上 |
| custom | `gemm_bf16_wmma_async_kernel`（Prefill） | 5.3% | 同上 |
| cuBLAS | — | — | 15.107 ms |

CP18B 中 Decode GEMV 占 93.1% 的时间已拆分并整体收缩：custom 单 token Decode 中位数
从 `37.65 ms` 降至 `15.05 ms`，与 cuBLAS（`15.10 ms`）持平。

## Nsight Compute（定点）

- `lm_head_bf16_kernel`（18992 blocks × 256）：memory throughput `69.45%`，
  compute `42.24%`，L2 hit `1.91%`；warp stall 58% 为 long-scoreboard L1TEX。
  属带宽受限（每 token 读 272 MB 权重），已接近 cuBLAS（97.4%），不再优化。
- `gemv_bf16_fast_kernel<8,1>`（112 blocks，K=896）：memory `43.19%`，compute
  `25.36%`；stall 69.8% long-scoreboard L1TEX（权重加载等待），small-grid
  latency-bound；该形状已快于 cuBLAS，收益已兑现。
- K/V 小网格（16 blocks）：memory `16.4%`，纯小网格延迟主导。

## 验收门槛

| 门槛 | 要求 | 实测 | 结果 |
|---|---:|---:|---|
| 七个 Decode GEMV 加权合计 | ≥ 85%（stretch 90%） | 115.8% | 通过（stretch 也通过） |
| tied LM Head | ≥ 80% | 97.37% | 通过 |
| 端到端 Decode | ≥ cuBLAS 85%（约 57.4 tok/s） | 101.6%（67.92） | 通过 |
| canonical 32-token Prefill | ≥ cuBLAS 80% | 87.6% | 通过 |
| p128 端到端 Prefill | ≥ 80% | 80.2% | 通过 |
| p512 端到端 Prefill | ≥ 90% | 92.2% | 通过 |
| Decode CV | ≤ 3% | 0.2293% | 通过 |

全部通过 → W16A16 默认 Linear kernel 由 cuBLAS 切换为 custom（commit `35a6250`），
cuBLAS 保留为显式基线；`M128N128K32` GEMM 冻结。

## 原始数据

Git 忽略目录 `results/`：

- `results/codex-cp19-canonical-w16a16-{custom,cublas,default}.json`
- `results/codex-cp19-linear-w16a16-p{32,128,512}.json`
- `results/codex-cp19-sweep-w16a16-{custom,cublas}-p{1,32,128,512}.json`
- `results/codex-cp19-decode-ctx-w16a16-{custom,cublas}-p{32,128,512}.json`
- `results/codex-cp19-nsys-w16a16-{custom,cublas}.nsys-rep/.sqlite` 及 stats CSV

审核用机器可读摘要：

- `benchmarks/results/codex-cp19-decode-gemv-lmhead-summary.json`
