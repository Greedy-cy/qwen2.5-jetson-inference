# CODEX-CP20：shape-aware W8A16 Decode GEMV 与 tensor-core Prefill GEMM

## 结论摘要

本检查点把 W8A16 Linear 从 CP17 的朴素 kernel（`linear_w8a16_kernel` 占 GPU
kernel 时间 94.3%，Prefill 仅 42.16 tok/s）重构为 shape-aware 实现：

- Decode 使用与 CP19 同构的 fast GEMV（`<8,1>` / `<2,4>`，INT8 权重 16 元素
  chunk + group-64 scale 复用 + shared input + FP32 FMA + warp shuffle）。
- Prefill 使用 bf16 WMMA GEMM（m32n64 / m128n128），cp.async 直载 INT8 raw
  tile 与 BF16 input tile，shared-memory 反量化（int8 × scale → bf16，打包
  `__floats2bfloat162_rn` 32-bit store）后进 tensor core。

结果（clean commit `8bd9293`，GPU 1020 MHz）：

- **Decode `80.01 tok/s`，为 W16 custom 的 `117.8%`**（67.92 tok/s），比 CP17 旧
  W8（33.57）快 `2.38x`；所有 context 长度下 decode 均超过 W16（115–118%）。
- **Prefill `1193.7 tok/s`（W16 的 `72.1%`）**，比 CP17 旧 W8（42.16）快 `28.3x`；
  TTFT 从 `758.96 ms` 降至 `26.81 ms`。prompt sweep 随长度改善：p1 `236%`、
  p32 `71.9%`、p128 `84.4%`、p512 `95.2%`。
- **Device weights `611.71 MiB`（W16 的 `64.9%`）**；能量 `0.2255 J/token`（W16
  的 `82.4%`）。
- NVTX：decode token 中位数 `12.86 ms`（W16 `15.05 ms`）；prefill `27.36 ms`
  （W16 `23.82 ms`）。
- 正确性：32/32 CTest；5-prompt top-1 与 W16 全部一致（5/5）；GEMV 与显式
  反量化 FP32 reference 逐位一致（max_abs_error=0）。
- **已知分歧（已接受并记录）**：canonical 16-token greedy 与 W16 有 2/16 处翻词
  （token 7 `Language`→`Model`、token 16 `It`→`that`），根因是 tensor-core
  GEMM 必须把反量化权重舍入到 BF16（每元素 ~2^-9 扰动，24 层传导后 ~1e-3
  logit 噪声），旧 fp32-FMA GEMM 是精确的。旧 W8 在该 prompt 与 W16 16/16 一致。

CP20 验收门槛（decode ≥85%、prefill p32/p128/p512 ≥70/80/85%、内存 ≤70%、CV
≤3%、top-1 5/5）全部通过；greedy 16/16 门槛不满足但经确认接受为 W8 量化行为。

## 测试身份与边界

- 测量提交：`8bd9293d6e06b2ac0f4c27c6fad6a6d1b5b4f1e8`
- 测量时工作树：clean
- 模型：`Qwen2.5-0.5B-Instruct`；W8A16 archive SHA-256
  `730d0ca9a60b55a63534c8edce19353d8f55fda07e6e5fc04ec88470bcf57814`
- 精度：INT8 weights（group-size 64）+ BF16 scales + BF16 activations/KV/logits，
  FP32 accumulation
- 设备：Jetson Orin Nano Super 8GB，Jetson Linux R36.4.7，CUDA 12.6，SM87
- 功耗模式：MAXN_SUPER；GPU 锁定 1020 MHz（`scripts/lock_clocks.sh`）
- W8A16 无 cuBLAS INT8 基线；对照为 W16 custom（CP19 同协议数据）。`--linear-kernel`
  仅作用于 BF16 路径，W8 Linear 恒为 custom。
- sweep 结果曾因两个 runner 并发争用 GPU 被污染，已用串行重测替换。

## 实现内容

### Decode GEMV（`gemv_w8a16_fast_kernel`）

- `K ≤ 1024`：`<8,1>`（每 block 8 rows、每 row 1 warp）；`K > 1024`（down_proj
  K=4864）：`<2,4>`（每 block 2 rows、每 row 4 warps）。
- 权重按 16 元素（`int4`）向量加载；group-size 64 下 16 元素 chunk 不跨 group，
  每个 chunk 复用 1 个 BF16 scale（FP32 展开）。
- 反量化保持 FP32 精度（`int8 × bf16_scale` 在 FP32 中精确，无 bf16 舍入）——
  与 CP15 的 fp32-FMA 语义逐位一致。
- input 协作载入 shared（与 CP19 相同），FP32 FMA + warp shuffle reduction +
  bias 融合。
- 非 16 对齐、K%16≠0、K%64≠0（不完整 group）或指针未对齐时回退
  `linear_w8a16_kernel`。

### Prefill GEMM（tensor-core）

- `tokens%32==0`：`gemm_w8a16_wmma_async_kernel`（BlockM=32, BlockN=64,
  BlockK=64, 256 threads）。
- `tokens%128==0 && out%128==0`：`gemm_w8a16_wmma_async_m128n128_kernel`
  （BlockM=128, BlockN=128, BlockK=32, 512 threads, 4 accumulators）。
- 双缓冲：cp.async 直载 BF16 input tile + INT8 raw weight tile；每个 stage 在
  shared 中反量化 raw → BF16 tile（打包 `__floats2bfloat162_rn`，每 2 元素 1
  个 32-bit store，反量化指令数比标量版少约 35%），再 bf16 `mma_sync`。
- **数值语义**：反量化结果必须舍入到 BF16 才能进 `mma`，与旧 fp32-FMA 路径每
  元素差 ~2^-9 相对误差 —— 这是 greedy 分歧的来源（见正确性）。
- 其余形状回退 `linear_w8a16_kernel`。

### 微基准工具修复

`linear_benchmark` 的 W8 对照路径此前把 INT8 原始 buffer 直接 reinterpret 成
BF16 喂 cuBLAS（读取量翻倍，在 4864-row 层越界触发 illegal memory access）。
现改为 host 侧显式反量化（int8×scale → bf16）后再做 cuBLAS 对照，并保留
sampled 反量化 reference 误差。

## 正确性

- 32/32 CTest 通过。新增：
  - `GemmFastPathsMatchDequantizedReferenceAtPrefillShapes`：tokens 32/128/512 ×
    Q/K/V/gate/up/down 真实形状，对照显式反量化 FP32 reference。
  - `GemvFastPathMatchesFallbackForAlignedK`：K=1024/1216/896/912、out=896/24/96，
    覆盖 `<8,1>`/`<2,4>`/fallback 分支。
- GEMV 与 FP32 反量化 reference 逐位一致（max_abs_error=0），与 CP15 语义相同。
- 5 个固定 prompt top-1 与 W16 全部一致：`9707 / 80285 / 6689 / 95456 / 785`。
- **Canonical greedy 分歧（已接受）**：新 W8 在 canonical 32-token prompt 上
  16-token greedy 为 `[80285,12,4086,44,320,34253,4903,8,374,264,943,315,3460,
  4128,1614,429]`，W16 为 `[...11434,4903,...,4128,1614]`：token 7 由
  `Language`(11434) 翻为 `Model`(4903)，token 16 由 `is` 翻为其他。旧 W8
  （fp32-FMA GEMM）在该 prompt 与 W16 16/16 一致。分歧仅源于 tensor-core GEMM
  的 bf16 反量化舍入经 24 层传导的 ~1e-3 噪声，属边界翻词（top-1 5/5 不受影响）。
  用户确认接受（方案 a），不引入 hi/lo 拆分或回退 fp32-FMA。

## Canonical 端到端（32 prompt + 128 output，5/20）

| Path | Prefill tok/s | TTFT | Decode tok/s | Decode CV | Total latency | J/token |
|---|---:|---:|---:|---:|---:|---:|
| W8 custom (CP20) | 1193.68 | 26.81 ms | 80.01 | 0.1336% | 1614.1 ms | 0.226 |
| W16 custom (CP19) | 1656.25 | 19.32 ms | 67.92 | 0.2293% | 1889.3 ms | 0.274 |
| W8 旧 (CP17) | 42.16 | 758.96 ms | 33.57 | 0.012% | 4541.8 ms | 0.522 |

## Prompt sweep（1 token 输出，5/20，串行重测）

| Prompt | W8 custom | W16 custom | attainment |
|---:|---:|---:|---:|
| 1 | 33.6 tok/s | 14.2 tok/s | 236.4% |
| 32 | 1192.8 tok/s | 1658.7 tok/s | 71.9% |
| 128 | 1881.4 tok/s | 2228.2 tok/s | 84.4% |
| 512 | 985.1 tok/s | 1034.8 tok/s | 95.2% |

Prefill 随 prompt 变长而逼近 W16：反量化指令开销随 token 摊销，长 prompt 更偏
DRAM-bound，W8 减半的权重流量开始兑现。p1 更快是因为 W16 走 generic WMMA
（M=1）而 W8 走 256-thread 逐 row fallback。

## Decode context sweep（128 token 输出，5/20，串行重测）

| Prompt | W8 custom | W16 custom | attainment |
|---:|---:|---:|---:|
| 32 | 79.93 tok/s | 67.71 tok/s | 118.1% |
| 128 | 77.22 tok/s | 65.79 tok/s | 117.4% |
| 512 | 68.18 tok/s | 59.11 tok/s | 115.3% |

所有 context 长度下 W8 decode 均超过 W16（权重流量减半，decode 为
memory-bound）。

## 微基准（20/100，CUDA Event P50，七个 Linear 合计）

| Tokens | W8 GEMM 7-sum | W16 GEMM 7-sum | W8/W16 |
|---:|---:|---:|---:|
| 32 | 0.811 ms | 0.5033 ms | 161% |
| 128 | 1.460 ms | 1.0244 ms | 143% |
| 512 | 4.572 ms | 3.2661 ms | 140% |

| Tokens | W8 GEMV 7-sum | W16 GEMV 7-sum | W8/W16 |
|---:|---:|---:|---:|
| 1 | 0.289 ms | 0.386 ms | 75% |

反量化打包把 GEMM 7-sum 从初版（1.156/1.831/5.776 ms）降到 70%/80%/79%。
GEMM 与 W16 的剩余差距是 bf16 tensor-core 路径固有的反量化指令成本（每元素
I2F+FMUL+F2F，约 5 指令/element vs mma ~0.25 指令/element），已接近该路径的
实际下限；未再做 hi/lo 拆分等折衷。

## Nsight Systems（32 prompt + 16 output ×4）

| 项 | 值 |
|---|---:|
| decode token 中位数 | 12.86 ms（W16 15.05 ms） |
| prefill 中位数 | 27.36 ms（W16 23.82 ms） |
| `gemv_w8a16_fast_kernel<8,1>` GPU 时间占比 | 38.0% |
| `lm_head_bf16_kernel` | 27.8% |
| `gemv_w8a16_fast_kernel<2,4>` | 14.4% |
| `gemm_w8a16_wmma_async_kernel`（prefill） | 9.4% |
| `attention_decode_bf16_kernel` | 2.7% |

## Nsight Compute（用户手动执行，`--set full`）

命令（修正后）：

```text
sudo /usr/local/cuda-12.6/bin/ncu --set full \
  --kernel-name "regex:gemv_w8a16_fast_kernel|lm_head_bf16_kernel" \
  --launch-count 2 --target-processes all build/llm_infer generate ...
```

- `lm_head_bf16_kernel`（18992 blocks × 256，duration `3.50 ms`）：
  memory throughput `69.64%`，compute `41.89%`，L2 hit `1.91%`；warp 每发指
  令 17.4 cycle，其中 57.8% 为 long-scoreboard L1TEX stall；occupancy 61%
  （56 regs 受限）。与 CP19 的 profile（69.45% / 3.51 ms）一致 —— W8 的
  tied LM Head 与 W16 共用同一 BF16 融合 kernel，无变化，仍为带宽受限。
- `gemv_w8a16_fast_kernel<8,1>`（112 blocks × 256，duration `23.84 us`）：
  memory throughput `32.91%`，compute `37.46%`，L2 hit `2.86%`；warp 每发指
  令 16.0 cycle，其中 51% 为 long-scoreboard L1TEX stall；eligible warps
  `0.82`/scheduler，3.5 waves/SM。与 CP19 的 BF16 GEMV 同属 small-grid
  latency-bound 形态；W8 以一半的权重字节达到相同形态并反超 W16 decode。
- `<2,4>`（down_proj）实例未进入本次 profiled 集合（结构相同，4 warps/row）。

## 内存与能耗

| Path | Device weights | Archive | J/token |
|---|---:|---:|---:|
| W8 custom | 611.71 MiB | 612.71 MiB | 0.226 |
| W16 custom | 942.29 MiB | 943.29 MiB | 0.274 |

W8A16 较 W16A16 节省 330.6 MiB（35.1%），且能量/token 更低。

## 验收门槛

| 门槛 | 要求 | 实测 | 结果 |
|---|---:|---:|---|
| Decode vs W16 | ≥ 85% | 117.8% | 通过 |
| Prefill p32 vs W16 | ≥ 70% | 71.9% | 通过 |
| Prefill p128 vs W16 | ≥ 80% | 84.4% | 通过 |
| Prefill p512 vs W16 | ≥ 85% | 95.2% | 通过 |
| Device weights | ≤ W16 的 70% | 64.9% | 通过 |
| Decode CV | ≤ 3% | 0.1336% | 通过 |
| 5-prompt top-1 | = W16 | 5/5 | 通过 |
| canonical greedy | = W16 16/16 | 14/16 | 未满足，已确认接受（见正确性） |

## 原始数据

Git 忽略目录 `results/`：

- `results/codex-cp20-canonical-w8a16-custom.json`
- `results/codex-cp20-sweep-w8a16-custom-p{1,32,128,512}.json`
- `results/codex-cp20-decode-ctx-w8a16-custom-p{32,128,512}.json`
- `results/codex-cp20-nsys-w8a16-custom.nsys-rep/.sqlite`
- （待补）`results/codex-cp20-ncu-w8a16.txt`

审核用机器可读摘要：

- `benchmarks/results/codex-cp20-w8a16-gemv-gemm-summary.json`
