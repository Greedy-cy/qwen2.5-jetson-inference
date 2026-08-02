# CODEX-CP18A：W16A16 Custom 对照基线与差距量化

## 结论摘要

本检查点固定 CUDA W16A16（BF16）数据类型、模型、输入、时钟和测量协议，只切换
`--linear-kernel custom|cublas`，建立后续自研算子优化的零点。此阶段没有修改 kernel。

- Canonical（32-token Prefill、128-token Decode）中，custom 为 `64.37 prefill tok/s`
  和 `27.76 decode tok/s`；cuBLAS 为 `1910.15 prefill tok/s` 和 `67.75 decode tok/s`。
- custom 的端到端 Prefill/Decode 吞吐分别达到 cuBLAS 的 `3.37%`/`40.97%`；总延迟
  为 `5072.10 ms` vs `1891.19 ms`。
- 真实 Qwen 线性层微基准显示：custom Decode 线性层加权耗时达到 cuBLAS 吞吐的
  `41.86%`，而一层 Prefill GEMM 在 p32/p128/p512 仅为 `2.18%/0.74%/0.60%`。
- Nsight Systems 中 custom 的 BF16 GEMV/GEMM 分别占 GPU kernel 时间 `51.4%/45.9%`；
  cuBLAS 的 Ampere BF16 Tensor Core GEMM 合计占 `82.9%`。

因此新路线应拆成两个小检查点：先把 Prefill 的非 Tensor-Core 朴素 tiled GEMM 替换为
真正利用 SM87 Tensor Core 的自研 BF16 GEMM，再单独优化 Decode GEMV 与 tied LM Head。

## 测试身份与边界

- 数据提交：`65d75edd16e436e327d57d17c8f0ac4cab087d23`
- 测量时工作树：clean
- 模型：`Qwen2.5-0.5B-Instruct`
- W16A16 含义：BF16 weight/activation/KV/logits，敏感归约和 cuBLAS compute 使用 FP32
- 模型归档 SHA-256：`f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`
- 模型归档：`943.29 MiB`；device weights：`942.29 MiB`
- 设备：Jetson Orin Nano Super 8GB，Jetson Linux R36.4.7，CUDA 12.6，SM87
- 功耗模式：MAXN_SUPER
- 时钟：每组测试临时 `jetson_clocks`，trap/finally 恢复；结束后 governor 为
  `schedutil`、GPU FreqOverride 为 `0`
- Nsight Systems：`2024.5.4.34-245434855735v0`
- Prefill：项目当前唯一的矩阵化 Prefill 基线，不包含 serial/伪 Prefill 对比

两条路径只在线性层实现上不同：

- `custom`：Decode 使用自研 `gemv_bf16_kernel`；Prefill 使用 16x16 朴素
  `gemm_bf16_tiled_kernel`，尚未使用 Tensor Core。
- `cublas`：使用 `cublasGemmEx`，BF16 输入/权重/输出、FP32 accumulation，由库选择
  SM87 Ampere BF16 Tensor Core kernel。

## 测量协议

Canonical：batch=1、固定 32-token prompt、128-token greedy 输出、
`max-seq-len=256`、5 次 warmup、20 次测量、tegrastats 100 ms。

Prompt sweep：prompt=`1/32/128/512`，输出 1 token，`max-seq-len=512`，每点 5/20。
128/512 token 由固定 32-token 序列重复得到。

真实形状微基准：Q/K/V/O、gate/up/down 与 tied LM Head；每个 case 同时执行 custom 和
cuBLAS，20 次 warmup、100 次 CUDA Event 测量。GEMM 分别测 `M=32/128/512`，GEMV
测 `M=1`。一层 GEMM 合计为七个 Transformer Linear 的 P50 之和；Decode 加权合计为
`24 × 一层七个 GEMV + tied LM Head GEMV`。

Nsight：32-token prompt、16-token 输出、1 次 warmup、3 次测量；trace 初始化，因此
CUDA API 时间不作为正式性能数字，只使用 GPU kernel summary 和 NVTX 阶段中位数定位瓶颈。

## Canonical 端到端结果

| Linear path | TTFT mean | TTFT p50/p95 | Prefill tok/s | Decode tok/s | Decode CV | Total latency |
|---|---:|---:|---:|---:|---:|---:|
| custom | 497.15 ms | 497.17 / 497.28 ms | 64.37 | 27.76 | 0.0176% | 5072.10 ms |
| cuBLAS | 16.75 ms | 16.75 / 16.77 ms | 1910.15 | 67.75 | 0.0153% | 1891.19 ms |

| 指标 | custom 相对 cuBLAS |
|---|---:|
| Prefill throughput attainment | 3.37% |
| Decode throughput attainment | 40.97% |
| TTFT | 29.68x |
| Total latency | 2.68x |

custom 平均功率更低（`13.31 W` vs `18.16 W`），但因执行更久，单位输出 token 能耗更高：
`0.527 J/token` vs `0.268 J/token`，即约为 cuBLAS 的 `1.97x`。两条 Decode CV 都远低于
`3%` 门槛。

## Prompt sweep

| Prompt | custom TTFT | cuBLAS TTFT | custom Prefill | cuBLAS Prefill | custom attainment |
|---:|---:|---:|---:|---:|---:|
| 1 | 258.19 ms | 14.33 ms | 3.87 tok/s | 69.81 tok/s | 5.55% |
| 32 | 497.20 ms | 16.79 ms | 64.36 tok/s | 1906.45 tok/s | 3.38% |
| 128 | 1952.26 ms | 46.06 ms | 65.56 tok/s | 2778.86 tok/s | 2.36% |
| 512 | 8045.67 ms | 455.48 ms | 63.64 tok/s | 1124.08 tok/s | 5.66% |

所有 sweep TTFT CV 均低于 `0.10%`。p1 仍走矩阵化 GEMM 路径；custom 16x16 tile 在
M=1 时存在大量无效线程，因此这里不是 Decode GEMV 的性能数字。p512 时 cuBLAS Linear
已很快，causal attention 在 TTFT 中占比上升，所以端到端 attainment 高于纯 GEMM 比例。

## 真实 Qwen 形状微基准

### Decode GEMV（P50，M=1）

| Linear | Shape (N x K) | custom | cuBLAS | custom throughput attainment |
|---|---:|---:|---:|---:|
| q_proj | 896 x 896 | 0.0521 ms | 0.0367 ms | 70.39% |
| k_proj | 128 x 896 | 0.0147 ms | 0.0180 ms | 122.66% |
| v_proj | 128 x 896 | 0.0140 ms | 0.0181 ms | 128.93% |
| o_proj | 896 x 896 | 0.0498 ms | 0.0300 ms | 60.31% |
| gate_proj | 4864 x 896 | 0.3124 ms | 0.1119 ms | 35.83% |
| up_proj | 4864 x 896 | 0.3128 ms | 0.1112 ms | 35.55% |
| down_proj | 896 x 4864 | 0.1779 ms | 0.1169 ms | 65.71% |
| tied LM Head | 151936 x 896 | 9.5268 ms | 2.7387 ms | 28.75% |

一层七个 GEMV P50 合计为 `0.9337 ms` vs `0.4428 ms`。按 24 层并加一次 LM Head
加权后为 `31.9363 ms` vs `13.3670 ms`，custom 达到 cuBLAS 的 `41.86%`，与端到端
Decode 的 `40.97%` 高度一致。最大机会点是 LM Head 和 gate/up projection；小尺寸
K/V projection 已略快于 cuBLAS，不应作为首要优化对象。

### Prefill GEMM（一层七个 Linear 的 P50 合计）

| Tokens | custom | cuBLAS | custom throughput attainment | custom slowdown |
|---:|---:|---:|---:|---:|
| 32 | 20.1516 ms | 0.4393 ms | 2.18% | 45.87x |
| 128 | 79.6883 ms | 0.5932 ms | 0.74% | 134.33x |
| 512 | 317.8669 ms | 1.8999 ms | 0.60% | 167.31x |

custom tiled GEMM 耗时近似随 token 数线性增长；cuBLAS 则能随 M 增大提高 Tensor Core
利用率，因此差距继续扩大。以 p32 为例，Q/O GEMM 分别慢约 `32.0x/41.6x`，三个 MLP
GEMM 慢约 `51.6x/52.0x/55.7x`。

### 正确性

微基准逐 case 比较 custom、cuBLAS 和 sampled FP32 reference。所有 case 均完成；
custom 与 cuBLAS 的最大差异相对输出峰值不超过 `0.006579`。此前三项端到端 BF16
正确性测试也全部通过，包括 custom/cuBLAS 对照与 Prefill 后继续 Decode 16 token。

## Nsight Systems 结论

| Path | GPU kernel time 主项 | 占比 | NVTX Prefill median | NVTX Decode token median |
|---|---|---:|---:|---:|
| custom | `gemv_bf16_kernel` | 51.4% | 497.98 ms | 37.62 ms |
| custom | `gemm_bf16_tiled_kernel` | 45.9% | 同上 | 同上 |
| cuBLAS | Ampere BF16 Tensor Core GEMM（两类） | 82.9% | 17.10 ms | 15.03 ms |

其他 custom kernel 单项均低于 `1%`，说明在 Linear 改善前优化 attention、RMSNorm、RoPE
或 argmax 不会改变结论。两条 trace 均只在初始化阶段出现 9 次 `cudaMalloc`，未观察到
逐 token 动态分配。

原始 trace 位于被 Git 忽略的：

- `results/codex-cp18a-nsys-w16a16-custom.nsys-rep`
- `results/codex-cp18a-nsys-w16a16-cublas.nsys-rep`

## CP18A 验收与下一步

CP18A 已完成：相同精度、相同 workload、相同提交下的 custom/cuBLAS canonical、prompt
sweep、真实形状微基准和 Nsight 证据齐全，稳定性与正确性门槛通过。这里记录的是优化前
baseline，不能作为“custom 已接近 cuBLAS”的成果表述。

建议下一检查点 CP18B 只处理 Prefill BF16 GEMM：设计 SM87 Tensor Core 路径，覆盖
Q/K/V/O 与 MLP 的真实非整齐维度，保留 FP32 accumulation，并以 p32/p128/p512 的
一层七 GEMM 合计和端到端 Prefill 对照 cuBLAS。阶段目标可设为真实形状合计至少达到
cuBLAS 的 `70%`，stretch goal 为 `80%`；正确性不退化。达到或定位新瓶颈后停止，再
决定是否进入 Decode GEMV/LM Head 优化。

原始数据均在 Git 忽略的 `results/`：

- `results/codex-cp18a-canonical-w16a16-{custom,cublas}.json`
- `results/codex-cp18a-sweep-w16a16-{custom,cublas}-p{1,32,128,512}.json`
- `results/codex-cp18a-bf16-linear-p{32,128,512}.json`
- `results/codex-cp18a-nsys-w16a16-{custom,cublas}.{nsys-rep,stats.txt}`
