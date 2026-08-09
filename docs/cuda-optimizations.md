# CUDA 算子设计与优化

## 1. 优化方法

项目先以 BF16 cuBLAS 建立同设备、同数据类型、同模型图的上限基线，再用 Nsight
Systems 定位 Prefill/Decode 的阶段占比，用 Nsight Compute 观察 memory throughput、
occupancy、warp stall 与 shared-memory wavefront。优化只针对真实 Qwen shape，并通过
端到端吞吐确认，避免只追求孤立 kernel 的峰值。

Qwen2.5-0.5B 的关键维度包括 hidden size 896、intermediate size 4864、GQA Q/K/V/O、
gate/up/down 和 `151936 x 896` tied LM Head。

## 2. BF16 Prefill GEMM

Prefill 输入为 `[tokens, hidden]`，Linear 是多行矩阵乘。自研路径采用 BF16 Tensor
Core WMMA 和 FP32 accumulator：

- p32 使用 `M32N64K64` tile；
- p128/p512 使用 `M128N128K32` tile；
- BF16 A/B tile 用 `cp.async` 搬入 shared memory；
- pipeline 让下一 K tile 的 global-memory copy 与当前 tile MMA 重叠；
- bias 与 BF16 output conversion 融合；
- tail 路径处理非整齐 shape。

最终 BF16 custom 在 p32/p128/p512 分别达到 cuBLAS 的 86.74%、80.11%、93.20%。
p128 达到预设 80% 门槛后，没有继续为单个 microbenchmark 引入更复杂的 swizzle 分支。

## 3. Shape-aware BF16 Decode GEMV

batch=1 Decode 的 Linear 每次只处理一行，核心从 GEMM 变为 GEMV，主要受权重读取带宽
而非 Tensor Core 峰值限制。旧实现每个 row 使用 256-thread block 和多轮 shared-memory
reduction，存在同步与低效 reduction 开销。

最终 kernel 按 K 自动调度：

- `K <= 1024`：一个 warp 计算一个 output row，每 block 处理 8 rows；
- `K > 1024`：多个 warp 协作一个 row，每 block 处理 2 rows，覆盖 down projection；
- BF16 weight/input 使用 16-byte 对齐向量加载；
- input 由 block 协作载入 shared memory，供多个 output rows 复用；
- FP32 FMA 后用 warp shuffle reduction；
- bias 在写回 BF16 前融合；
- 非对齐 K 走显式安全 tail。

一层七个 Linear 的 GEMV 合计耗时为 0.3779 ms，cuBLAS 为 0.4453 ms；canonical
Decode 达到 68.19 tok/s，cuBLAS baseline 为 67.70 tok/s。

## 4. Tied LM Head + argmax

LM Head 为 `151936 x 896` GEMV，在每个 Decode step 都扫描约 136M 个 BF16 元素，是
不能忽略的带宽开销。自研实现分两阶段：

1. 第一阶段同时写完整 BF16 logits 和每个 block 的 `(max_value, max_index)`；
2. 第二阶段只归约 block maxima，并在相同最大值时选择最小 token index。

归约 workspace 预分配，推理循环中没有 `cudaMalloc`。完整 logits 仍被保留，因此
`last_logits_host()` 不因融合而改变语义。W16/W8 custom 共用这一实现；cuBLAS baseline
继续使用 `cublasGemmEx + argmax_bf16`。

p32 下 custom LM Head 为 3.2811 ms，cuBLAS + argmax 为 3.1745 ms，即 96.75%。Nsight
显示它在 W8 Decode 中占 GPU kernel 时间 28.3%，因为 W8 Transformer GEMV 已经更快。

## 5. W8A16 Decode GEMV

W8A16 的目标是减少每 token 必须读取的 Linear weight，而不是生成完整反量化矩阵：

- 一个 warp 负责一个 output row，每 block 8 warps/8 rows；
- 每 lane 用 `char4` 连续读取 4 个 INT8 weights；
- 同步读取 4 个 BF16 activations；
- 每 128 个 K 覆盖两个 group-size 64 quant groups；
- scale 由指定 lane 读取并用 shuffle 广播；
- input 放入 shared memory，被 8 个 output rows 复用；
- 在线执行 `int8 * bf16_scale * bf16_activation`，FP32 accumulation；
- bias 融合后写 BF16，K tail 单独处理。

真实七个 Linear 的合计 GEMV 为 0.2896 ms，相对显式反量化 reference 的 0.4448 ms
为 1.54x。端到端 Decode 达到 80.25 tok/s，相对 BF16 custom 提高 17.68%。

## 6. W8A16 Prefill GEMM

Prefill 不能简单复用 GEMV，否则 token 维度没有形成 Tensor Core 计算。自研 fused
dequant GEMM：

- BF16 activation 以 async copy 搬入 shared memory；
- INT8 weight 向量加载，在 tile 内结合 BF16 scale 反量化到 shared BF16；
- WMMA BF16 Tensor Core 计算，FP32 accumulator；
- tile 与 BF16 路径一致，group-size 64 与 K tile 显式对齐；
- 不生成完整 BF16 dequantized weight；
- bias 和 BF16 output conversion 融合。

该路径降低了存储和 HBM/DRAM 权重流量，但多了 scale load、整数转换、反量化和 shared
写入，且 dequant producer 与 MMA consumer 的流水尚不如纯 BF16 GEMM。最终 W8
Prefill 在 p32/p128/p512 为 BF16 custom 的 72.28%/84.66%/94.00%。因此 W8A16 的
准确定位是 Decode、模型体积与能效优化，而不是所有序列长度下的 Prefill 加速。

## 7. Profiling 结论

32 prompt + 16 output 的 Nsight Systems 短 trace：

| 路径 | Prefill NVTX median | Decode NVTX median | kernel 时间主要构成 |
|---|---:|---:|---|
| BF16 cuBLAS | 17.173 ms | 226.231 ms | BF16 GEMM family 72.1% |
| BF16 custom | 19.837 ms | 223.879 ms | GEMV 60.6%，LM Head 25.2%，GEMM 5.2% |
| W8A16 custom | 27.186 ms | 191.183 ms | W8 GEMV 52.4%，LM Head 28.3%，W8 GEMM 9.4% |

这个结果解释了端到端行为：W8 GEMV 缩短 Decode；共享 BF16 LM Head 的绝对耗时基本
不变，因此相对占比上升；Prefill 的下一候选瓶颈仍是 fused dequant GEMM。

## 8. 公平对比原则

- 相同设备、功耗模式、频率、模型归档和 token IDs；
- W16 custom/cuBLAS 共享 attention、非线性、KV Cache 和采样路径；
- 分开报告 TTFT/Prefill 与 Decode/TPOT；
- microbenchmark 使用 20 warmup/100 repeat，端到端使用 5/20；
- 结果绑定 clean commit、模型 SHA-256 和系统信息；
- 性能结果必须同时通过数值、top-1、KV 延续和稳定性检查。
