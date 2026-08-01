# CODEX-CP04｜CUDA FP32 与 W8A32 串行参考

## 结论

Checkpoint 4 已通过。在 Jetson Orin Nano Super 8GB 上，现有 CUDA FP32 串行路径达到 22.878 tok/s，现有 W8A32 串行路径达到 33.052 tok/s。W8A32 相对 CUDA FP32 的 decode、TTFT 和端到端加速都约为 1.45 倍，同时 device weights 减少 53.19%，能耗/token 降低 27.98%。

这两条路径仍会对 32 个 prompt token 逐个调用 `forward_token()`。它们是矩阵化 Prefill 之前的 CUDA 参考，不是真正的 Prefill 实现。

## 正式测试协议

- 模型：Qwen2.5-0.5B-Instruct。
- 代码 commit：`fc0dd53ae23f8cd767eb5df612cef560276a0161`。
- 工作负载：固定 32-token prompt、固定生成 128 token、`max-seq-len=256`、batch=1、greedy argmax、不提前停止。
- 协议：5 次 warmup、20 次正式测量，进程固定在 CPU 0-5，OpenBLAS/OMP 环境线程数为 1。
- 设备：Jetson Orin Nano Super 8GB、MAXN_SUPER、CUDA 12.6.68、Release 构建。
- 每组测试前启用 `jetson_clocks`，结束后恢复。最终确认 CPU governor 为 `schedutil`、EMC `FreqOverride=0`、风扇为动态控制。
- FP32：各层 Linear 使用自研 FP32 GEMV；tied LM Head 使用 cuBLAS GEMV。
- W8A32：各层 Linear 使用 group INT8 weight × FP32 activation 自研 GEMV；embedding、激活、KV Cache 和 tied LM Head 仍为 FP32。

## 正式结果

| 指标 | CUDA FP32 | CUDA W8A32 | W8A32 相对 FP32 |
|---|---:|---:|---:|
| TTFT mean | 1388.867 ms | 958.335 ms | 1.449× |
| TTFT p95 | 1389.076 ms | 958.663 ms | — |
| 串行 prompt 吞吐 mean | 23.040 tok/s | 33.391 tok/s | 1.449× |
| 串行 prompt 吞吐 p05 | 23.034 tok/s | 33.374 tok/s | — |
| Decode 吞吐 mean | 22.878 tok/s | 33.052 tok/s | 1.445× |
| Decode 吞吐 p05 | 22.874 tok/s | 33.041 tok/s | — |
| TPOT mean | 43.711 ms | 30.255 ms | 1.445× |
| 总延迟 mean | 6.940 s | 4.801 s | 1.446× |
| Decode CV | 0.0093% | 0.0167% | 均通过 ≤3% |
| 模型归档 | 1885.59 MiB | 883.16 MiB | -53.16% |
| Device weights | 1884.59 MiB | 882.16 MiB | -53.19% |
| 峰值 RSS | 4092.55 MiB | 2085.98 MiB | -49.03% |
| 平均 VDD_IN | 14.650 W | 15.254 W | +4.12% |
| 能耗/token | 0.793 J | 0.571 J | -27.98% |
| GPU 最高温度 | 69.343 °C | 71.625 °C | +2.282 °C |

W8A32 的平均功耗略高，但完成相同工作所需时间明显更短，因此能耗/token 更低。量化收益不能只看瞬时功耗。

与六线程 CPU FP32 基线相比：

- CUDA FP32 decode 加速 1.723 倍，TTFT 加速 1.673 倍，能耗/token 降低 34.88%。
- CUDA W8A32 decode 加速 2.489 倍，TTFT 加速 2.424 倍，能耗/token 降低 53.10%。

## Nsight Systems 短序列设计

短 trace 使用相同的 32-token prompt，但只生成 8 token，所以包含：

- 32 次 prompt `forward_token()`；
- 7 次 decode `forward_token()`，因为首个生成 token 来自最后一个 prompt forward；
- 共 39 次 forward；
- 每次 forward 有 388 次 kernel launch 和 1 次 argmax D2H；
- prompt 阶段共发射 12,416 个 kernel。

原始 trace 使用 Nsight Systems 2024.5.4，采集 `cuda,nvtx`，NVTX 范围为 `qwen2.prefill`、`qwen2.decode` 和 `qwen2.forward_token`。

## Prompt 阶段 GPU kernel 分解

下表的百分比以 `qwen2.prefill` 范围内的 GPU kernel 时间为分母。Launch 是 CPU Runtime API 时间，单独统计，不能再加到 GPU kernel 时间上。

| 类别 | FP32 时间 / 占比 | W8A32 时间 / 占比 |
|---|---:|---:|
| 各层 Linear GEMV | 1180.747 ms / 83.78% | 750.219 ms / 76.84% |
| Tied LM Head GEMV | 174.129 ms / 12.35% | 174.494 ms / 17.87% |
| Causal attention | 10.178 ms / 0.72% | 10.213 ms / 1.05% |
| Argmax kernel | 13.852 ms / 0.98% | 13.856 ms / 1.42% |
| RMSNorm/RoPE/KV/残差/SwiGLU/Embedding | 30.482 ms / 2.16% | 27.606 ms / 2.83% |
| GPU kernel 合计 | 1409.389 ms | 976.388 ms |
| Prefill NVTX wall | 1447.652 ms | 1026.354 ms |

### 1. GEMV 是当前主耗时

W8A32 将各层 Linear GEMV 的 GPU 时间从 1180.747 ms 降到 750.219 ms，即 1.574 倍加速。这是 W8A32 端到端提升的主要来源。

但它仍是逐 token GEMV。后续矩阵化 Prefill 的核心不是继续优化 32 次独立 GEMV，而是让一个层的多个 token 共同进入 GEMM。

### 2. 重复 LM Head 已成为明确浪费

当前 32-token prompt 会执行 32 次 tied LM Head 和 32 次 argmax，但实际上只有最后一个 prompt token 的结果用于生成首 token。

- FP32：32 次 LM Head 为 174.129 ms，32 次 argmax 为 13.852 ms。只保留最后一次，预计可消除约 182.106 ms GPU kernel 工作，约占该 trace 的 12.58% TTFT。
- W8A32：预计可消除约 182.464 ms，约占 TTFT 的 17.78%。

W8A32 的 tied LM Head 仍来自 FP32 embedding，没有被 INT8 量化。其绝对时间几乎没有变化，因此在各层 GEMV 变快后，LM Head 占比从 12.35% 上升到 17.87%。这是典型的 Amdahl 定律现象。

### 3. Kernel launch 开销不可忽略

| Launch 指标 | FP32 | W8A32 |
|---|---:|---:|
| Prompt launch 次数 | 12,416 | 12,416 |
| `cudaLaunchKernel` API 总时间 | 85.649 ms | 85.064 ms |
| 平均每次 launch | 6.898 μs | 6.851 μs |
| 占 Prefill NVTX wall | 5.92% | 8.29% |

INT8 kernel 变快后，launch 总时间几乎不变，所以其占比进一步上升。矩阵化 Prefill 会同时减少 Linear 调用次数和大量逐 token kernel launch。

每个 token 结束还有一次 `cudaMemcpyAsync` 将 argmax 结果复制到 pageable host memory。Nsight 将此前 stream 的等待时间计入该 API，所以它的 API duration 不能解释成纯 4-byte D2H 成本；它表示当前每 token 都建立了一个主机同步边界。

### 4. Attention 目前不是主瓶颈

32-token prompt 下 attention 只占 GPU kernel 时间的 0.72%（FP32）或 1.05%（W8A32）。这不代表长 prompt 下也永远很小，但说明当前阶段优先级应是矩阵化 Linear、移除重复 LM Head/argmax 和减少 launch，而不是先实现 FlashAttention。

## 测量异常记录

第一次 CUDA FP32 正式启动在产生任何测量样本前遇到共享内存瞬时压力，`cudaMalloc` 返回 OOM。runner 已恢复时钟且没有保留部分结果。随后检查发现系统已回收缓存、模型驻留页为 0；未杀死进程、未手工清缓存。FP32 和 W8A32 冒烟均成功后，重新执行完整 5/20 协议并得到本报告中的有效结果。

这也说明 8GB 统一内存设备上的 FP32 路径有较明显内存压力：正式运行峰值 RSS 约 4.00 GiB，同时 device weights 约 1.84 GiB。

## 数据与复现

正式命令遵循：

```bash
sudo -v
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 0-5 python3 tools/run_benchmark.py \
  --binary build/llm_infer \
  --model models/qwen2.5-0.5b-instruct \
  --backend cuda --precision fp32 \
  --max-new-tokens 128 --max-seq-len 256 \
  --warmup 5 --repeat 20 --telemetry-interval-ms 100 \
  --lock-clocks --output results/cuda-fp32-serial.json
```

W8A32 使用相同命令，将 precision 改为 `w8a32`、输出文件改为 `cuda-w8a32-serial.json`。

原始 benchmark JSON 位于忽略的 `results/`；原始 `.nsys-rep` 和导出 SQLite 位于忽略的 `profiles/`；审核后的数据位于 `benchmarks/results/codex-cp04-cuda-fp32-w8a32-serial-summary.json`。摘要保存了模型与 trace SHA-256。

后续 A16 路线按项目约定使用 BF16：W16A16 表示 BF16 权重/激活基线，W8A16 表示 INT8 Linear 权重与 BF16 激活，不采用 FP16。
