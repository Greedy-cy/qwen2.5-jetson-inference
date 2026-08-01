# CODEX-CP03｜CPU FP32 串行基线

## 结论

Checkpoint 3 已通过。Qwen2.5-0.5B-Instruct 在 Jetson Orin Nano Super 8GB 上使用 CPU FP32、旧串行 prompt 路径时，六线程正式结果为：TTFT 2.323 s、decode 13.282 tok/s、TPOT 75.338 ms、端到端 11.891 s。decode CV 为 2.410%，低于 3% 验收门槛。

六线程相对单线程的 decode 加速为 2.339 倍，六核扩展效率为 38.98%。这是后续矩阵化 Prefill、CUDA BF16（W16A16）和 W8A16 优化的固定 CPU 对照，不应把它解读成理想的六核线性扩展。

## 测试对象与协议

- 模型：Qwen2.5-0.5B-Instruct，FP32 归档 1885.59 MiB。
- 模型 SHA-256：`a58918c8fcbe0332c3a09d233f14425d1ecf32facca658b92203ff89d81c0b63`。
- 代码 commit：`42ec8a9dc5490635054d2a7eb2f283d1a4d9484a`。
- 构建：CMake Release，OpenBLAS `/usr/lib/aarch64-linux-gnu/libopenblas.so`。
- 设备：Jetson Orin Nano Super 8GB，MAXN_SUPER，CUDA 12.6.68。
- 工作负载：固定 32-token prompt，固定生成 128 token，`max-seq-len=256`，batch=1，greedy argmax，不提前停止。
- 单线程诊断：CPU 0，OpenBLAS/OMP 各 1 线程，1 次 warmup、5 次测量。
- 正式基线：CPU 0-5，OpenBLAS/OMP 各 6 线程，5 次 warmup、20 次测量。
- 每组运行前由 runner 保存当前状态并启用 `jetson_clocks`，退出时在 `finally` 中恢复。运行后验证 CPU governor 为 `schedutil`、EMC `FreqOverride=0`、风扇为动态控制。

当前还没有独立 Prefill API。这里的 `serial_legacy` 会逐个处理 32 个 prompt token，并在每一步执行完整 LM Head 与 argmax。因此表中的“prefill tok/s”只是旧 prompt 阶段的等效吞吐，不是真正 GEMM 化 Prefill 的吞吐。

## 指标口径

- TTFT：从 prompt 阶段开始到获得首个生成 token 的时间。
- Prefill tok/s：`32 / TTFT`；当前仅用于评价旧串行 prompt 路径。
- Decode tok/s：首 token 之后的 127 次 decode 除以 decode 时间。
- TPOT：decode 时间除以 127；越低越好。
- p05 吞吐：较慢端吞吐，用来观察性能掉速。
- p95 延迟：较慢端延迟，用来观察尾部抖动。
- CV：标准差除以均值；正式 decode 要求不超过 3%。
- 能耗/token：正式测量窗口的 VDD_IN 功耗积分除以生成 token 总数。

## 结果

| 指标 | 1 线程诊断（1/5） | 6 线程正式（5/20） | 六线程效果 |
|---|---:|---:|---:|
| TTFT mean | 5567.989 ms | 2323.041 ms | 2.397× |
| TTFT p95 | 5570.740 ms | 2315.510 ms | — |
| 串行 prompt 吞吐 mean | 5.747 tok/s | 13.782 tok/s | 2.398× |
| 串行 prompt 吞吐 p05 | 5.744 tok/s | 12.534 tok/s | — |
| Decode 吞吐 mean | 5.679 tok/s | 13.282 tok/s | 2.339× |
| Decode 吞吐 p05 | 5.677 tok/s | 12.191 tok/s | — |
| TPOT mean | 176.085 ms | 75.338 ms | 2.339× |
| TPOT p95 | 176.139 ms | 79.313 ms | — |
| 总延迟 mean | 27.931 s | 11.891 s | 2.349× |
| Decode CV | 0.016% | 2.410% | 通过 ≤3% |
| 平均 VDD_IN | 9.074 W | 13.121 W | +44.6% |
| 能耗/token | 1.978 J | 1.219 J | 降低 38.4% |
| CPU 最高温度 | 61.593 °C | 69.656 °C | +8.063 °C |
| 峰值 RSS | 2059.48 MiB | 2059.27 MiB | 基本不变 |

六线程 TTFT 有一个 2553.136 ms 的最慢样本，使均值高于 p95；完整 20 轮的 TTFT CV 仍为 2.277%，decode CV 为 2.410%，所以不触发重跑。

## 分析

1. 六线程将 decode 从 5.679 提高到 13.282 tok/s，但只获得 2.339 倍加速。batch=1 的逐 token Linear 主要是 GEMV，权重读取和内存带宽限制使其无法随核心数线性扩展。
2. 六线程平均板级功耗提高 44.6%，但吞吐提升更多，因此能耗/token 降低 38.4%。这说明仅看瞬时功耗会误判能效。
3. 旧 prompt 阶段的吞吐与 decode 接近，且 32 token TTFT 仍需 2.323 s。这与“逐 token prompt”特征一致，也给后续矩阵化 Prefill 提供了明确优化目标。
4. 模型进程峰值 RSS 约 2.01 GiB，其中 FP32 模型归档本身约 1.84 GiB；KV Cache 为 6 MiB，当前内存主要由权重占据。

## 复现命令

runner 内部使用非交互 `sudo -n` 控制时钟，因此执行者先单独运行 `sudo -v`，密码不会写入脚本或结果。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 0 python3 tools/run_benchmark.py \
  --binary build/llm_infer \
  --model models/qwen2.5-0.5b-instruct \
  --backend cpu --precision fp32 \
  --prefill-mode serial \
  --max-new-tokens 128 --max-seq-len 256 \
  --warmup 1 --repeat 5 --telemetry-interval-ms 100 \
  --lock-clocks --output results/cpu-fp32-1thread-serial.json
```

```bash
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6
taskset -c 0-5 python3 tools/run_benchmark.py \
  --binary build/llm_infer \
  --model models/qwen2.5-0.5b-instruct \
  --backend cpu --precision fp32 \
  --prefill-mode serial \
  --max-new-tokens 128 --max-seq-len 256 \
  --warmup 5 --repeat 20 --telemetry-interval-ms 100 \
  --lock-clocks --output results/cpu-fp32-6threads-serial.json
```

原始结果保存在忽略目录 `results/`；审核后的轻量数据位于 `benchmarks/results/codex-cp03-cpu-fp32-serial-summary.json`。后续对比必须引用这里记录的 commit、模型 SHA 和 workload，避免混用不同测试口径。
