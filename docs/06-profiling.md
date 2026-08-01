# 06 Benchmark 与 Profiling

标准口径是 batch=1、32-token prompt、128-token greedy output、5 次 warmup、20 次测量。
JSON 报告包含 TTFT、端到端 p50/p95、decode tok/s、RSS、archive、workspace、KV Cache
和 device weight 字节数。

Nsight Systems 使用 NVTX 标记 prefill、decode 和 token forward：

```bash
MODEL_DIR="$PWD/models/qwen2.5-0.5b-instruct" bash scripts/profile.sh
nsys stats profiles/qwen_fp32.nsys-rep
```

分析顺序：先确认没有 decode 内 `cudaMalloc/cudaMemcpy(H2D)`，再按总时长排序 kernel，
最后判断瓶颈是权重带宽、归约效率还是 launch 数量。Jetson 调优完成后，在 RTX 3090
云机用相同 JSON 协议重跑，不直接比较不同 prompt、dtype 或计时边界的数字。
