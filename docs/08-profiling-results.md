# 08 Nsight Systems 实测

在 Jetson Orin 上分别采集 FP32 与 INT8 的 CUDA+NVTX trace。短生成仅用于定位 kernel
占比，标准吞吐仍以 `07-jetson-results.md` 的 32+128、5/20 结果为准。

| Kernel 类别 | FP32 时间占比 | INT8 时间占比 |
|---|---:|---:|
| 自研 GEMV | 83.1% | 73.5% |
| cuBLAS tied LM head | 13.1% | 21.1% |
| Attention | 0.7% | 1.1% |
| Argmax | 1.0% | 1.5% |

原始报告位于 `profiles/qwen_fp32.nsys-rep` 和 `profiles/qwen_int8.nsys-rep`。两条路径
都由 Linear/LM head 主导；Attention 在 batch=1 decode 中不是首要瓶颈。INT8 将自研
Linear 占比降至 73.5%，因此下一步应优化或量化仍为 FP32 的 tied LM head，并减少
每 token 的 24 层 kernel launch 数量。

复现：

```bash
MODEL_DIR="$PWD/models/qwen2.5-0.5b-instruct" \
  PRECISION=fp32 OUTPUT_NAME=qwen_fp32 bash scripts/profile.sh

MODEL_DIR="$PWD/models/qwen2.5-0.5b-instruct" \
  PRECISION=int8 OUTPUT_NAME=qwen_int8 bash scripts/profile.sh
```
