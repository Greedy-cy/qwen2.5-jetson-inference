# Qwen2.5 端侧推理框架与 CUDA 算子优化

面向 Jetson Orin Nano Super 的 C++17/CUDA Qwen2.5 batch=1 推理框架。项目以
`Qwen2.5-0.5B-Instruct` 为固定模型，实现矩阵化 Prefill、KV Cache 和自回归 Decode，
并保留四条公开路径：

| Backend | Precision | Linear 实现 | Activation / KV | Accumulation |
|---|---|---|---|---|
| CPU | FP32 | OpenBLAS + CPU kernels | FP32 | FP32 |
| CUDA | W16A16 | custom | BF16 | FP32 |
| CUDA | W16A16 | cuBLAS baseline | BF16 | FP32 |
| CUDA | W8A16 | custom group-64 weight-only | BF16 | FP32 |

这里的 A16/W16 均指 **BF16**，不使用 FP16。CUDA FP32 和旧 W8A32 实验路径已从
最终运行时移除，使项目集中展示 BF16 与 W8A16 的真实优化路线。

## 主要结果

设备为 Jetson Orin Nano Super 8GB、CUDA 12.6、MAXN_SUPER。统一 workload 为
32-token prompt + 128-token output，5 次 warmup + 20 次测量。

| 路径 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | TPOT (ms) | Device weights |
|---|---:|---:|---:|---:|---:|
| CPU FP32 reference | 353.36 | 90.56 | 13.51 | 74.00 | - |
| CUDA BF16 cuBLAS | 16.79 | 1906.25 | 67.70 | 14.77 | 942.29 MiB |
| CUDA BF16 custom | 19.43 | 1647.29 | 68.19 | 14.66 | 942.29 MiB |
| CUDA W8A16 custom | 26.81 | 1193.53 | 80.25 | 12.46 | 611.71 MiB |

自研 BF16 Decode 达到 cuBLAS baseline 的 100.72%；W8A16 Decode 相对自研 BF16
提高 17.68%，device weights 减少 35.08%，canonical energy/token 减少 19.09%。
W8A16 Prefill 仍慢于 BF16，因为 fused dequantize GEMM 尚未达到纯 BF16 GEMM 的效率；
项目不会把量化描述成所有阶段都加速。

完整协议、prompt/context sweep、功耗和 Nsight 结论见
[最终性能报告](docs/performance.md)。

## 构建

目标平台固定为 Linux aarch64、CUDA 12.6、SM87；项目不维护跨平台兼容层。

```bash
cd /path/to/qwen2.5-jetson-inference
bash scripts/bootstrap.sh

.venv/bin/python tools/download_model.py \
  --output models/source/Qwen2.5-0.5B-Instruct

.venv/bin/python tools/export_model.py \
  --source models/source/Qwen2.5-0.5B-Instruct \
  --output models/qwen2.5-0.5b-instruct \
  --precision all \
  --group-size 64

export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --parallel "$(nproc)"
ctest --test-dir build --output-on-failure
```

导出目录包含：

- `model.fp32.qbin`：CPU FP32 reference；
- `model.w16a16.qbin`：CUDA BF16；
- `model.w8a16.qbin`：group-size 64 INT8 Linear weights + BF16 scale/非量化权重。

## 单次推理

CUDA BF16 默认使用自研 Linear：

```bash
./build/llm_infer generate \
  --model models/qwen2.5-0.5b-instruct \
  --backend cuda \
  --precision w16a16 \
  --prompt "请简要介绍 CUDA。" \
  --max-new-tokens 32 \
  --max-seq-len 256
```

显式选择 cuBLAS baseline：

```bash
./build/llm_infer generate \
  --model models/qwen2.5-0.5b-instruct \
  --backend cuda \
  --precision w16a16 \
  --linear-kernel cublas \
  --prompt "请简要介绍 CUDA。" \
  --max-new-tokens 32
```

W8A16 固定使用自研在线反量化 kernel：

```bash
./build/llm_infer generate \
  --model models/qwen2.5-0.5b-instruct \
  --backend cuda \
  --precision w8a16 \
  --prompt "请简要介绍 CUDA。" \
  --max-new-tokens 32
```

CPU reference：

```bash
./build/llm_infer generate \
  --model models/qwen2.5-0.5b-instruct \
  --backend cpu \
  --precision fp32 \
  --prompt "请简要介绍 CUDA。" \
  --max-new-tokens 32
```

CUDA FP32、CPU 非 FP32、W8A16 + cuBLAS 等无效组合会显式报错。

## Benchmark

runner 使用固定 token IDs，记录逐轮 TTFT、Prefill/Decode 吞吐、TPOT、总延迟、
mean/p50/p95/CV、RSS、GPU 内存、系统 RAM、频率、温度、功率与 energy/token，并绑定
Git commit、模型 SHA-256、CUDA 和功耗模式。

```bash
mkdir -p results
python3 tools/run_benchmark.py \
  --binary build/llm_infer \
  --model models/qwen2.5-0.5b-instruct \
  --backend cuda \
  --precision w16a16 \
  --linear-kernel custom \
  --warmup 5 \
  --repeat 20 \
  --lock-clocks \
  --output results/w16-custom.json
```

`--lock-clocks` 会在测量期间临时启用 Jetson 锁频，并在退出路径恢复原状态；脚本和
结果中不保存 sudo 密码。正式数据应在 clean commit 上生成，dirty worktree 默认被拒绝。

## 设计与验证

- [运行时架构](docs/architecture.md)：模型归档、Prefill/Decode、KV Cache 与数据布局；
- [CUDA 优化](docs/cuda-optimizations.md)：BF16 Tensor Core、shape-aware GEMV、融合
  LM Head、W8A16 在线反量化；
- [最终性能报告](docs/performance.md)：统一 benchmark、内存/功耗与 Nsight；
- [简历表述](docs/resume.md)：中英文可核验 bullets；
- [面试讲解](docs/interview-guide.md)：核心设计选择、瓶颈与限制。

## 项目边界

当前版本固定 batch=1 greedy generation，不实现 FlashAttention、chunked Prefill、
continuous batching、多 GPU 或通用跨平台后端。W8A16 对 5 个固定 prompt 的首 token
top-1 为 5/5，但 canonical 16-token greedy 与 BF16 为 14/16 一致；该已知差异来自
INT8 在线反量化到 BF16 后的舍入误差，详见最终性能报告。
