# CODEX-CP16B：Jetson-only 代码精简报告

## 检查点边界

- 基线提交：`91281c4728a233063c478b71073a2f4be1f8c58d`
- 代码提交：`043a1d10ab650eefb046afb8d19c8149f292247d`
- 目标设备：Jetson Orin Nano Super 8GB
- 目标系统：Jetson Linux，`aarch64`
- CUDA 架构：SM87
- 本检查点只做目标设备收敛，未开始 CP17 正式性能测试。

## 保留的运行路径

| Backend | Precision | Activation / KV Cache | Accumulation |
|---|---|---|---|
| CPU | FP32 | FP32 | FP32 |
| CUDA | FP32 | FP32 | FP32 |
| CUDA | W16A16 | BF16 | FP32 |
| CUDA | W8A16 | BF16 | FP32 |

所有路径使用矩阵化 Prefill 和单 token Decode；未重新引入 serial/伪 Prefill。

## 删除和收敛内容

- CMake 明确拒绝非 Linux/aarch64 构建。
- CUDA 架构固定为 SM87，不再接受环境变量或外部 CMake 值覆盖。
- CUDA Runtime、cuBLAS 与 NVTX 作为目标环境的必需依赖。
- 删除 Windows 文件映射实现，只保留 POSIX `open/fstat/mmap/munmap`。
- 删除 Windows 对齐分配实现，只保留 `posix_memalign/free`。
- 删除无 CUDA 编译时的 buffer 回退和 `INFER_WITH_CUDA` 条件分支。
- 删除 Windows RSS 回退，只保留 Linux `getrusage`。
- benchmark runner 显式检查 `uname -m == aarch64` 与
  `/etc/nv_tegra_release`，不再输出通用桌面平台信息。
- 删除目标机上恒真的 NVTX 与 `MADV_SEQUENTIAL` 条件包装。

净变化：9 个文件，新增 21 行，删除 90 行。

## 验证

### Release 构建与测试

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 6
ctest --test-dir build --output-on-failure
```

结果：`28/28` 测试通过，总测试时间 `5.87 s`。

### 真实 Qwen2.5-0.5B-Instruct 冒烟

测试配置：1-token prompt，1-token greedy 输出，`max-seq-len=16`。
这些数字只验证端到端可运行，不作为正式性能结论。

| Path | TTFT | 首 token 文本 |
|---|---:|---|
| CPU FP32，6 threads | 2171.38 ms | `/API` |
| CUDA FP32 | 310.98 ms | `/API` |
| CUDA W16A16（BF16） | 243.54 ms | `/API` |
| CUDA W8A16 | 258.36 ms | `/API` |

四条路径均成功加载完整归档、执行 Prefill，并产生相同首 token。

### Benchmark runner 冒烟

W8A16、1 次测量、无 telemetry 的最小 runner 测试成功，JSON 写入被
Git 忽略的 `results/codex-cp16b-runner-smoke.json`。该测试验证了 Jetson
平台约束、模型 SHA-256、provenance 收集和结果 schema；不纳入 CP17 数据。

## 结论

代码库现在明确是 Jetson Orin Nano Super 专用推理项目，不再维护未测试的
Windows、桌面 Linux/x86 或无 CUDA 构建路径。CP16B 验收通过，等待确认后再进入 CP17。
