# Codex Checkpoint 11：CUDA BF16 Linear GEMV/GEMM

## 结论

Checkpoint 11 已完成。CUDA A16 继续使用 **BF16**，不是 FP16。本步实现了自研 BF16 GEMV/GEMM 和 cuBLAS `cublasGemmEx` 对照路径，并使用 Qwen2.5-0.5B-Instruct 的真实 BF16 权重完成正确性与 5/20 微基准。

- 实现 commit：`61a03a5a836d0ee52e1abdc9bb69a44b1d94020d`
- BF16 模型 SHA-256：`f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`
- 目标设备：Jetson Orin Nano Super 8GB
- 功耗模式：MAXN_SUPER
- 正式测量：临时锁定 `jetson_clocks`，退出后恢复原状态
- 协议：5 次 warmup、20 次测量、CUDA Event 计时
- 完整回归：22/22 通过

## 实现内容

### 自研 BF16 GEMV

- 一个 CUDA block 负责一个输出行。
- K 为偶数时使用 `__nv_bfloat162` 成对读取 weight 和 activation。
- K 为奇数时使用安全的标量路径，避免未对齐 BF162 访问。
- block 内使用 FP32 累加与 FP32 reduction。
- bias 参与 FP32 结果计算，最终以 RNE 写回 BF16。

### 自研 BF16 GEMM

- 使用 16×16 tiled CUDA kernel。
- 输入和权重从 BF16 转换为 FP32 shared tile。
- 乘加使用 FP32 accumulator。
- 支持 tokens、输出维度和 K 非 16 整除的边界。
- bias 广播后以 RNE 写回 BF16。

### cuBLAS 对照

- GEMV 和 GEMM 均使用 `cublasGemmEx`。
- A/B/C 数据类型为 `CUDA_R_16BF`。
- compute type 为 `CUBLAS_COMPUTE_32F`。
- cuBLAS 输出 BF16 后使用 BF16 bias kernel 完成广播加法。

### 运行时选择

新增：

```text
--linear-kernel custom|cublas
```

旧 `--cublas` 保留为兼容别名。FP32 原有默认保持 custom；W16A16 根据本步真实权重 Linear 结果暂定默认 cuBLAS。CP12 端到端接通后必须重新验证这个默认选择。

## 正确性验证

### 完整小矩阵 reference

测试 `CudaBFloat16.LinearCustomAndCublasMatchFp32Reference` 覆盖：

- BF162 偶数 K 路径。
- 奇数 K 标量路径。
- 非 tile 整齐的 tokens/out/K。
- 有 bias 和无 bias。
- tied LM Head 的无 bias 语义。
- custom GEMV、cuBLAS GEMV、custom GEMM、cuBLAS GEMM 全部对 CPU FP32 reference。

### 真实 Qwen 维度

微基准直接读取 `model.w16a16.qbin` 中 layer 0 的真实 tensor：

- Q/O：896×896。
- K/V：128×896。
- gate/up：4864×896。
- down：896×4864。
- tied LM Head：151936×896。

真实形状对 custom/cuBLAS 全输出进行比较，并对首行、中间行、末行进行 CPU FP32 抽样 reference。最大 custom/cuBLAS 绝对差异按该 case 输出峰值归一化后为 0.4566%；抽样 FP32 绝对误差按输出峰值归一化后的最大值为 0.1442%。

这里使用“按输出峰值归一化”，避免 K projection 大幅值 bias 导致 BF16 单个 ULP 的绝对值看起来很大。

## 锁钟 5/20 结果

以下均为 p50，单位为毫秒。

| Linear | shape [out,in] | GEMV custom | GEMV cuBLAS | GEMV custom/cuBLAS | GEMM custom | GEMM cuBLAS | GEMM custom/cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 896×896 | 0.052192 | 0.036480 | 1.43× | 1.087072 | 0.033792 | 32.17× |
| k_proj | 128×896 | 0.014752 | 0.020704 | 0.71× | 0.167328 | 0.023776 | 7.04× |
| v_proj | 128×896 | 0.014528 | 0.019648 | 0.74× | 0.167456 | 0.025184 | 6.65× |
| o_proj | 896×896 | 0.050688 | 0.032064 | 1.58× | 1.086816 | 0.026624 | 40.82× |
| gate_proj | 4864×896 | 0.312832 | 0.114592 | 2.73× | 5.834496 | 0.112736 | 51.75× |
| up_proj | 4864×896 | 0.312832 | 0.111776 | 2.80× | 5.834240 | 0.111872 | 52.15× |
| down_proj | 896×4864 | 0.178048 | 0.117856 | 1.51× | 5.969504 | 0.107936 | 55.31× |
| tied LM Head | 151936×896 | 9.529440 | 2.734752 | 3.48× | — | — | — |

K/V 的小 GEMV 自研 kernel 更快，但 gate/up、tied LM Head 等主要耗时项明显由 cuBLAS 占优。

## Linear-only 聚合

这些数值只用于 kernel 选择，不能当作端到端模型延迟：

- Decode Linear 调用估算：`24 × 每层 7 个 Linear + 1 × tied LM Head`
  - custom：31.9904 ms
  - cuBLAS：13.6096 ms
  - cuBLAS 相对加速：2.3506×
- 32-token Prefill 单层 7 个 Linear GEMM 之和：
  - custom：20.1469 ms
  - cuBLAS：0.44192 ms
  - cuBLAS 相对加速：45.5895×

估算不包含 embedding、RMSNorm、RoPE、attention、KV Cache、残差、SwiGLU、argmax 和 kernel 间交互。

## 可复现命令

```bash
./build/bf16_linear_benchmark \
  --archive models/qwen2.5-0.5b-instruct/model.w16a16.qbin \
  --tokens 32 \
  --warmup 5 \
  --repeat 20 \
  --json results/codex-cp11-bf16-linear-5w20-locked.json
```

原始大结果保存在忽略的 `results/`；审核后的轻量摘要保存在 `benchmarks/results/codex-cp11-bf16-linear-summary.json`。

## 本步没有声称的内容

- 尚未将 BF16 weights/workspace/KV Cache 接入端到端 Qwen2。
- 尚未得到真实 W16A16 TTFT 或 decode tok/s。
- 2.35× 和 45.59× 是 Linear-only 聚合，不是端到端加速。
- W16A16 默认 cuBLAS 是当前候选，CP12 必须通过端到端 A/B 再确认。
