# CODEX-CLEANUP-PREFILL：矩阵化 Prefill 基线整理

## 结论

本次工程整理通过。当前运行时将完整矩阵化 Prefill 作为唯一正式 Prompt 路径，不再
保留逐 token 调用 Decode 来模拟 Prefill 的生产接口、CLI 开关或测试入口。

这次整理不是编号 checkpoint，也没有进入 BF16/W16A16，因此使用
`CODEX-CLEANUP-PREFILL` 命名。后续 A16 路径仍按既定约定使用 BF16。

## 整理决策

- `prefill(prompt_tokens)` 永远一次处理完整 Prompt。
- CPU FP32 进入 `prefill_cpu_fp32()`；CUDA FP32/W8A32 进入
  `prefill_cuda()`。
- Prompt 长度为 1 时也走矩阵化 Prefill，不再隐式回退到单 token Decode。
- 单 token 实现改名为私有 `decode_token()`，只允许
  `decode_next()` 调用。
- `generate()` 的状态机固定为
  `reset → prefill → decode_next loop`。
- Prefill workspace 始终按 `max-seq-len` 预分配，容量不足显式报错。
- CPU 后端当前明确只接受 FP32；W8A32 由 CUDA 路径支持。

Prompt=1 的矩阵化路径可能比旧 serial reference 有更多调度开销。这是一个明确的工程
取舍：最终项目优先保持真实 Prefill 语义和单一生产路径，不为了极短 Prompt 重新暴露
“Decode 伪装成 Prefill”的分支。

## 删除内容

- `PrefillMode::kAuto/kSerial/kBatched`。
- `RuntimeOptions::prefill_mode`。
- `--prefill-mode` CLI 和 benchmark runner 参数。
- 公开的 `forward_token(token, position)`。
- `effective_prefill_mode()`。
- `qwen2.prefill.serial` 与 `qwen2.prefill.batched` 分支。

新的 benchmark JSON 使用：

```json
{
  "configuration": {
    "prefill_implementation": "matrixized"
  }
}
```

## 保留内容

- CPU FP32、CUDA FP32 和 CUDA W8A32 的矩阵化 Prefill 实现。
- CUDA FP32 custom/cuBLAS 两种 GEMM 路径。
- 私有的单 token CPU/CUDA Decode 实现。
- 历史 `docs/codex-cp03-*.md` 至 `docs/codex-cp08-*.md` 报告及审核结果。

历史报告中的 serial 数据是优化前后的证据，绑定其记录的 commit。报告里的
`--prefill-mode` 命令不代表当前 CLI。

## 回归测试

Release 增量构建通过；benchmark runner self-test 通过；CTest 共 16 项全部通过。

新的长期 Prefill/Decode 回归覆盖：

- Prompt 长度 `1/2/31/32/127/128/512`。
- 空 Prompt、token 越界、KV 容量、重复 Prefill、未 Prefill 直接 Decode。
- 连续多次 `generate()` 的 reset 与确定性。
- CPU 矩阵化 Prefill 后连续 16-token Decode 的状态确定性。
- CUDA FP32 与 CPU FP32 在 7 种 Prompt 长度上的 top-1 和 logits cosine
  （门槛 `0.999999`）。
- CUDA Prefill 后连续 16-token Decode 与 CPU 路径一致，验证 KV Cache 衔接。
- CUDA custom GEMM 与 cuBLAS GEMM 的 Prefill 和后续 Decode 一致性。

测试命令：

```bash
python3 tools/run_benchmark.py --self-test
cmake --build build -j6
ctest --test-dir build --output-on-failure
```

结果：`16/16 passed`，CTest 总时间 `2.22 s`。

## 真实模型 Smoke Test

使用 Qwen2.5-0.5B-Instruct、固定 32-token IDs、2-token 输出、无 warmup、单次测量，
以下四条路径均成功，并确认 JSON 中
`prefill_implementation == "matrixized"`：

- CPU FP32
- CUDA FP32 custom
- CUDA FP32 cuBLAS
- CUDA W8A32

原始 smoke JSON 位于忽略目录：

- `results/codex-cleanup-prefill-cpu-fp32-smoke.json`
- `results/codex-cleanup-prefill-cuda-fp32-custom-smoke.json`
- `results/codex-cleanup-prefill-cuda-fp32-cublas-smoke.json`
- `results/codex-cleanup-prefill-cuda-w8a32-smoke.json`

这些运行没有 warmup，目的是验证真实归档和端到端路径，不作为正式性能数字。

## 性能证据继承

本次没有修改 Prefill 数值 kernel，当前唯一生产路径就是 Checkpoint 8 已正式测量的
batched/matrixized 路径。正式 5 warmup + 20 repeats 数据绑定 commit
`d667fb113619ef133dea7f54085216799cf1ae72`：

- CPU FP32 P32/P128/P512 TTFT 加速：`6.50x/8.29x/6.07x`。
- CUDA FP32 cuBLAS P32/P128/P512 TTFT 加速：`34.42x/47.80x/30.83x`。
- CUDA W8A32 P32/P128/P512 TTFT 加速：`1.37x/1.38x/1.39x`。
- 28 组正式结果 TTFT CV 全部不超过 3%。

正式报告：
`docs/codex-cp08-prefill-benefit-validation.md`。

审核摘要：
`benchmarks/results/codex-cp08-prefill-sweep-summary.json`。
