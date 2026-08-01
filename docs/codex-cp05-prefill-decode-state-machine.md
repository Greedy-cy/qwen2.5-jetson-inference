# CODEX-CP05｜Prefill/Decode API 状态机拆分报告

## 本阶段结论

Checkpoint 5 已完成独立的 `prefill()`、`decode_next()` 和 `PrefillMode` 接口，但计算路径仍然是原有的逐 token 串行实现。本阶段只拆分状态机，不宣称已经实现矩阵化 Prefill，也不产生新的正式性能基线。

`generate()` 现在统一执行：

```text
reset
  -> prefill(prompt_tokens)
  -> decode_next(first_generated_token)
  -> decode_next(...)
```

## 接口语义

- `reset()`：将内部 position 归零并清除 prefill 完成标记。
- `prefill(prompt_tokens)`：只允许在 fresh/reset 状态调用；处理完整非空 prompt，写入对应 KV Cache，将 position 推进到 prompt 长度，并返回最后一个 prompt token 的 top-1。
- `decode_next(token)`：只允许在成功 prefill 后调用；在当前 position 处理一个输入 token，复用已有 KV Cache，随后 position 加一。
- `forward_token(token, position)`：保留为显式 position 的无状态参考接口，不修改新状态机的 position。
- `generate()`：先校验总容量，再 reset、prefill 和循环 decode_next；重复调用不会继承上一次请求的 position。

容量超过 `max-seq-len`、空 prompt、重复 prefill、prefill 前 decode、或 KV Cache 已满时都会显式报错。

## PrefillMode 的阶段性行为

- `serial`：调用现有逐 token 路径。
- `auto`：Checkpoint 5 暂时解析为 `serial`，因为 batched 算子尚未实现。
- `batched`：明确报错 `batched prefill is not implemented in checkpoint 5`，不会静默回退或把串行执行记录成 batched。

benchmark JSON 同时记录：

- `prefill_mode`：用户请求的模式。
- `effective_prefill_mode`：本次实际执行的模式。

历史 CP02、CP03、CP04 复现命令已经显式增加 `--prefill-mode serial`，因此后续改变 `auto` 默认行为不会污染旧基线。

## 正确性验证

Release 构建后共有 13 个测试，全部通过；本阶段新增 5 个测试：

1. `auto/serial/batched` 的解析和字符串输出。
2. 新 serial prefill 与旧显式 position 路径逐步比较。
3. Prefill 后继续 decode 16 token，每一步 top-1 和完整 logits 完全一致，用于验证 KV Cache 和 position 衔接。
4. 空 prompt、单 token、容量边界、非法状态转换、reset 和连续 generate。
5. 显式 batched 请求失败时 position 与状态不发生变化。

正式 `Qwen2.5-0.5B-Instruct` 的最小冒烟结果：

- CPU FP32、`prefill-mode=serial`：3-token prompt 加 2-token 输出成功。
- CUDA W8A32、`prefill-mode=auto`：成功，JSON 记录请求 `auto`、实际 `serial`。
- 显式 `prefill-mode=batched`：按设计拒绝执行。
- CUDA FP32 在分配 device weights 时遇到 Jetson 统一内存 OOM，尚未进入 prefill；这是 CP04 已记录的设备内存压力，不属于本阶段状态机错误。

上述均为 warmup=0、repeat=1 的功能冒烟，不能作为性能数字引用。

## 本阶段未做

- 没有批量 embedding、RMSNorm、RoPE、attention 或 MLP。
- 没有 GEMM Prefill。
- 没有改变 KV Cache 的 FP32 数据类型。
- 没有改变模型输出精度或建立新的性能结果。
- 没有实现 A16；后续 A16 路线按项目约定使用 BF16，而非 FP16。

下一 checkpoint 是 CPU FP32 矩阵化 Prefill。开始前需要单独确认。
