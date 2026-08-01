# CODEX-CP09：BF16 归档与类型系统

## 结论

Checkpoint 9 通过。项目新增 BF16 tensor 类型和 W16A16 归档：
`model.w16a16.qbin`。本项目中的 A16 明确表示 BF16，不是 FP16。

本 checkpoint 只完成归档、类型解析、导出、检查和旧格式兼容；BF16 CUDA 算子尚未
实现。若请求 W16A16 推理，运行时会明确报错，不会把 BF16 payload 误当成 FP32。

实现 commit：`295c6f3e32c754cf06e787510807ffb0f7e60937`。

## 实现内容

- 新增 `DType::kBFloat16`，元素大小为 2 字节，归档字符串为
  `bfloat16`。
- 精度枚举统一为 `FP32/W8A32/W16A16`。
- W16A16 对应文件名 `model.w16a16.qbin`。
- `QWENBIN1` 版本仍为 1；dtype 位于 JSON tensor record，因此无需破坏旧文件格式。
- C++ mmap 加载器同时识别 `float32/int8/bfloat16`。
- CLI 接受 `--precision w16a16`；`inspect` 可查看 BF16 tensor 和未来 BF16
  KV Cache 规划。
- 导出器使用 round-to-nearest-even 将 FP32 表示转换为 BF16 16-bit payload。
- 导出后重新打开归档，验证每个 tensor 的 shape、nbytes、256-byte 对齐、文件边界
  和 SHA-256。
- 导出报告包含逐 tensor BF16 最大/平均重构误差。

## BF16 转换

BF16 保留 FP32 的 8-bit exponent，截短 mantissa 到 7 bit。导出器不是简单截断，
而是执行 round-to-nearest-even：

1. 读取 FP32 的 32-bit 表示。
2. 根据低 16 位和目标最低有效位添加 rounding bias。
3. 取高 16 位作为 BF16 payload。
4. 对 NaN 保证保留非零 mantissa。

自测覆盖可精确表示值、正负零、正负无穷、NaN、tie-to-even 和刚超过 tie 的边界。

## 真实模型归档

源模型：

- `Qwen2.5-0.5B-Instruct`
- safetensors：`988,097,824` bytes
- SHA-256：
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`

W16A16 归档：

- 文件：`model.w16a16.qbin`
- 文件大小：`989,114,112` bytes
- payload：`988,065,536` bytes
- tensor：`290`
- dtype：`290 bfloat16 / 0 float32 / 0 int8`
- SHA-256：
  `f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`
- 相对 FP32 文件大小：`50.0265%`
- 相对 FP32 缩小：`49.9735%`

FP32 归档为 `1,977,179,648` bytes，因此满足“不超过 FP32 约 55%”的验收目标。

## Tensor 覆盖与误差

以下类型均确认进入 BF16 归档：

- embedding：`model.embed_tokens.weight`
- final norm：`model.norm.weight`
- layer norm：`model.layers.0.input_layernorm.weight`
- bias：`model.layers.0.self_attn.q_proj.bias`
- 所有 Linear 权重

模型配置 `tie_word_embeddings=true`，因此没有单独的 `lm_head.weight`；
运行时 tied LM Head 复用 BF16 embedding，未漏导出 LM Head 权重。

源 safetensors 本身保存 BF16。导出器先精确展开到 FP32，再按 RNE 编回 BF16，因此：

- 最大 BF16 重构误差：`0.0`
- tensor 平均误差的平均值：`0.0`

这里的零误差表示 BF16 payload 被无损保留，不表示 BF16 相对训练前 FP32 权重没有量化误差。

## 兼容性

新版 C++ 加载器成功检查三种现有归档：

| 归档 | Tensor 组成 | Payload | inspect |
|---|---:|---:|---|
| FP32 | 290 FP32 | 1884.59 MiB | 通过 |
| W8A32 | 168 INT8 + 290 FP32 scale/non-linear | 882.16 MiB | 通过 |
| W16A16 | 290 BF16 | 942.29 MiB | 通过 |

CPU FP32 真实模型端到端 smoke 通过。CUDA W8A32 兼容 smoke 已正确选择旧
`model.int8.qbin` 并进入 device weight 分配，但当前 Jetson 的 NvMap 在约 1 GiB
连续 buffer 申请时返回 OOM；测试时系统有约 4.5 GiB 文件页缓存。该现象未通过清缓存
或修改系统状态规避。旧归档解析、INT8 kernel 回归和此前正式 W8A32 端到端结果均通过。

## 测试

```bash
python3 tools/export_model.py --self-test
cmake --build build -j6
ctest --test-dir build --output-on-failure
```

结果：`19/19 passed`，失败 `0`。

新增测试：

- `DType.BFloat16SizeAndNames`
- `ModelArchive.LoadsBFloat16Tensor`
- `ExportModel.BFloat16SelfTest`

W16A16 推理边界测试返回：

```text
error: W16A16 uses BF16; archive support is ready but BF16 runtime kernels are not implemented
```

## 数据归档

审核摘要：

- `benchmarks/results/codex-cp09-bf16-archive-summary.json`

忽略的原始数据：

- `results/codex-cp09-bf16-export.log`
- `models/qwen2.5-0.5b-instruct/export_report.json`
- `models/qwen2.5-0.5b-instruct/model.w16a16.qbin`

下一步在用户确认后进入 Checkpoint 10：实现 BF16 activation、KV Cache、Prefill 和
Decode 所需的批量基础算子。
