# Codex Checkpoint 14：W8A16（BF16 activation）模型导出

## 结论

Checkpoint 14 已完成。项目现在能够从 Qwen2.5-0.5B-Instruct 的原始 BF16 safetensors 导出：

```text
model.w8a16.qbin
```

本 checkpoint 只建立 W8A16 归档，不接入推理 kernel，也不报告 token/s。Decode GEMV、Prefill GEMM、端到端接通和性能测试分别属于后续 CP15、CP16、CP17。

- 实现 commit：`b745a9a793a867515abad145c476e257b50cb0bf`
- 原始 safetensors SHA-256：`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- W8A16 archive SHA-256：`730d0ca9a60b55a63534c8edce19353d8f55fda07e6e5fc04ec88470bcf57814`
- group size：64，当前显式拒绝其他值。
- Linear weight：INT8。
- group scale：BF16。
- 目标 activation、KV Cache、workspace：BF16（在后续 runtime checkpoint 接通）。
- embedding、norm、bias、tied LM Head：BF16。
- 完整回归：26/26 通过。

## 导出规则

所有二维 Linear weight，除 embedding 与独立 `lm_head.weight` 外，使用按输入维度分组的对称 INT8 量化：

```text
group_size = 64
axis = 1
q = round(weight / scale)
q ∈ [-127, 127]
scale = max(abs(group)) / 127
```

每个量化权重记录包含：

```json
{
  "scheme": "symmetric_group_w8a16",
  "group_size": 64,
  "axis": 1,
  "scale_tensor": "<weight>.scale",
  "scale_dtype": "bfloat16"
}
```

scale shape 为：

```text
[out_features, ceil(in_features / 64)]
```

Qwen2.5-0.5B-Instruct 使用 tied embedding/LM Head，原始文件没有单独的 `lm_head.weight`。因此 `model.embed_tokens.weight` 以 BF16 保留，并同时作为 tied LM Head。导出器的合成测试也覆盖了存在独立 `lm_head.weight` 的情况，确保它不会被 INT8 量化。

## 为什么误差必须使用 BF16 scale 重新计算

量化时先在 FP32 中计算 scale，但真正写入归档的是 BF16 scale。如果用尚未舍入的 FP32 scale 计算误差，报告会比实际 runtime 读到的权重更乐观。

本实现的误差口径为：

1. 计算 FP32 group scale。
2. 将 scale round-to-nearest-even 转为 BF16。
3. 再把已落盘 BF16 scale 解码为 FP32。
4. 用 `INT8 weight × decoded BF16 scale` 重构权重。
5. 逐 tensor 记录 max/mean absolute error。

因此报告反映的是后续 W8A16 kernel 实际会读取到的数据。

## 真实模型归档

| 项目 | 数值 |
|---|---:|
| 源 tensors | 290 |
| INT8 Linear tensors | 168 |
| BF16 原始非量化 tensors | 122 |
| BF16 scale tensors | 168 |
| 总 archive records | 458 |
| FP32 records | 0 |
| Payload | 641,421,056 bytes |
| Archive | 642,469,632 bytes（约 612.71 MiB） |

168 个 Linear 的组成是：

- 每层 Q/K/V/O Projection：4 × 24 = 96。
- 每层 gate/up/down Projection：3 × 24 = 72。
- 总计：168。

独立验证逐项检查了 290 个源 tensor：所有应量化 Linear 均为 INT8，所有 scale 均为 BF16，scale shape 均匹配 group 数，所有非量化 tensor 均为 BF16，没有缺失或多余记录。

## 归档大小对比

| Archive | Bytes | 相对 FP32 |
|---|---:|---:|
| FP32 | 1,977,179,648 | 100.00% |
| W8A32 | 926,064,128 | 46.84% |
| W16A16 BF16 | 989,114,112 | 50.03% |
| W8A16 BF16 | 642,469,632 | 32.49% |

W8A16 archive：

- 比 FP32 小 67.51%。
- 比 W16A16 小 35.05%。
- 比现有 W8A32 小 30.62%，因为 scale、embedding、norm 和 bias 从 FP32 改为 BF16。

旧归档没有被重新生成或覆盖：

- W8A32 SHA-256：`6315f60dce24ee6f4e2b9609c69df222c2d02e78f37f1c1f21d13d8b20793d90`
- W16A16 SHA-256：`f8b6b16a3a85d70083bfd7f822a7d587332982b0f0e27691512efa844855b2a6`

## 重构误差

真实模型 168 个量化 Linear：

- 最大 group-wise absolute error：`0.0067138671875`。
- 验收门槛：`≤0.007`。
- 逐 tensor mean absolute error 的均值：`0.0001134105702`。
- BF16 scale/非量化 tensor 最大转换误差：`0.0000197039917`。

误差最大的 10 个 tensor：

| Tensor | Max abs error | Mean abs error |
|---|---:|---:|
| `layers.8 q_proj` | 0.0067138672 | 0.0001311472 |
| `layers.0 q_proj` | 0.0057373047 | 0.0003193163 |
| `layers.0 k_proj` | 0.0044860840 | 0.0003286342 |
| `layers.21 up_proj` | 0.0040283203 | 0.0001096117 |
| `layers.23 o_proj` | 0.0034790039 | 0.0000987800 |
| `layers.22 gate_proj` | 0.0030822754 | 0.0001070753 |
| `layers.2 up_proj` | 0.0030517578 | 0.0000928265 |
| `layers.15 up_proj` | 0.0030517578 | 0.0001099491 |
| `layers.5 up_proj` | 0.0030212402 | 0.0000923673 |
| `layers.11 up_proj` | 0.0027160645 | 0.0001024890 |

审核汇总 JSON 保留全部 168 个 Linear 的逐 tensor max/mean reconstruction error，而不是只保留全局最大值。

## 自动化测试

新增 `ExportModel.W8A16SelfTest`，构造真实 QWENBIN1 归档并验证：

- group size 64。
- `K=70` 的非整除 group 边界。
- 全零 group。
- INT8 weight 与 BF16 scale。
- scale shape 与量化元数据。
- embedding、norm、bias、独立 LM Head 为 BF16。
- 使用落盘 BF16 scale 重新计算误差。
- tensor SHA-256 和 archive bounds。
- 最大重构误差门槛。

全量结果：

```text
26 tests passed, 0 failed
```

## 使用方式

```bash
python3 tools/export_model.py \
  --source models/source/Qwen2.5-0.5B-Instruct \
  --output models/qwen2.5-0.5b-instruct \
  --precision w8a16 \
  --group-size 64
```

`--precision all` 现在会生成 FP32、W8A32、W16A16 和 W8A16 四种归档。

## 归档位置

受版本控制的审核结果：

- `benchmarks/results/codex-cp14-w8a16-export-summary.json`
- 其中 `per_tensor_reconstruction_errors` 包含全部 168 个量化 tensor。

忽略版本控制的真实模型文件与原始导出报告：

- `models/qwen2.5-0.5b-instruct/model.w8a16.qbin`
- `models/qwen2.5-0.5b-instruct/export_report.json`

本 checkpoint 到此停止。下一 checkpoint 是 W8A16 GEMV/GEMM kernel，不在本次提交中启动。
