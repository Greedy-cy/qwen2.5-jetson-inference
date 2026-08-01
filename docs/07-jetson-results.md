# 07 Jetson Orin 实测结果

## 环境与口径

- Jetson Orin Nano Super 8GB，MAXN_SUPER，SM 8.7
- Ubuntu 22.04、JetPack 6.2.1、CUDA 12.6
- Qwen2.5-0.5B-Instruct，batch=1，纯 argmax greedy
- 32-token chat prompt，固定执行 128 个生成 step
- 5 次 warmup、20 次测量，`max-seq-len=256`

## 结果

| 指标 | FP32 | INT8 W8A32 | 变化 |
|---|---:|---:|---:|
| Decode 吞吐 | 22.82 tok/s | 32.97 tok/s | +44.44% |
| 平均 TTFT | 1392.40 ms | 960.80 ms | -31.00% |
| 平均总耗时 | 6957.15 ms | 4813.36 ms | -30.81% |
| 总耗时 p50 | 6957.03 ms | 4813.32 ms | -30.81% |
| 总耗时 p95 | 6962.30 ms | 4814.42 ms | -30.85% |
| 模型文件 | 1,977,179,648 B | 926,064,128 B | -53.16% |
| Device 权重 | 1,976,131,072 B | 925,015,552 B | -53.19% |
| 峰值 RSS | 4,148,916 KiB | 2,131,800 KiB | -48.62% |

原始数据位于被 Git 忽略的 `results/jetson-orin-fp32.json` 和
`results/jetson-orin-int8.json`。

## 正确性

- C++ tokenizer 在英文、中文、emoji、连续空格和 chat template 上与 Transformers
  逐 token 一致（5/5）。
- FP32 首 token logits 对 Transformers：最大绝对误差 `3.67e-5`、平均绝对误差
  `6.17e-6`、cosine similarity `0.99999988`，top-1 一致。
- INT8 首 token logits：cosine similarity `0.9998098`，top-1 与 FP32/Transformers
  一致；16-token 短生成与 FP32 完全一致。
- 最大 group-wise 权重重构误差 `0.006586`，低于 `0.007` 验收线。

## 结论

INT8 的第一版标量 kernel 只有 18.5 tok/s。改为 `char4` 读取后，一次加载 4 个
int8 权重并复用一次 group scale，吞吐提高到 32.97 tok/s，超过相对 FP32 1.25×
目标。当前 tied LM head 仍使用 cuBLAS FP32，下一阶段可研究词表裁剪、量化 LM head
或 fused top-k，以继续降低每 token 的输出投影成本。
