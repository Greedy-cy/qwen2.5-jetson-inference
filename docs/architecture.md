# 运行时架构

## 1. 目标与公开路径

项目针对 Jetson Orin Nano Super 上的 `Qwen2.5-0.5B-Instruct`、batch=1 greedy
generation。CPU FP32 负责正确性参考，CUDA BF16 cuBLAS 负责性能上限对照，自研 CUDA
BF16/W8A16 负责算子优化。

最终公开矩阵：

| Backend | Precision | Linear kernel |
|---|---|---|
| CPU | FP32 | OpenBLAS / CPU reference |
| CUDA | W16A16 (BF16) | custom，默认 |
| CUDA | W16A16 (BF16) | cuBLAS，显式 baseline |
| CUDA | W8A16 | custom，唯一实现 |

运行时主动拒绝 CUDA FP32、CPU 非 FP32、CPU Linear kernel 选项和 W8A16 + cuBLAS，
避免 CLI 选项看似有效、实际却走了其他路径。

## 2. 模型归档与类型

导出工具把 Hugging Face 权重转换为可直接解析的 qbin 归档，并保存 tensor name、shape、
dtype 与 payload。运行时校验 Qwen2 配置和每个 tensor 的 shape/dtype，再建立权重视图。

| 归档 | Linear weight | 其他权重 | Activation / KV | 累加 |
|---|---|---|---|---|
| FP32 | FP32 | FP32 | FP32 | FP32 |
| W16A16 | BF16 | BF16 | BF16 | FP32 |
| W8A16 | INT8 + BF16 group scale | BF16 | BF16 | FP32 |

W8A16 只量化 Linear weight，group-size 固定为 64；embedding、RMSNorm、bias、tied
LM Head 保持 BF16。运行时不创建完整 BF16 反量化权重副本，反量化发生在 GEMV/GEMM
kernel 内部。

CPU 通过 mmap 读取归档；CUDA 初始化时一次性上传并按运行时布局打包。workspace、
KV Cache、logits 和归约临时区按 `max-seq-len` 预分配，token 循环内不执行
`cudaMalloc`。

## 3. 推理状态机

对外核心接口为：

- `prefill(prompt_tokens)`：一次处理完整 prompt，构建全部层的 KV Cache，仅对最后一行
  hidden state 执行最终 RMSNorm、LM Head 和 argmax；
- `decode_next(token)`：读取并追加 KV Cache，计算下一个 token；
- `generate()`：reset 后执行一次 Prefill，再循环 Decode。

矩阵化 Prefill 是唯一 prompt 路径，不保留逐 token forward 伪 Prefill。

```text
token IDs
   |
   v
batched embedding [tokens, hidden]
   |
   v
N x Transformer layer
   |-- RMSNorm -> Q/K/V GEMM -> batched RoPE
   |-- causal GQA + write all prompt K/V
   |-- O GEMM -> residual
   `-- RMSNorm -> gate/up GEMM -> SwiGLU -> down GEMM -> residual
   |
   v
last hidden row -> RMSNorm -> tied LM Head -> logits + first token
                                             |
                                             v
                                  single-token Decode loop
```

Prefill 主张量使用 `[tokens, hidden]`。causal attention 对第 `t` 个 query 只访问
`[0, t]`，在线完成 score、softmax 和 value reduction，不持久化完整
`[tokens, tokens]` attention matrix。

## 4. KV Cache

每层分别保存 K/V，布局覆盖 `max_seq_len`、KV heads 和 head dimension。Prefill 按绝对
position 批量写入 prompt K/V；Decode 在当前位置追加一行，并对已有 `[0, position]`
执行 attention。

位置推进规则：

1. reset 后 position 为 0；
2. Prefill 成功后 position 等于 prompt 长度；
3. 每次 Decode 成功后加 1；
4. prompt 为空或总长度超过容量时显式报错，不静默切换实现。

这保证 Prefill 的最后 logits 与随后 16-token Decode 都可以对 reference 检查，从而不仅
验证首 token，也验证 KV Cache 状态。

## 5. CPU 与 CUDA 数据流

CPU FP32 路径使用 OpenBLAS SGEMM 实现 batched Linear，CPU kernels 实现 RMSNorm、
RoPE、GQA attention、residual 与 SwiGLU。它不承担最终性能竞争，主要提供易检查的
数值参考。

CUDA BF16 路径保留同一模型层级数据流，但用 BF16 减少权重/activation/KV 带宽，
并在 dot product、RMS、softmax 和归约中使用 FP32 accumulation。W16 custom 与
cuBLAS 共享非 Linear 算子，以保证 Linear kernel 对照公平。

CUDA W8A16 复用 BF16 workspace、KV Cache、attention、非线性与 fused LM Head；只把
Transformer Linear 替换为 group-64 INT8 weight 的在线反量化 GEMV/GEMM。这使 W8 与
W16 的性能差异可以明确归因于 Linear weight 格式和 kernel。

## 6. 内存所有权

- 模型归档：只读映射/权重视图；
- device weights：模型加载时分配，模型析构时释放；
- Prefill workspace：按 `max-seq-len` 预分配并跨层复用；
- KV Cache：按层、最大序列长度一次分配；
- logits：保留完整 vocabulary BF16/FP32 输出，`last_logits_host()` 行为不因融合 argmax
  改变；
- argmax workspace：保存 block maxima，避免每 token 动态分配。

## 7. 正确性策略

测试覆盖归档 shape/dtype、RMSNorm、RoPE、attention、GEMV/GEMM、bias/tail、重复最大值、
Prefill 后连续 Decode、reset 与非法路径。BF16 custom 与 cuBLAS 检查 logits cosine、
top-1 和 16-token greedy；W8A16 检查在线反量化误差、top-1、KV 延续和无完整反量化副本。

量化后的 sequence-level greedy 对数值边界更敏感：W8A16 固定 prompt 首 token top-1
为 5/5，但 canonical 16-token 为 14/16。项目把它作为已知精度限制公开记录。
