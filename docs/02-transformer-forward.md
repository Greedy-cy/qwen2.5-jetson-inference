# 02 Qwen2 Transformer 前向

Qwen2.5-0.5B-Instruct 使用 hidden size 896、24 层、14 个 query head、2 个 KV head、
head dimension 64 和 intermediate size 4864。实现从 `config.json` 读取这些参数。

关键公式：

- RMSNorm：`y = x / sqrt(mean(x²) + eps) * weight`
- RoPE：把 head 的前后半维组成旋转对，角频率为 `theta^(-2i/head_dim)`
- GQA：每 7 个 query head 共享一个 KV head
- SwiGLU：`down(silu(gate(x)) * up(x))`

CPU 与 CUDA 使用同一权重名称和执行顺序。调试时应先比较单层输出，再比较最终 logits；
仅比较生成文本无法定位误差首次出现的位置。
