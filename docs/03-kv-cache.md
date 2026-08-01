# 03 KV Cache 与内存规划

K/V 布局为 `[layer, kv_head, max_sequence, head_dim]`，K 和 V 使用两个连续区域。
Qwen2.5-0.5B 在 `max_sequence=2048` 时 FP32 KV Cache 约 48MiB：

```text
2 * 24 layers * 2 kv_heads * 2048 positions * 64 dims * 4 bytes
```

workspace、KV Cache、argmax 输出和 device 权重都在模型构造时完成分配。每个 token
只更新当前位置的 K/V，attention 仅读取 `[0, position]`。`max-seq-len` 是实际容量，
超过容量会显式报错而不是越界写入。
