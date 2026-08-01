# 04 CUDA 映射

- RMSNorm：一个 block 处理一个 hidden vector，shared memory reduction 求平方和。
- GEMV：一个 block 处理权重的一行，线程遍历 K 维并归约。
- RoPE：一个 block 对应一个 attention head，线程对应旋转对。
- Attention：一个 block 对应一个 query head，动态 shared memory 保存时间维 scores。
- Argmax：单 block 跨词表遍历并归约，只向 CPU 回传一个 token id。
- Tiled FP32 GEMM：`16x16` block 复用 input/weight shared-memory tile，接口按
  `[tokens, in] x [out, in]^T` 输出 `[tokens, out]`。
- Batched W8A32：二维 grid 同时覆盖 token 与输出行，沿 K 维以 `char4` 加载并复用
  group scale，为后续矩阵化 prefill 提供算子基础。

`--cublas` 可将 FP32 Linear 切换到 cuBLAS；tied LM head 默认始终使用 cuBLAS，以免
151936 行输出投影的自研 kernel launch 组织方式主导端到端结果。自研与 cuBLAS 路径
必须先在随机矩阵和真实维度上做误差比较。
