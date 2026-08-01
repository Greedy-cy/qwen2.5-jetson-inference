# 01 模型格式与 mmap

官方 safetensors 主要以 BF16 保存。`tools/export_model.py` 直接读取其 8 字节 header
长度和 JSON index，不依赖 PyTorch；BF16 通过将 16 位载荷移动到 FP32 高 16 位完成
精确展开。

`QWENBIN1` 使用固定 1MiB 元数据区和 256 字节对齐的数据区。这样导出器可流式写入
每个 tensor，运行时也可通过 `mmap_base + data_offset + tensor.offset` 得到零拷贝
CPU view。CUDA 后端只在模型初始化阶段执行 H2D，decode 内不再读取文件或搬运权重。

检查命令：

```bash
./build/llm_infer inspect --model models/qwen2.5-0.5b-instruct --precision fp32
```
