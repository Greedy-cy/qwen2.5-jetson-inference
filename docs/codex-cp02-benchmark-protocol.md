# CODEX-CP02｜统一 Benchmark 协议

## 测量边界

主性能表测量模型已加载、prompt 已 tokenized 后的稳态推理：

```text
prefill start -> first token ready -> decode remaining tokens -> request complete
```

- Engine TTFT：从 prefill 开始到第一个输出 token 可用。
- Prefill tok/s：`prompt_tokens * 1000 / TTFT_ms`。
- TPOT：`decode_ms / (generated_tokens - 1)`。
- Decode tok/s：`(generated_tokens - 1) * 1000 / decode_ms`。
- Total latency：TTFT 与后续 decode 的总时间。

第一个输出 token 属于 TTFT，不计入 decode tok/s。模型加载、tokenizer、网络和排队时间不计入
Engine TTFT；模型加载耗时可作为独立冷启动指标记录。

## Canonical workload

- Qwen2.5-0.5B-Instruct
- batch size 1
- 32 个固定 token ID
- 128 个输出 token
- greedy argmax，禁用 EOS 提前停止
- `max-seq-len=256`
- 5 次 warmup，20 次正式测量

固定输入对应 chat prompt `Explain-CUDA-memory-bandwidth-LLM.`，token ID 为：

```text
151644,8948,198,2610,525,264,10950,17847,
13,151645,198,151644,872,198,840,20772,
12,80285,12,17269,12,7053,3098,12,
4086,44,13,151645,198,151644,77091,198
```

benchmark 使用 `--token-ids` 直接输入这些 ID，避免 tokenizer、chat template、shell 编码造成差异。

## C++ benchmark JSON v2

`llm_infer benchmark` 输出：

- `configuration`：backend、precision、线程、warmup/repeat、KV 容量和当前 prefill 模式。
- `tokens`：prompt IDs、prompt/output 数量。
- `samples`：每轮 TTFT、decode、total、TPOT、prefill tok/s、decode tok/s。
- `statistics`：每个指标的 mean、p50、p95、min、max、标准差和 CV。
- `memory`：归档、device weights、workspace、KV Cache 和峰值 RSS。

`--telemetry-markers` 在 warmup 后和正式测量后向 stderr 输出 marker。runner 只统计两个 marker
之间的温度、内存和功耗，避免把模型加载与 warmup 混入正式能耗。

## Jetson runner

正式运行前先授权本终端使用 sudo：

```bash
sudo -v
```

运行 CUDA FP32 canonical benchmark：

```bash
python3 tools/run_benchmark.py \
  --model models/qwen2.5-0.5b-instruct \
  --backend cuda --precision fp32 \
  --lock-clocks \
  --output results/cuda-fp32-serial.json
```

runner 执行以下工作：

1. 拒绝未显式允许的 dirty worktree。
2. 记录 commit、模型 SHA-256、CUDA、Jetson BSP、功耗模式和完整命令。
3. 临时保存并锁定 Jetson 时钟，在退出路径中恢复。
4. 使用 `tegrastats` 采样 RAM、GPU 频率/利用率、温度和 VDD_IN。
5. 计算正式测量窗口的能量和每生成 token 能量。
6. 将 runner provenance/telemetry 与 C++ benchmark JSON 合并并原子写入结果文件。

CPU 正式基线需额外固定线程和 affinity：

```bash
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 taskset -c 0-5 \
python3 tools/run_benchmark.py \
  --model models/qwen2.5-0.5b-instruct \
  --backend cpu --precision fp32 \
  --lock-clocks \
  --output results/cpu-fp32-6threads-serial.json
```

## 结果有效性

- 正式 decode 指标要求 CV 不超过 3%；超出时检查温度、频率和后台负载后整组重跑。
- 不挑选单次最好结果，保留全部原始 sample。
- Jetson 使用统一内存；RSS、device allocation 和 tegrastats RAM 分开报告，不能直接相加。
- Nsight trace 用于解释瓶颈，不代替 canonical benchmark 数字。
- BF16/W8A16 或 batched prefill 必须先通过正确性验证，再进入主性能表。
