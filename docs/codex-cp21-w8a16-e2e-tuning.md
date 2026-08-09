# CODEX-CP21-W8A16-E2E-TUNING：端到端收口与一次集中调优

## 结论摘要

本检查点以 CP20 的 shape-aware W8A16 GEMV/GEMM 为基线，完成端到端正确性、
性能、内存与能效收口，并依据 Nsight 中 GPU 时间占比最高的 W8 Decode GEMV
执行且只执行一次集中调优。

- CP20 trace 中 `gemv_w8a16_fast_kernel<8,1>` 与 `<2,4>` 合计占 GPU kernel
  时间约 52.4%，高于 BF16 tied LM Head 的 27.8%，因此选择 GEMV scale 读取为
  唯一调优对象。
- 实验将同一 group-size 64 scale 从 4 个 lane 重复加载改为 4-lane 子组单 lane
  加载并 shuffle 广播，同时移除 `<8,1>` 路径一次 shared store/load。
- 该实验导致七个 GEMV P50 合计 `0.2895 -> 0.3351 ms`（慢 15.7%），canonical
  Decode `80.63 -> 73.05 tok/s`（下降 9.4%），因此已完整回退，不把性能回退
  留在最终 kernel 中。
- 最终代码树恢复为 CP20 kernel；最终复测为 `1195.84 prefill tok/s`、
  `80.35 decode tok/s`、TTFT `26.76 ms`、TPOT `12.45 ms`，Decode CV
  `0.0324%`。
- W8 Decode 为 CP19 W16 custom（67.92 tok/s）的 `1.183x`；device weights
  `611.71 MiB`，为 W16 的 `64.9%`。CP20 正式能耗为 `0.2255 J/token`，低于
  W16 的 `0.274 J/token`。

结论：W8A16 已满足端到端性能、内存与稳定性目标；本轮唯一集中调优被数据否决，
冻结 CP20 W8 kernel，进入 CP22 最终集中 Benchmark。

## 提交与测试身份

- CP20 基线代码：`8bd9293`，CP20 报告终点：`2aa30cd`
- 被拒绝实验：`ad39b61`（warp 内 scale 广播）
- 回退提交：`1e53d11`
- 最终代码树与 `2aa30cd` 一致；实验与回退保留在 Git 历史中便于追溯
- 模型：`Qwen2.5-0.5B-Instruct`
- W8 archive SHA-256：
  `730d0ca9a60b55a63534c8edce19353d8f55fda07e6e5fc04ec88470bcf57814`
- 设备：Jetson Orin Nano Super 8GB，CUDA 12.6，SM87，MAXN_SUPER
- 测试时 GPU 1020 MHz、EMC 3199 MHz；结束后恢复 GPU 306--1020 MHz 动态频率、
  EMC 2133 MHz
- 全量 Release CTest：32/32 通过

## 正确性收口

- W8 GEMV 与显式反量化 FP32 reference：测试形状 max_abs_error `0`
- 20/100 真实形状微基准抽样最大相对误差：`0.001117 < 0.007`
- 5 个固定 prompt 首 token top-1：W8 与 W16 `5/5` 一致
- Prefill 后继续 Decode 的 KV Cache 状态测试通过
- archive/device weights/runtime workspace 中不存在完整 BF16 反量化权重副本
- canonical 16-token greedy 与 W16 为 `14/16` 一致。差异来自 Tensor Core GEMM
  前在线反量化权重舍入到 BF16，经 24 层传播后造成边界 token 翻转；该量化行为已
  接受并必须在最终文档中如实描述，不声明逐 token 完全一致

## 最终端到端结果

### Canonical：32 prompt + 128 output，5/20

| 指标 | CP21 最终复测 | CP20 正式结果 |
|---|---:|---:|
| Prefill | 1195.84 tok/s | 1193.68 tok/s |
| Decode | 80.35 tok/s | 80.01 tok/s |
| TTFT | 26.76 ms | 26.81 ms |
| TPOT | 12.45 ms | 12.50 ms |
| Total latency | 1607.38 ms | 1614.1 ms |
| Decode CV | 0.0324% | 0.1336% |
| Device weights | 611.71 MiB | 611.71 MiB |
| Archive | 612.71 MiB | 612.71 MiB |
| Workspace | 9.91 MiB | 9.91 MiB |
| Peak RSS | 1496.04 MiB | -- |

### CP20 已完成的 prompt/context sweep（同一最终 kernel）

| Prompt | W8 Prefill | W16 Prefill | W8/W16 |
|---:|---:|---:|---:|
| 1 | 33.6 tok/s | 14.2 tok/s | 236.4% |
| 32 | 1192.8 tok/s | 1658.7 tok/s | 71.9% |
| 128 | 1881.4 tok/s | 2228.2 tok/s | 84.4% |
| 512 | 985.1 tok/s | 1034.8 tok/s | 95.2% |

| Context | W8 Decode | W16 Decode | W8/W16 |
|---:|---:|---:|---:|
| 32 | 79.93 tok/s | 67.71 tok/s | 118.1% |
| 128 | 77.22 tok/s | 65.79 tok/s | 117.4% |
| 512 | 68.18 tok/s | 59.11 tok/s | 115.3% |

## 一次集中调优的判定

### 假设

group-size 64 下，每 4 个连续的 16-weight chunk 共用一个 scale。原 kernel 的
4 个 lane 分别加载同一个 BF16 scale，理论上可通过单 lane load + shuffle 将 scale
load 数量降低 4 倍。

### 实测

| 指标 | CP20 baseline | scale broadcast | 变化 |
|---|---:|---:|---:|
| 七个 W8 GEMV P50 合计 | 0.2895 ms | 0.3351 ms | +15.7%（回退） |
| Canonical Decode | 80.63 tok/s | 73.05 tok/s | -9.4%（回退） |
| TPOT | 12.40 ms | 13.69 ms | +10.4%（回退） |
| Total latency | 1601.90 ms | 1765.26 ms | +10.2%（回退） |
| Prefill | 1198.13 tok/s | 1196.93 tok/s | 基本不变 |

scale tensor 很小且访问高度连续，缓存已能有效服务重复 load；shuffle、lane owner
判断及依赖链增加的指令成本反而主导。实验还验证了 partial-warp 循环必须使用
`__activemask()`，全 warp mask 会在 K=896 的最后一轮导致未定义同步行为。

### 决策

实验提交已通过 `1e53d11` 回退。按照“一次集中调优”约束，本检查点不再尝试
rows/block、QKV 融合或其他 GEMV 变体。保留真实失败数据，冻结已达到目标的 CP20
kernel。

## 验收

| 门槛 | 要求 | 实测 | 结果 |
|---|---:|---:|---|
| W8 Decode vs W16 custom | >=1.05x | 1.183x | 通过 |
| W8 p32/p128/p512 Prefill | >=70% | 71.9% / 84.4% / 95.2% | 通过 |
| Device weights vs W16 | <=66% | 64.9% | 通过 |
| Energy/token vs W16 | 不高于 W16 | 0.2255 vs 0.274 J | 通过 |
| Decode CV | <=3% | 0.0324% | 通过 |
| 5-prompt top-1 | 5/5 | 5/5 | 通过 |
| canonical greedy | 16/16 | 14/16 | 未满足，量化边界行为已接受并记录 |

## 数据文件

Git 忽略目录 `results/`：

- `codex-cp21-final-linear-w8a16-p1.json`
- `codex-cp21-final-canonical-w8a16.json`
- `codex-cp21-w8-gemv-broadcast-linear-p1-clean.json`
- `codex-cp21-w8-gemv-broadcast-canonical.json`

机器可读摘要：

- `benchmarks/results/codex-cp21-w8a16-e2e-tuning-summary.json`
