# 前 50k 训练结果性能对比：Dense vs Hidden-only MoE vs Entropy PhaseB

生成时间：2026-06-22，口径为 `global_step <= 50000`。

## 0. 核心结论

只看前 50k，模型质量指标排序非常清楚：

```text
entropy_phaseB 最好  >  hidden-only MoE  >>  dense
```

但论文能否站住，取决于如何解释“质量收益”和“系统代价”的关系：

- `entropy_phaseB` 在 50k 附近 BPB = 0.833，比 `hidden-only` 的 0.923 低约 9.8%，比 `dense` 的 1.749 低约 52.4%。
- `hidden-only` 也明显优于 `dense`，50k 附近 BPB = 0.923，比 dense 低约 47.2%。
- 但是 MoE 的 wall-clock 代价很重：`hidden-only` 和 `entropy_phaseB` 的 wps 只有 dense 的 25.8% / 27.4%，显存 active 从 52% 升到 77%。
- MoE 的记录 FLOPs 反而低于 dense：dense 约 `6.19e13`，hidden-only 约 `2.51e13`，entropy_phaseB 约 `2.67e13`。这说明瓶颈不是纯算子 FLOPs，而是 MoE dispatch、expert parallel 通信、显存/allocator 压力。
- `entropy_phaseB` 形成了很强的 entropy 专家分工，但专家负载严重不均衡：0/1 两个专家承担约 76.9% load，6/7 几乎空闲。

一句话：如果论文主张是“entropy-aware patch routing 提升质量”，当前 50k 数据支持这个方向；如果主张是“质量-效率同时优越”，当前系统效率数据还不够，需要解决 MoE 吞吐、显存和负载均衡问题。

## 1. 数据来源与口径

读取文件：

| 模型类型 | 日志文件 | 50k 口径 |
|---|---|---|
| dense | `dense100k.out` | 只截取 `step <= 50000` |
| hidden-only MoE | `hidden100k.out` | 只截取 `step <= 50000` |
| entropy PhaseB | `entropy_phaseB_100k.out` | 只截取 `step <= 50000` |

结构化补充：

- `blt1b_warmstart/entropy_bands_phaseB_100k/metrics.jsonl`：可用于 entropy PhaseB 的 50k MoE 专家分析。
- `blt1b_warmstart/hidden_only_blt1b_10000step_lr5e6_balance0/metrics.jsonl`：只用于 hidden-only 10k warmup 的 MoE 参考，不能等价替代 hidden-only 50k。
- dense/hidden 100k 继续训练目录当前未在工作区保留对应 `metrics.jsonl`，所以三模型共同表按 stdout 解析。

主统计窗口：

- “50k 性能”使用最后 100 条训练记录均值，即约 49k-50k 区间。
- 每个 step 只统计一次 `loss_avg/bpb_avg/wps/iter/mem/flops`，避免两个 rank 重复计数。

## 2. 实验配置身份

| 维度 | dense | hidden-only MoE | entropy PhaseB |
|---|---:|---:|---:|
| 总参数量 | 4.534B | 7.130B | 7.130B |
| MoE experts | 0 | 8 | 8 |
| top-k | 0 | 2 | 2 |
| expert parallel size | 1 | 2 | 2 |
| MoE FFN dim multiplier | - | 0.5 | 0.5 |
| hidden-state routing | 无 MoE | true | false |
| patch entropy feature | 无 MoE | false | true |
| patch feature bias | 无 MoE | false | true |
| balance cost | byte（MoE disabled） | byte | byte |
| balance loss weight | 0.05（MoE disabled） | 0.0 | 0.0 |
| init checkpoint | dense 10k | hidden-only 10k | entropy PhaseA 1k |

重要说明：

- dense 与 hidden-only 都从各自 10k checkpoint 继续。
- entropy PhaseB 从 PhaseA 1k checkpoint 继续，起点不完全等价。
- 因此 50k 绝对 BPB 是“当前训练设置下的模型性能”，不是严格同一起点的最终公平性证明。

## 3. 前 50k 质量指标

| 模型 | 50k step BPB | 49k-50k BPB 均值 | 49k-50k loss 均值 | 前 100 条 BPB 均值 | 变化 |
|---|---:|---:|---:|---:|---:|
| dense | 1.734 | 1.749 | 1.231 | 2.111 | -0.362 |
| hidden-only MoE | 0.930 | 0.923 | 0.652 | 0.828 | +0.096 |
| entropy PhaseB | 0.832 | 0.833 | 0.589 | 0.448 | +0.385 |

相对性能：

| 对比 | BPB 差值 | 相对降低 | 结论 |
|---|---:|---:|---|
| hidden-only vs dense | -0.826 | 47.2% lower | patch-level MoE 质量显著优于 dense |
| entropy PhaseB vs dense | -0.916 | 52.4% lower | entropy routing 在当前 50k 下最强 |
| entropy PhaseB vs hidden-only | -0.090 | 9.8% lower | entropy 侧信息相对 hidden-only 有额外收益 |

论文含义：

- “MoE 是否有用”：50k 数据强支持。hidden-only 相比 dense 的 BPB 降幅非常大。
- “entropy-aware 是否有用”：50k 数据支持。entropy PhaseB 相比 hidden-only 又降低约 9.8% BPB。
- “是否已能作为最终主结果”：还不能。因为 entropy PhaseB 起点不同，并且有路由/负载/显存风险。

## 4. 训练效率与系统成本

| 模型 | 49k-50k wps | 相对 dense wps | 49k-50k iter(s) | 相对 dense iter | 49k-50k FLOPs | 相对 dense FLOPs | active 显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 2275.2 | 1.000x | 0.664 | 1.000x | 6.191e13 | 1.000x | 52% |
| hidden-only MoE | 587.3 | 0.258x | 2.810 | 4.230x | 2.512e13 | 0.406x | 77% |
| entropy PhaseB | 623.4 | 0.274x | 2.147 | 3.231x | 2.667e13 | 0.431x | 77% |

关键解释：

- MoE 记录 FLOPs 只有 dense 的约 40%-43%，但 wall-clock 更慢 3.2x-4.2x。
- 这说明当前实现瓶颈不是密集矩阵计算，而是 MoE 路由、expert dispatch、all-to-all、FSDP/EP 组合、显存碎片或 allocator retry。
- entropy PhaseB 比 hidden-only 略快：wps 高约 6.1%，iter time 低约 23.6%。这可能和路由模式、负载形态或后续日志阶段有关，但还不能单独作为“entropy routing 更高效”的结论。
- 如果论文要强调 efficiency，必须补充 active FLOPs、dispatch latency、all-to-all bytes、GPU utilization、memory reserve/fragmentation 等指标。

## 5. 50k checkpoint 与运行稳定性

| 模型 | 50k 日志状态 | 50k checkpoint 可用性 | 后续运行风险 |
|---|---|---|---|
| dense | 到 50k 并继续训练 | 日志显示 50k 保存成功 | 无异常，最终也到 100k |
| hidden-only MoE | 到 50k | 50k 保存失败：`CheckpointException` | 50k 训练行可用，但 checkpoint 不可复现 |
| entropy PhaseB | 到 50k | `blt1b_warmstart/entropy_bands_phaseB_100k/checkpoints/0000050000` 存在 | 50k 后继续训练时 OOM / ChildFailedError |

稳定性结论：

- dense 工程稳定性最好。
- hidden-only 的 50k 指标可用于曲线对比，但不能作为可加载 checkpoint 结果，除非修复保存失败或重新导出。
- entropy PhaseB 的 50k checkpoint 存在，适合优先做 held-out eval；但 50k 后 OOM 说明显存余量不足。

## 6. MoE 路由与专家分工

### 6.1 Entropy PhaseB 50k 专家指标

来自 `blt1b_warmstart/entropy_bands_phaseB_100k/metrics.jsonl`，只取 `global_step <= 50000`，最后 100 条均值。

| 指标 | entropy PhaseB | 解释 |
|---|---:|---|
| router entropy | 1.267 | 低于 `ln(8)=2.079`，路由较集中 |
| active experts | 6.82 | 平均不是 8 个专家都有效活跃 |
| load imbalance | 3.107 | 负载不均衡明显 |
| max load fraction | 0.388 | 最重专家接近 39% load |
| min load fraction | 0.0002 | 最轻专家几乎空闲 |
| balance loss | 1.288 | 不均衡强，但当前 weight=0 |
| top2 same-pair fraction | 0.9997 | Top-2 几乎固定成专家对 |
| entropy-selected expert corr | 0.909 | 专家选择与 patch entropy 高度相关 |
| EP all-to-all bytes | 4,198,400 | 每条记录约 4.2MB 通信 |

专家负载：

| expert | load fraction | unit fraction | patch entropy | patch length | 倾向 |
|---:|---:|---:|---:|---:|---|
| 0 | 0.3843 | 0.2889 | 0.683 | 7.997 | 低 entropy / 长 patch |
| 1 | 0.3843 | 0.2889 | 0.683 | 7.997 | 低 entropy / 长 patch |
| 2 | 0.1045 | 0.1860 | 1.606 | 3.245 | 中 entropy / 中短 patch |
| 3 | 0.1044 | 0.1859 | 1.606 | 3.245 | 中 entropy / 中短 patch |
| 4 | 0.0110 | 0.0237 | 2.650 | 2.663 | 高 entropy / 短 patch |
| 5 | 0.0110 | 0.0237 | 2.650 | 2.663 | 高 entropy / 短 patch |
| 6 | 0.0002 | 0.0015 | 1.403 | 0.410 | 几乎空闲 |
| 7 | 0.0002 | 0.0015 | 1.403 | 0.410 | 几乎空闲 |

结论：

- entropy PhaseB 确实学到了非常清晰的 entropy band 分工。
- 但它不是健康均衡的 MoE：0/1 负载过高，6/7 基本空闲。
- `top2_same_pair_fraction=0.9997` 说明路由几乎退化成固定专家对，而不是自由组合的 8-expert Top-2。

### 6.2 Hidden-only 参考指标

hidden-only 的 50k 继续训练没有可用 `metrics.jsonl`，只能用 `hidden_only_blt1b_10000step_lr5e6_balance0/metrics.jsonl` 作为 10k warmup 参考：

| 指标 | hidden-only 10k warmup | entropy PhaseB 50k |
|---|---:|---:|
| router entropy | 2.070 | 1.267 |
| active experts | 7.92 | 6.82 |
| max load fraction | 0.417 | 0.388 |
| min load fraction | 0.011 | 0.0002 |
| balance loss | 0.0178 | 1.288 |
| top2 same-pair fraction | 0.119 | 0.9997 |
| entropy-selected expert corr | 0.0065 | 0.909 |

这组参考说明：

- hidden-only 的路由更分散、更接近使用全部专家。
- entropy PhaseB 的路由更可解释，但也更集中、更像 entropy-band hard routing。
- 对论文来说，这是一个双刃剑：可解释性增强，但 MoE 负载均衡和专家利用率变差。

## 7. 论文层面的判断

### 可以主张的内容

1. Patch-level MoE 在 50k 训练窗口显著优于 dense。
   hidden-only 和 entropy PhaseB 都远低于 dense BPB。

2. Entropy-aware routing 在 50k 训练窗口优于 hidden-only routing。
   entropy PhaseB 的 49k-50k BPB 为 0.833，hidden-only 为 0.923，相对降低约 9.8%。

3. Entropy side feature 产生了可解释专家专门化。
   专家 0/1 接长、低 entropy patch；2/3 接中 entropy；4/5 接高 entropy；这直接对应论文里的 patch-native routing 叙事。

### 不能直接主张的内容

1. 不能说当前 MoE 更高效。
   虽然 FLOPs 更低，但实际 wps 只有 dense 的约四分之一，显存压力也更高。

2. 不能说 entropy PhaseB 已经是健康负载均衡的 MoE。
   专家 6/7 几乎空闲，0/1 承担主要负载。

3. 不能把这三组当作严格公平最终结果。
   dense/hidden 从 10k 继续，entropy PhaseB 从 PhaseA 1k 继续；并且 hidden 50k checkpoint 保存失败。

4. 不能忽略 entropy PhaseB 的初始化 warning 和 50k 后 OOM。
   日志中 `patch_feature_router` 有大量 suspiciously large initialization warning；50k 后继续训练出现 CUDA OOM。

## 8. 建议的下一步

1. 立即对 `entropy_bands_phaseB_100k/checkpoints/0000050000` 做 held-out eval。
   这是当前最有论文价值的 checkpoint：50k BPB 最好，且文件存在。

2. 重新跑或修复 hidden-only 50k checkpoint 保存。
   只有 stdout 指标不够支撑最终论文表格，需要可加载 checkpoint 和 held-out BPB。

3. 增加 entropy PhaseB 的负载均衡控制实验。
   建议至少补 `moe_balance_loss_weight=0.01` 和 `0.05`，观察 BPB 是否保持、专家 6/7 是否恢复使用。

4. 做 “quality vs system cost” 双表。
   主质量表报告 BPB/loss；系统表报告 wps、iter、FLOPs、显存、allocation retries、all-to-all bytes、max/min expert load。

5. 检查 `patch_feature_router` 初始化。
   entropy PhaseB 的 router 参数分布 warning 可能导致 routing 过早 hard partition，既可能解释强 specialization，也可能解释负载塌缩。

6. 如果论文主打 A-conference 贡献，当前最稳的叙事应是：
   “Entropy-aware patch routing improves modeling quality and yields interpretable expert specialization, but naive entropy-band routing requires additional load-balancing/system work to recover efficient expert utilization.”
