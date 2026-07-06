# Dense / Hidden / Entropy PhaseB Training Results Comparison

生成时间：2026-06-22 18:09 左右（Asia/Shanghai）

## 1. 数据来源与文档依据

本报告读取并对比了以下训练输出：

| 结构标签 | 日志文件 | 可用结构化指标 | 说明 |
|---|---|---|---|
| dense | `dense100k.out` | stdout 训练行 | 当前工作区未找到对应 100k `metrics.jsonl` 目录，按 stdout 解析 |
| hidden-only | `hidden100k.out` | stdout 训练行 | 用户描述中提到 `entropy`，但给定文件名和配置实际是 hidden-only PatchMoE |
| entropy_phaseB | `entropy_phaseB_100k.out` | stdout 训练行 + `blt1b_warmstart/entropy_bands_phaseB_100k/metrics.jsonl` | 额外包含 MoE 专家、负载、entropy bucket 等指标 |

相关文档结论：

- `README.md`：BLT 使用 byte-level 动态 patch，patch 由 next-byte entropy 等信号决定。
- `PatchMoE_Internal_Proposal.md`：PatchMoE 目标是把 sparse expert routing 放到 patch-level global Transformer；早期对比包括 dense BLT、hidden-only patch routing 和 entropy/length/cost-aware routing。
- `Byte_Patch_MoE_A_Conference_Roadmap.md`：主对比维度包括 BPB/loss、吞吐、内存、expert load imbalance、router entropy、load-balancing loss、expert specialization 等。

## 2. 三个运行的配置身份

| 维度 | dense | hidden-only | entropy_phaseB |
|---|---:|---:|---:|
| run name | `dense_blt1b_100000step_from10k_lr5e6` | `hidden_only_blt1b_100000step_from10k_lr5e6_balance0` | `entropy_bands_phaseB_100k` |
| 目标 step | 100000 | 100000 | 100000 |
| 实际日志末尾 step | 100000 | 50000 | 50440 |
| MoE experts | 0 | 8 | 8 |
| top-k | 0 | 2 | 2 |
| expert parallel size | 1 | 2 | 2 |
| MoE FFN dim multiplier | - | 0.5 | 0.5 |
| hidden-state routing | 无 MoE | true | false |
| patch entropy feature | 无 MoE | false | true |
| patch feature bias | 无 MoE | false | true |
| patch length / byte features | false / false | false / false | false / false |
| balance cost | byte（MoE disabled） | byte | byte |
| balance loss weight | 0.05（MoE disabled） | 0.0 | 0.0 |
| init checkpoint | dense 10k | hidden-only 10k | entropy PhaseA 1k |

重要可比性限制：

- dense 完整跑到 100000 step 并在日志中成功保存。
- hidden-only 在 50000 step 保存 checkpoint 时发生 `torch.distributed.checkpoint.api.CheckpointException`，没有完成 100k。
- entropy_phaseB 在本次读取时仍停留在约 50440 step，日志和 `metrics.jsonl` 都未到 100k；50000 checkpoint 在日志中保存成功。
- 三者起点不同：dense/hidden-only 从各自 10k checkpoint 继续，entropy_phaseB 从 PhaseA 1k checkpoint 继续。因此绝对 BPB 不能直接解释为同一训练阶段的公平最终成绩。

## 3. 共同训练指标对比

以下统计按 stdout 中每 10 step 的训练行解析。每个 step 只统计一次 `loss_avg/bpb_avg/wps/iter/mem`，避免 2 个 rank 重复计数。

| 结构 | 状态 | 记录 step 数 | 末尾 step | 末尾 BPB | 最近 100 条 BPB 均值 | 最近 100 条 loss 均值 | 最近 100 条 wps | 最近 100 条 iter(s) | 显存 active | 运行耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 完成 | 10000 | 100000 | 1.625 | 1.599 | 1.135 | 2260.6 | 0.669 | 52% | 19:07:25 |
| hidden-only | 50k 保存失败 | 5000 | 50000 | 0.930 | 0.923 | 0.652 | 587.3 | 2.810 | 77% | 1 day, 9:55:58 |
| entropy_phaseB | 未到 100k | 5044 | 50440 | 0.762 | 0.829 | 0.586 | 619.8 | 2.154 | 77% | 1 day, 10:34:26 |

### 训练曲线变化

| 结构 | 前 100 条 BPB 均值 | 最近 100 条 BPB 均值 | 变化 | 解释 |
|---|---:|---:|---:|---|
| dense | 2.111 | 1.599 | -0.511 | dense 在 10k -> 100k 期间持续下降，是唯一完整收敛样本 |
| hidden-only | 0.828 | 0.923 | +0.096 | 50k 前窗口较起始窗口变差，并最终在 50k 保存时报错 |
| entropy_phaseB | 0.448 | 0.829 | +0.382 | 从 PhaseA 1k 继续后，训练 BPB 明显上升；可能是阶段切换、router 结构或数据/初始化差异共同导致 |

### Rank 间波动

| 结构 | rank 间 BPB gap 均值 | rank 间 BPB gap 最大值 | 观察 |
|---|---:|---:|---|
| dense | 0.097 | 0.694 | dense rank 波动最大，可能与样本字节分布差异有关 |
| hidden-only | 0.056 | 0.521 | MoE 两个 rank 的训练 BPB 差异较小 |
| entropy_phaseB | 0.056 | 0.500 | 与 hidden-only 接近 |

## 4. 系统效率与资源占用

| 结构 | 最近 100 条 wps | 相对 dense wps | 最近 100 条 iter(s) | 相对 dense iter | active 显存 | 现象 |
|---|---:|---:|---:|---:|---:|---|
| dense | 2260.6 | 1.00x | 0.669 | 1.00x | 52% | 最高吞吐、最低显存 |
| hidden-only | 587.3 | 0.26x | 2.810 | 4.20x | 77% | EP + MoE dispatch 带来明显开销 |
| entropy_phaseB | 619.8 | 0.27x | 2.154 | 3.22x | 77% | 比 hidden-only 略快，但仍远慢于 dense |

系统层面的主要结论：

- MoE 结构显著增加显存占用：hidden-only 和 entropy_phaseB 的 active 显存约 77%，dense 约 52%。
- MoE 结构显著降低吞吐：hidden-only/entropy_phaseB 只有 dense 的约 26%-27% wps。
- entropy_phaseB 比 hidden-only 最近窗口 wps 高约 5.5%，iter 时间低约 23%，但二者运行阶段和完成状态不同，不能过度解释为结构必然更快。
- hidden-only 和 entropy_phaseB 日志中均反复出现 `15 CUDA memory allocation retries`；dense 没有该现象。这说明 MoE 路径在当前显存压力下更接近 allocator 紧张状态。

## 5. Entropy PhaseB 的 MoE 路由与专家负载

这一节来自 `blt1b_warmstart/entropy_bands_phaseB_100k/metrics.jsonl` 最近 100 条记录。dense 没有 MoE；hidden-only 的 100k 运行当前没有对应 metrics 文件，因此无法做同粒度专家指标对比。

| 指标 | 最近 100 条均值 | 解读 |
|---|---:|---|
| router entropy | 1.263 | 低于 8 expert 的最大熵 `ln(8)=2.079`，路由分布较集中 |
| active experts | 6.86 | 平均不是 8 个专家都充分活跃 |
| load imbalance | 3.124 | 专家负载明显不均衡 |
| max load fraction | 0.391 | 最重专家承担约 39% byte-cost load |
| min load fraction | 0.0003 | 最轻专家几乎空闲 |
| balance loss | 1.297 | 负载不均衡信号很强，但当前 balance loss weight 为 0.0 |
| top2 same-pair fraction | 0.9997 | Top-2 几乎固定成专家对，而不是自由组合 |
| entropy selected expert corr | 0.909 | 专家选择与 patch entropy 强相关 |
| EP all-to-all bytes | 4,198,400 | 每条记录约 4.2MB all-to-all 通信量 |
| patch length mean | 6.009 | 平均 patch 长度稳定在约 6 bytes |
| patch entropy mean | 1.100 | 当前窗口 patch entropy 均值约 1.10 |
| reserved memory | 94.58% | 显存 reservation 很高，allocator 压力明显 |

### 专家负载分布

| expert | load fraction | unit assignment fraction | patch entropy 均值 | patch length 均值 | 角色倾向 |
|---:|---:|---:|---:|---:|---|
| 0 | 0.3891 | 0.2936 | 0.664 | 8.012 | 低 entropy / 长 patch |
| 1 | 0.3890 | 0.2935 | 0.664 | 8.010 | 低 entropy / 长 patch |
| 2 | 0.1025 | 0.1818 | 1.567 | 3.299 | 中 entropy / 中短 patch |
| 3 | 0.1024 | 0.1817 | 1.567 | 3.297 | 中 entropy / 中短 patch |
| 4 | 0.0082 | 0.0231 | 2.570 | 2.111 | 高 entropy / 短 patch |
| 5 | 0.0082 | 0.0231 | 2.570 | 2.111 | 高 entropy / 短 patch |
| 6 | 0.0003 | 0.0016 | 1.475 | 0.430 | 几乎空闲 |
| 7 | 0.0003 | 0.0016 | 1.475 | 0.430 | 几乎空闲 |

Entropy PhaseB 的行为很清楚：它确实形成了 entropy-driven specialization，但负载严重集中在 0/1 两个低 entropy 专家上。4/5 只接收少量高 entropy 短 patch，6/7 基本闲置。由于 `moe_balance_loss_weight=0.0`，这种 specialization 没有被负载均衡项纠正。

## 6. 关键对比结论

1. dense 是唯一完整 100k 样本。
   dense 从 10k 继续训练到 100k，BPB 最近窗口均值从前 100 条的 2.111 降到 1.599，吞吐约 2261 wps，显存 52%。它是当前最稳定、工程风险最低的 baseline。

2. hidden-only 的训练质量指标低于 dense，但运行没有完成。
   hidden-only 在 50k 附近 BPB 最近窗口为 0.923，显著低于 dense 的 1.599；但它在 50k 保存 checkpoint 时失败，且没有 100k 结果。因此只能说明“50k 前训练 loss/BPB 表现较好”，不能作为 100k 完整结论。

3. entropy_phaseB 在 50k 附近训练 BPB 最低，但存在负载塌缩风险。
   entropy_phaseB 最近窗口 BPB 为 0.829，低于 hidden-only 的 0.923 和 dense 的 1.599；但它还没有到 100k，并且专家负载高度不均衡，0/1 专家合计承担约 77.8% load，6/7 几乎空闲。

4. MoE 带来的系统代价很大。
   hidden-only 和 entropy_phaseB 的吞吐只有 dense 的约四分之一，active 显存从 52% 升至 77%，reserved memory 接近 95%。如果目标是质量/效率 tradeoff，后续必须把 MoE dispatch、EP 通信和显存压力纳入主表，而不仅是训练 BPB。

5. entropy_phaseB 的专家专门化是“强但不平衡”。
   专家 0/1 对应低 entropy、长 patch；2/3 对应中 entropy；4/5 对应高 entropy、短 patch；6/7 基本没有有效负载。这支持“patch entropy 能驱动可解释专家分工”，但不支持“当前配置已达到良好负载均衡”。

## 7. 建议

1. 不要把当前三条结果写成完整 100k 公平对比。
   更准确的表述是：dense 已完成 100k；hidden-only 在 50k checkpoint 保存失败；entropy_phaseB 当前约 50.4k，仍需继续到 100k。

2. 优先修复 hidden-only 的 50k checkpoint 保存失败。
   失败点是 `torch.distributed.checkpoint.api.CheckpointException`。否则 hidden-only 不能作为可复现实验基线。

3. 对 entropy_phaseB 增加或恢复负载均衡约束。
   当前 `moe_balance_loss_weight=0.0`，而 `balance_loss/load_imbalance/max_load_fraction` 都显示负载明显塌缩。建议尝试 `moe_balance_loss_weight=0.01/0.05`，或加入 entropy-band capacity/temperature 约束。

4. 为 dense/hidden 100k 运行补齐 `metrics.jsonl`。
   目前 dense/hidden 100k 只能从 stdout 解析共同训练指标，缺少 patch/MoE JSON 指标，不利于后续自动制表和专家专门化对比。

5. 后续主表建议同时报告：
   BPB/loss、完成 step、是否成功保存 checkpoint、wps、iter time、active/reserved memory、CUDA allocation retries、router entropy、load imbalance、max/min expert load、top2 same-pair fraction、expert entropy/length specialization。
