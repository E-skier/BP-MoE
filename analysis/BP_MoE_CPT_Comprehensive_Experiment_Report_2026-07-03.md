# BP-MoE CPT 综合实验分析报告

日期：2026-07-03  
仓库：`/data1/pengfeigao/BP-MoE`  
实验主题：BLT-1B PatchMoE CPT 中 entropy-guided routing 的质量收益与 pair-level byte-load collapse

---

## 1. 执行摘要

当前实验支持以下结论：

1. **entropy routing 是目前最强的质量信号。**  
   历史 PhaseB 结果显示，50k 附近质量排序为：

   ```text
   entropy PhaseB > hidden-only MoE >> dense
   ```

   spec 中记录的代表性数值为：

   ```text
   entropy PhaseB BPB ~= 0.833
   hidden-only MoE BPB ~= 0.923
   ```

   本地 `entropy_bands_phaseB_100k` 的 50k held-out 摘要也给出：

   ```text
   held-out BPB = 0.8425948453
   ```

2. **entropy routing 的质量收益伴随显著 pair-level byte-load collapse。**  
   历史 PhaseB 中：

   ```text
   top2_same_pair_fraction ~= 0.9997
   experts 0/1 承担约 76.9% byte load
   experts 6/7 几乎空闲
   ```

   新 BCFP 实验中，A1 clean baseline 也复现了 collapse：

   ```text
   pair max byte fraction = 0.9190
   pair min byte fraction = 0.0
   pair dead count = 2.48
   ```

3. **目前还没有找到“质量不掉 + byte load 均衡”的方案。**  
   C/D 系列能消除 dead pair，但 10k 后仍出现 pair drift 或质量退化：

   - C001/C002 10k：无 dead pair，但 pair0 回到约 0.706，train BPB 比 A1 差约 2.49%。
   - D 6GPU 10k：与 C 10k 基本重合，EMA bias 未形成有效控制。
   - D 4GPU 10k：pair load 较好，但 train BPB 明显更差。

4. **当前最核心的论文发现是结构性张力：**

   ```text
   entropy specialization 提升 LM 质量
   byte-level utilization balance 要求负载均衡
   二者在 web text 的 entropy/length 分布下存在冲突
   ```

   更准确地说，当前问题不是简单工程 bug，而是 entropy/length specialization 与 byte-mass balance 的目标不一致。

---

## 2. 实验与证据汇总

### 2.1 历史 entropy / hidden / dense 对照

| Run | Step | 训练 BPB 口径 | 关键诊断 | 结论 |
|---|---:|---:|---|---|
| `entropy_phaseB_100k` | 50480 | tail train BPB 0.7898 | top2 same pair 0.9966, entropy-selected corr 0.9257 | 质量强，强 entropy specialization |
| `entropy_bal001_50k` | 50000 | tail train BPB 0.8570 | top2 same pair 0.9998, entropy-selected corr 0.8830 | 质量仍强，但 pair-level collapse 明显 |
| `hidden_only_10k` | 10000 | tail train BPB 0.9578 | top2 same pair 0.0931, entropy corr 约 0 | 没有 entropy specialization，质量弱 |
| `dense_10k` | 10000 | tail train BPB 2.2375 | 非 MoE 对照 | 当前本地训练口径较差，仅作弱对照 |

额外 held-out：

```text
entropy_bands_phaseB_100k/heldout_eval/0000050000:
  BPB = 0.8425948453
  held-out bytes = 2,381,597
```

### 2.2 BCFP A/C/D screening 结果

| Run | Step | tail train BPB | pair max | pair min | dead pair | CV | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 clean | 5k | 0.846875 | 0.9190 | 0.0000 | 2.48 | 1.584 | 复现 collapse |
| C001 5k | 5k | 0.853906 | 0.4177 | 0.1470 | 0 | 0.407 | 早期负载好，质量轻微退化 |
| C002 5k | 5k | 0.853906 | 0.4177 | 0.1470 | 0 | 0.407 | 与 C001 5k 基本相同 |
| C001 10k | 10k | 0.867969 | 0.7059 | 0.0716 | 0 | 1.055 | 后期 pair0 回流，质量退化 |
| C002 10k | 10k | 0.867969 | 0.7059 | 0.0716 | 0 | 1.055 | 与 C001 10k 基本相同 |
| D 4GPU 10k | 10k | 0.890625 | 0.4461 | 0.0897 | 0 | 0.591 | 负载较好，但质量明显更差 |
| D 6GPU 10k | 10k | 0.867969 | 0.7066 | 0.0711 | 0 | 1.057 | 与 C 10k 基本重合，不通过 |

### 2.3 C 系列 fixed held-out 结果

| Run | Step | fixed held-out BPB | bytes | 备注 |
|---|---:|---:|---:|---|
| C001 | 5k | 0.6908252671 | 18,889,898 | 早期 checkpoint |
| C002 | 5k | 0.6907679867 | 18,889,898 | 与 C001 5k 接近 |
| C001 | 10k | 0.5858054749 | 18,889,898 | 比 C002 10k 略好 |
| C002 | 10k | 0.5859060593 | 18,889,898 | 与 C001 10k 接近 |

注意：D 4GPU/6GPU 10k checkpoint 因 screening 失败已清理，未做 held-out eval。

### 2.4 strong-bias grid 结果

已完成：

```text
experiments/bcfp_cpt/D_strong_bias_grid_2k_6gpu_nockpt
```

配置：

```text
ema in {0.8, 0.9}
lr in {0.2, 0.5}
max_byte_fraction in {0.45, 0.50}
steps = 2000
checkpoint = disabled
```

8 个组合均跑到 2k，未出现 OOM/retry，未留下 checkpoint 目录。尾部 5 个 log 点的核心结果：

| Run | tail BPB | tail loss | last pair max | last pair min | last pair byte distribution | last dynamic bias |
|---|---:|---:|---:|---:|---|---|
| ema08 lr02 max045 | 0.84453125 | 0.59850646 | 0.54342226 | 0.14105400 | [0.1710, 0.5434, 0.1414, 0.1442] | [-0.00192, -0.00192, +0.00242, +0.00143] |
| ema08 lr02 max050 | 0.84453125 | 0.59850559 | 0.54308394 | 0.14105400 | [0.1713, 0.5431, 0.1414, 0.1442] | [-0.00192, -0.00192, +0.00242, +0.00143] |
| ema08 lr05 max045 | 0.84453125 | 0.59848375 | 0.54342226 | 0.14121015 | [0.1710, 0.5434, 0.1419, 0.1437] | [-0.00475, -0.00475, +0.00589, +0.00361] |
| ema08 lr05 max050 | 0.84453125 | 0.59844782 | 0.54292779 | 0.14118413 | [0.1715, 0.5429, 0.1418, 0.1438] | [-0.00475, -0.00475, +0.00589, +0.00361] |
| ema09 lr02 max045 | 0.84453125 | 0.59847368 | 0.54363046 | 0.14123618 | [0.1707, 0.5436, 0.1422, 0.1434] | [-0.00612, -0.00256, +0.00635, +0.00233] |
| ema09 lr02 max050 | 0.84453125 | 0.59848652 | 0.54363046 | 0.14115810 | [0.1707, 0.5436, 0.1421, 0.1435] | [-0.00612, -0.00256, +0.00635, +0.00233] |
| ema09 lr05 max045 | 0.84453125 | 0.59843976 | 0.54521796 | 0.14105400 | [0.1692, 0.5452, 0.1431, 0.1425] | [-0.01523, -0.00634, +0.01561, +0.00596] |
| ema09 lr05 max050 | 0.84453125 | 0.59846341 | 0.54521796 | 0.14102798 | [0.1692, 0.5452, 0.1430, 0.1426] | [-0.01523, -0.00634, +0.01561, +0.00596] |

结论：strong-bias grid 没有把 routing 拉到目标上限。`max_byte_fraction=0.45` 的组合仍停在 pair max ~= 0.543-0.545；`max_byte_fraction=0.50` 也没有明显优于 0.45。bias 量级从约 0.002 增加到约 0.015，但 pair distribution 几乎不变，说明 D 失败不只是原始 bias 太弱、太慢。

与旧 C/D 10k run 的 step 1800-2000 窗口对比，strong-bias 2k 的 pair 分布和 C/D6 早期形态基本一致：

| Run | 2k window BPB | 2k window loss | 2k window pair max | 2k last pair distribution |
|---|---:|---:|---:|---|
| C001 10k | 0.89453125 | 0.63468422 | 0.53910215 | [0.1711, 0.5433, 0.1399, 0.1457] |
| C002 10k | 0.89609375 | 0.63468259 | 0.53910215 | [0.1711, 0.5433, 0.1399, 0.1457] |
| D6 10k | 0.89296875 | 0.63419667 | 0.53998699 | [0.1712, 0.5432, 0.1417, 0.1439] |
| D4 10k | 0.89609375 | 0.63503266 | 0.49901366 | [0.4927, 0.2117, 0.1432, 0.1524] |

注意：strong-bias 2k 的训练 BPB 低于旧 C/D 2k，但这可能来自 restart/data order/logging window 差异；在没有 fixed held-out 或严格同 seed 对照前，不能声称 strong bias 带来质量提升。

---

## 3. 关键现象：质量好与负载均衡形成 tradeoff

当前结果呈现稳定模式：

```text
不强控 routing:
  质量较好
  但 byte load collapse / pair drift 明显

强控 utilization:
  pair load 变好
  但 LM BPB 变差
```

最典型对比：

```text
C001/C002/D6 10k:
  tail BPB ~= 0.86797
  pair0 ~= 0.706
  no dead pair，但 max pair 超过 screening 阈值

D 4GPU 10k:
  pair max ~= 0.446
  tail BPB ~= 0.89063
```

因此，当前还没有达到 spec 中期望的 finalist 条件：

```text
pair load 达标
held-out/train quality 不显著劣化
entropy specialization 保留
```

---

## 4. Root-cause 分析

### 4.1 不是单纯 expert-level 问题，而是 pair-level byte-mass 问题

BCFP pair mode 保证：

```text
top2_same_pair_fraction = 1.0
pair 内两个 experts 权重约 0.5 / 0.5
```

因此主要问题不在 pair 内，而在 **哪个 pair 被分配到多少 byte mass**。

### 4.2 entropy specialization 与 patch length 强耦合

D 6GPU 10k 最后一步诊断：

| Pair | byte fraction | unit fraction | avg patch length | avg entropy |
|---:|---:|---:|---:|---:|
| 0 | 0.7066 | 0.2056 | 23.07 | 0.281 |
| 1 | 0.0711 | 0.0828 | 5.77 | 0.628 |
| 2 | 0.1023 | 0.1446 | 4.75 | 0.862 |
| 3 | 0.1199 | 0.5670 | 1.42 | 1.948 |

这说明：

```text
pair3 处理大量 patch，但 patch 很短；
pair0 处理 patch 数量不多，但 patch 极长；
byte load 因此集中到 pair0。
```

也就是说，collapse 不是简单的 assignment count collapse，而是 **low-entropy long-patch byte mass collapse**。

### 4.3 Web text 数据分布会放大这个问题

当前训练主要是 FineWeb-style web text。Web text 常见：

- boilerplate / 模板文本；
- 列表、导航、重复格式；
- 低 entropy 长片段；
- 高 entropy 短片段。

entropy router 自然会把低 entropy 长 patch 送到低 entropy pair。由于这些 patch 长，byte mass 会被放大，导致 pair-level byte load 不均。

因此，负载不均衡至少部分来自真实数据分布，而不是纯实现错误。

### 4.4 C 的问题：aux balance 与 LM objective 冲突

C 使用 byte auxiliary loss。它能让 5k 早期负载明显改善：

```text
C001/C002 5k:
  pair max ~= 0.418
  pair min ~= 0.147
  dead pair = 0
```

但到 10k：

```text
pair0 回到 ~= 0.706
tail BPB 比 A1 5k 差约 2.49%
```

说明 aux loss 没有形成稳定的长期约束，同时会干扰 LM objective。

### 4.5 D 的问题：EMA bias 在当前机制下不改变 routing

D 6GPU 10k 与 C 10k 几乎重合：

```text
D 6GPU pair0 = 0.7066
C001 pair0 = 0.7059
D 6GPU tail BPB = C001/C002 tail BPB
```

D 6GPU 最终 dynamic bias 量级约为：

```text
[-0.0022, -0.0012, +0.0029, +0.0006]
```

这个量级相对 entropy/hidden routing logits 太小，基本没有改变决策边界。

strong-bias grid 将 EMA、lr 和 max threshold 调强后，dynamic bias 最高达到约：

```text
[-0.0152, -0.0063, +0.0156, +0.0060]
```

但 step 2k 的 routing 仍几乎固定在：

```text
[0.17, 0.54, 0.14, 0.14]
```

因此当前证据更支持：router 的 entropy/length prior 或 batch/data pattern 主导了决策边界，EMA pair bias 在这个量级和更新方式下不足以改变 routing。

D 4GPU 则显示另一个方向：负载变好，但质量显著变差。这意味着如果控制足够强，它可能通过把 patch 送到次优 pair 来换取 balance。当前 C/D/strong-D 结果共同指向同一个 tradeoff：弱控制不改变路由，强控制可能损伤 LM objective。

---

## 5. 数据结构调整是否可能缓解

结论：**有机会缓解训练期 collapse，但不能保证解决真实 web distribution 下的长期负载均衡。**

### 5.1 可能有效的方向

1. **过滤低质量 / 低 entropy / 模板化 web text**

   如果大量低 entropy 长 patch 来自 boilerplate、重复模板或 HTML 残留，清洗这些内容可能同时改善质量与负载。

2. **entropy-length stratified sampling**

   按 entropy bucket 与 patch-length bucket 做采样，使每个 batch 的 byte mass 更均匀。目标不是让 assignment 数均匀，而是让各 entropy/length regime 的 byte mass 不在局部 batch 中极端偏斜。

3. **curriculum sampling**

   前 1k-3k steps 使用 balanced entropy/length sampling，防止 router 早期锁死；随后逐渐 anneal 回 natural web distribution。

4. **domain mixture**

   加入代码、数学、论坛、百科、书籍等不同结构数据，可能改变 entropy/length 分布，减少单一 web text 中模板化低 entropy 长 patch 的支配。

### 5.2 已完成的数据诊断与 ablation 启动

新增离线诊断脚本：

```text
scripts/diagnostics/analyze_entropy_length_distribution.py
scripts/diagnostics/build_entropy_length_balanced_source.py
```

原始 fineweb web-text 样本诊断：

```text
analysis/data_diagnostics/fineweb_entropy_length_distribution_2k_per_file.md
```

每个可用 shard 抽 2000 行，合计 18000 行：

| bucket | patch % | byte % | 结论 |
|---|---:|---:|---|
| low entropy | 53.71% | 83.11% | byte mass 极度集中 |
| medium entropy | 36.91% | 14.84% | patch 多但 byte 少 |
| high entropy | 9.37% | 2.05% | 几乎不贡献 byte mass |
| low/long | 16.00% | 40.36% | 最大单元 |
| low/medium | 24.25% | 32.55% | 第二大单元 |

这直接支持当前结构性解释：web text 中低 entropy 长/中 patch 只占约 40% patch units，却占约 73% byte mass，质量最优的 entropy specialization 天然会制造 byte-load imbalance。

已生成两个 balanced source：

1. row-dominant source：

   ```text
   fineweb_edu_10bt_entropy_length_balanced_2kpf_1kpb
   analysis/data_diagnostics/fineweb_entropy_length_balanced_2kpf_1kpb_distribution.md
   ```

   结果：low entropy byte share 仅从 83.11% 降到 80.18%，说明文档级 dominant-bucket 重采样力度不够。

2. top-cell-fraction source：

   ```text
   fineweb_edu_10bt_entropy_length_topcell_2kpf_1kpb
   analysis/data_diagnostics/fineweb_entropy_length_topcell_2kpf_1kpb_distribution.md
   ```

   结果：low entropy byte share 降到 77.61%，short-patch byte share 从 25.98% 提高到 31.34%。这仍不是真正均衡，但比 row-dominant 更适合作为 curriculum/data ablation。

A1 top-cell 早期短跑已停止：

```text
experiments/bcfp_cpt/A1_topcell_balanced_2k_6gpu_nofused_clip0_nockpt
```

step 350 仍为 `pair_max ~= 0.91-0.96`、`pair_min = 0`、dead pair 约 2.5，和 natural A1 的 collapse 形态没有分离，且 BPB 更高。因此单靠数据源调整无法修复未校准 A1。

更相关的 D ablation 已完成：

```text
experiments/bcfp_cpt/D_topcell_balanced_10k_6gpu_nofused_clip0_nockpt
metrics: experiments/bcfp_cpt/D_topcell_balanced_10k_6gpu_nofused_clip0_nockpt/metrics.jsonl
```

关键结果：

| Step | train BPB | pair byte distribution | pair max | pair min | dead pair | hidden residual |
|---:|---:|---|---:|---:|---:|---:|
| 100 | 0.89453125 | [0.2453, 0.4704, 0.1555, 0.1288] | 0.4704 | 0.1288 | 0.0 | 0.0 |
| 1000 | 0.84765625 | [0.3689, 0.2297, 0.1093, 0.2921] | 0.3689 | 0.1093 | 0.0 | 0.0 |
| 3000 | 0.84765625 | [0.1210, 0.0666, 0.0648, 0.7476] | 0.7476 | 0.0633 | 0.0 | 0.2499 |
| 5000 | 0.68750000 | [0.0703, 0.1667, 0.5995, 0.1635] | 0.5995 | 0.0703 | 0.0 | 0.25 |
| 7500 | 0.34570313 | [0.4867, 0.2329, 0.1730, 0.1074] | 0.4867 | 0.1074 | 0.0 | 0.25 |
| 10000 | 0.33007813 | [0.1609, 0.1519, 0.1529, 0.5344] | 0.5344 | 0.1497 | 0.0 | 0.25 |

尾部 20 个 log 点：

```text
tail BPB mean = 0.3410
tail pair max mean = 0.4848
tail pair min mean = 0.1139
tail dead pair = 0.0
tail pair max > 0.55: 4/20 records
tail pair max > 0.60: 2/20 records
```

相比 natural D/C 在 10k 最后一条记录的 `[0.706, 0.071, 0.103, 0.119]`，top-cell 数据源没有复现固定 pair0 `~=0.70` 的最终 collapse，且没有 dead pair。但它也没有稳定解决负载均衡：中途出现过 pair3 `0.7476`，尾部仍有 4/20 个记录超过 0.55。

注意：top-cell source 只有约 6602 条重采样记录，10k 训练会多轮重复该源。因此该 run 的 train BPB 不能和 natural web train BPB 直接比较，必须做 fixed natural held-out eval。

### 5.3 数据调整的限制

如果目标部署/评估分布就是 web text，那么 natural distribution 本身可能就是 byte-mass 偏斜的。此时即使训练时通过重采样获得更均衡 routing，回到 natural distribution 后仍可能重新出现 pair0-heavy。

因此，数据调整要同时报告：

```text
balanced-train distribution 上的 pair load
natural held-out distribution 上的 pair load
natural held-out BPB
```

否则只能说明训练被人为平衡，不能说明真实 workload 被解决。

---

## 6. 论文叙事建议

当前最适合的论文主线不是“我们已经解决负载均衡”，而是：

> Entropy routing is a strong quality signal for byte-level PatchMoE CPT, but its quality gain arises from entropy/length specialization that is naturally byte-imbalanced on web text. This creates a fundamental tension between quality-optimal specialization and utilization-balanced routing.

中文表达：

> entropy routing 的质量收益来自有意义的 entropy/length specialization；但 web text 中不同 entropy/length regime 的 byte mass 极不均匀，导致质量最优 routing 自然形成 pair-level byte-load collapse。

可以将贡献拆成：

1. 证明 entropy routing 相比 hidden-only/dense 具有明显质量优势。
2. 证明 entropy routing 的 top-2 实际退化成 pair-level specialization。
3. 发现 byte load collapse 与 entropy/length specialization 绑定，而不是简单实现 artifact。
4. 系统比较 BCFP init、byte aux balance、loss-free EMA bias，展示质量-负载 tradeoff。
5. 提出后续方向：data curriculum / distribution-aware routing / capacity-aware entropy specialization。

---

## 7. 推荐下一步实验

### 7.1 不建议把 8 个 strong-bias 组合全部长跑

strong-bias 2k grid 已完成。它没有显示出配置间有意义差异：

```text
所有组合 tail BPB = 0.84453125
last pair max ~= 0.543-0.545
last pair distribution ~= [0.17, 0.54, 0.14, 0.14]
```

因此不建议把 8 个组合都推进到 10k。若必须验证后期漂移，只需要选 1 个代表性配置：

```text
D_strongbias_ema09_lr05_max045_2k_6gpu_nockpt
```

理由：这是 bias 最大、目标最严格的组合。如果它 5k/10k 仍漂移，基本可以停止当前 EMA bias 路线；如果它能阻止 drift，再补 fixed natural held-out 判断质量代价。

更优先的下一步不是继续扩大 bias grid，而是做数据侧诊断和 curriculum ablation。

### 7.2 做数据侧 ablation

建议新增三个短跑：

1. `natural_web`：当前数据，作为对照。
2. `entropy_length_balanced`：按 entropy bucket × patch length bucket 平衡 byte mass。
3. `curriculum_balanced_to_natural`：前 2k balanced，之后 anneal 回 natural。

每个 run 记录：

```text
train BPB
fixed natural held-out BPB
pair byte max/min
pair unit max/min
entropy/length profile per pair
per bucket BPB
```

### 7.3 诊断数据分布

需要先生成一个数据分布报告：

```text
P(entropy_bucket, length_bucket)
byte_mass(entropy_bucket, length_bucket)
router_selected_pair(entropy_bucket, length_bucket)
BPB contribution by bucket
```

这会直接回答：collapse 是否主要由 web text 中低 entropy 长 patch 引起。

---

## 8. 当前结论边界

已经有证据支持：

- entropy routing 提供强质量信号；
- entropy routing 导致高度 pair-level specialization；
- web text 中 entropy/length 与 byte mass 的耦合会导致 pair-level load imbalance；
- C/D 当前方案未能同时保质量与负载均衡。

尚未证明：

- strong EMA bias 在所有可能参数下都失败；但当前 2k grid 已显示，在 ema 0.8/0.9、lr 0.2/0.5、max 0.45/0.50 范围内没有实质改善；
- 数据 curriculum 一定能解决 collapse；
- 在真实 natural held-out distribution 上可以长期维持 balanced load 且不损失 BPB。

因此，当前论文结论应保持为：

```text
Entropy routing is quality-optimal among current variants, but byte-balanced
entropy-specialized PatchMoE remains unresolved. The unresolved issue appears
structural: useful entropy/length specialization is naturally imbalanced in
web-text byte mass.
```

---

## 9. 2026-07-04 后续推进记录

### 9.1 Pure top-cell fixed natural held-out

`D_topcell_balanced_10k_6gpu_nofused_clip0_nockpt` 已完成 expanded natural held-out eval：

```text
heldout shard 00002 BPB = 0.7670614693
heldout shard 00003 BPB = 0.7509227498
combined weighted BPB ~= 0.75906
```

这明显弱于 C001/C002 10k fixed held-out 的约 `0.586`。因此 pure top-cell 不适合继续长跑；它主要证明“把训练分布做简单/重复”会让训练 BPB 看起来很好，但不能迁移到 natural held-out。

### 9.2 Light data mix 方向

当前更合理的方向是少量补偿而不是 pure top-cell：

```text
natural FineWeb 80-90%
entropy/length compensated source 10-20%
fixed natural held-out eval 作为主判据
```

已存在一个 in-flight run：

```text
experiments/bcfp_cpt/D_mix90_bal10_screenlike_5k_4gpu_2to5_nofused_clip0_ckptfinal
```

它使用：

```text
fineweb_edu_10bt: 0.9
fineweb_edu_10bt_entropy_length_balanced_2kpf_1kpb: 0.1
GPU: 2,3,4,5
target steps: 5000
```

截至 step 850，训练仍在推进，早期 pair load 没有 dead pair，但波动仍大：

```text
step 850 BPB = 0.91796875
pair byte = [0.5504, 0.1906, 0.1256, 0.1334]
pair max = 0.5504
pair min = 0.1256
dead pair = 0
```

这个 run 跑完后应先看 fixed natural held-out，不应只看 train BPB。

### 9.3 已准备的下一组 top-cell light-mix grid

已新增：

```text
experiments/bcfp_cpt/D_lightmix_topcell_grid_5k_6gpu_0to5/
```

包含：

```text
launch_one.sh
queue.sh
README.md
```

队列会顺序跑：

```text
D_mix90_topcell10_screenlike_5k_6gpu_0to5_nofused_clip0_ckptfinal
D_mix80_topcell20_screenlike_5k_6gpu_0to5_nofused_clip0_ckptfinal
```

二者都使用 D EMA-bias、GPU `0-5`、5k steps、final checkpoint，并在训练后跑 expanded natural held-out eval。启动前必须确认当前 4GPU run 已结束，避免和 GPU `2-5` 冲突。

### 9.4 D mix90/bal10 5k 最终结果

`D_mix90_bal10_screenlike_5k_4gpu_2to5_nofused_clip0_ckptfinal` 已完成训练与 fixed natural held-out eval：

```text
pipeline complete: 2026-07-05 03:06:29 CST
train step: 5000
eval bytes: 18,889,898
held-out weighted BPB = 0.6879106866
```

held-out shard：

```text
chunk 00002 BPB = 0.6921464695
chunk 00003 BPB = 0.6835998496
```

训练侧 tail 指标：

| Window | BPB | pair byte distribution | pair max | pair min | CV | dead |
|---:|---:|---|---:|---:|---:|---:|
| last | 0.843750 | [0.1899, 0.5319, 0.1064, 0.1718] | 0.5319 | 0.1064 | 0.6629 | 0 |
| tail 5 | 0.836719 | [0.2926, 0.3950, 0.1658, 0.1466] | 0.4990 | 0.1119 | 0.6225 | 0 |
| tail 20 | 0.854883 | [0.2668, 0.2771, 0.2315, 0.2246] | 0.4789 | 0.1165 | 0.5770 | 0 |

同口径 held-out 对比：

| Run | Step | held-out BPB | 结论 |
|---|---:|---:|---|
| D mix90/bal10 | 5k | 0.6879106866 | 比 C 5k 略好，但幅度很小 |
| C001 | 5k | 0.6908252671 | baseline |
| C002 | 5k | 0.6907679867 | baseline |
| C001 | 10k | 0.5858054749 | 更长训练，不与 5k 直接等价 |
| C002 | 10k | 0.5859060593 | 更长训练，不与 5k 直接等价 |
| D pure top-cell | 10k | 0.7590629725 | 明显失败 |

解释：`90% natural + 10% row-balanced` 在 held-out 质量上没有伤害，甚至比 C 5k 略好约 `0.4%` 相对 BPB；但训练负载指标没有显著优于 C 5k/D6 tail20。它消除了 dead pair，但 pair max/CV 仍停在 `~0.48/~0.58` 水平，说明 row-balanced light mix 不是负载问题的突破口。
