# Entropy PhaseB Balance 0.01 当前结论汇总

生成时间：2026-06-24。

## 1. 新实验状态

`balance=0.01` 已完成 50k 训练并成功保存 checkpoint：

`blt1b_warmstart/entropy_bands_phaseB_bal001_50k/entropy_bands_phaseB_bal001_50k/checkpoints/0000050000`

checkpoint 文件完整：

- `.metadata`
- `__0_0.distcp`
- `__1_0.distcp`
- `params.json`
- `train_state_00000.json`
- `train_state_00001.json`

这解决了“有 stdout 指标但缺少可加载 checkpoint”的核心问题之一。当前还没有发现 `balance=0.01` 的 held-out eval 输出。

## 2. 50k 训练质量对比

下表使用各自 `metrics.jsonl` 中 `global_step <= 50000` 的最后 100 条记录均值。

| run | 50k train BPB | 49k-50k BPB mean | 49k-50k loss mean | checkpoint |
| --- | ---: | ---: | ---: | --- |
| Entropy PhaseB, balance=0.00 | 0.832031 | 0.833242 | 0.588890 | yes |
| Entropy PhaseB, balance=0.01 | 0.832031 | 0.833438 | 0.602194 | yes |

结论：

- `balance=0.01` 没有破坏训练 BPB，最终 50k BPB 与原 PhaseB 一样都是 `0.832031`。
- 最后 100 条均值几乎相同：`0.833438` vs `0.833242`，差值只有 `+0.000196 BPB`。
- 训练 loss 均值略高：`0.602194` vs `0.588890`。这提示 BPB 口径下质量基本持平，但 loss 维度不如原始 PhaseB。

## 3. 系统成本对比

| run | wps mean | iter mean | FLOPs mean | max active GiB | alloc retries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entropy PhaseB, balance=0.00 | 623.63 | 2.1591 | 2.668e13 | 61.111 | 15 |
| Entropy PhaseB, balance=0.01 | 459.24 | 3.3020 | 1.965e13 | 61.111 | 15 |

结论：

- `balance=0.01` 的训练吞吐明显下降：`459 wps` vs `624 wps`，约低 `26%`。
- iter time 从 `2.16s` 增到 `3.30s`，慢约 `53%`。
- 显存 active 和 allocation retries 基本没变，说明不是显存峰值改善带来的权衡。

## 4. 专家负载与路由行为

| run | active experts | balance loss | load imbalance | max load | min load | expert 6 | expert 7 | top2 same pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Entropy PhaseB, balance=0.00 | 6.82 | 1.2877 | 3.1067 | 0.3883 | 0.000247 | 0.000247 | 0.000247 | 0.999737 |
| Entropy PhaseB, balance=0.01 | 6.82 | 1.2624 | 3.0939 | 0.3867 | 0.000247 | 0.000247 | 0.000247 | 0.999419 |

专家负载均值：

| run | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balance=0.00 | 0.3843 | 0.3843 | 0.1045 | 0.1044 | 0.0110 | 0.0110 | 0.00025 | 0.00025 |
| balance=0.01 | 0.3824 | 0.3821 | 0.1081 | 0.1083 | 0.0093 | 0.0093 | 0.00025 | 0.00025 |

结论：

- `balance=0.01` 基本没有恢复专家 6/7 使用。
- 负载分布仍然是 0/1 主载、2/3 次载、4/5 少量高 entropy、6/7 几乎空闲。
- `top2_same_pair_fraction` 仍接近 1，说明路由仍几乎固定成专家对。
- router entropy 和 entropy-selected correlation 基本不变，说明 entropy-band specialization 仍然很强。

## 5. balance=0.05 当前进度

当前检测到：

`blt1b_warmstart/entropy_bands_phaseB_bal005_50k/entropy_bands_phaseB_bal005_50k/metrics.jsonl`

截至当前读取：

| run | last step | BPB last | BPB mean100 | active experts mean100 | expert 6/7 mean100 | checkpoint 50k |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| balance=0.05 | 20640 | 0.890625 | 0.825273 | 7.16 | 0.000270 | not yet |

这个实验还不能作为最终结论，因为未到 50k，也没有最终 checkpoint。早期现象是：

- BPB mean100 暂时不差，但和 50k 不可直接比较。
- active experts 略高于 0.01/0.00，但 expert 6/7 仍几乎空闲。
- 目前尚未看到强 balance weight 使 6/7 真正恢复使用。

## 6. Held-out 状态

已有原 PhaseB 50k 的 quick held-out eval：

| run | held-out bytes | held-out BPB | held-out loss |
| --- | ---: | ---: | ---: |
| Entropy PhaseB balance=0.00 | 2,381,597 | 0.842595 | 0.584042 |

注意：这个 held-out 不是完整 50k JSONL 字节量，而是之前 `max_n_batches=2000` 的快速 eval。`balance=0.01` 目前还没有 held-out eval，所以不能进入最终论文质量表。

## 7. 当前论文结论

当前最稳的结论是：

1. `balance=0.01` 保住了 Entropy PhaseB 的 50k 训练 BPB。
   它没有造成明显质量退化，说明轻量 balance loss 与 entropy routing 兼容。

2. `balance=0.01` 没有解决专家塌缩。
   专家 6/7 仍几乎不用，top-2 仍几乎固定专家对。因此它不能作为“负载均衡修复成功”的证据。

3. `balance=0.01` 系统成本更差。
   吞吐明显低于原 PhaseB，同时显存和 allocation retry 没改善。

4. 对论文主叙事有利的是质量稳定和 checkpoint 可加载；不利的是系统效率与负载均衡没有改善。

建议主叙事仍保持谨慎：

> Entropy-aware patch routing gives strong modeling quality and interpretable specialization. A small balance loss preserves quality but is insufficient to fix expert utilization; stronger or structurally different balancing is needed for system-efficient MoE.

## 8. 下一步

1. 立即对 `balance=0.01` checkpoint 跑 held-out eval。
2. 等 `balance=0.05` 到 50k 后再用同一窗口和 held-out 口径比较。
3. 如果 `balance=0.05` 仍不能恢复 6/7，需要考虑不是 loss weight 不够，而是 entropy-band initialization 或 top-2 pair locking 太强。
4. 最终论文表格必须只纳入有 checkpoint 和 held-out eval 的 run。
