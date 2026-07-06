# AAAI 正式实验执行路线：Entropy-Guided Tokenizer-Free PatchMoE

**项目：** BP-MoE / PatchMoE  
**基线模型：** BLT-1B dense checkpoint  
**状态：** 前期机制探索完成；本文件定义正式实验，不再进行无边界的机制扫参。  

## 1. 论文目标与范围冻结

### 1.1 核心主张

> 在 tokenizer-free、dynamic byte-patch language models 中，真实 patch entropy 是有效的 MoE routing side feature。采用 sparse PatchMoE 后，在匹配的 **active training FLOPs** 下，entropy-guided routing 能获得优于 dense BLT 与 hidden-only MoE 的 held-out byte-level modeling quality。

本篇论文要证明：

1. sparse PatchMoE 是否能在 matched active training compute 下优于 dense BLT；
2. entropy routing 是否稳定优于 hidden-only routing；
3. entropy 的作用是否独立于 patch length，且不是任意额外 feature 的偶然正则化；
4. entropy routing 是否诱导稳定、可解释的 entropy–length-conditioned expert specialization；
5. entropy 收益是否不依赖于 fixed paired Top-2 的结构约束。

### 1.2 不作为本文核心目标的事项

本轮正式实验**不再以负载均衡为优化目标**，也不主张：

- 所有 expert 的原始 byte mass 必须均匀；
- wall-clock throughput 更高；
- 显存或 all-to-all 通信更优；
- 高效 distributed MoE execution；
- expert capacity scheduling、EMA-bias 或 byte auxiliary loss；
- continual domain adaptation；
- 大规模多域预训练或 MoE serving。

这些指标仍需记录，并作为透明的 limitation / systems diagnostics 报告。

### 1.3 关于“负载”的解释原则

在 dynamic byte-patch 模型中，至少区分：

\[
B_g=\text{pair/expert }g\text{ 的原始 byte coverage},
\]
\[
U_g=\text{pair/expert }g\text{ 的 valid patch-unit utilization},
\]
\[
T_g=\text{pair/expert }g\text{ 的真实 dispatch + FFN runtime}.
\]

不得把 `byte fraction` 直接等价为 MoE runtime bottleneck。低 entropy 的长 patches 可带来高 byte coverage，但其 patch-unit 工作量未必高。正式论文中只报告 routing concentration，不把 byte-uniformity 当作模型质量选择准则。

---

## 2. 正式主架构：F-PatchMoE

### 2.1 架构定义

正式主架构采用：

```text
4 full-width experts
+ Top-1 routing
+ dense-copy function-preserving initialization
```

简称 **F-PatchMoE**。

```yaml
moe_num_experts: 4
moe_top_k: 1
moe_ffn_dim_multiplier: 1.0
expert_init: dense_copy
paired_routing: false
moe_balance_loss_weight: 0.0
capacity_control: false
```

### 2.2 选择该架构的原因

原先 paired Top-2 为：

```text
8 half-width experts + fixed paired Top-2
```

其实际行为接近：

```text
4 macro-experts + Top-1 routing
```

因此会混淆 entropy routing effect 与 predefined pair structure effect。

F-PatchMoE 与 paired Top-2 的 expert FFN 容量对齐：

| 架构 | 总 expert FFN 容量 | 每个 patch 激活 FFN 容量 |
|---|---:|---:|
| 8 half-width experts + paired Top-2 | 4× dense FFN | 1× dense FFN |
| 4 full-width experts + free Top-1 | 4× dense FFN | 1× dense FFN |

F-PatchMoE 的每个 expert 都完整复制 dense FFN，因此任意 Top-1 assignment 在 step 0 都可保持 dense function。

### 2.3 Step-0 function-preserving 条件

对 dense FFN \(F_{\mathrm{dense}}\)，初始化：

\[
E_g(\mathbf h)\leftarrow F_{\mathrm{dense}}(\mathbf h),\quad g=1,\dots,4.
\]

Top-1 routing 输出：

\[
y_{\mathrm{moe}}(\mathbf h)=E_{\arg\max_g z_g}(\mathbf h)=F_{\mathrm{dense}}(\mathbf h).
\]

这意味着：**所有 router modes 在 t=0 必须与 dense model 数值连续。**

---

## 3. 唯一允许的工程 Gate（不是新一轮研究探索）

### G0：Full-width Top-1 等价性测试

使用固定 BLT-1B dense checkpoint 和真实 preprocessed patches，验证：

```text
A. dense FFN 与 F-PatchMoE FFN 的 max_abs_error
B. relative L2 error
C. full-model logits cosine similarity
D. small fixed eval shard 的 BPB delta
E. Top-1 selected gate weight = 1
F. invalid patches (patch_length <= 0) 不进入 router 或 expert dispatch
```

**验收阈值：**

```text
fp32 reference relative L2 error <= 1e-4
small-shard BPB delta 数值上可忽略
invalid-patch dispatch count = 0
```

若 G0 失败，禁止启动正式训练；先修实现。

### G1：Run manifest 与可复现性登记

每次训练开始时生成 `run_manifest.json`：

```json
{
  "git_commit": "...",
  "dense_init_checkpoint": "...",
  "architecture": "...",
  "router_mode": "...",
  "seed": 42,
  "world_size": 6,
  "expert_parallel_size": 2,
  "data_manifest_sha256": "...",
  "entropy_preprocess_manifest_sha256": "...",
  "optimizer_config_sha256": "...",
  "compute_budget": "...",
  "valid_patch_dispatch_enabled": true
}
```

无 manifest 的 run 不进入最终主表。

---

## 4. 数据、评估与公平比较协议

### 4.1 Series A：受控正式主实验

Series A 继续采用既有、已验证的数据流：

```text
FineWeb-Edu-10BT: chunk 00000 + chunk 00001
```

目的：保持与前期机制结论的可比性，且不将数据迁移变量混入 architecture/router 因果比较。

所有模型必须固定：

```text
same training shards
same sample order seed
same entropy preprocessing version
same entropy-model checkpoint
same patching threshold
same sequence length
same global batch policy
same optimizer and LR schedule
same FSDP / EP / precision configuration
```

禁止：

```text
0–50k FineWeb -> 50–100k math -> 100–150k code
```

不得把 sequential domain switching 作为本轮正式主实验的一部分。

### 4.2 Series B：同分布数据扩展验证

Series A 主结论形成后，重新从同一 dense checkpoint 出发，扩大到：

```text
FineWeb-Edu-10BT: chunk 00000–00007
```

Series B 只验证主结论是否依赖于两个特定 chunks；不新增训练机制或调参。

### 4.3 固定 held-out protocol

建立两个不可变 held-out manifest：

```text
heldout_natural_core
heldout_natural_extended
```

每个 checkpoint 必须在完全相同的 eval bytes、patching config、masking policy、max length 和 BPB 实现上评估。

训练 tail BPB 仅用于诊断，不能作为最终质量结论。

---

## 5. Active Training FLOPs 协议

### 5.1 主图

主图为：

\[
\text{Held-out BPB}\quad \text{vs.}\quad \text{Cumulative Active Training FLOPs}.
\]

不使用 “same number of steps” 直接代替 compute matching。

### 5.2 必须纳入统一计数器的组成

`active_compute_counter` 至少应计入：

```text
1. local byte encoder / decoder
2. global attention，使用实际 patch count
3. dense FFN 或被选择的 MoE expert FFN
4. router projection
5. output / embedding projections
6. 一个明确且各模型一致的 forward+backward accounting rule
```

MoE 只计入 activated expert FFN 的计算，不能将 4 个 full-width experts 的总容量视为每个 patch 的 active compute。

### 5.3 Compute budget 的预注册方式

每种 architecture 先运行同一固定 200 batches 的无更新/短 profiling，记录：

```text
mean_active_train_flops_per_step
mean_valid_bytes_per_step
mean_valid_patches_per_step
```

定义预注册预算：

```text
C_low   ≈ 当前 5k-equivalent compute
C_mid   ≈ 当前 10k-equivalent compute
C_high  ≈ 当前 50k-equivalent compute
```

对 architecture \(r\) 的目标训练步数：

\[
N_r(C)=\left\lfloor C/\overline{\mathrm{FLOPs}}_r^{\mathrm{step}}\right\rfloor.
\]

论文主图按 cumulative active training FLOPs 比较；附录同时报告 same-training-bytes 视角。

不得单独依赖 stdout 里的 `flops` 字段作为最终计算证据；必须用统一 counter，并以 profiler sanity check 验证相对量级。

---

## 6. Series A 主结果矩阵

### 6.1 Main runs

所有 main runs 均从**完全相同的 BLT-1B dense checkpoint**启动。

| ID | 方法 | Router 输入 | 作用 |
|---|---|---|---|
| A-Dense | Dense BLT CPT | 无 MoE | dense compute-quality reference |
| A-H | F-PatchMoE | hidden only | sparse hidden-only baseline |
| A-E | F-PatchMoE | patch entropy only | entropy 的独立信息量 |
| A-HE | F-PatchMoE | hidden + patch entropy | 主候选方法 |

固定：

```text
Data: FineWeb-Edu chunk 00000 + 00001
Seeds: {42, 43, 44}
Checkpoints: C_low, C_mid, C_high
Same dense start / same stream / same infra config
```

正式规模：

```text
4 methods × 3 seeds × C_high budget
```

中间预算点从同一 run 的 checkpoints 读取，不单独重跑。

### 6.2 Router modes 的语义定义

```text
A-H:
  hidden router trainable
  entropy absent
  length absent

A-E:
  hidden router contribution disabled in code
  aligned real patch entropy is the only router input

A-HE:
  hidden router trainable
  aligned real patch entropy feature trainable
  no length feature in main method

A-Dense:
  original dense BLT FFN
```

注意：`entropy-only` 不能通过“hidden router 权重很小”实现；应在逻辑上关闭 hidden contribution。

### 6.3 Main claim 成立的最低条件

每个 checkpoint 输出：

```text
held-out BPB / loss
cumulative active FLOPs
training bytes / valid patches consumed
total parameters / active parameters
router entropy
expert valid-patch share
expert byte coverage
expert entropy and length profiles
wps / iter time / peak memory
```

最小结论条件：

```text
1. A-E 或 A-HE 在 C_mid、C_high 上优于 A-H；
2. A-E 或 A-HE 在 C_high 上优于 A-Dense；
3. 三个 seed 的 improvement direction 一致；
4. 同 seed 配对差值的 95% CI excludes zero；
5. 结论来自固定 held-out，而不是 tail train BPB。
```

若 A-E 和 A-HE 都优于 A-H，则选择效果更稳定、叙事更简洁的一项作为 paper 主方法，另一项作为强 ablation。

---

## 7. Series A：因果与结构鲁棒性实验

这些是正式证据，但无需全部跑至 C_high。

### 7.1 Feature causality ablations

| ID | Router | Budget | Seeds | 要回答的问题 |
|---|---|---:|---:|---|
| B-L | length-only | C_mid | 3 | entropy 是否只是 length proxy |
| B-HL | hidden + length | C_mid | 3 | entropy 相对 length 的增益 |
| B-HE-shuf | hidden + shuffled entropy | C_mid | 3 | entropy 与原 patch 对齐是否必要 |
| B-E-shuf | shuffled entropy only | C_mid | 2–3 | 更强的 entropy-only 对照 |
| B-rand | random feature | C_low/C_mid | 2 | 可选：排除任意输入维度收益 |

`shuffled entropy` 实现必须满足：

```text
- 只对 valid patches 进行 permutation；
- entropy 边缘分布完全不变；
- patch length、mask、batch composition 不变；
- entropy 与原 patch hidden/context 的对应关系被破坏；
- global step、seed、rank、layer policy 被记录并可复现。
```

主要因果结论要求：

\[
\text{real entropy}>\text{shuffled entropy},\qquad
\text{entropy-only}>\text{length-only}.
\]

### 7.2 Paired Top-2 structural sanity control

| ID | Architecture | Router | Budget | Seeds |
|---|---|---|---:|---:|
| S-PH | 8 half-width + paired Top-2 | hidden-only | C_mid | 2 |
| S-PE | 8 half-width + paired Top-2 | entropy-guided | C_mid | 2 |

该控制必须使用相同 dense checkpoint、相同 data/eval manifests 与统一 compute accounting。

判定：

```text
F-PatchMoE 仍有 entropy gain:
  主结论不依赖 pairing，可采用 F-PatchMoE 为正式方法。

只有 paired Top-2 有 entropy gain:
  pairing 必须纳入方法定义；不得宣称自由 expert routing 的一般规律。
```

---

## 8. Series B：同分布扩展验证

Series A 主方法确定后，**不进行新机制搜索**，从同一 dense checkpoint 重启：

| ID | 方法 | Seeds | Budget |
|---|---|---:|---:|
| B-Dense | Dense BLT | 2–3 | C_high |
| B-H | F-PatchMoE hidden-only | 2–3 | C_high |
| B-Entropy | Series A selected entropy method | 2–3 | C_high |

数据：

```text
FineWeb-Edu chunks 00000–00007
```

目标：证明 entropy gain 不依赖于 chunk 00000/00001 的偶然统计特性。

禁止从 Series A checkpoint 接着更换 chunks；Series B 所有方法必须从 original dense checkpoint 重新 upcycle。

---

## 9. 日志、checkpoint 与分析产物

### 9.1 每 step / 每 N steps 记录

```text
global_step
cumulative_active_flops
cumulative_valid_bytes
cumulative_valid_patches
train_loss
train_bpb
learning_rate
wps
iter_time
allocated_memory
reserved_memory
router_entropy
expert_i_valid_patch_fraction
expert_i_byte_fraction
expert_i_entropy_mean
expert_i_length_mean
dead_expert_count
top1_assignment_entropy
```

### 9.2 每个 checkpoint 生成

```text
checkpoint/
heldout_metrics.json
compute_accounting.json
expert_profile.csv
entropy_expert_heatmap.png
length_expert_heatmap.png
routing_stability_inputs.npz
run_manifest.json
```

### 9.3 最终图表

1. Held-out BPB vs cumulative active training FLOPs；
2. Dense / Hidden / Entropy / Hidden+Entropy 主结果表；
3. entropy、length、shuffled entropy 的 feature ablation 表；
4. entropy–expert specialization heatmap；
5. expert length profile 与 cross-seed stability；
6. systems transparency table：active FLOPs、wps、memory、expert usage。

系统表固定说明：

> Active training compute is matched. Wall-clock throughput and load-balanced sparse execution are reported for transparency but are not optimization objectives of this work.

---

## 10. 统计与报告规则

对同一 seed 的模型差值定义：

\[
\Delta_s=\mathrm{BPB}_{\mathrm{baseline},s}-\mathrm{BPB}_{\mathrm{method},s}.
\]

报告：

```text
mean ± standard deviation
paired bootstrap 95% confidence interval
all individual seed results
```

禁止：

```text
- 只报告 best seed；
- 只报告 best checkpoint；
- 用训练 tail BPB 替代 held-out；
- 在看到曲线后新增一个“更有利”的比较 checkpoint；
- 静默删除不稳定 run。
```

---

## 11. 结果决策树

### 情形 A：Entropy-only 最好

```text
A-E > A-H and A-Dense
A-E > B-L and B-E-shuf
```

主方法命名：**Entropy-Only F-PatchMoE**。

结论：patch entropy 本身包含 hidden-only router 未利用的 routing information。

### 情形 B：Hidden+Entropy 最好

```text
A-HE > A-E, A-H, A-Dense
A-HE > B-HL and B-HE-shuf
```

主方法命名：**Entropy-Guided F-PatchMoE**。

结论：entropy 是对 contextual hidden representation 的互补，而不是替代。

### 情形 C：Entropy 只在 paired Top-2 中有效

```text
S-PE gain exists, but A-E/A-HE gain does not
```

处理：将 pairing 纳入方法本体，收束叙事为 function-preserving paired sparse upcycling；不得宣称 entropy 对 free-expert routing 普适有效。

### 情形 D：Series A 有优势、Series B 消失

处理：不 cherry-pick；先核查 data manifest、entropy preprocessing、compute counter 与 evaluation protocol；在解释前不扩展到跨域训练或系统路线。

---

## 12. 明确不做事项

在 Series A 与 Series B 完成前，不做：

```text
- byte auxiliary balance coefficient sweep
- EMA byte-bias sweep
- 强行将 byte fraction 拉到均匀
- code/math/multilingual training mixture
- sequential domain switching
- Megatron/DeepSpeed stack migration
- larger-scale model training
- distributed expert-parallel systems optimization
- complement-bank Top-2 作为主线
```

这些可在论文后续工作或下一篇系统论文中处理。

---

## 13. Codex 实施任务

### Task 1 — F-PatchMoE

- 支持 `moe_num_experts=4`、`top_k=1`、`ffn_dim_multiplier=1.0`；
- 增加 `expert_init=dense_copy`；
- 将每层 dense FFN 的 `w1/w2/w3` 复制到四个 full-width experts；
- 实现 G0 unit + integration test；
- `patch_length <= 0` 的位置不进入 router 或 expert dispatch，也不产生 router gradient。

### Task 2 — Router mode 枚举

```text
router_mode in {
  hidden,
  entropy,
  hidden_entropy,
  length,
  hidden_length,
  entropy_shuffled,
  hidden_entropy_shuffled
}
```

每种 mode 必须通过代码路径严格关闭不需要的贡献，不能依赖“小权重”。

### Task 3 — Active FLOPs accounting

- 实现统一 counter；
- 每 step 记录 cumulative active FLOPs；
- 用固定 batch profiler sanity check；
- 每 checkpoint 输出 `compute_accounting.json`。

### Task 4 — Manifests 与启动器

创建：

```text
aaai_dense
aaai_fw4_hidden
aaai_fw4_entropy
aaai_fw4_hidden_entropy
aaai_fw4_length
aaai_fw4_hidden_length
aaai_fw4_entropy_shuffled
aaai_fw4_hidden_entropy_shuffled
aaai_pair8_hidden_control
aaai_pair8_entropy_control
```

每个 launcher 支持：

```text
--seed
--compute-budget {low,mid,high}
--data-manifest
--resume
--output-dir
```

### Task 5 — 一键评估与汇总

提供：

```bash
python -m experiments.aaai.generate_report \
  --run-root <RUN_ROOT> \
  --heldout-manifest <HELDOUT_MANIFEST> \
  --output-dir <OUTPUT_DIR>
```

固定输出：

```text
main_results.csv
feature_ablation.csv
compute_curves.csv
expert_profiles.csv
all_seed_summary.csv
fig_bpb_vs_compute.pdf
fig_entropy_specialization.pdf
fig_length_specialization.pdf
table_system_transparency.md
```

---

## 14. 正式启动顺序

### Step 0：实现认证

```text
G0 full-width Top-1 equivalence
G1 manifest recording
200-batch active-compute profiling
fixed held-out pipeline validation
```

### Step 1：Series A main runs

启动：

```text
A-Dense: seeds 42, 43, 44
A-H:     seeds 42, 43, 44
A-E:     seeds 42, 43, 44
A-HE:    seeds 42, 43, 44
```

每个 run 自动保存 `C_low / C_mid / C_high` checkpoints。

### Step 2：Series A causality + structural controls

```text
B-L
B-HL
B-HE-shuf
S-PH
S-PE
```

### Step 3：Series B

选定 Series A 中最强 entropy method 后，从 original dense checkpoint 重启：

```text
B-Dense
B-H
B-Entropy
```

使用 expanded FineWeb-Edu chunks。

---

## 15. 最终论文表述模板

### 主结论

> Dynamic patch entropy is a routing-relevant uncertainty signal for sparse conditional computation in tokenizer-free language models. At matched active training compute, entropy-guided PatchMoE improves held-out byte-level modeling quality over dense BLT and hidden-only sparse routing.

### 机制结论

> The improvement is not explained by patch length alone or by adding an arbitrary router feature: it weakens when entropy is shuffled across patches and is accompanied by stable entropy–length-conditioned expert specialization.

### 透明限制

> We optimize modeling quality under matched active compute. Sparse execution utilization, expert load balancing, and wall-clock throughput are measured and reported, but are not optimization objectives of this work.

---

## 16. 正式实验完成条件

```text
[ ] G0 passed
[ ] unified compute counter profiler-validated
[ ] all Series A main runs reached C_high
[ ] every main checkpoint evaluated on fixed held-out
[ ] all seed-level outputs retained
[ ] causality ablations completed
[ ] paired structural control completed
[ ] Series B completed or explicitly recorded as unavailable
[ ] reporting pipeline reproduces all figures/tables from raw metrics
```

只有上述条件满足后，才可以在论文中使用：

> Entropy-guided tokenizer-free PatchMoE outperforms dense and hidden-only baselines at matched active training compute.

---

## 17. 强制 Checkpoint 策略：每 1k 保存，最新状态可恢复

### 17.1 目标与硬约束

由于训练任务可能无人值守、服务器磁盘空间有限，所有正式实验必须满足：

```text
- 每 1,000 个 global optimizer steps 保存一次 full checkpoint；
- 任意时刻必须保留一个“最新且已验证可恢复”的 checkpoint；
- 新 checkpoint 写入、校验或提交失败时，旧的最新可恢复 checkpoint 必须仍然存在；
- 不允许按固定数量盲目删除 checkpoint；
- 不允许将未完成写入的目录标记为 latest；
- resume 默认只能恢复 latest_valid，而不能恢复未验证目录。
```

这里的“最新”指：

> 最近一个完成写入、完成完整性校验、并已成功更新 latest 指针的 checkpoint。

训练运行时可能存在一个正在写入的更高 step 的 staging checkpoint；它在校验完成前**不是** latest，也不得被用于恢复。

### 17.2 保留策略

正式默认策略：

```yaml
checkpoint:
  every: 1000                    # global optimizer steps
  keep_last_complete: 2          # latest + fallback
  keep_milestones: []            # 主实验不额外永久保留 milestone，避免磁盘膨胀
  save_optimizer_state: true
  save_rng_state: true
  save_dataloader_state: true
  save_router_state: true
  atomic_commit: true
  verify_after_save: true
  latest_pointer: checkpoints/LATEST.json
  staging_dir: checkpoints/.staging
```

`keep_last_complete=2` 是默认下限，因为：

```text
latest checkpoint：正常恢复点；
previous checkpoint：新 checkpoint 写入中断、损坏、OOM、NFS/metadata 异常时的 fallback。
```

**禁止将 `keep_last_complete=1` 用于正式实验。**  
只保留一个 checkpoint 时，无法同时满足“新 checkpoint 正在写入”与“旧 checkpoint 仍可恢复”；若写入失败或目录损坏，训练可能失去唯一恢复点。

### 17.3 磁盘空间预检

每次 checkpoint 前必须进行预检：

\[
\mathrm{free\_space}
\ge
1.25\times \widehat S_{\mathrm{checkpoint}}
+
S_{\mathrm{safety}},
\]

其中：

```text
S_checkpoint: 最近一个完整 checkpoint 的实际目录大小；
S_safety: 至少 20 GiB，或 max(20 GiB, 10% of S_checkpoint)。
```

建议实现：

```text
required_free_bytes =
ceil(1.25 * latest_complete_checkpoint_size)
+ max(20 GiB, 0.10 * latest_complete_checkpoint_size)
```

若空间不足：

1. 只删除比“最近两个 complete checkpoints”更旧的 **已验证 complete** checkpoint；
2. 再次检查空间；
3. 若仍不足，记录明确错误并停止在安全状态；
4. **绝不删除 `LATEST.json` 指向的 checkpoint；**
5. **绝不删除唯一 fallback checkpoint；**
6. **绝不在空间不足时开始写新 checkpoint。**

训练应在日志中明确输出：

```text
[checkpoint] preflight step=12000
[checkpoint] latest_complete_size=...
[checkpoint] free_space=...
[checkpoint] required_free=...
[checkpoint] retention_pruned=[...]
[checkpoint] decision=proceed|abort_safe
```

### 17.4 原子保存协议

每次 `global_step % 1000 == 0` 时，严格执行以下顺序：

```text
1. 创建唯一 staging 目录：
   checkpoints/.staging/0000012000.<uuid>/

2. 将 model / optimizer / scheduler / RNG / dataloader /
   router state / run_manifest 保存到 staging 目录。

3. 写入 checkpoint_manifest.json，至少包含：
   - global_step
   - cumulative_active_flops
   - cumulative_valid_bytes
   - git_commit
   - source_dense_checkpoint
   - model/router config SHA256
   - data manifest SHA256
   - file list
   - each file's size and SHA256
   - save timestamp

4. 在同一进程或独立 verifier 中验证：
   - 关键文件存在；
   - 文件大小非零；
   - hash 与 manifest 一致；
   - model state 可以被重新 load；
   - optimizer / scheduler state 可读；
   - RNG 和 dataloader state 存在；
   - global_step 与 manifest 一致。

5. 验证成功后，原子 rename：
   checkpoints/.staging/0000012000.<uuid>/
   ->
   checkpoints/0000012000/

6. 原子更新 checkpoints/LATEST.json：
   先写 checkpoints/.LATEST.tmp.<uuid>
   再 os.replace(...) 为 checkpoints/LATEST.json。

7. 只有 LATEST.json 成功更新后，才执行 retention pruning：
   保留两个最新 complete checkpoints；
   删除其余 complete checkpoints；
   staging 下失败目录保留为 failed/，或按独立清理规则删除。
```

关键原则：

```text
staging directory != valid checkpoint
complete directory + valid manifest != latest
only LATEST.json defines the default resume target
```

### 17.5 LATEST.json 格式

```json
{
  "schema_version": 1,
  "global_step": 12000,
  "checkpoint_relpath": "0000012000",
  "checkpoint_manifest_sha256": "<sha256>",
  "git_commit": "<commit>",
  "created_at_utc": "<timestamp>",
  "status": "complete_and_verified"
}
```

恢复时：

```bash
python -m bytelatent.train \
  --resume checkpoints/LATEST.json
```

训练代码必须：

1. 读取 `LATEST.json`；
2. 验证引用目录存在；
3. 验证 checkpoint manifest；
4. 再加载 checkpoint；
5. 若 latest 无效，自动扫描按 step 降序的 complete checkpoint；
6. 选择最新一个可验证的 fallback；
7. 在日志中显式输出恢复来源。

例如：

```text
[resume] LATEST step=12000 verified=true
[resume] restoring checkpoints/0000012000
```

或：

```text
[resume] LATEST step=12000 invalid: missing optimizer shard
[resume] fallback step=11000 verified=true
[resume] restoring checkpoints/0000011000
```

### 17.6 必须保存的状态

每个 full checkpoint 必须包括：

```text
model state
optimizer state
scheduler state
AMP / scaler state（若使用）
RNG states: Python / NumPy / PyTorch CPU / all CUDA ranks
dataloader / sampler state
global_step
cumulative active FLOPs
cumulative valid bytes / patches
router state
entropy shuffling state（若为 shuffled-entropy ablation）
run_manifest.json
checkpoint_manifest.json
```

不允许只保存 model weights 后将其称作可恢复 checkpoint。

### 17.7 分布式训练约束

对 FSDP / DCP / EP 分布式 checkpoint：

```text
- 所有 rank 必须成功写完各自 shard；
- rank 0 只能在收到所有 rank success signal 后写 complete marker；
- rank 0 只能在 complete marker 存在且 verifier 通过后更新 LATEST.json；
- 任意 rank 失败则本次 checkpoint 视为失败；
- 失败的 staging checkpoint 不能进入 retention list；
- barrier 超时必须保留上一个 LATEST，不得继续更新。
```

建议完整状态机：

```text
WRITING
  -> ALL_RANKS_WRITTEN
  -> VERIFIED
  -> COMMITTED
  -> LATEST_UPDATED
  -> RETENTION_PRUNED
```

只有 `LATEST_UPDATED` 后的 checkpoint 才视为正式保存成功。

### 17.8 训练结束与异常处理

必须额外支持：

```text
- SIGTERM / SIGINT：在安全退出前尝试保存 emergency checkpoint；
- CUDA OOM：不覆盖或删除现有 latest；记录失败原因；
- checkpoint exception：不更新 LATEST；训练可选择安全停止；
- periodic checkpoint save failure：默认停止训练，避免继续训练到无法恢复的状态。
```

对于无人值守正式训练：

```text
checkpoint failure -> abort_safe
```

优先保证已有可恢复状态，不允许“继续跑但不再保存”。

### 17.9 运行命名与目录结构

每个 run 的 checkpoint 目录必须如下：

```text
<output_dir>/
  run_manifest.json
  metrics.jsonl
  checkpoints/
    LATEST.json
    0000001000/
      checkpoint_manifest.json
      ...
    0000002000/
      checkpoint_manifest.json
      ...
    .staging/
    failed/
```

其中：

```text
- checkpoint 目录名始终为 10 位 global_step；
- 不允许混用 epoch、local rank step 或 micro-step；
- 一个 output_dir 只能对应一个 run identity；
- resume 到新 output_dir 时，必须在 manifest 中记录 parent run 与 source checkpoint。
```

### 17.10 Codex 验收测试

在正式训练前，必须通过以下自动化测试：

#### Test CKPT-1：1k cadence

使用小模型跑到 3,050 global steps，断言：

```text
0000001000 exists and is verified
0000002000 exists and is verified
0000003000 exists and is verified
LATEST points to 0000003000
```

在 retention 执行后，断言：

```text
only 0000002000 and 0000003000 remain complete
```

#### Test CKPT-2：写入中断保护

模拟 step 2,000 checkpoint 写入中断：

```text
LATEST still points to 0000001000
0000001000 remains loadable
0000002000 is absent from complete list
```

#### Test CKPT-3：损坏 latest fallback

模拟 `LATEST.json` 指向的 3,000 checkpoint 缺失一个 shard：

```text
resume automatically falls back to 0000002000
fallback event is recorded in log and metadata
```

#### Test CKPT-4：跨 rank 完整性

在多 rank tiny run 中模拟一个 rank 保存失败：

```text
LATEST is not advanced
no incomplete directory is marked complete
previous latest remains recoverable
```

#### Test CKPT-5：磁盘空间 guard

通过 mock 或临时空间阈值模拟磁盘不足：

```text
older-than-fallback checkpoints are pruned first
LATEST and fallback are never deleted
if still insufficient, training exits with abort_safe
```

### 17.11 该策略的正式实验含义

正式主实验每个 1k checkpoint 都应保存以下可比较内容：

```text
same fixed held-out BPB
cumulative active training FLOPs
cumulative training bytes
router / expert profiles
run and checkpoint manifests
```

因此，即使训练在 50k 之前异常终止，也能够：

```text
- 从最新 verified checkpoint 恢复；
- 或使用最新 checkpoint 进入正式 compute-quality 曲线；
- 不丢失最近 1k steps 以上的训练进度；
- 不因磁盘清理误删唯一可恢复状态。
```
