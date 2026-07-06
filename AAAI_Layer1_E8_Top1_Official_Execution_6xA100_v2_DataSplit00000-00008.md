# AAAI 第一层正式实验执行规范
## E8 Full-Width Top-1 Entropy-Guided PatchMoE（6×A100：GPU 0–5）

**项目：** BP-MoE  
**训练根目录：** `/data1/pengfeigao/BP-MoE`  
**执行资源：** 单节点 6×A100，使用 GPU `0,1,2,3,4,5`  
**本文档范围：** 仅覆盖论文第一层、必须完成的主结果实验。  
**本文档不覆盖：** Top-2 扩展、paired Top-2 结构控制、byte-load balance、EMA bias、跨域数据混合、多数据集训练、系统优化。

---

## 1. 第一层的唯一论文问题

本层只回答：

> 在 tokenizer-free 的动态 byte-patch language model 中，真实 patch entropy 是否能作为有效的稀疏 MoE routing signal，使 **8 个自由专家** 在匹配 active training FLOPs 的条件下，获得优于 dense BLT 和 hidden-only MoE 的 held-out BPB？

本层最终希望获得的可投稿结论是：

> Entropy-guided routing enables eight independently initialized experts to self-organize across low-, medium-, and high-entropy patch regimes, improving held-out byte-level modeling quality at matched active training compute.

---

## 2. 研究边界：本层不解决什么

以下内容必须记录，但不是本层的优化目标，也不能作为主结果失败标准：

```text
- 原始 byte mass 在专家之间是否均匀；
- expert/pair 的 byte-load balance；
- expert-parallel all-to-all 优化；
- wall-clock throughput 优于 dense；
- GPU 利用率最优；
- Top-2 compositional routing；
- 多语言 / code / math 混合训练；
- 训练数据按 domain 顺序切换。
```

必须遵守的解释原则：

```text
byte coverage != valid patch-unit work != runtime bottleneck
```

因此：

```text
不以 max_byte_fraction 作为 stop criterion；
不在这一层继续调 byte auxiliary loss 或 EMA byte bias；
不把 entropy–length specialist 的非均匀 byte coverage 直接写为“系统负载失败”。
```

---

## 3. 正式主架构：E8 Full-Width Free Top-1

### 3.1 架构定义

所有 MoE 主实验使用：

```yaml
model:
  moe_num_experts: 8
  moe_top_k: 1
  moe_ffn_dim_multiplier: 1.0
  moe_layer_frequency: 2
  paired_routing: false
  expert_init: dense_copy
```

解释：

```text
8 个 expert：彼此独立，不固定两两成 pair；
full-width：每个 expert FFN 的中间宽度等于原 dense FFN；
Top-1：每一个有效 patch 在每个 MoE layer 只激活一个 expert；
moe_layer_frequency=2：仅每隔一个 global Transformer layer 替换为 MoE；
dense_copy：每个 full-width expert 初始复制该层 dense FFN。
```

### 3.2 为什么这能与 dense 做 active-FLOPs 对齐

在每一个 MoE layer：

```text
Dense:
  1 × full-width dense FFN

E8 full-width Top-1:
  1 × selected full-width expert FFN
```

所以每个有效 patch 在 MoE FFN 上的活跃计算量近似相同：

\[
C_{\mathrm{FFN}}^{\mathrm{E8\text{-}Top1}}
\approx
C_{\mathrm{FFN}}^{\mathrm{Dense}}.
\]

8 个 experts 增加的是**总参数容量**，而不是每个 patch 的激活 FFN 计算量。

`moe_layer_frequency=2` 的目的只是控制总参数量、optimizer state 和显存；它不改变“MoE layer 内 Top-1 full-width expert 与 dense FFN 同活跃计算量”的事实。

### 3.3 Dense-to-MoE function preservation

每个 MoE layer 的 dense FFN 记为：

\[
F_{\mathrm{dense}}(\mathbf h).
\]

初始化时，对 \(e\in\{1,\ldots,8\}\)：

\[
E_e(\mathbf h)\leftarrow F_{\mathrm{dense}}(\mathbf h).
\]

Top-1 routing 选择 \(e^\star\)：

\[
y_{\mathrm{MoE}}(\mathbf h)
=
E_{e^\star}(\mathbf h)
=
F_{\mathrm{dense}}(\mathbf h).
\]

因此，在 step 0，任意 router assignment 都不应破坏 dense BLT 输出。

---

## 4. 强制工程 Gate：通过后才启动正式训练

这不是探索实验，而是新主架构的正确性认证。

### G0.1 Full-width expert copy test

对同一批真实 patch hidden states：

```text
dense FFN output
vs.
每一个 expert 的 FFN output
```

验收：

```text
fp32 reference relative L2 error <= 1e-4
max absolute error only reflects expected numerical precision
```

### G0.2 Full-model Top-1 equivalence test

在 router mode 为 `hidden`、`entropy`、`hidden_entropy` 的情况下，用同一小批真实输入比较：

```text
dense BLT full-model output
vs.
E8 Top-1 MoE full-model output
```

验收：

```text
small fixed shard BPB delta is numerically negligible
logits cosine similarity is effectively 1.0
```

### G0.3 Valid-patch-only dispatch test

当：

```text
patch_length <= 0
```

该位置必须：

```text
- 不进入 router；
- 不进入 expert dispatch；
- 不产生 router gradient；
- 不计入 valid patch utilization；
- 不计入 entropy / length specialization statistics。
```

验收：

```text
invalid_patch_dispatch_count = 0
```

### G0.4 Six-GPU memory smoke test

使用与正式训练完全相同的：

```text
world_size = 6
moe_ep_size = 2
precision / FSDP / optimizer
global batch and sequence length
```

运行至少 200 optimizer steps。

验收：

```text
no CUDA OOM
no FSDP / DCP shard error
no NCCL timeout
peak reserved memory leaves >= 10% headroom on every GPU
```

若显存不足：

```text
优先修正 FSDP / optimizer-state sharding；
不得为了绕过 OOM 把 expert 数从 8 改成 4；
不得把 full-width 改回 half-width；
不得改变 Top-1。
```

---

## 5. 6×A100 运行拓扑与固定环境

### 5.1 GPU 分配

所有第一层正式训练使用：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
```

启动方式：

```bash
$PROJECT_ROOT/.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=6 \
  -m bytelatent.train \
  ...
```

### 5.2 固定拓扑

使用当前已验证的 6-GPU训练栈；默认建议：

```yaml
distributed:
  world_size: 6
  moe_ep_size: 2
  data_parallel_groups: 3
```

即：

```text
6 ranks = 3 个 data-parallel replicas × 每组 2 个 expert-parallel ranks
```

不得在主表中混合：

```text
4 GPU、6 GPU、不同 EP size、不同 FSDP mode
```

所有 Dense / H8 / E8 / HE8 使用完全相同的：

```text
GPU 集合
world size
EP size
FSDP strategy
precision
global batch size
gradient accumulation
optimizer
learning rate schedule
```

### 5.3 环境记录

每个 run 启动前记录：

```bash
nvidia-smi
nvidia-smi topo -m
git rev-parse HEAD
```

将输出写入：

```text
<run_dir>/environment.txt
<run_dir>/run_manifest.json
```

不得默认设置 `NCCL_P2P_DISABLE=1` 或 `NCCL_IB_DISABLE=1`，除非当前六卡稳定 launcher 已经必须使用该设置；若使用，必须写入 manifest。

---

## 6. 数据与预处理固定协议

### 6.1 Series A 训练数据：使用已准备的 9 个 FineWeb-Edu chunks

正式第一层不再只使用历史筛选阶段的 `chunk 00000 + 00001`。

已准备的 `chunk 00000–00008` 足以支持当前 6×A100、50k-equivalent active-compute 的正式实验；此阶段**不需要为了启动训练再新增其他数据集或下载更多 FineWeb-Edu chunks**。

为避免训练/验证泄漏，并保留一个不参与模型选择的最终测试集，固定划分为：

```text
train:
  FineWeb-Edu-10BT chunk 00000–00006

dev / heldout_natural_core:
  FineWeb-Edu-10BT chunk 00007

locked final test / heldout_natural_extended:
  FineWeb-Edu-10BT chunk 00008
```

用途：

```text
chunk 00000–00006:
  所有 Dense / H8 / E8 / HE8 的正式 CPT 训练流

chunk 00007:
  每个 1k checkpoint、C_low/C_mid/C_high budget checkpoint 的固定开发集评估；
  用于生成 compute-quality 曲线、诊断训练稳定性和选择论文主方法

chunk 00008:
  不参与超参数选择、不参与中途调参；
  仅在每个方法的 C_high checkpoint 与最终选定 checkpoint 上运行最终报告评估
```

所有 run 必须一致：

```text
same train/dev/test manifests
same sample order seed
same interleaved shard sampler
same sharding behavior
same entropy-preprocessed Arrow files
same entropy model checkpoint
same entropy patching threshold
same max sequence length
same tokenizer / byte preprocessing configuration
```

训练采样要求：

```text
- 不允许先完整训练 chunk 00000、再 00001、再依次切换；
- train shards 00000–00006 必须从 step 0 开始交错采样；
- 每个 shard 的采样权重按可用训练 bytes 近似成比例；
- 使用固定 global data-order seed；
- 所有方法、所有 seed 在对应的 data-order seed 下看到相同训练流；
- 不允许因某个方法结果更好而改变数据文件、shuffle 逻辑或 shard 权重。
```

禁止：

```text
- 训练过程中切换到 math / code / multilingual；
- 50k 之后硬切换新数据集；
- 不同方法使用不同训练 chunks；
- 把 chunk 00007 或 00008 混回训练；
- 用 chunk 00008 频繁评估或据此调参；
- 为 entropy router 单独增加更优数据。
```

为什么不是把 00000–00008 全部用作训练：

```text
- 需要严格、从未参与训练的自然分布 held-out；
- chunk 00007 可承担开发/曲线评估；
- chunk 00008 可作为锁定的最终测试集；
- 这比只使用 00000+00001 更有代表性，也比“所有 chunk 都训练、无独立测试”更适合 AAAI 证据链。
```

若一次完整 Series A 的实际累计训练 bytes 明显小于 chunk 00000–00006 的唯一可用 bytes，则不需要更多训练数据；
若未来 C_high 之后仍要进行远长于当前预算的 CPT，或训练 bytes 接近/超过当前 train split 的唯一 bytes，才扩展到新的未使用 FineWeb-Edu chunks，并重新冻结新的 train/dev/test manifest。

### 6.2 固定 held-out

建立并冻结：

```text
heldout_natural_core
```

要求：

```text
- 与训练数据严格 disjoint；
- 由固定文件 manifest 定义；
- 评估 bytes 数量固定；
- 所有 checkpoint 使用相同 patching configuration；
- 所有方法使用完全相同的 BPB 实现。
```

每次正式评估输出：

```text
heldout_bpb
heldout_loss
evaluated_valid_bytes
evaluated_valid_patches
eval_manifest_sha256
```

训练 tail BPB 只用于诊断，不能进入主结果表。

---

## 7. Router 设计：不预设专家身份，但让 entropy 有足够表达能力

### 7.1 禁止的 router 形式

不要使用：

```text
一个 nonnegative scalar entropy
+ 一个 bias=False 线性层
```

因为若：

\[
z_e = a_e H,\quad H>0,
\]

则 expert 排序可能与 entropy 数值无关，只由 \(a_e\) 的大小决定，无法自然形成 low/mid/high entropy 分段专家。

### 7.2 正式 entropy branch

使用全局标准化后的 entropy：

\[
\hat H =
\operatorname{clip}
\left(
\frac{H-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}},
-4,4
\right).
\]

其中：

```text
μ_train 与 σ_train：
只由 Series A 训练集的有效 patches 离线统计；
写入 entropy_stats.json；
所有 seed、所有主方法共享；
不得从 held-out 数据估计。
```

Entropy branch 采用小型可学习 MLP：

```python
EntropyRouter(
    Linear(1, 32, bias=True),
    SiLU(),
    Linear(32, 8, bias=True),
)
```

路由模式：

```text
hidden:
  z = hidden_router(h)

entropy:
  z = entropy_router(H_hat)

hidden_entropy:
  z = hidden_router(h) + entropy_router(H_hat)
```

要求：

```text
- 不使用人工 entropy bands；
- 不指定“expert 0 对应低熵”或其他固定 expert role；
- 所有 8 个 experts 初始化为相同 dense FFN；
- entropy specialization 必须由 CPT 中 router/expert co-adaptation 学得。
```

### 7.3 早期对称性破除与存活约束

因为 8 个 experts 在 step 0 完全相同，必须使用固定、轻量的训练稳定机制，避免专家永久没有梯度。

正式固定设置：

```yaml
router:
  initial_jitter: 0.01
  jitter_warmup_steps: 1000
  jitter_final: 0.0

moe:
  balance_cost: patch
  balance_loss_weight: 0.001
```

解释：

```text
- jitter 只在前 1k steps 用于打破完全对称；
- balance 只按有效 patch count，而不是 byte mass；
- balance 项只用于避免 dead expert；
- 所有 MoE variants 使用完全相同的固定值；
- 不把这个轻量 regularizer 作为论文方法贡献；
- 不扫描 byte balance、EMA bias 或其他强控制策略。
```

若 G0/G1 显示 `0.001` 仍导致永久 dead experts，可唯一允许的稳定性调整为：

```text
0.001 -> 0.003
```

该调整必须对 H8/E8/HE8 全部同步应用，并在 run manifest 标记；不可按某个方法单独选择。

---

## 8. 第一层正式 run 矩阵

### 8.1 主结果 run

| Run ID | Architecture | Experts | Top-k | Router | 角色 |
|---|---|---:|---:|---|---|
| `A-Dense` | Dense BLT | 0 | — | — | dense 质量-计算基线 |
| `A-H8` | E8 full-width MoE | 8 | 1 | hidden-only | sparse hidden-only baseline |
| `A-E8` | E8 full-width MoE | 8 | 1 | entropy-only | entropy 的独立信息量 |
| `A-HE8` | E8 full-width MoE | 8 | 1 | hidden + entropy | entropy 是否补充 hidden context |

所有 MoE run：

```yaml
moe_num_experts: 8
moe_top_k: 1
moe_ffn_dim_multiplier: 1.0
moe_layer_frequency: 2
expert_init: dense_copy
paired_routing: false
balance_cost: patch
balance_loss_weight: 0.001
```

### 8.2 种子

正式种子固定为：

```text
42
43
44
```

即：

```text
4 methods × 3 seeds = 12 个正式 C_high 训练 run
```

每一个 seed 下的四个方法必须共享：

```text
same data-order seed
same initial dense checkpoint
same training input stream
same preprocessing
same evaluation manifest
```

### 8.3 不在本层运行的实验

以下不属于本文档：

```text
length-only
hidden+length
shuffled entropy
paired half-width Top-2
full-width Top-2
E4 expert count control
math / code / multilingual data
byte balance / EMA bias
```

这些将在主结果完成后作为第二层或后续 ablation 单独执行。

---

## 9. Active Training FLOPs：正式对齐协议

### 9.1 论文主图

主图为：

\[
\text{Held-out BPB}
\quad \text{vs.} \quad
\text{Cumulative Active Training FLOPs}.
\]

不能用：

```text
- 相同步数；
- stdout 中单一 FLOPs 数字；
- 相同 wall-clock 时间；
```

替代统一 active FLOPs accounting。

### 9.2 Counter 必须包含

每个 training step 必须累计：

```text
1. local byte encoder / decoder；
2. global Transformer attention，按实际 valid patch count；
3. dense FFN 或所选 Top-1 expert FFN；
4. router projection / entropy MLP；
5. embedding / output projection；
6. 统一定义下的 forward + backward cost。
```

每个 run 写入：

```text
cumulative_active_flops
cumulative_valid_bytes
cumulative_valid_patches
```

### 9.3 三个预注册 compute budget

首先对 `A-Dense` 做 200-step profiling，定义：

\[
\bar F_{\mathrm{Dense}}^{\mathrm{step}}
=
\text{mean active train FLOPs per Dense step}.
\]

然后固定：

```text
C_low  = 5,000  × F_dense_step
C_mid  = 10,000 × F_dense_step
C_high = 50,000 × F_dense_step
```

每个 run 都训练至：

```text
cumulative_active_flops >= C_high
```

Dense 的 nominal target 为 50k steps；MoE 因 router 的小额额外计算，可能在略少于 50k step 时到达同一个 `C_high`。

要求：

```text
- 不根据 BPB 选择训练长度；
- 每个 run 使用同一个 C_low/C_mid/C_high；
- 记录实际 steps、实际 bytes、实际 patches；
- 以 nearest compute-boundary checkpoint 做 C_low/C_mid/C_high 比较。
```

### 9.4 特殊 budget checkpoint

除每 1k regular checkpoint 外，当首次跨越：

```text
C_low
C_mid
C_high
```

必须额外保存一个 budget checkpoint，即使其 step 不是 1k 的整数倍。

budget checkpoint 在 held-out eval 完成前不能被 retention 删除。

---

## 10. Checkpoint 与无人值守容错策略

### 10.1 保存频率与保留

正式每个 run：

```yaml
checkpoint:
  every_global_steps: 1000
  save_budget_boundaries: true
  keep_last_complete: 2
  atomic_commit: true
  verify_after_save: true
  resume_default: checkpoints/LATEST.json
```

含义：

```text
每 1k global optimizer steps 保存一次；
始终保留最近两个已验证、可恢复的 full checkpoint；
新 checkpoint 未完整写入和校验前，旧 latest 不得删除；
新 checkpoint 验证成功后才更新 LATEST.json；
最新 checkpoint 损坏时自动回退到次新 checkpoint。
```

### 10.2 必须保存的状态

```text
model
optimizer
scheduler
AMP/scaler state（如适用）
all RNG states
dataloader/sampler state
router state
entropy-shuffle state（本层未用，保留接口）
global step
cumulative active FLOPs
cumulative valid bytes/patches
run_manifest
checkpoint_manifest
```

### 10.3 磁盘安全规则

checkpoint 保存前：

```text
1. 预估新 checkpoint 完整大小；
2. 预留 1.25 × checkpoint_size + safety margin；
3. 仅删除比两个最新 complete checkpoint 更旧的已验证目录；
4. 不得删除 current latest；
5. 不得删除唯一 fallback；
6. 空间仍不足时 abort_safe。
```

checkpoint failure：

```text
不更新 LATEST；
保留旧 checkpoint；
停止训练而非继续无恢复点运行。
```

---

## 11. 启动顺序：六张 A100 上的正式执行

### 11.1 目录变量

```bash
export PROJECT_ROOT=/data1/pengfeigao/BP-MoE
export RUN_ROOT=$PROJECT_ROOT/experiments/aaai_layer1_e8_top1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
cd "$PROJECT_ROOT"
```

### 11.2 先执行一次 Gate

```bash
$PROJECT_ROOT/.venv/bin/python -m pytest \
  bytelatent/test_patch_moe.py \
  -q
```

然后运行：

```text
G0 full-width dense-copy equivalence
G0 valid-patch-only dispatch
G0 six-GPU 200-step smoke
200-step active-FLOPs profiling
fixed held-out evaluator smoke
```

Gate 全部成功后，冻结：

```text
git commit hash
model config
data manifest hash
entropy stats hash
C_low/C_mid/C_high
```

### 11.3 单个正式 run 的 launcher 模板

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

$PROJECT_ROOT/.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=6 \
  -m bytelatent.train \
  --config "$PROJECT_ROOT/configs/aaai_layer1/aaai_e8_entropy.yaml" \
  --seed 42 \
  --output-dir "$RUN_ROOT/A-E8/seed42" \
  --data-manifest "$PROJECT_ROOT/manifests/aaai_series_a_train.json" \
  --heldout-manifest "$PROJECT_ROOT/manifests/heldout_natural_core.json" \
  --compute-budget-json "$RUN_ROOT/compute_budgets.json" \
  --checkpoint-every 1000 \
  --keep-last-complete 2 \
  --checkpoint-verify true \
  --save-budget-boundaries true
```

对应 config：

```text
aaai_dense.yaml
aaai_e8_hidden.yaml
aaai_e8_entropy.yaml
aaai_e8_hidden_entropy.yaml
```

### 11.4 运行队列

在同一组 GPU `0–5` 上，一次只跑一个 six-GPU distributed run。

执行顺序：

```text
Seed 42:
  A-Dense
  A-H8
  A-E8
  A-HE8

Seed 43:
  A-Dense
  A-H8
  A-E8
  A-HE8

Seed 44:
  A-Dense
  A-H8
  A-E8
  A-HE8
```

每个 run 完成后立刻：

```text
1. 校验 LATEST checkpoint 可恢复；
2. 对 C_low / C_mid / C_high budget checkpoint 执行 held-out evaluation；
3. 写入 results.json；
4. 不根据当前 BPB 调整其他 run 的超参数；
5. 再启动下一个 run。
```

---

## 12. 正式评估与主结果提取

### 12.1 每个预算点的强制输出

每个 run 在 C_low、C_mid、C_high 都必须产生：

```json
{
  "run_id": "...",
  "seed": 42,
  "budget": "C_mid",
  "global_step": 9997,
  "cumulative_active_flops": "...",
  "cumulative_valid_bytes": "...",
  "cumulative_valid_patches": "...",
  "heldout_bpb": "...",
  "heldout_loss": "...",
  "checkpoint_path": "...",
  "checkpoint_manifest_sha256": "..."
}
```

### 12.2 主比较

主表：

| Method | Router | C_low BPB | C_mid BPB | C_high BPB | Active FLOPs | Train bytes |
|---|---|---:|---:|---:|---:|---:|
| Dense | — | | | | | |
| H8 | hidden | | | | | |
| E8 | entropy | | | | | |
| HE8 | hidden + entropy | | | | | |

每个数值报告：

```text
mean ± std over seeds {42,43,44}
```

主图：

```text
held-out BPB vs. cumulative active training FLOPs
```

### 12.3 第一层成功条件

本层通过的最低条件：

```text
1. E8 或 HE8 在 C_mid 与 C_high 上优于 H8；
2. E8 或 HE8 在 C_high 上优于 Dense；
3. 3 个 seed 的改善方向一致；
4. 结果以 fixed held-out BPB 为准；
5. E8/H8/HE8 的训练与评估 protocol 完全一致；
6. G0 function-preserving tests 已归档。
```

当且仅当上述条件成立，论文可以主张：

> Entropy-guided tokenizer-free PatchMoE improves held-out BPB over dense BLT and hidden-only sparse routing at matched active training compute.

---

## 13. 8-expert entropy–length specialization 分析

本节是第一层的重要分析产物。目标不是证明专家均匀，而是证明 8 个自由 experts 出现可解释、可复现、并且具有功能意义的分工。

### 13.1 Routing specialization：必须做

在 C_high fixed held-out 上，对有效 patches 统计：

\[
P(e \mid H\text{-bin}, \ell\text{-bin}).
\]

使用训练集预先冻结的分箱边界：

```text
entropy bins:
  low / medium / high
  由 train valid-patch entropy 的三分位数确定

length bins:
  short / medium / long
  由 train valid-patch length 的三分位数确定
```

输出：

```text
expert_entropy_heatmap.png      # 8 × 3
expert_length_heatmap.png       # 8 × 3
expert_entropy_length_heatmap.png # 8 × 3 × 3
expert_profile.csv
```

每个 expert 至少报告：

```text
valid-patch share
byte coverage
mean entropy
mean patch length
entropy-bin routing share
length-bin routing share
```

目标观察模式：

```text
部分 experts:
  偏向 low entropy + long patches

部分 experts:
  偏向 medium entropy / medium length patches

部分 experts:
  偏向 high entropy + short patches
```

不得要求：

```text
每个 expert 都占相同 patch count；
每个 expert 都占相同 byte mass；
expert ID 在不同 seed 中完全一致。
```

### 13.2 Cross-seed stability：必须做

expert index 在不同 seed 中可置换，因此禁止直接比较：

```text
seed42 expert0 vs seed43 expert0
```

正确协议：

```text
1. 对每个 expert 生成 3×3 entropy-length routing profile；
2. 使用 Hungarian matching 在不同 seed 间匹配 experts；
3. 计算 matched profile correlation / Jensen-Shannon similarity；
4. 报告 mean matched similarity；
5. 展示至少一组 seed-to-seed matched heatmap。
```

### 13.3 Functional specialization：C_high 后执行

只有 “router 常送给某个 expert” 还不能证明 “该 expert 更擅长”。

必须在最终 checkpoint 上运行 forced-expert intervention：

```text
1. 从 held-out 中抽取：
   low-entropy long-patch bucket
   medium-entropy medium-patch bucket
   high-entropy short-patch bucket

2. 保持全部其他层 native routing；

3. 对选定的一个 MoE layer：
   针对该 bucket 的有效 patches，
   分别强制 route 到 expert 0...7；

4. 计算该 bucket 对应 byte span 的 NLL / BPB change；

5. 在多个 representative MoE layers 上重复并平均。
```

定义：

\[
\Delta_{e,\mathcal B}
=
\operatorname{BPB}(
\text{force expert } e \text{ on bucket }\mathcal B
)
-
\operatorname{BPB}(
\text{native routing}
).
\]

若某 expert 对 bucket \(\mathcal B\) 的 \(\Delta\) 最小，且与 routing profile 的主要偏好一致，则可以说：

> The expert is functionally better suited to that entropy–length regime.

这项干预只在：

```text
E8 / HE8 主方法的 C_high checkpoint
```

上执行；建议至少覆盖 seeds 42 与 43。

---

## 14. 透明但非主张的系统日志

每个 run 仍需记录：

```text
wps
iteration time
allocated memory
reserved memory
router entropy
expert valid-patch share
expert byte fraction
dead-expert count
```

论文中可在小表中透明呈现：

> Active training compute is matched. Wall-clock throughput and expert utilization are reported for transparency, but efficient load-balanced sparse execution is outside the scope of this work.

不得写：

```text
entropy routing is faster；
entropy routing has better system utilization；
byte-uniform expert balance is achieved；
```

除非未来有独立、完整的系统实验支持。

---

## 15. Codex 实施清单

### Task L1-1：E8 full-width free Top-1

- 支持 8 个 full-width experts；
- `top_k=1`；
- `moe_ffn_dim_multiplier=1.0`；
- 不存在 fixed pair constraint；
- 每隔一个 global Transformer layer 使用 MoE；
- 在每个 MoE layer 复制 dense `w1/w2/w3` 到全部 8 个 experts。

### Task L1-2：Router modes

实现严格模式：

```text
hidden
entropy
hidden_entropy
```

要求：

- entropy-only 不调用 hidden router；
- entropy branch 使用训练集全局 z-score entropy；
- entropy branch 使用带 bias 的 MLP；
- 不存在人工 entropy band -> expert ID 映射；
- invalid patches 不进入任何 route path。

### Task L1-3：Training stability

- 前 1k step 的 router jitter schedule；
- 固定 patch-count balance loss；
- 记录 dead-expert count；
- 不实现或启用 byte balance / EMA byte bias。

### Task L1-4：Active FLOPs

- 实现所有方法共用的 active compute counter；
- 生成 `compute_budgets.json`；
- 支持在 C_low/C_mid/C_high 保存 budget checkpoints；
- 输出每个 checkpoint 的 FLOPs、bytes、patches。

### Task L1-5：Checkpoint safety

- 每 1k global steps full checkpoint；
- atomic staging -> verify -> commit；
- `LATEST.json`；
- retain latest + fallback；
- disk preflight；
- checkpoint save failure -> abort_safe。

### Task L1-6：Evaluation and analysis

输出：

```text
main_results.csv
all_seed_results.csv
bpb_vs_active_flops.csv
expert_profiles.csv
expert_entropy_heatmap.png
expert_length_heatmap.png
expert_entropy_length_heatmap.png
cross_seed_matching.json
forced_expert_intervention.csv
system_transparency.csv
```

---

## 16. 最终完成判据

以下项目全部完成后，第一层正式实验才算结束：

```text
[ ] E8 full-width Top-1 G0 tests passed
[ ] six-GPU 200-step smoke passed
[ ] compute budgets frozen
[ ] A-Dense: seeds 42, 43, 44 complete
[ ] A-H8: seeds 42, 43, 44 complete
[ ] A-E8: seeds 42, 43, 44 complete
[ ] A-HE8: seeds 42, 43, 44 complete
[ ] C_low/C_mid/C_high fixed held-out evaluation complete
[ ] compute-quality curves generated
[ ] entropy-length heatmaps generated
[ ] cross-seed expert matching complete
[ ] forced-expert intervention complete
[ ] checkpoint resume has been tested from LATEST
```

未完成上述条件前：

```text
不将训练 tail BPB 写成最终主结果；
不根据单个 seed 宣称 entropy specialization；
不将 Top-2、byte balance 或多数据集实验混入本层结论。
```
