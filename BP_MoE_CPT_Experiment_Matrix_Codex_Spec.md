# BLT-1B PatchMoE CPT: Byte-Calibrated Function-Preserving Upcycling
## Codex 实施与实验矩阵执行规范

**目标分支建议：** `feature/byte-calibrated-pair-upcycling`  
**基础仓库：** `E-skier/BP-MoE`  
**目标：** 在不牺牲 BLT-1B CPT 质量优势的前提下，消除 entropy-guided PatchMoE 的 expert-pair collapse，并形成可用于 AAAI 的公平、可复现实验闭环。

---

## 0. 本文档的结论与边界

### 0.1 已知实验事实

当前 entropy PhaseB 的结果表明：

- 50k 附近模型质量排序为 `entropy PhaseB > hidden-only MoE >> dense`。
- entropy PhaseB 的 BPB 为 0.833，优于 hidden-only MoE 的 0.923。
- entropy routing 与 expert selection 强相关，`entropy-selected expert corr = 0.909`。
- Top-2 几乎总是同一 pair，`top2_same_pair_fraction = 0.9997`。
- 有效负载严重坍缩到 experts 0/1：它们合计承担约 76.9% byte load；6/7 几乎空闲。
- 当前 `balance_cost=byte`，但 `balance_loss_weight=0.0`，因此 byte balance 只被记录，未参与训练。

因此，本任务不是重新证明 entropy 有用，而是解决：

> 在 dense BLT-1B upcycling 到 paired PatchMoE 时，如何保留 dense 功能、保留 entropy specialization，同时避免早期 entropy-prior 导致的 pair-level load collapse。

### 0.2 本任务明确不做的事情

- 不重置整个 expert FFN。
- 不对 dense FFN 注入大幅随机噪声。
- 不删除 entropy feature。
- 不把 8 个 half-width experts 强行当作 8 个独立路由单元做均匀化。
- 不在本阶段接入 Megatron / DeepEP / 新的 EP dispatcher。
- 不直接跑新的 50k CPT，除非 5k/10k screening 达到下文通过阈值。

### 0.3 主方法名称

实现中统一采用：

```text
Byte-Calibrated Function-Preserving Pair Upcycling (BCFP-Upcycling)
```

其组成包括：

1. **Function-preserving paired expert initialization**：每个 macro-pair 重建 dense FFN。
2. **Byte-calibrated soft entropy prior**：用 byte mass 校准 entropy routing prior，而不是硬 entropy band。
3. **Valid-patch-only dispatch**：padding / invalid patches 完全不进入 MoE router 与 expert execution。
4. **Pair-level utilization control**：比较小权重的 byte auxiliary balance 或 loss-free EMA bias。
5. **Late hidden residual release**：先让 entropy prior 稳定，再允许 hidden router 进行细粒度修正。

---

# 1. 术语与不变量

## 1.1 Expert 与 Pair

当前模型：

```text
num_experts = 8
top_k = 2
expert_ffn_dim_multiplier = 0.5
```

定义四个 macro-pairs：

```text
pair 0 = experts (0, 1)
pair 1 = experts (2, 3)
pair 2 = experts (4, 5)
pair 3 = experts (6, 7)
```

在 BCFP 模式下，一个有效 patch 的两个 active experts 必须来自同一 pair：

```text
Top-2(i) = {2g, 2g+1},  g ∈ {0,1,2,3}
```

这不是新的假设；当前 entropy PhaseB 的实测 `top2_same_pair_fraction≈0.9997` 已经说明现有模型实际上处于该工作模式。新实现只是将其显式化、可控化、可测试化。

## 1.2 Dense FFN 与 paired partition

对 dense SwiGLU FFN：

\[
F(h) = W_2[\operatorname{SiLU}(W_1h)\odot W_3h].
\]

沿 intermediate dimension 将通道划分为两个互补集合 \(A\) 和 \(B\)：

\[
F(h) = F_A(h) + F_B(h).
\]

对每一个 pair \(g\)，初始化为：

```text
expert 2g     ← dense channel subset A
expert 2g + 1 ← dense channel subset B
```

每一个 pair 都必须复制同一组 A/B partition；不能让 pair 0、pair 1、pair 2、pair 3 分别继承 dense FFN 的不同局部片段。

因为当前 implementation 对 Top-2 gate weights 做归一化，pair 内 logits 相同会令两个 gate weights 各为约 0.5。因此要在 warmstart 时保证：

```text
expert output scale = 2.0
```

推荐实现方式：对每个 half-width expert 的 `W2` 权重乘以 2。于是：

\[
0.5(2F_A(h)) + 0.5(2F_B(h)) = F(h).
\]

若实际 combine rule 与上述不同，必须以 t=0 numerical equivalence test 的结果为准，不允许凭经验修改 scale。

## 1.3 Load 的唯一主口径

对于有效 patch \(i\)，长度为 \(\ell_i\)，pair assignment 为 \(g_i\)。

pair byte share 定义为：

\[
q_g^{byte} =
\frac{\sum_i \ell_i \mathbf{1}[g_i=g]}
{\sum_i \ell_i}.
\]

所有训练决策、通过阈值和报告主指标以 **pair-level byte share** 为准；expert-level fraction 只作附属诊断。

---

# 2. 代码改造范围

## 2.1 预期文件

| 文件 | 责任 | 必需修改 |
|---|---|---|
| `bytelatent/base_transformer.py` | MoE routing、dispatch、balance、metrics | 主要实现位置 |
| `bytelatent/test_patch_moe.py` | 单元测试 | 扩展为强制验收测试 |
| `scripts/prepare_blt1b_patchmoe_warmstart.py` 或等价现有 warmstart 脚本 | dense→MoE checkpoint 转换 | 加 paired function-preserving export 与验证 manifest |
| `scripts/calibrate_patchmoe_router.py` | 新建 | 从 calibration data 计算 byte-calibrated entropy prior |
| `scripts/analyze_patchmoe_cpt.py` | 新建 | 汇总 5k/10k/50k 的 quality、load、specialization 与系统指标 |
| `bytelatent/configs/*.yaml` 或现有训练 config 路径 | 配置 | 新增 A/B/C/D 四个可复现实验配置 |
| 训练入口，例如 `bytelatent/train.py` | schedule hook | 支持 step-based router schedule 与 dynamic pair bias 更新 |

代码必须保持默认行为兼容：旧配置未开启 BCFP 时，当前 MoE 行为不应变化。

---

# 3. Phase 0：必须先完成的实现与单元测试

> **本阶段不跑 CPT。所有条目通过前，不允许开始 A–D 训练矩阵。**

## 3.1 修复 valid-patch-only dispatch

### 问题

当前 `SparseMoEFeedForward.forward()` 会：

1. 将 `[B, P, D]` flatten 为 `[B×P, D]`；
2. 对所有 positions 计算 router logits 和 Top-K；
3. 对所有 positions 进行 expert dispatch；
4. 仅在 balance / metrics 阶段以 `patch_lengths > 0` 过滤。

这意味着 padding 或 invalid patch 可能参与 router/expert 计算，污染梯度、专家统计和 EP 通信；这也可能解释 experts 6/7 的平均 patch length 为 0.410。

### 目标行为

令：

```python
valid_mask = flat_patch_lengths > 0
```

仅对 `valid_mask=True` 的 positions 执行：

```text
router logits
entropy prior
Top-K
expert dispatch
router / expert gradient
byte balance
pair metrics
```

invalid positions 必须满足：

```text
MoE output = 0
不参与 expert execution
不参与 all-to-all / dispatch accounting
不参与 balance loss
不参与 router gradient
```

### 推荐伪代码

```python
flat_output = torch.zeros_like(flat_x)
valid_mask = flat_patch_lengths > 0

if valid_mask.any():
    valid_x = flat_x[valid_mask]
    valid_lengths = flat_patch_lengths[valid_mask]
    valid_entropies = flat_patch_entropies[valid_mask]

    valid_output, routing_state = self._route_and_dispatch(
        valid_x,
        valid_lengths,
        valid_entropies,
    )
    flat_output[valid_mask] = valid_output
else:
    routing_state = self._empty_routing_state(...)

self._record_metrics(routing_state)
return flat_output.reshape_as(x)
```

不要通过将 invalid position 的 feature 强行置零来“掩盖”；它们必须在 router 前被移除。

## 3.2 实现 pair-level router mode

新增配置字段：

```yaml
model:
  moe_routing_granularity: pair  # choices: expert, pair
  moe_pair_size: 2
```

### Pair mode 的 routing 规则

- router 对 `num_pairs = num_experts / 2` 输出 logits；
- pair score expand 为两个相同 expert logits；
- Top-2 由于 pair members 同分，必须选中同一 pair 的两个 experts；
- 对于同一 pair 的两个 experts，normalized gate weights 必须是 `0.5, 0.5`（允许数值误差）；
- pair mode 仍然输出 8 个 half-width experts 的 per-expert metrics，同时额外输出 4 个 pair metrics。

建议新建内部 helper，而不是复制整个 `SparseMoEFeedForward`：

```python
class PairRouter(nn.Module):
    def forward(
        self,
        hidden: Tensor,
        patch_lengths: Tensor,
        patch_entropies: Tensor,
    ) -> Tensor:
        # returns [N_valid, num_pairs]
        ...
```

`SparseMoEFeedForward` 根据 `moe_routing_granularity` 选择 legacy expert router 或 pair router。

## 3.3 实现 byte-calibrated soft entropy prior

### 新配置字段

```yaml
model:
  moe_entropy_prior_mode: gaussian_pairs  # none | linear_legacy | gaussian_pairs
  moe_entropy_prior_scale: 0.5
  moe_entropy_prior_hidden_scale: 0.0
  moe_entropy_prior_calibration_path: /path/to/router_calibration.json
  moe_entropy_prior_trainable: false
```

### 先验定义

对 pair \(g\) 的 entropy center \(c_g\)、width \(\sigma_g\)、static bias \(b_g\)：

\[
z_g^{prior}(H)
= \alpha\left[-\frac{(H-c_g)^2}{2\sigma_g^2}\right]+b_g.
\]

其中：

- \(H\)：patch entropy；
- \(\alpha\)：`moe_entropy_prior_scale`；
- `centers`, `widths`, `static_bias` 必须作为 checkpoint state 保存；
- 一组 pair logits 再 expand 为两个相同 expert logits。

### Hidden residual

hidden router 不要删除，但在 B/C/D 中用缩放控制：

\[
z_g(h,H) = z_g^{prior}(H) + \gamma z_g^{hidden}(h) + b_g^{dynamic}.
\]

- `γ = moe_entropy_prior_hidden_scale`；
- initialization 时 `γ=0.0`；
- Phase D 才按 schedule ramp up；
- hidden residual weights 默认使用 zero initialization，避免一开始打破 pair 内同分关系。

## 3.4 新建 calibration script

新文件：

```text
scripts/calibrate_patchmoe_router.py
```

### 输入

- 与 CPT 完全相同的 preprocessing / entropy patcher；
- 一个固定 calibration split，建议 50k–200k sequence 或约 50M–200M bytes；
- `num_pairs=4`；
- `patch_lengths` 与 `patch_entropies`。

### 输出

```json
{
  "version": 1,
  "num_pairs": 4,
  "num_experts": 8,
  "pair_size": 2,
  "entropy_centers": [...],
  "entropy_widths": [...],
  "static_pair_bias": [...],
  "target_byte_share": [0.25, 0.25, 0.25, 0.25],
  "calibrated_byte_share": [...],
  "entropy_prior_scale": 0.5,
  "calibration_num_valid_patches": ...,
  "calibration_total_bytes": ...,
  "data_fingerprint": "..."
}
```

### 校准过程

1. 仅统计 valid patches：`patch_length > 0`。
2. 以 **byte-weighted quantiles** 计算 4 个初始 entropy centers；推荐 quantiles：`[0.125, 0.375, 0.625, 0.875]`。
3. 由相邻 centers 距离初始化 width；width 不得小于 `min_width`，避免过硬 band。
4. 使用 fixed centers/widths，迭代优化 `static_pair_bias`，使 pair byte share 尽可能接近 `[0.25]*4`。
5. 存储校准后 pair byte share 和最终 bias。

### 重要约束

- calibration 使用 byte share，不使用 patch count。
- calibration 不使用 hidden state。
- calibration 不能访问 held-out evaluation split。
- 必须保存 data fingerprint，避免后续训练时误用不同 patching threshold 对应的 prior。

## 3.5 实现 pair-level metrics

新增日志 keys：

```text
moe_pair_0_byte_fraction ... moe_pair_3_byte_fraction
moe_pair_0_unit_fraction ... moe_pair_3_unit_fraction
moe_pair_load_cv
moe_pair_max_byte_fraction
moe_pair_min_byte_fraction
moe_pair_dead_count
moe_pair_router_entropy
moe_top2_same_pair_fraction
moe_valid_patch_fraction
moe_invalid_positions_skipped
moe_pair_entropy_mean_0 ...
moe_pair_length_mean_0 ...
moe_entropy_pair_mi
moe_dynamic_pair_bias_0 ...
```

其中：

```text
pair_dead_count = count(pair_byte_fraction < 0.005)
```

`moe_entropy_pair_mi` 采用 entropy bins 与 pair assignment 的 mutual information；只对 valid patches 计算。

## 3.6 单元测试：必须新增

在 `bytelatent/test_patch_moe.py` 中新增或替换以下测试。

### T1：invalid patches 不进入 router / expert

构造：

```python
x_a = random([1, 4, D])
x_b = x_a.clone()
x_b[:, invalid_positions, :] = very_large_random_values
patch_lengths = [[4, 3, 0, 0]]
```

要求：

```text
valid positions 的 outputs 在 x_a / x_b 下完全一致或仅有浮点误差；
invalid positions 的 MoE output 恒为 0；
router metrics 的 routed_units = 2；
invalid positions 不改变 expert assignment / load；
invalid positions 不引入 expert/router gradient。
```

### T2：pair router 一定返回同 pair Top-2

```text
for random valid input:
  top2_same_pair_fraction == 1.0
  pair members 的 normalized top weights 均为约 0.5
```

### T3：dense → paired-MoE t=0 function preservation

使用 FP32 小模型和真实 BLT FFN 形状各测试一次。

固定 one pair route，比较 dense FFN 与 paired MoE：

```text
FP32 relative L2 error ≤ 1e-5
BF16 relative L2 error ≤ 5e-3
```

同时比较 one-step model logits 与 BPB：

```text
|BPB_dense - BPB_paired_moe| ≤ 0.005
```

阈值若因现有数值实现不合理，可放宽，但必须在 commit 中说明原因，且不能只报告隐藏层近似而不报告 output/logit/BPB 连续性。

### T4：calibration byte share

加载 calibration JSON 后，在 calibration split 上执行 no-grad router：

```text
max_abs(pair_byte_share - 0.25) ≤ 0.05
no pair_byte_share < 0.10
```

### T5：checkpoint round-trip

保存、加载后以下完全一致：

```text
entropy centers
entropy widths
static pair bias
dynamic pair bias
EMA load state
router schedule state
```

---

# 4. Warmstart 生成规范

## 4.1 统一起点

A/B/C/D 必须从**同一个原始 BLT-1B dense checkpoint**生成各自 warmstart，不允许分别从旧的 PhaseA/PhaseB checkpoint 开始。

推荐目录：

```text
initial_weights/
  blt_1b_dense_source/
  bcfp_pair_base/
  bcfp_pair_calibrated/
```

每个 warmstart 目录中保存：

```text
model checkpoint
warmstart_manifest.json
t0_equivalence_report.json
router_calibration.json (B/C/D only)
git commit hash
config snapshot
```

## 4.2 Warmstart manifest 格式

```json
{
  "source_dense_checkpoint": "...",
  "num_experts": 8,
  "pair_size": 2,
  "num_pairs": 4,
  "top_k": 2,
  "expert_ffn_dim_multiplier": 0.5,
  "expert_init": "paired_partition_replicated_per_pair",
  "pair_member_output_scale": 2.0,
  "routing_granularity": "pair",
  "hidden_router_init": "zero",
  "entropy_prior_mode": "gaussian_pairs",
  "t0_relative_l2_error": ...,
  "t0_bpb_delta": ...
}
```

## 4.3 绝对禁止

- 不允许 warmstart script 静默 fallback 到 random expert init。
- 不允许 `patch_feature_router` 沿用 PyTorch default initialization。
- 不允许自动从旧 PhaseA entropy-band checkpoint 继承 router，除非该 run 明确标为 legacy baseline A。
- 不允许 t=0 equivalence report 缺失仍启动 B/C/D CPT。

---

# 5. 实验协议：全部 run 的共同控制变量

## 5.1 数据与训练公平性

A/B/C/D 在 screening 阶段必须使用：

```text
same dense BLT-1B source checkpoint
same train data snapshot
same entropy-preprocessed Arrow shards
same patching threshold / entropy model
same seq_len / batch_size / grad accumulation
same optimizer / LR / scheduler / warmup
same number of GPUs = 8 A100 80GB
same FSDP / EP configuration
same random seed for first-pass comparison
same checkpoint interval
```

只有本文明确列出的 router / balance / schedule 参数允许改变。

## 5.2 分阶段预算

| 阶段 | 每个配置 | seed 数 | 目的 |
|---|---:|---:|---|
| Smoke | 100–300 steps | 1 | 验证 t=0、OOM、metrics、checkpoint |
| Screening | 5,000 steps | 1 | 排除明显 collapse / quality regression |
| Confirmation | 10,000 steps | 2 | 验证稳定性和重复性 |
| Finalist | 50,000 steps | 3 | 论文主表与 held-out 评估 |

筛选阶段未通过的 run 不进入下一阶段。

## 5.3 日志频率

```yaml
log_step: 50                 # screening
metrics_flush_every: 50
checkpoint.every: 1000       # screening
checkpoint.keep: 3
profile.every: 500
```

每个 log record 都必须包含当前 global step、seed、run ID、git commit、config hash。

---

# 6. A–D 实验矩阵

## 6.1 总览

| Run | 定位 | Expert init | Router prior | Balance / utilization control | Hidden residual | 预期作用 |
|---|---|---|---|---|---|---|
| A | Legacy reproduction | 当前 paired partition | 当前 entropy bands | none | 当前 | 复现高质量与 collapse |
| B | Initialization-only fix | function-preserving paired init | byte-calibrated soft prior | none | zero / disabled | 检验 calibration 是否已缓解 collapse |
| C | Auxiliary-balance candidate | 同 B | 同 B | byte aux schedule | zero / disabled | 检验小权重 byte balance 的质量-负载折中 |
| D | Loss-free final candidate | 同 B | 同 B | EMA byte-aware pair bias | delayed ramp | 检验不干扰 LM loss 的 utilization control |

### 关于 Run A

Run A 不是最终公平 baseline；它用于确认新代码没有改写旧结论。若 valid-patch-only dispatch 修复改变 A 的绝对结果，必须同时保留：

```text
A0 = strict legacy reproduction（旧 dispatch）
A1 = valid-patch-clean reproduction（新 dispatch，无 calibration）
```

后续 B/C/D 与 **A1** 比较，而非 A0。

---

## 6.2 Run A：Legacy / clean reproduction

### 目标

复现：

```text
quality good
entropy specialization strong
pair load collapse persists
```

### 配置

```yaml
model:
  moe_num_experts: 8
  moe_top_k: 2
  moe_pair_size: 2
  moe_routing_granularity: pair
  moe_ffn_dim_multiplier: 0.5

  moe_entropy_prior_mode: linear_legacy
  moe_entropy_prior_scale: 2.0       # 与旧 entropy-band run 一致；若原始值不同，使用原始值
  moe_entropy_prior_hidden_scale: 0.0

  moe_balance_cost: byte
  moe_balance_loss_weight: 0.0
  moe_pair_bias_mode: none
```

### 通过条件

本 run 不以平衡为目标；只需确认：

```text
1. quality trajectory 与旧 run 方向一致；
2. pair 0 dominant / pair 3 near-dead 的现象可复现；
3. 新 metrics 正常记录；
4. valid-mask 修复后无 invalid dispatch。
```

### 失败解释

若 A1 不再 collapse，优先结论是：**旧结果的 collapse 大量来自 invalid patch dispatch 或 metric contamination**。此时不要继续跑 B/C/D，先完成 root-cause 分析与 A0/A1 对比报告。

---

## 6.3 Run B：Byte-calibrated soft prior（无 balance）

### 目标

隔离验证：仅靠更合理的 initialization / entropy prior 是否可以将：

```text
legacy hard entropy partition
```

转换为：

```text
soft entropy specialization with non-dead pairs
```

### 配置

```yaml
model:
  moe_num_experts: 8
  moe_top_k: 2
  moe_pair_size: 2
  moe_routing_granularity: pair
  moe_ffn_dim_multiplier: 0.5

  moe_entropy_prior_mode: gaussian_pairs
  moe_entropy_prior_calibration_path: ${paths.router_calibration}
  moe_entropy_prior_scale: 0.5
  moe_entropy_prior_hidden_scale: 0.0
  moe_entropy_prior_trainable: false

  moe_balance_cost: byte
  moe_balance_loss_weight: 0.0
  moe_pair_bias_mode: none
```

### Screening 成功条件（5k）

```text
- no pair dead: min pair byte fraction ≥ 0.01
- max pair byte fraction ≤ 0.70
- pair load CV 明显小于 A1
- entropy–pair MI > 0，且 entropy / length profile 仍可解释
- train BPB 不劣于 A1 超过 1.5% relative
- 无 OOM、无 checkpoint failure
```

### 解释

若 B 已经通过，则论文中可以将核心问题归因于：

> byte-uncalibrated / hard entropy initialization，而不是 entropy feature 本身。

若 B 仍 collapse，则需要 C/D 的显式 utilization control。

---

## 6.4 Run C：Byte auxiliary-balance schedule

### 目标

检验标准 byte-balance auxiliary loss 在 paired CPT 中能否维持质量并避免 dead pair。

### 配置

```yaml
model:
  # inherit all Run B settings
  moe_balance_cost: byte
  moe_balance_loss_weight: 0.01
  moe_balance_schedule: linear_decay
  moe_balance_start_step: 0
  moe_balance_peak_step: 250
  moe_balance_decay_start_step: 1000
  moe_balance_end_step: 5000
  moe_balance_final_weight: 0.002
```

### 实现规则

balance loss 必须以 valid patches 的 byte mass 计算，并在所有训练 ranks 上 all-reduce 后得到 global load density。

目标定义：

\[
\mathcal L_{byte} = E\sum_{g=1}^{E}\left(p_g^{byte}-\frac{1}{E}\right)^2.
\]

在 pair mode 中，\(E=4\)；不要错误地以 8 个 individual experts 为平衡目标。

### 推荐额外小 grid

Screening 可以并行尝试：

```text
C-002: peak weight 0.002
C-005: peak weight 0.005
C-010: peak weight 0.010
```

只保留其中 held-out BPB 最好、且 pair load 达标的一组进入 10k。

### Screening 成功条件（5k）

```text
- min pair byte fraction ≥ 0.03
- max pair byte fraction ≤ 0.60
- pair dead count = 0
- BPB 相对 B 不恶化超过 1.0%
- entropy–pair MI 至少保留 B 的 70%
- router entropy 不得直接逼近均匀随机而丧失 specialization
```

### 失败解释

若 C load 改善但 BPB 明显下降，说明 auxiliary gradient 与 language modeling objective 冲突；将 D 作为主候选方案，不继续加大 aux weight。

---

## 6.5 Run D：Loss-free EMA byte-aware pair bias + late hidden residual

### 目标

在不向 LM loss 注入强 balance gradient 的情况下，只通过 router bias 防止 pair 长期饿死；同时在训练稳定后逐步允许 hidden residual 做细粒度修正。

### 配置

```yaml
model:
  # inherit all Run B settings
  moe_balance_cost: byte
  moe_balance_loss_weight: 0.0

  moe_pair_bias_mode: ema_byte_floor
  moe_pair_bias_ema: 0.95
  moe_pair_bias_update_interval: 20
  moe_pair_bias_lr: 0.05
  moe_pair_bias_clip: 1.5
  moe_pair_min_byte_fraction: 0.05
  moe_pair_max_byte_fraction: 0.55

  moe_entropy_prior_hidden_scale: 0.0
  moe_hidden_residual_ramp_start_step: 1000
  moe_hidden_residual_ramp_end_step: 3000
  moe_hidden_residual_final_scale: 0.25
```

### Loss-free dynamic bias update

每 `update_interval` steps：

1. 统计所有 ranks 的 global pair byte share \(q_g\)；
2. 更新 EMA：

\[
\bar q_g\leftarrow \rho\bar q_g+(1-\rho)q_g;
\]

3. 只在 pair 超出利用范围时更新 bias：

\[
\Delta b_g=
\eta\left(
\max(0, q_{min}-\bar q_g)
-
\max(0, \bar q_g-q_{max})
\right);
\]

4. 更新并中心化：

\[
b_g\leftarrow\operatorname{clip}(b_g+\Delta b_g,-b_{clip},b_{clip}),
\qquad b\leftarrow b-\operatorname{mean}(b).
\]

### 重要要求

- `dynamic_pair_bias` 是 buffer，不参与 autograd。
- 必须进入 checkpoint state。
- pair byte share 必须先 all-reduce 后再更新。
- 只有 valid patches 进入统计。
- 每次 bias update 记录 pre/post bias、EMA load 与真实 load。

### Screening 成功条件（5k）

```text
- pair dead count = 0
- min pair byte fraction ≥ 0.05
- max pair byte fraction ≤ 0.55
- BPB 不劣于 B 超过 0.5% relative
- BPB 不劣于 A1 超过 1.0% relative
- entropy–pair MI ≥ B 的 80%
- dynamic bias 未长期卡在 clip boundary
- hidden residual ramp 后未重新引发 pair collapse
```

### D 是最终候选的条件

若 D 相对 C：

```text
quality ≥ C
load stability ≥ C
entropy specialization ≥ C
```

则 D 进入 50k finalist，并作为论文中的主训练稳定化机制。

---

# 7. 决策树

```text
Phase 0 tests fail
  → 修实现，不跑训练。

A1 clean baseline no longer collapses
  → 先研究 invalid-patch dispatch 影响；B/C/D 暂停。

B passes load + quality thresholds
  → calibration is sufficient; C/D 作为 robustness ablation。

B collapses, C passes
  → aux byte balance is primary stabilization mechanism。

B collapses, C hurts quality, D passes
  → loss-free EMA byte-aware pair bias is primary method。

C and D both fail
  → 检查 warmstart t=0 equivalence、prior scale/width、EP global all-reduce、entropy patch distribution；
     不允许直接提高 balance weight 或随机重置 experts。
```

---

# 8. 必须交付的分析产物

## 8.1 每个 run 的目录结构

```text
experiments/bcfp_cpt/
  A0_legacy/
  A1_clean/
  B_calibrated/
  C_byte_aux_002/
  C_byte_aux_005/
  C_byte_aux_010/
  D_ema_bias/
    config_resolved.yaml
    warmstart_manifest.json
    router_calibration.json
    t0_equivalence_report.json
    metrics.jsonl
    stdout.log
    checkpoints/
    analysis/
      summary.json
      pair_load_curve.png
      pair_entropy_heatmap.png
      pair_length_heatmap.png
      quality_vs_load.png
      systems_profile.json
```

## 8.2 `summary.json` 必填字段

```json
{
  "run_id": "D_ema_bias_seed42",
  "git_commit": "...",
  "seed": 42,
  "train_steps": 5000,
  "source_dense_checkpoint": "...",
  "warmstart_manifest": "...",
  "router_calibration": "...",
  "t0_bpb_delta": 0.0,
  "last_100_train_bpb": 0.0,
  "heldout_bpb": 0.0,
  "pair_byte_fraction": [0, 0, 0, 0],
  "pair_unit_fraction": [0, 0, 0, 0],
  "pair_load_cv": 0.0,
  "pair_dead_count": 0,
  "entropy_pair_mi": 0.0,
  "top2_same_pair_fraction": 1.0,
  "router_entropy": 0.0,
  "wps_last_100": 0.0,
  "iter_time_last_100": 0.0,
  "gpu_max_allocated_gb": 0.0,
  "gpu_max_reserved_gb": 0.0,
  "ep_alltoall_bytes_last_100": 0.0,
  "checkpoint_roundtrip_pass": true,
  "oom": false
}
```

## 8.3 Held-out evaluation

每个通过 5k screening 的 run 必须：

1. 加载 checkpoint；
2. 在固定 expanded held-out split 上评估；
3. 记录 BPB、loss、pair load、entropy/length profiles；
4. 保存 evaluation config hash；
5. 禁止从 stdout training BPB 推断最终质量。

---

# 9. 50k finalist 选择与复现协议

## 9.1 入围条件

从 B/C/D 中选择最多两个候选进入 50k。候选必须同时满足：

```text
1. 5k 与 10k 无 OOM / checkpoint failure；
2. t=0 functional equivalence pass；
3. held-out BPB 相对 A1 不恶化超过 1%；
4. min pair byte fraction ≥ 0.03；
5. max pair byte fraction ≤ 0.60；
6. pair dead count = 0；
7. entropy specialization 可见且量化 MI 非零；
8. dynamic control 没有直接将所有 pair 强行推向随机均匀路由。
```

## 9.2 50k 主结果

对入围的每个候选跑 3 seeds：

```text
seed = 42, 43, 44
steps = 50000
```

同时复跑以下公平对照：

```text
A1 clean entropy baseline: 3 seeds
hidden-only MoE: 3 seeds, same dense source checkpoint and same CPT budget
Dense BLT continuation: 3 seeds, same data and training budget
```

## 9.3 AAAI 最终主表应报告

| 类别 | 指标 |
|---|---|
| Quality | train BPB、held-out BPB、loss、mean ± std over 3 seeds |
| Routing | pair byte share、pair load CV、dead pair count、router entropy、entropy–pair MI |
| Specialization | pair entropy means、pair length means、cross-seed profile correlation |
| System | wps、iter time、active/reserved memory、EP all-to-all bytes、checkpoint success rate |
| Fairness | source checkpoint、training bytes、model parameters、active parameters、seed count |

---

# 10. Codex 验收清单

Codex 完成后，必须逐条确认：

- [ ] legacy expert-mode routing 不受影响。
- [ ] `moe_routing_granularity=pair` 可用，Top-2 100% 同 pair。
- [ ] invalid patches 不进入 router、experts、balance、metrics、EP accounting。
- [ ] warmstart 从同一 dense BLT-1B checkpoint 生成四个 pair 的完整 A/B partitions。
- [ ] paired MoE 在 t=0 可重建 dense FFN，数值测试通过。
- [ ] `patch_feature_router` 不再使用不可控默认初始化；soft entropy prior 从 calibration JSON 加载。
- [ ] calibration 使用 byte-weighted quantiles 与 byte share，而不是 patch count。
- [ ] `static_pair_bias`、`dynamic_pair_bias`、EMA state、schedule state 都能 checkpoint round-trip。
- [ ] byte balance 在 pair mode 下以 4 个 pairs 而非 8 个 half-experts 计算。
- [ ] D 的 dynamic bias 在 all-reduce 后更新，且不进入 autograd。
- [ ] metrics.jsonl 包含所有 pair-level、valid-mask、system 指标。
- [ ] A0/A1/B/C/D 均有独立 resolved config、warmstart manifest、分析输出。
- [ ] 100–300 step smoke run 无 OOM、无 deadlock、无 checkpoint failure。
- [ ] 5k screening 的自动 summary / pass-fail 判定可由 `scripts/analyze_patchmoe_cpt.py` 生成。

---

# 11. 实施顺序（严格执行）

```text
1. 实现 valid-patch-only dispatch + 单测 T1。
2. 实现 pair router + 单测 T2。
3. 更新 warmstart script，完成 paired function-preserving initialization + 单测 T3。
4. 实现 calibration script + 单测 T4。
5. 实现 checkpoint state + 单测 T5。
6. 实现 metrics / analysis script。
7. 跑 100–300 step A1 smoke。
8. 跑 A1 5k；确认 clean baseline 的真实行为。
9. 生成 calibration JSON，跑 B 5k。
10. 只在 B 仍 collapse 时跑 C grid 与 D。
11. 对通过者跑 10k × 2 seeds。
12. 选 finalist，跑 50k × 3 seeds + fixed held-out eval。
```

**停止规则：** 任意一步未达到验收条件，先修该步骤；不要用更长训练、更多随机 seed 或更大 balance weight 掩盖实现问题。
