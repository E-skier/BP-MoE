# Byte-Patch MoE Research Roadmap for A-Conference Submission

> Working title: **PatchMoE: Patch-Native Sparse Expert Routing for Byte-Level Language Models**  
> Alternative titles: **ByteMoE**, **BLT-MoE**, **PatchFlow-MoE**, **Entropy-Aware Patch Mixture-of-Experts**

---

## 0. Executive Decision

This project should **not** aim to reproduce Meta-scale BLT pretraining or compete with industrial foundation models. The feasible and publishable path is:

> **Design and evaluate a patch-native Mixture-of-Experts architecture for byte-level language models, where dynamic byte patches become the unit of sparse expert routing, compute allocation, load balancing, and system scheduling.**

The central claim is:

> Current MoE language models are token-native, while recent byte-level models such as BLT are patch-native. Once the computation unit changes from fixed tokens to dynamic byte patches, MoE routing, load balancing, expert specialization, and serving efficiency must be redesigned.

This is a realistic A-conference direction because it combines:

1. **A timely architecture shift**: token-based LMs → byte/patch-based LMs.
2. **A mature scaling mechanism**: dense Transformer → sparse MoE Transformer.
3. **A clear missing interface**: MoE routing is still largely designed around tokens, not dynamic byte patches.
4. **A feasible experimental scope**: controlled BLT-1B continued-pretraining and active-FLOP-matched PatchMoE experiments on 2x6000 Pro plus 6xA100, with strong ablations and system analysis.

---

## 1. Research Positioning

### 1.1 Background

Modern language models usually depend on tokenizers. Tokenization introduces several limitations:

- fixed vocabulary dependency;
- language/script bias;
- poor robustness to typos, noise, and rare strings;
- extra preprocessing and vocabulary engineering;
- mismatch between semantic complexity and compute allocation.

Byte-level models avoid fixed tokenization by operating on raw bytes. However, naive byte-level autoregressive modeling is expensive because byte sequences are much longer than token sequences. Recent byte/patch-level architectures address this by grouping bytes into latent patches and performing most expensive computation at the patch level.

BLT demonstrates that byte-level LMs can use dynamically sized patches as the main computation units. Patches are determined using next-byte entropy, allocating more compute where the sequence is complex and using longer patches where the sequence is predictable.

In parallel, MoE LLMs such as Mixtral show that sparse expert activation can scale model capacity without proportionally increasing active computation. However, mainstream MoE routing is still token-centric: each token representation is routed to a small number of experts.

The gap is therefore natural:

> **Byte-level LMs have moved the computation unit from tokens to dynamic patches, but MoE architectures still assume token-level routing and token-count-based load balancing.**

### 1.2 Core Research Question

> **How should sparse expert routing be redesigned when the basic computation unit of a language model becomes a dynamic byte patch rather than a fixed token?**

A more A-conference-style formulation:

> **Can dynamic byte patches serve as a better conditional-computation interface for sparse expert language models than conventional token-level routing?**

### 1.3 Thesis Statement

This project argues that byte-patch representations provide a more expressive and system-efficient routing unit for MoE language models because each patch carries not only semantic information but also structural uncertainty, byte length, local entropy, script/code patterns, and compute demand.

---

## 2. Proposed Research Direction

### 2.1 One-Line Proposal

Design **PatchMoE**, a byte-level language model that routes dynamic byte patches to sparse experts using entropy-aware, length-aware, and load-aware expert selection.

### 2.2 What Is New?

Existing MoE LLMs usually route **tokens**. Existing byte-level LMs usually use **dense patch processing**. PatchMoE combines these two directions but does not simply add MoE layers to BLT. The contribution is to redesign the MoE mechanism around dynamic byte patches.

The novelty should be framed around four points:

1. **Patch-native routing**  
   The router operates on latent byte-patch representations instead of tokenizer-derived token embeddings.

2. **Entropy-aware expert selection**  
   Routing uses byte-level uncertainty signals, such as next-byte entropy, patch boundary confidence, and local byte complexity.

3. **Length-/cost-aware load balancing**  
   Expert load is measured by bytes, FLOPs, and patch complexity, not merely by number of routed units.

4. **Patch-aware serving analysis**  
   Dynamic patches alter batching, dispatch granularity, expert imbalance, KV footprint, and all-to-all communication.

---

## 3. Proposed Method: PatchMoE

### 3.1 Architecture Overview

PatchMoE contains five components:

1. **Byte input stream**  
   The model consumes raw UTF-8 bytes directly.

2. **Local byte encoder**  
   A lightweight local encoder maps short byte spans into contextual byte representations.

3. **Dynamic patcher**  
   Bytes are grouped into variable-length patches based on entropy, boundary confidence, or learned segmentation.

4. **Patch-level latent Transformer with MoE FFN layers**  
   The expensive global computation is performed over patch representations. Selected FFN layers are replaced with sparse MoE layers.

5. **Local byte decoder**  
   Patch-level hidden states are decoded back into byte-level predictions.

The key design is not merely inserting MoE into the Transformer. The router must understand that each routed unit is a **variable-length byte patch** with different compute cost and information density.

---

### 3.2 Patch-Level MoE Routing

For each patch \(p_i\) at layer \(\ell\), the router selects Top-\(K\) experts:

\[
\mathcal{E}^{\star}_{i,\ell}
= \operatorname{TopK}_{e \in \mathcal{E}_{\ell}}
S_{i,\ell,e}.
\]

The routing score should combine model relevance and patch-level side information:

\[
S_{i,\ell,e}
=
R_{i,\ell,e}
+ \alpha \cdot \phi_{\mathrm{entropy}}(p_i)
+ \beta \cdot \phi_{\mathrm{length}}(p_i)
+ \gamma \cdot \phi_{\mathrm{type}}(p_i)
- \eta \cdot C_{e}(t).
\]

Where:

- \(R_{i,\ell,e}\): learned router relevance score;
- \(\phi_{\mathrm{entropy}}(p_i)\): entropy or uncertainty feature;
- \(\phi_{\mathrm{length}}(p_i)\): patch length feature;
- \(\phi_{\mathrm{type}}(p_i)\): byte pattern feature, such as natural language, code-like segment, digit-heavy span, punctuation, UTF-8 script pattern;
- \(C_e(t)\): expert congestion/load price.

The key question is whether this side-information-aware router improves quality-efficiency tradeoffs compared with a standard learned router.

---

### 3.3 Cost-Aware Load Balancing

Token-level MoE usually balances the number of tokens per expert. This is insufficient for byte-patch MoE because one patch may represent 2 bytes and another may represent 20 bytes.

PatchMoE should define expert load using a weighted cost:

\[
L_e
=
\sum_{i: e \in \mathcal{E}^{\star}_{i}}
\omega_i,
\]

where \(\omega_i\) can be:

- \(1\): patch-count load;
- \(|p_i|\): byte-count load;
- \(|p_i| \cdot H(p_i)\): entropy-weighted byte load;
- estimated FLOPs or latency cost.

The load-balancing loss should minimize imbalance under this cost definition:

\[
\mathcal{L}_{\mathrm{balance}}
=
\sum_e
\left(
\frac{L_e}{\sum_{e'}L_{e'}}
- \frac{1}{|\mathcal{E}|}
\right)^2.
\]

This can become one of the main technical contributions:

> **PatchMoE replaces token-count balancing with byte-/entropy-/cost-aware expert balancing.**

---

### 3.4 Expert Specialization Hypothesis

PatchMoE should test whether experts naturally specialize by patch structure.

Potential expert specialization types:

- low-entropy predictable continuations;
- high-entropy semantic transitions;
- code-like spans;
- digits, formulas, symbols;
- punctuation-heavy spans;
- multilingual or non-Latin UTF-8 patterns;
- noisy or typo-heavy text;
- long predictable patches;
- short uncertain patches.

The paper should not only report benchmark scores. It should visualize and quantify expert specialization.

Possible measurements:

- expert usage by patch entropy bucket;
- expert usage by patch length bucket;
- expert usage by script/language;
- expert usage by natural language vs code;
- expert mutual information with patch features;
- entropy of expert assignment distribution;
- stability of expert assignment across layers.

---

### 3.5 System-Aware Extension

A stronger version of the paper should connect architecture and systems.

PatchMoE changes the serving problem because routing units are variable-length patches rather than tokens. This affects:

- expert capacity planning;
- GPU load imbalance;
- all-to-all communication volume;
- batching efficiency;
- memory bandwidth pressure;
- KV cache footprint;
- decoding step granularity;
- p95 inference latency.

System-side mechanisms:

1. **Patch bucketing**  
   Group patches by length or entropy to reduce padding and dispatch imbalance.

2. **Expert capacity by byte-cost**  
   Set capacity using byte-weighted load rather than patch count.

3. **Entropy-aware batching**  
   Avoid mixing extremely short high-entropy patches with long low-entropy patches when it harms GPU utilization.

4. **Communication accounting**  
   Measure all-to-all bytes per generated byte, not merely tokens/s.

5. **Serving latency decomposition**  
   Break latency into local encoder, patcher, latent Transformer, expert dispatch, all-to-all, decoder, and memory/KV cost.

This system analysis can distinguish the work from a pure modeling paper.

---

## 4. Minimal Viable Paper Scope

### 4.1 Main Claim

A feasible main claim is:

> **Patch-native MoE improves the quality-efficiency tradeoff of byte-level language models by routing variable-length byte patches according to both semantic representation and byte-level complexity, while reducing active computation and exposing new expert specialization patterns.**

### 4.2 What the Paper Does Not Need to Claim

Avoid claiming:

- PatchMoE beats GPT-class models.
- PatchMoE is a new SOTA foundation model.
- PatchMoE fully replaces tokenization in all settings.
- PatchMoE reproduces industrial BLT scaling.
- PatchMoE is already production-ready.

Instead, claim:

- It provides a new conditional-computation interface for byte-level LMs.
- It improves controlled quality-compute Pareto curves.
- It reveals different routing/load-balancing behavior from token MoE.
- It provides evidence that byte patches are meaningful MoE routing units.

---

## 5. Experimental Plan

## 5.1 Hardware Assumption

Available hardware:

- one 2-GPU 6000 Pro server;
- one 6-GPU A100 server;
- enough for controlled BLT-1B continued-pretraining, active-FLOP-matched PatchMoE comparisons, and smaller multi-seed ablations;
- not enough for full-scale 8B/4T-byte BLT pretraining or broad 1B-scale sweeps over every router feature.

The old 8xA100 assumption is no longer the planning target. The six-A100 server should not be treated as an eight-way expert-parallel run unless the expert count changes. With the current 8-expert PatchMoE implementation, the preferred six-A100 layout is:

- `NPROC_PER_NODE=6`;
- `EP_SIZE=2`;
- 8 experts partitioned over the 2 expert-parallel ranks;
- 3 data-parallel replicas over the six ranks.

Do not set `EP_SIZE=6` with 8 experts; the launcher correctly rejects this because the expert count is not divisible by 6. The 2x6000 Pro machine should use `NPROC_PER_NODE=2`, `EP_SIZE=2` for MoE runs and should be reserved for reproducibility, queue backfilling, short validation, held-out evaluation, and mechanism ablations.

Therefore, the experimental design must emphasize **controlled comparison**, not absolute scale.

---

## 5.2 Model Scales

The formal plan now uses four compute tiers. Budgets must be reported in training bytes and active FLOPs per byte, not just optimizer steps, because changing from 2 GPUs to 6 GPUs changes the global batch.

### Tier F0: Locked Pilot Evidence

Purpose: preserve the currently verified result as the starting hypothesis.

| Item | Value |
|---|---:|
| Model family | BLT-1B continued-pretraining / warm-start PatchMoE |
| Required variants | dense BLT-1B, hidden-only PatchMoE, entropy-only PatchMoE |
| Current data | `fineweb_edu_10bt` entropy-preprocessed chunks 00000-00001 |
| Current conclusion | entropy-only > dense BLT-1B > hidden-only under same-FLOP controls |
| Paper role | pilot evidence only until reproduced with frozen held-out evaluation |

### Tier F1: Reproducibility And Variance

Purpose: turn the pilot conclusion into defensible evidence.

| Item | Value |
|---|---:|
| Hardware | 2x6000 Pro |
| Model family | BLT-1B active-FLOP-matched triad |
| Variants | dense BLT-1B, hidden-only PatchMoE, entropy-only PatchMoE |
| Experts / Top-K | 8 experts / Top-2 for MoE variants |
| Parallel layout | `NPROC_PER_NODE=2`, `EP_SIZE=2` |
| Data | train on chunks 00000-00001; evaluate on disjoint held-out chunks 00002-00003 when available |
| Seeds | at least 3 short seeds before any optional ablation is promoted |
| Role | variance bars, failure-mode detection, checkpoint/eval reproducibility |

### Tier F2: Main Formal Result

Purpose: produce the primary paper table.

| Item | Value |
|---|---:|
| Hardware | 6xA100 |
| Model family | BLT-1B active-FLOP-matched triad |
| Variants | dense BLT-1B, hidden-only PatchMoE, entropy-only PatchMoE |
| Experts / Top-K | 8 experts / Top-2 for MoE variants |
| Parallel layout | `NPROC_PER_NODE=6`, `EP_SIZE=2`, 3 DP replicas |
| Budget | one fixed training-byte budget shared by all variants; optionally a longer entropy-only continuation |
| Data | use the same chunk split first; expand to more FineWeb-Edu chunks only after the triad is complete |
| Role | main BPB/loss, active FLOPs per byte, throughput, memory, and expert-load table |

### Tier F3: Mechanism Ablations

Purpose: explain why entropy helps without spending A100 weeks on nonessential variants.

| Item | Value |
|---|---:|
| Hardware | 2x6000 Pro by default; 6xA100 only for finalist ablations |
| Model family | Stage-1 candidate and/or shorter BLT-1B warm-start runs |
| Mandatory ablations | hidden-only vs entropy-only |
| Conditional ablations | length, byte type, entropy+type, entropy+length, congestion price, router z-loss |
| Promotion rule | run on 6xA100 only if it changes BPB, load stability, or specialization beyond entropy-only |
| Role | ablation table and expert-specialization figures |

### Tier F4: Systems And Robustness Evidence

Purpose: support the patch-native MoE claim beyond BPB.

| Item | Value |
|---|---:|
| Hardware | 6xA100 for profile; 2x6000 Pro for held-out eval/backfill |
| Variants | dense BLT-1B and entropy-only PatchMoE first; hidden-only if capacity allows |
| Metrics | active FLOPs/byte, bytes/sec, patches/sec, memory, load imbalance, dispatch overhead |
| Robustness | typo/noise, rare string/code-like spans, non-ASCII byte patterns |
| Role | system table, robustness appendix, specialization analysis |

---

## 5.3 Baselines

The paper needs clean baselines. Suggested baselines:

### Baseline A: Dense Token Transformer

A conventional tokenizer-based dense Transformer with matched active parameters.

Purpose:

- represents standard token-based modeling;
- establishes whether byte-level modeling is competitive under controlled compute.

### Baseline B: Token-Level MoE Transformer

A Mixtral-style token MoE with the same number of experts and active FFN compute.

Purpose:

- tests whether patch-level routing is better than standard token-level routing.

### Baseline C: Dense Byte/Patch Transformer

A BLT-like dense byte-patch model without MoE.

Purpose:

- isolates the gain from sparse experts rather than byte patches alone.

### Baseline D: Naive Patch-MoE

A patch MoE where the router uses only hidden representations, without entropy/length/cost features.

Purpose:

- isolates the contribution of entropy-/length-aware routing.

### Proposed: PatchMoE

Full model:

- dynamic byte patches;
- patch-level sparse experts;
- entropy-aware routing;
- byte-/cost-aware load balancing;
- optional patch-aware batching/serving optimization.

---

## 5.4 Main Comparisons

The main formal table should now be the BLT-1B active-FLOP-matched triad, because the current verified signal is `entropy-only > dense BLT-1B > hidden-only` on `fineweb_edu_10bt` chunks 00000-00001.

| Model | Init | Unit | Sparse? | Routing feature | MoE experts / Top-K | Training bytes | Active FLOPs/byte | BPB/loss | Throughput | Load imbalance |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Dense BLT-1B | official dense BLT-1B | patch | no | none | 0 / 1 | matched | measured |  |  | n/a |
| Hidden-only PatchMoE | BLT-1B warm-start, active-matched experts | patch | yes | patch hidden only | 8 / 2 | matched | measured |  |  |  |
| Entropy-only PatchMoE | BLT-1B warm-start, active-matched experts | patch | yes | patch hidden + entropy | 8 / 2 | matched | measured |  |  |  |

Secondary tables should be conditional:

| Table | Variants | Purpose | Hardware |
|---|---|---|---|
| Stage-1 mechanism ablation | hidden, entropy, length, byte type, entropy+type, entropy+length | explain router side features | 2x6000 Pro first |
| Cost/load ablation | byte balance, entropy-byte balance, congestion price, z-loss | load stability and collapse control | 2x6000 Pro first |
| Systems profile | dense BLT-1B, entropy-only PatchMoE, hidden-only if capacity allows | throughput, dispatch overhead, memory | 6xA100 |
| Token baseline appendix | dense token Transformer, token MoE | context only, not the top-line claim | smallest stable scale |

The fairness rule:

> Compare under matched training bytes and measured active FLOPs per byte. Within the BLT-1B triad, keep the data order, optimizer, sequence length, batch shape, checkpoint cadence, and evaluation set identical. Across 2-GPU and 6-GPU runs, do not compare by step count alone; report consumed bytes and active FLOPs/byte.

---

## 5.5 AAAI Workload Assessment And Parallel Execution

### AAAI Workload Assessment

The redesigned formal plan is sufficient for an AAAI-style submission if it is executed as a mechanism paper rather than a scale paper. The workload is credible because it combines:

1. a BLT-1B active-FLOP-matched main triad;
2. reproduction and seed-aware variance on a separate machine;
3. held-out validation on disjoint FineWeb-Edu chunks;
4. targeted router/load-balancing ablations;
5. expert-specialization analysis;
6. throughput, memory, and dispatch-overhead profiling;
7. robustness checks on byte-sensitive inputs.

Minimum acceptable AAAI evidence:

| Evidence block | Minimum requirement | Paper role |
|---|---|---|
| Main quality table | six-A100 BLT-1B triad: dense, hidden-only, entropy-only | primary claim |
| Reproducibility | at least one reproduced triad run on 2x6000 Pro; preferably 3 short seeds | confidence / error bars |
| Held-out evaluation | chunks 00002-00003 or another frozen disjoint held-out split via native `bytelatent.eval` | prevents train-split overclaim |
| Downstream benchmarks | OpenCompass table for MMLU, BoolQ, OpenBookQA, ARC, HellaSwag, PIQA; aggregate with `scripts/patchmoe/summarize_opencompass_results.py` | A-conference standard benchmark evidence |
| Compute accounting | training bytes, active FLOPs/byte, wall time, peak memory | fairness |
| Mechanism ablation | hidden-only vs entropy-only plus at least two optional controls | explains why entropy helps |
| Expert analysis | entropy-bucket utilization, load imbalance, specialization plots | patch-native MoE evidence |
| Systems profile | dense BLT-1B vs entropy-only PatchMoE on 6xA100 | efficiency claim |
| Robustness | typo/noise, rare strings, code-like spans, non-ASCII bytes | byte-level motivation |

This is likely enough work for AAAI if the final results remain positive and the paper is framed as a controlled conditional-computation study. It is not enough if the paper claims broad foundation-model scaling, tokenizer replacement, or general SOTA. The claim must stay narrow:

> Entropy-aware patch routing improves the quality-efficiency tradeoff of BLT-style byte-patch models under matched active compute, and exposes patch-structured expert behavior.

Hard submission gate:

- do not submit with only the current chunks 00000-00001 pilot result;
- do not submit without frozen held-out evaluation;
- do not submit without active-FLOP accounting;
- do not submit if entropy-only only beats hidden-only but not dense BLT-1B in the formal triad;
- do not promote byte-type/congestion/z-loss to the main claim unless they beat entropy-only or explain a clear stability issue.

### Two-Server Parallel Execution Plan

The two servers should be used concurrently. The six-A100 machine is the scarce main-result machine; the 2x6000 Pro machine should continuously remove uncertainty around it.

| Phase | 2x6000 Pro server | 6xA100 server | Can run in parallel? | Dependency |
|---|---|---|---|---|
| P0: readiness | verify data, held-out preprocessing, checkpoint load, eval scripts | six-A100 CUDA smoke with `NPROC_PER_NODE=6`, `EP_SIZE=2` | yes | none |
| P1: pilot reproduction | triad short seeds on chunks 00000-00001; held-out eval on 00002-00003 | idle or smoke/profile dry run | yes | current pilot checkpoints/data |
| P2: main triad | backfill failed seed, monitor eval, run short mechanism checks | formal dense BLT-1B, hidden-only, entropy-only triad | yes | P0 smoke plus frozen config |
| P3: mechanism ablations | length, byte type, entropy+type, entropy+length, congestion, z-loss | continue/finish main triad | yes | P1 confirms entropy-only signal |
| P4: systems profile | held-out eval, robustness eval, analysis aggregation | dense vs entropy-only throughput/memory/dispatch profile | yes | at least one completed formal checkpoint |
| P5: data-scale confirmation | evaluate expanded-heldout or smaller controls | expanded chunk confirmation or longer entropy-only continuation | yes | F2 main triad stable |
| P6: paper assembly | plots, tables, seed summaries, failure cases | optional final rerun/profile only | yes | all mandatory evidence blocks |

Scheduling rules:

- Never block the six-A100 machine on optional ablations while the BLT-1B triad is incomplete.
- Keep 2x6000 Pro occupied with reproducibility, held-out evaluation, short ablations, and analysis jobs.
- Promote a 2x6000 Pro ablation to 6xA100 only if it changes held-out BPB, load stability, or expert-specialization interpretation.
- Record every cross-server comparison by training bytes and active FLOPs/byte, not by step count.
- Use the same held-out split for both servers so server-to-server differences are diagnosable.

---

## 5.6 Datasets

Use a mixture that tests byte-level advantages.

### Pretraining / Language Modeling Data

Recommended:

- C4 subset;
- SlimPajama subset;
- The Pile subset;
- English Wikipedia subset;
- GitHub/code subset;
- multilingual subset, if available.

The dataset does not need to be huge. It needs to be diverse enough to expose:

- natural language;
- code;
- symbols;
- rare words;
- multilingual scripts;
- noisy strings.

### Evaluation Data

Use several categories:

1. **Language modeling**
   - validation BPB;
   - validation negative log-likelihood;
   - optionally token-normalized perplexity for compatibility.

2. **Downstream reasoning/knowledge**
   - run through OpenCompass using `bytelatent.opencompass.ByteLatentOpenCompassModel`;
   - MMLU;
   - BoolQ;
   - OpenBookQA;
   - ARC-Easy and ARC-Challenge;
   - HellaSwag;
   - PIQA;
   - optional WinoGrande.

   Each formal checkpoint should write an OpenCompass run under `runs/opencompass/<run_name>`, then merge all completed runs with `scripts/patchmoe/summarize_opencompass_results.py` to produce `opencompass_long.csv` and `opencompass_paper_table.csv`.

3. **Robustness**
   - character noise;
   - typos;
   - casing perturbation;
   - Unicode perturbation;
   - rare string handling;
   - code formatting perturbation.

4. **Byte-specific evaluation**
   - spelling-sensitive tasks;
   - transliteration or non-Latin scripts;
   - code identifiers;
   - numeric strings;
   - corrupted text recovery.

For an A-conference paper, robustness and byte-specific evaluation are important because they justify why byte-level modeling matters. OpenCompass should be used for the standard downstream benchmark table, while native PatchMoE evaluation remains authoritative for BPB, active FLOPs/byte, routing, expert-load, and system metrics.

---

## 5.7 Metrics

### Modeling Metrics

- Bits per byte (BPB);
- validation loss;
- downstream accuracy;
- robustness degradation under noise;
- long-tail token/string accuracy.

### Efficiency Metrics

- active parameters;
- active FLOPs per byte;
- generated bytes per second;
- patches per second;
- effective tokens/bytes per second;
- GPU memory usage;
- memory bandwidth proxy;
- training throughput.

### MoE Metrics

- expert load imbalance;
- load-balancing loss;
- dropped patch rate, if using capacity limits;
- Top-K expert entropy;
- expert utilization distribution;
- expert specialization by entropy/length/type;
- all-to-all communication bytes;
- expert dispatch latency.

### System Metrics

- p50/p95 inference latency;
- end-to-end decoding latency;
- local encoder latency;
- latent Transformer latency;
- MoE dispatch latency;
- local decoder latency;
- communication overhead;
- KV/cache memory footprint.

---

## 6. Ablation Plan

The paper must include strong ablations. Suggested ablations:

### 6.1 Routing Features

Compare in priority order:

1. hidden-only router;
2. hidden + entropy;
3. hidden + patch length;
4. hidden + byte type;
5. hidden + entropy + byte type;
6. hidden + entropy + length;
7. hidden + entropy + congestion price / z-loss, only if load collapse or instability appears.

The mandatory formal contrast is hidden-only vs entropy-only. Length, byte-type, congestion, and z-loss are mechanism ablations, not top-line variants, unless they beat entropy-only on held-out BPB or materially improve load stability at matched active FLOPs.

Expected result:

- entropy-aware routing improves quality-efficiency over hidden-only routing;
- additional side features either explain specialization or are reported as negative/neutral controls;
- no optional feature should consume six-A100 main-run budget before the BLT-1B triad is reproduced.

### 6.2 Load Balancing Definition

Compare:

1. patch-count balancing;
2. byte-count balancing;
3. entropy-weighted balancing;
4. latency/FLOP-weighted balancing.

Expected result:

- patch-count balancing is insufficient because patches have variable byte lengths;
- byte-/cost-aware balancing reduces expert imbalance and dispatch latency.

### 6.3 Number of Experts

Compare:

- 4 experts;
- 8 experts;
- 16 experts;
- optional 32 experts at small scale.

Expected result:

- more experts improve capacity but increase routing imbalance and communication overhead;
- patch-aware load balancing becomes more important as expert count increases.

### 6.4 Top-K Routing

Compare:

- Top-1;
- Top-2;
- optional Top-4.

Expected result:

- Top-2 improves quality and stability;
- Top-1 improves throughput but may produce unstable specialization.

### 6.5 Patch Policy

Compare:

1. fixed-size byte patches;
2. entropy-based dynamic patches;
3. learned patch boundaries;
4. oracle or upper-bound segmentation, if feasible.

Expected result:

- dynamic patching improves quality/compute tradeoff;
- MoE benefits more from dynamic patches than from fixed patches.

### 6.6 Expert Placement / Serving

If system experiments are included:

1. random expert placement;
2. balanced expert placement;
3. entropy-aware expert placement;
4. patch-bucketed dispatch.

Expected result:

- patch-aware scheduling reduces p95 latency and all-to-all traffic.

---

## 7. Expected Figures for the Paper

### Figure 1: Architecture Overview

Show:

- raw byte stream;
- local encoder;
- entropy-based dynamic patches;
- patch-level latent Transformer;
- sparse MoE experts;
- local decoder.

Caption message:

> PatchMoE replaces token-level expert routing with dynamic byte-patch routing.

### Figure 2: Routing Unit Comparison

Compare:

- token MoE: token → experts;
- dense BLT: byte patches → dense Transformer;
- PatchMoE: byte patches → sparse experts.

Caption message:

> The routing unit changes from tokenizer-defined tokens to entropy-defined byte patches.

### Figure 3: Quality–Compute Pareto

x-axis:

- active FLOPs per byte;
- or active parameters.

 y-axis:

- BPB or validation loss.

Expected:

- PatchMoE should dominate or improve over dense byte model and token MoE at similar active compute.

### Figure 4: Expert Specialization Heatmap

Rows:

- experts.

Columns:

- entropy bucket;
- patch length bucket;
- text/code/script category.

Expected:

- different experts specialize to different patch complexity regimes.

### Figure 5: Load Balancing and Dispatch Cost

Show:

- expert load distribution under patch-count vs byte-/cost-aware balancing;
- all-to-all bytes;
- dispatch latency;
- p95 latency.

### Figure 6: Robustness Evaluation

Show performance drop under:

- typos;
- byte corruption;
- Unicode noise;
- rare strings;
- code identifier perturbation.

Expected:

- byte-level PatchMoE should be more robust than token baselines.

---

## 8. Implementation Roadmap

## 8.1 Phase 1: Literature and Reproduction

Duration: 2–3 weeks.

Tasks:

1. Read and summarize:
   - BLT;
   - Fast BLT;
   - ByT5;
   - MEGABYTE;
   - Switch Transformer;
   - GShard;
   - GLaM;
   - Mixtral;
   - recent MoE training/system papers.

2. Reproduce or implement a small dense byte/patch Transformer.

3. Implement a toy MoE layer at patch level.

4. Confirm that the pipeline trains stably on 1B–5B bytes.

Deliverables:

- literature matrix;
- small-scale dense byte model;
- small-scale patch MoE model;
- first BPB curves.

Exit criteria:

- training loss decreases normally;
- patching pipeline works;
- MoE router does not collapse completely;
- expert load can be logged.

---

## 8.2 Phase 2: PatchMoE Core Mechanism

Duration: 4–6 weeks.

Tasks:

1. Implement entropy-aware routing.
2. Implement patch-length-aware routing.
3. Implement byte-/cost-aware load balancing.
4. Compare against hidden-only patch router.
5. Run Stage-1 model scale.

Deliverables:

- router feature ablation;
- load balancing ablation;
- expert specialization visualization;
- initial main result table.

Exit criteria:

- PatchMoE improves over naive Patch-MoE or dense patch baseline under matched active compute;
- expert utilization remains stable;
- specialization patterns are visible.

### Implementation checkpoint: 2026-06-01

Completed integration:

- PatchMoE is wired into the BLT global Transformer at the dynamic patch level.
- Router side features now cover patch length, patch entropy, and patch byte-type ratios: alpha, digit, whitespace, punctuation, and non-ASCII bytes.
- Patch-count, byte-count, and entropy-weighted byte balancing remain selectable controls.
- Cost-aware expert congestion pricing `C_e(t)` is implemented as an optional router logit penalty using the active patch cost definition.
- Router z-loss is implemented as an optional training auxiliary loss on learned router logits for collapse mitigation.
- Byte-type specialization metrics and analysis plots are available.
- Stage-0 end-to-end smoke runs pass for byte-type-only, entropy-plus-byte-type, entropy-plus-byte-type-plus-congestion, and congestion-plus-z-loss routing.
- Stage-1 matched-eval launch script now exposes the roadmap queue variants, including byte-type, congestion, and z-loss candidates.
- Formal analysis hooks now emit training summaries, held-out validation summaries, and exact-run-label merged `summary_with_heldout.csv` tables for Stage-1/200k multi-seed evidence.
- Stage-1 and 200k formal launchers now provide `--preflight` checks for data readiness, variant coverage, checkpoints, eval outputs, and pending runs before using GPUs.
- `scripts/patchmoe/launch_formal_single_variant.sh` provides a guarded single-variant launch path that defaults to command printing and refuses `MODE=run` when GPU memory or utilization is above threshold.
- `scripts/patchmoe/formal_status_report.py` emits CSV/JSON status for Stage-1, 200k, and external dense-compute controls, including done/partial/todo state and held-out BPB when available.
- BLT-1B warm-start integration has started: `scripts/patchmoe/prepare_blt1b_patchmoe_warmstart.py` converts the released dense BLT-1B checkpoint into a PatchMoE DCP initialization by preserving the byte/local/global/local-decoder trunk, expanding selected patch-level global FFNs into copied experts, and initializing new routers deterministically. `scripts/patchmoe/verify_blt1b_patchmoe_warmstart.py` verifies the full DCP key set and representative loaded tensors. The first guarded BLT-1B launcher uses the validated entropy router with byte-cost balancing on all 25 global FFN layers by default, keeps `moe_layer_frequency` as the interleaved-MoE ablation switch, and supports a locked one-time `MODE=wait` queue while GPUs are occupied.
- A CPU-side production-loader audit materialized and loaded the complete all-layer BLT-1B PatchMoE model (`10,589,816,008` parameters, `986` state keys). It exposed and fixed the inherited dense-only top-level RoPE reset assumption: warm-start loading now calls `model.reset_rope_embeddings()`, which resets all three non-persistent BLT RoPE buffers before training.
- All-layer BLT-1B PatchMoE now has an expert-parallel execution path. `model.moe_ep_size` partitions the eight experts across EP ranks, patch assignments use differentiable all-to-all dispatch and return collectives, and local experts are wrapped first on the orthogonal `expert_dp` mesh so outer FSDP continues to shard the shared BLT trunk without materializing every expert on every rank. Rank-local model and AdamW states save and restore through DCP while preserving complete consolidated expert keys. For the current hardware, the two-GPU smoke path can use `EP_SIZE=2`; the six-A100 formal path must set `EP_SIZE=2` explicitly so the 8 experts divide the EP group while the six ranks form three data-parallel replicas. CPU/gloo equivalence, gradient, and DCP regression tests pass; the six-A100 CUDA nested-FSDP smoke remains pending.

### Formal experiments queue: 2026-06-05 compute-aware redesign

Validated starting point:

- On `data/entropy_preprocessed_stage1/fineweb_edu_10bt` chunks 00000-00001, the observed ordering is `entropy-only > dense BLT-1B > hidden-only` under the same-FLOP controls.
- Treat this as pilot evidence. It becomes paper evidence only after frozen held-out evaluation and at least one reproduction run with identical accounting.

Main queue:

1. Freeze the BLT-1B triad and accounting: dense BLT-1B, hidden-only active-matched PatchMoE, and entropy-only active-matched PatchMoE. Every result table must include training bytes, active FLOPs/byte, wall time, peak memory, BPB/loss, and exact data split.
2. Run F1 on 2x6000 Pro: reproduce the triad on chunks 00000-00001 and evaluate on disjoint held-out chunks 00002-00003. Use this machine for multi-seed variance, checkpoint validation, and queue backfilling.
3. Run F2 on 6xA100: launch the same triad with `NPROC_PER_NODE=6` and `EP_SIZE=2`. Keep the 8-expert Top-2 MoE shape fixed. Compare variants only at the same training-byte budget.
4. After the triad completes, run one data-scale confirmation: either repeat the triad on an expanded FineWeb-Edu chunk set or continue only dense BLT-1B and entropy-only PatchMoE if the hidden-only gap is already stable.
5. Run F3 mechanism ablations only after the main ordering is stable: length, byte type, entropy+type, entropy+length, congestion price, and z-loss stay on 2x6000 Pro unless they materially improve BPB or load stability.
6. Run F4 systems profile on 6xA100: profile dense BLT-1B vs entropy-only PatchMoE first, then hidden-only if capacity allows.

Pending exit-criteria evidence:

- frozen held-out BPB/loss for the BLT-1B triad on chunks 00002-00003 or another disjoint held-out set;
- at least one reproduced triad run showing the same ordering as the pilot;
- six-A100 CUDA smoke for all-layer BLT-1B PatchMoE with `NPROC_PER_NODE=6`, `EP_SIZE=2`, nested FSDP materialization, and one optimizer step;
- main six-A100 active-FLOP-matched triad table;
- expert utilization and entropy-bucket specialization for entropy-only versus hidden-only;
- throughput, memory, and dispatch-overhead profile for dense BLT-1B versus entropy-only PatchMoE;
- seed-aware BLT-1B queue support if the current active-matched launcher does not expose seed-specific run names and `seed=` overrides.

---

## 8.3 Phase 3: Main Model Training

Duration: 4–8 weeks.

Tasks:

1. Train main PatchMoE model at 500M–800M active scale.
2. Train matched baselines.
3. Run downstream evaluations.
4. Run robustness evaluations.
5. Run efficiency and latency measurements.

Deliverables:

- main comparison table;
- quality-compute Pareto plot;
- robustness plot;
- efficiency breakdown.

Exit criteria:

- PatchMoE has a clear win in at least two of the following:
  - BPB/loss under matched compute;
  - robustness;
  - throughput/latency;
  - expert specialization/load stability;
  - memory/communication efficiency.

---

## 8.4 Phase 4: System Analysis

Duration: 3–5 weeks.

Tasks:

1. Measure expert dispatch overhead.
2. Measure all-to-all communication bytes.
3. Compare patch-count vs byte-cost capacity.
4. Implement patch bucketing.
5. Measure p95 inference latency.
6. Optionally emulate edge-cloud placement if connecting to StateFlow.

Deliverables:

- latency decomposition;
- all-to-all communication analysis;
- serving-oriented result section.

Exit criteria:

- system results explain why patch-aware MoE is different from token MoE;
- serving results are not merely engineering details but support the architectural claim.

---

## 8.5 Phase 5: Paper Writing

Duration: 4–6 weeks.

Tasks:

1. Write introduction around the missing interface between byte patches and MoE.
2. Write method section clearly.
3. Keep theory simple and focused.
4. Write experiments with controlled comparison.
5. Prepare figures early.
6. Run final sanity checks and statistical repeats.

Deliverables:

- full paper draft;
- supplementary material;
- code release plan;
- reproducibility checklist.

---

## 9. Proposed Paper Structure

## Abstract

State the problem:

- byte-level LMs remove tokenizers but shift computation to dynamic patches;
- MoE scales models but remains token-native;
- propose patch-native MoE routing.

State method:

- entropy-/length-/cost-aware patch routing;
- byte-/cost-aware load balancing.

State results:

- improved quality-compute tradeoff;
- better robustness;
- expert specialization;
- reduced serving overhead or better load balance.

---

## 1. Introduction

Recommended storyline:

1. Tokenization is no longer a fixed assumption.
2. Byte-level LMs are becoming viable through dynamic patching.
3. MoE is the dominant conditional computation mechanism for scaling LLMs.
4. But MoE routing is still token-native.
5. Dynamic byte patches are neither tokens nor fixed sequence chunks; they have variable length, entropy, and cost.
6. Therefore, patch-native MoE needs new routing and load balancing.
7. Introduce PatchMoE.

Contributions:

1. We formulate patch-native MoE for byte-level LMs.
2. We propose entropy-/length-/cost-aware expert routing.
3. We propose byte-/cost-aware expert load balancing.
4. We provide controlled training and serving analysis showing quality-efficiency benefits.
5. We analyze expert specialization under byte-patch routing.

---

## 2. Background and Motivation

Sections:

2.1 Tokenization and byte-level language modeling  
2.2 Dynamic byte patches  
2.3 Sparse MoE language models  
2.4 Why token-level MoE does not directly solve patch-level modeling  
2.5 Motivating measurements

Motivating measurements can include:

- patch length distribution;
- patch entropy distribution;
- correlation between patch entropy and prediction loss;
- imbalance caused by patch-count balancing.

---

## 3. PatchMoE Architecture

Sections:

3.1 Byte encoder and dynamic patching  
3.2 Patch-level latent Transformer  
3.3 Patch-native MoE layers  
3.4 Entropy-aware routing  
3.5 Byte-/cost-aware load balancing  
3.6 Training objective

---

## 4. Systems Implications

Optional but valuable.

Sections:

4.1 Patch length and expert capacity  
4.2 Patch bucketing and batching  
4.3 Expert dispatch and all-to-all communication  
4.4 Inference latency decomposition

This section can strengthen the A-conference submission if targeting MLSys, ASPLOS, OSDI-adjacent ML systems venues, or a systems-flavored NeurIPS/ICLR paper.

---

## 5. Experiments

Sections:

5.1 Experimental setup  
5.2 Main quality-compute comparison  
5.3 Routing feature ablation  
5.4 Load balancing ablation  
5.5 Robustness evaluation  
5.6 Expert specialization analysis  
5.7 Serving efficiency analysis  
5.8 Scaling trend

---

## 6. Related Work

Categories:

- byte-level and token-free LMs;
- dynamic patching;
- MoE LLMs;
- efficient MoE training/serving;
- adaptive computation;
- tokenizer robustness and multilingual modeling.

---

## 7. Conclusion

Reinforce:

> Dynamic byte patches are not just a tokenizer replacement; they are a new conditional-computation unit. PatchMoE shows that routing, balancing, and serving should be redesigned around this unit.

---

## 10. Venue Strategy

### 10.1 Best-Fit Venues

#### ICLR

Best if the paper emphasizes:

- architecture novelty;
- representation and routing mechanism;
- interpretability of expert specialization;
- clean ablations.

Risk:

- reviewers may demand stronger scale or broad benchmarks.

How to strengthen:

- show strong controlled comparisons;
- emphasize mechanism, not SOTA;
- include robustness and specialization.

#### NeurIPS

Best if the paper emphasizes:

- conditional computation;
- scaling behavior;
- expert specialization;
- theoretical or empirical analysis of routing/load balancing.

Risk:

- novelty may be seen as engineering if method is just “MoE added to BLT”.

How to strengthen:

- make the load-balancing and routing formulation clearly new;
- provide analysis of why token MoE assumptions fail for patches.

#### ICML

Best if the paper emphasizes:

- principled routing objective;
- compute allocation;
- learning dynamics;
- clean mathematical formulation.

Risk:

- system-heavy results may be less appreciated unless tied to learning objective.

How to strengthen:

- formulate routing as cost-constrained conditional computation;
- include rigorous ablations.

#### ACL / EMNLP

Best if the paper emphasizes:

- tokenizer-free modeling;
- multilingual robustness;
- rare words;
- noise robustness;
- code/morphology/string-level tasks.

Risk:

- MoE systems analysis may be less central.

How to strengthen:

- focus on language robustness and tokenizer limitations.

#### MLSys

Best if the paper emphasizes:

- serving efficiency;
- dynamic patch batching;
- expert dispatch;
- p95 latency;
- memory/communication bottlenecks.

Risk:

- may require stronger systems implementation and deployment evidence.

How to strengthen:

- include full latency decomposition and GPU utilization analysis.

---

## 10.2 Recommended Primary Target

The recommended target is:

> **ICLR or NeurIPS if the architecture and empirical results are strong; MLSys if the system analysis becomes the strongest part.**

For the current StateFlow background, the most natural long-term path is:

1. **ICLR/NeurIPS version**: Patch-native MoE architecture and routing.
2. **MLSys/TMC extension**: distributed serving of byte-patch MoE over edge-cloud networks.

Do not mix too much edge-cloud orchestration into the first paper. The first paper should establish PatchMoE as a model/system primitive. StateFlow can become the follow-up deployment framework.

---

## 11. Relation to StateFlow and PhD Direction

This research can connect naturally to the broader PhD story, but it should remain independently publishable.

### 11.1 Relation to StateFlow

StateFlow assumes MoE sparse expert routing at token/layer level. PatchMoE changes the basic unit:

- from token to byte patch;
- from token-count load to byte-/cost-aware load;
- from token-level expert dispatch to patch-level expert dispatch;
- from fixed token generation to variable patch generation.

Therefore, PatchMoE can be positioned as the next-generation model substrate for StateFlow-like distributed inference.

### 11.2 Research Chain

A coherent PhD chain could be:

1. **StateFlow**  
   Session-continuous distributed MoE inference over edge-cloud networks.

2. **PatchMoE / ByteFlow-MoE**  
   Byte-patch-native sparse inference and routing.

3. **CogFlow**  
   Memory-continuous distributed agent execution over dynamic model/state units.

PatchMoE should not replace StateFlow. It upgrades the computation unit that StateFlow may eventually orchestrate.

---

## 12. Risks and Mitigation

### Risk 1: Implementation Complexity Too High

Problem:

- BLT-like architecture + MoE + dynamic patching may be difficult to implement from scratch.

Mitigation:

- begin with a simplified patch Transformer;
- use fixed-size patches first;
- add entropy-based patching later;
- add MoE only after dense patch model is stable;
- keep first model small.

---

### Risk 2: PatchMoE Does Not Improve BPB

Problem:

- MoE may not improve language modeling loss at small scale.

Mitigation:

- emphasize robustness, load balance, and compute efficiency;
- test noisy/multilingual/code tasks where byte-level modeling helps;
- compare under matched active compute, not total params;
- use expert specialization analysis as supporting evidence.

---

### Risk 3: Router Collapse

Problem:

- experts may collapse to a few dominant experts.

Mitigation:

- use load-balancing loss;
- add router z-loss (implemented as an optional PatchMoE auxiliary loss, default off);
- use capacity factor;
- use noisy Top-K routing;
- warm up with dense FFN or progressive sparsification;
- start with Top-2 rather than Top-1.

---

### Risk 4: Reviewers Say “This Is Just BLT + MoE”

Problem:

- novelty may be questioned.

Mitigation:

The paper must repeatedly emphasize:

- patch-native routing is different from token routing;
- patch length and entropy change load definition;
- byte-/cost-aware balancing is necessary;
- expert specialization is structured by byte-patch properties;
- serving metrics change under variable-length patches.

The method section should avoid appearing as a simple module insertion.

---

### Risk 5: Experiments Too Small for A Conference

Problem:

- the available 2x6000 Pro plus 6xA100 setup cannot match industrial scale.

Mitigation:

- present controlled scaling laws across 3 sizes;
- include strong baselines;
- include multiple seeds at small scale;
- include robustness and systems analysis;
- release code and configurations;
- be explicit that this is a mechanism study.

---

## 13. Concrete Weekly Plan

### Month 1: Foundation

Week 1:

- finalize literature review;
- define exact architecture variants;
- build data pipeline for raw bytes;
- implement metrics: BPB, patch length, patch entropy.

Week 2:

- implement fixed-patch dense byte Transformer;
- train tiny model;
- log patch statistics.

Week 3:

- implement dynamic patching;
- compare fixed vs dynamic patches;
- stabilize dense patch baseline.

Week 4:

- implement MoE FFN layer;
- run toy PatchMoE;
- debug expert load logging.

### Month 2: Core Method

Week 5:

- implement hidden-only patch router;
- train Stage-0 and Stage-1 models.

Week 6:

- add entropy-aware routing;
- run ablation.

Week 7:

- add patch-length-aware routing;
- run ablation.

Week 8:

- add byte-/cost-aware balancing;
- produce first main plots.

### Month 3: Baselines

Week 9:

- train dense token baseline.

Week 10:

- train token-MoE baseline.

Week 11:

- train dense byte/patch baseline.

Week 12:

- compare all baselines under matched active compute.

### Month 4: Main Training

Week 13–14:

- train main PatchMoE model.

Week 15:

- train or finish main baselines.

Week 16:

- run downstream and robustness evaluations.

### Month 5: System and Analysis

Week 17:

- measure throughput and latency.

Week 18:

- measure expert communication and load imbalance.

Week 19:

- expert specialization visualization.

Week 20:

- scaling trend experiments.

### Month 6: Paper

Week 21:

- write introduction and method.

Week 22:

- write experiments.

Week 23:

- write related work and conclusion.

Week 24:

- polish, rebuttal-style self-review, release checklist.

---

## 14. Minimum Publishable Result

The project is publishable if it can show all of the following:

1. PatchMoE trains stably.
2. PatchMoE improves over dense byte/patch baseline or token-MoE under matched active compute.
3. Entropy-/length-aware routing improves over hidden-only routing.
4. Byte-/cost-aware load balancing reduces imbalance or latency compared with patch-count balancing.
5. Experts specialize according to byte-patch properties.
6. The method has a clear advantage on robustness, code, multilingual, noisy text, or efficiency.

The strongest version shows:

- better BPB/loss;
- better robustness;
- lower active compute;
- lower p95 latency;
- interpretable expert specialization;
- consistent scaling trend.

---

## 15. Final Research Identity

The identity of this work should be:

> **A conditional-computation architecture for tokenizer-free language modeling.**

Not:

> “A bigger byte-level LLM.”

Not:

> “A distributed serving system only.”

Not:

> “BLT plus MoE.”

The correct framing is:

> **When language models move from tokens to dynamic byte patches, sparse expert routing must also move from token-native to patch-native. PatchMoE provides this missing interface.**

---

## 16. Reference Pointers

This roadmap is based on the following research directions and papers:

1. Byte Latent Transformer: Patches Scale Better Than Tokens, arXiv:2412.09871.  
   https://arxiv.org/abs/2412.09871

2. Fast Byte Latent Transformer, arXiv:2605.08044.  
   https://arxiv.org/abs/2605.08044

3. ByT5: Towards a token-free future with pre-trained byte-to-byte models, arXiv:2105.13626.  
   https://arxiv.org/abs/2105.13626

4. Mixtral of Experts, arXiv:2401.04088.  
   https://arxiv.org/abs/2401.04088

5. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.

6. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.

7. GLaM: Efficient Scaling of Language Models with Mixture-of-Experts.

8. Recent MoE systems work on expert parallelism, communication-efficient MoE training, and Megatron-Core MoE implementations.

---

## 17. Immediate Next Actions

The next practical steps are:

1. Decide the exact name: PatchMoE, ByteMoE, or ByteFlow-MoE.
2. Write a two-page internal proposal.
3. Implement the smallest fixed-patch dense byte model.
4. Add patch-level MoE.
5. Run the first 1B-byte experiment.
6. Generate three early plots:
   - patch length/entropy distribution;
   - BPB training curve;
   - expert load distribution.

Only after these three plots are available should the project move to larger training.
