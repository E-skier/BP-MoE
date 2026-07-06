# Entropy PhaseB Paper Experiments

This document describes the next experiments for the paper-critical 50k checkpoint:

`blt1b_warmstart/entropy_bands_phaseB_100k/checkpoints/0000050000`

The launcher is:

`scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh`

It does not modify training code. It only launches existing `bytelatent.train` and `bytelatent.eval` entrypoints with controlled overrides.

## Goals

1. Run held-out eval for the existing Entropy PhaseB 50k checkpoint.
2. Rerun hidden-only 50k with a real checkpoint artifact, because stdout metrics are not enough for final paper tables.
3. Add Entropy PhaseB balance-loss controls at `moe_balance_loss_weight=0.01` and `0.05`.
4. Produce two read-only tables:
   - Quality: train BPB/loss and held-out BPB/loss.
   - System cost: WPS, iteration time, FLOPs, memory, allocation retries, all-to-all bytes, and expert load balance.

## Preflight

Always start with:

```bash
ACTION=preflight scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

This checks:

- training config and entropy-preprocessed training data
- existing Entropy PhaseB 50k checkpoint
- PhaseA init checkpoint for balance-control reruns
- hidden-only warm-start DCP
- held-out source file and held-out arrow status
- current GPU guard state

Default GPU guard refuses launch if a selected GPU is above `MAX_GPU_USED_MB=20000` or `MAX_GPU_UTIL_PCT=20`.

## Immediate Held-Out Eval

Full 50k held-out eval:

```bash
ACTION=eval-phaseb50k MODE=run GPU=1 EVAL_CUDA_VISIBLE_DEVICES=1 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

Output:

`runs/entropy_phaseb_paper_heldout_eval/entropy_bands_phaseB_100k/0000050000/validation.json`

For a fast smoke eval, cap batches:

```bash
ACTION=eval-phaseb50k MODE=run GPU=1 EVAL_CUDA_VISIBLE_DEVICES=1 EVAL_MAX_BATCHES=2000 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

Use the full run for paper numbers. The capped run is only for checking that the checkpoint loads and the eval path works.

## Training Targets

Default targets:

```text
phaseb_balance_w001
phaseb_balance_w005
hidden_only_w000_50k
```

Variants:

| variant | purpose | init checkpoint | key routing/balance setting |
| --- | --- | --- | --- |
| `phaseb_repro_w000` | optional Entropy PhaseB baseline rerun | PhaseA 1k checkpoint | entropy side feature, balance weight 0.00 |
| `phaseb_balance_w001` | balance-control test | PhaseA 1k checkpoint | entropy side feature, balance weight 0.01 |
| `phaseb_balance_w005` | stronger balance-control test | PhaseA 1k checkpoint | entropy side feature, balance weight 0.05 |
| `hidden_only_w000_50k` | hidden-only checkpoint recovery | hidden-only active-matched DCP | hidden-state router, no entropy side feature, balance weight 0.00 |

All default runs target 50k steps and save checkpoints every 5k steps with `checkpoint.dump.keep=3`.

## Serial Launch

Print the command first:

```bash
ACTION=train VARIANT=phaseb_balance_w001 MODE=print GPUS=1,2 NPROC_PER_NODE=2 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

Launch:

```bash
ACTION=train VARIANT=phaseb_balance_w001 MODE=run GPUS=1,2 NPROC_PER_NODE=2 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

Then repeat for:

```bash
ACTION=train VARIANT=phaseb_balance_w005 MODE=run GPUS=3,4 NPROC_PER_NODE=2 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh

ACTION=train VARIANT=hidden_only_w000_50k MODE=run GPUS=5,6 NPROC_PER_NODE=2 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

## Parallel Launch

Open separate shells and assign disjoint GPU pairs:

```bash
ACTION=train VARIANT=phaseb_balance_w001 MODE=run GPUS=1,2 NPROC_PER_NODE=2 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

```bash
ACTION=train VARIANT=phaseb_balance_w005 MODE=run GPUS=3,4 NPROC_PER_NODE=2 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

```bash
ACTION=train VARIANT=hidden_only_w000_50k MODE=run GPUS=5,6 NPROC_PER_NODE=2 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

If you intentionally launch on a busy GPU after manual inspection:

```bash
FORCE_GPU=1 ACTION=train VARIANT=phaseb_balance_w001 MODE=run GPUS=1,2 NPROC_PER_NODE=2 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

## Eval New Checkpoints

After a training run reaches 50k, evaluate it:

```bash
ACTION=eval MODE=run GPU=1 EVAL_CUDA_VISIBLE_DEVICES=1 \
  CKPT_DIR=blt1b_warmstart/paper_phaseb_experiments/entropy_bands_phaseB_balance001_50k/checkpoints/0000050000 \
  RUN_DIR=blt1b_warmstart/paper_phaseb_experiments/entropy_bands_phaseB_balance001_50k \
  LABEL=entropy_bands_phaseB_balance001_50k STEP=0000050000 \
  scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

Repeat with the corresponding run name for `balance005` and `hidden_only`.

## Monitoring

Each launched job writes:

- training log: `<run_dir>/train.log`
- training PID: `<run_dir>/train.pid`
- training metrics: `<run_dir>/metrics.jsonl`
- checkpoints: `<run_dir>/checkpoints/00000xxxxx`
- eval log: `<eval_dir>/eval.log`
- eval PID: `<eval_dir>/eval.pid`
- eval metrics: `<eval_dir>/validation.json`

Useful checks:

```bash
nvidia-smi
tail -f blt1b_warmstart/paper_phaseb_experiments/entropy_bands_phaseB_balance001_50k/train.log
tail -n 1 blt1b_warmstart/paper_phaseb_experiments/entropy_bands_phaseB_balance001_50k/metrics.jsonl
```

## Summary Tables

Generate the markdown tables:

```bash
ACTION=summarize scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
```

Default output:

`runs/entropy_phaseb_paper_tables.md`

Quality table fields:

- `train_bpb`: `bpb/interval_across_gpus`
- `train_loss`: `loss/interval_across_gpu`
- `heldout_bpb`: computed from `validation.json` loss sum and byte count
- `heldout_loss`: held-out mean loss
- `heldout_bytes`: total evaluated held-out bytes

System cost table fields:

- `wps`: `speed/wps`
- `iter_s`: `speed/curr_iter_time`
- `FLOPS`: `speed/FLOPS`
- `max_active_gib`: `memory/max_active_gib`
- `max_reserved_gib`: `memory/max_reserved_gib`
- `alloc_retries`: `memory/num_alloc_retries`
- `all_to_all_bytes`: `moe/ep_all_to_all_bytes_mean`
- `active_experts`: `moe/active_experts_mean`
- `max_load`: `moe/max_load_fraction_mean`
- `min_load`: `moe/min_load_fraction_mean`
- `expert6_load`: `moe/expert_6_load_fraction_mean`
- `expert7_load`: `moe/expert_7_load_fraction_mean`
- `load_imbalance`: `moe/load_imbalance_mean`

For paper claims, compare quality and system cost together. A run is not table-ready unless it has both a loadable checkpoint and a held-out `validation.json`.
