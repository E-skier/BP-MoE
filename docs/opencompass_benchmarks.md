# OpenCompass Benchmark Evaluation

This project keeps BPB/loss, routing, and systems metrics in native PatchMoE
evaluation scripts. OpenCompass is added as an optional downstream benchmark
path for paper-style tables such as MMLU, BoolQ, OpenBookQA, ARC, HellaSwag,
and PIQA.

## What This Integration Provides

- `bytelatent.opencompass.ByteLatentOpenCompassModel`: an OpenCompass custom
  model adapter for BLT/PatchMoE checkpoints.
- `apps/opencompass/patchmoe_benchmarks.py`: reusable OpenCompass model config.
- `scripts/patchmoe/run_opencompass_benchmarks.sh`: launcher for benchmark runs.
- `scripts/patchmoe/preflight_opencompass_benchmarks.py`: checks the
  OpenCompass environment, config, checkpoint layout, and required local data
  files before GPU inference.
- `scripts/patchmoe/prepare_opencompass_extra_data.py`: prepares OpenBookQA,
  PIQA, and BoolQ files that OpenCompass 0.5.x does not auto-download reliably.
- `scripts/patchmoe/summarize_opencompass_results.py`: converts OpenCompass
  `summary/*.csv` outputs into long-form results and a paper-table CSV.

The adapter implements the OpenCompass custom model surface: `generate`,
`get_ppl`, `get_ppl_tokenwise`, and `get_token_len`. When OpenCompass is
installed, it also registers `ByteLatentOpenCompassModel` with the OpenCompass
`MODELS` registry. It uses the native
BLT/PatchMoE checkpoint loader and `PackedCausalTransformerGenerator`, so
benchmark inference uses the same model path as `bytelatent.eval`.

OpenCompass calls the discriminative multiple-choice path `get_ppl`, but its
HuggingFace backend returns mean cross-entropy loss rather than `exp(loss)`.
`ByteLatentOpenCompassModel` follows that convention so lower-is-better PPL
ranking is consistent with OpenCompass' built-in models.

## Install OpenCompass

OpenCompass is optional and is not added to the core project dependencies. Install
it in the environment used for benchmark inference:

```bash
./venv/bin/python -m pip install -U opencompass
./venv/bin/python -m pip install 'transformers==4.51.3'
./venv/bin/python -m pip uninstall -y torchvision
```

The local `./venv` benchmark environment has been validated with
`opencompass==0.5.2`, `mmengine==0.10.7`, `transformers==4.51.3`, and the
repository's existing torch build. `torchvision` was removed from that venv
because the installed build was incompatible with the local torch and blocked
OpenCompass text-only startup.

For full dataset coverage, follow OpenCompass data preparation instructions and
make sure the desired dataset configs are available. OpenCompass 0.5 documents
three evaluation stages: `infer`, `eval`, and `viz`; use `-m all` for a complete
run or split the stages when debugging.

## Run A Benchmark

For a consolidated checkpoint or a DCP checkpoint that can be consolidated:

```bash
CKPT_DIR=/path/to/checkpoint \
PYTHON_BIN=./venv/bin/python \
OPENCOMPASS_BIN=./venv/bin/opencompass \
RUN_NAME=entropy_only_blt1b \
DATASETS="mmlu_ppl hellaswag_ppl ARC_c_ppl ARC_e_ppl obqa_ppl piqa_ppl SuperGLUE_BoolQ_ppl" \
scripts/patchmoe/run_opencompass_benchmarks.sh
```

Before running the full suite for the first time, prepare the extra local
data files required by OpenCompass 0.5.x:

```bash
./venv/bin/python scripts/patchmoe/prepare_opencompass_extra_data.py
```

The launcher runs a preflight check by default. It verifies the repository
adapter import, OpenCompass `MODELS` registry visibility when available, config
path, checkpoint layout, OpenCompass package/CLI, and can optionally check
dataset names through an OpenCompass source checkout.

Useful controls:

```bash
# Smoke test command construction without running inference.
SKIP_PREFLIGHT=1 DRY_RUN=1 scripts/patchmoe/run_opencompass_benchmarks.sh /path/to/checkpoint

# Run a tiny real-inference smoke test on the first example of one dataset.
PATCHMOE_OC_TEST_RANGE="[0:1]" \
DATASETS="piqa_ppl" \
RUN_NAME=blt1b_piqa_smoke \
scripts/patchmoe/run_opencompass_benchmarks.sh /path/to/checkpoint

# Run only the preflight check.
./venv/bin/python scripts/patchmoe/preflight_opencompass_benchmarks.py /path/to/checkpoint \
  --opencompass-bin ./venv/bin/opencompass

# Check dataset config names against an OpenCompass source checkout.
OPENCOMPASS_ROOT=/path/to/opencompass \
./venv/bin/python scripts/patchmoe/preflight_opencompass_benchmarks.py /path/to/checkpoint \
  --opencompass-bin ./venv/bin/opencompass \
  --opencompass-root /path/to/opencompass

# Use a different OpenCompass entry point, for example a source checkout.
OPENCOMPASS_BIN="python /path/to/opencompass/run.py" \
scripts/patchmoe/run_opencompass_benchmarks.sh /path/to/checkpoint

# Override model-side limits.
PATCHMOE_OC_MAX_SEQ_LEN=4096 \
PATCHMOE_OC_MAX_OUT_LEN=128 \
PATCHMOE_OC_MAX_TOKENS=4224 \
scripts/patchmoe/run_opencompass_benchmarks.sh /path/to/checkpoint

# Override the entropy model used by BLT entropy patching.
PATCHMOE_ENTROPY_CKPT_DIR=/path/to/entropy_model \
PATCHMOE_ENTROPY_STATE_DICT_PATH=/path/to/entropy_model/consolidated.pth \
PATCHMOE_ENTROPY_ATTN_IMPL=sdpa \
scripts/patchmoe/run_opencompass_benchmarks.sh /path/to/checkpoint
```

If a dataset name fails, list available configs in the OpenCompass environment:

```bash
python /path/to/opencompass/tools/list_configs.py mmlu ARC hellaswag openbookqa BoolQ
```

Then pass the exact names through `DATASETS`.

## Summarize Results

OpenCompass writes CSV/TXT summaries under each run's `summary/` directory, often
under a timestamped run folder or a `latest` symlink. Merge one or more runs into
paper-ready CSV files with:

```bash
python3 scripts/patchmoe/summarize_opencompass_results.py \
  runs/opencompass/entropy_only_blt1b \
  runs/opencompass/hidden_only_blt1b \
  runs/opencompass/blt1b_baseline \
  --output-dir runs/opencompass/paper_summary
```

The script writes:

- `opencompass_long.csv`: one row per method/dataset/metric.
- `opencompass_paper_table.csv`: method rows with MMLU, BoolQ, OpenBookQA,
  ARC-E, ARC-C, HellaSwag, PIQA, and average columns.

If OpenCompass does not emit an aggregate MMLU row, the script averages
`lukaemon_mmlu_*` subject rows as a fallback.

## Paper Usage

Use OpenCompass for the downstream benchmark table only. The main PatchMoE claim
still depends on native held-out BPB, active FLOPs/byte, expert utilization,
routing specialization, and throughput/memory/dispatch profiling.
