"""OpenCompass config template for BLT/PatchMoE downstream benchmarks.

This file intentionally defines only the model. Pass dataset config names through
OpenCompass CLI, for example:

    opencompass apps/opencompass/patchmoe_benchmarks.py \
      --datasets mmlu_ppl hellaswag_ppl ARC_c_ppl ARC_e_ppl \
      -w outputs/opencompass_patchmoe
"""

from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl import ARC_c_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_ppl import ARC_e_datasets
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl import (
        BoolQ_datasets,
    )
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl import (
        hellaswag_datasets,
    )
    from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    from opencompass.configs.datasets.obqa.obqa_ppl import obqa_datasets
    from opencompass.configs.datasets.piqa.piqa_ppl import piqa_datasets

from bytelatent.opencompass import ByteLatentOpenCompassModel


_os = __import__("os")
CKPT_DIR = _os.environ.get("PATCHMOE_CKPT_DIR", "")
if not CKPT_DIR:
    raise RuntimeError(
        "Set PATCHMOE_CKPT_DIR to a consolidated BLT/PatchMoE checkpoint."
    )

models = [
    dict(
        type=ByteLatentOpenCompassModel,
        abbr=_os.environ.get("PATCHMOE_OC_ABBR", "patchmoe"),
        ckpt_dir=CKPT_DIR,
        max_seq_len=int(_os.environ.get("PATCHMOE_OC_MAX_SEQ_LEN", "2048")),
        max_out_len=int(_os.environ.get("PATCHMOE_OC_MAX_OUT_LEN", "128")),
        batch_size=int(_os.environ.get("PATCHMOE_OC_BATCH_SIZE", "1")),
        consolidate_if_needed=_os.environ.get("PATCHMOE_OC_CONSOLIDATE", "1") != "0",
        entropy_ckpt_dir=_os.environ.get("PATCHMOE_ENTROPY_CKPT_DIR") or None,
        entropy_state_dict_path=_os.environ.get("PATCHMOE_ENTROPY_STATE_DICT_PATH") or None,
        entropy_attn_impl=_os.environ.get("PATCHMOE_ENTROPY_ATTN_IMPL", "sdpa"),
        generator_kwargs=dict(
            dtype=_os.environ.get("PATCHMOE_OC_DTYPE", "bf16"),
            temperature=float(_os.environ.get("PATCHMOE_OC_TEMPERATURE", "0.0")),
            max_tokens=int(_os.environ.get("PATCHMOE_OC_MAX_TOKENS", "2176")),
            max_prompt_len=int(_os.environ.get("PATCHMOE_OC_MAX_PROMPT_LEN", "2048")),
        ),
        run_cfg=dict(
            num_gpus=int(_os.environ.get("PATCHMOE_OC_NUM_GPUS", "1")),
            num_procs=int(_os.environ.get("PATCHMOE_OC_NUM_PROCS", "1")),
        ),
    )
]

_DEFAULT_DATASETS = (
    "mmlu_ppl hellaswag_ppl ARC_c_ppl ARC_e_ppl "
    "obqa_ppl piqa_ppl SuperGLUE_BoolQ_ppl"
)
_DATASET_GROUPS = dict(
    mmlu_ppl=mmlu_datasets,
    hellaswag_ppl=hellaswag_datasets,
    ARC_c_ppl=ARC_c_datasets,
    ARC_e_ppl=ARC_e_datasets,
    obqa_ppl=obqa_datasets,
    piqa_ppl=piqa_datasets,
    SuperGLUE_BoolQ_ppl=BoolQ_datasets,
)
_DATASET_NAMES = _os.environ.get("PATCHMOE_OC_DATASETS", _DEFAULT_DATASETS).split()
_TEST_RANGE = _os.environ.get("PATCHMOE_OC_TEST_RANGE", "").strip()
_UNKNOWN_DATASETS = [name for name in _DATASET_NAMES if name not in _DATASET_GROUPS]
if _UNKNOWN_DATASETS:
    raise RuntimeError(
        "Unsupported PATCHMOE_OC_DATASETS entries: " + ", ".join(_UNKNOWN_DATASETS)
    )

datasets = []
_dataset_name = None
_dataset_cfg = None
for _dataset_name in _DATASET_NAMES:
    for _dataset_cfg in _DATASET_GROUPS[_dataset_name]:
        if _TEST_RANGE:
            _dataset_cfg = dict(_dataset_cfg)
            _dataset_cfg["reader_cfg"] = dict(_dataset_cfg.get("reader_cfg", {}))
            _dataset_cfg["reader_cfg"]["test_range"] = _TEST_RANGE
        datasets.append(_dataset_cfg)

del (
    _os,
    _DEFAULT_DATASETS,
    _DATASET_GROUPS,
    _DATASET_NAMES,
    _TEST_RANGE,
    _UNKNOWN_DATASETS,
    _dataset_name,
    _dataset_cfg,
    ARC_c_datasets,
    ARC_e_datasets,
    BoolQ_datasets,
    hellaswag_datasets,
    mmlu_datasets,
    obqa_datasets,
    piqa_datasets,
)
