#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data1/pengfeigao/BP-MoE

RUN_NAME=router_ablation_entropy_only_A5k_B15k_6gpu_c00000_c00005

INIT_CKPT_CANDIDATES=(
  "${PROJECT_ROOT}/initial_weights/blt_1b_patchmoe_entropy_bands_paired_dcp"
  "${PROJECT_ROOT}/initial_weights/blt_1b_patchmoe_entropy_bands_dcp"
  "${PROJECT_ROOT}/blt1b_warmstart/initial_weights/blt_1b_patchmoe_entropy_bands_paired_dcp"
  "${PROJECT_ROOT}/blt1b_warmstart/initial_weights/blt_1b_patchmoe_entropy_bands_dcp"
)

MODEL_OVERRIDES=(
  "model.moe_router_use_hidden_state=false"
  "model.moe_router_patch_feature_bias=true"
  "model.moe_router_normalize_patch_entropy=false"
  "model.moe_router_use_patch_entropy=true"
  "model.moe_router_use_patch_length=false"
  "model.moe_router_use_patch_byte_features=false"
)

source "${PROJECT_ROOT}/scripts/train_router_ablation_A5k_B15k_common.sh"
