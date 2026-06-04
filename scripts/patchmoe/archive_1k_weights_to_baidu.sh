#!/usr/bin/env bash
set -euo pipefail

BAIDUPCS_BIN="${BAIDUPCS_BIN:-/home/yuzhang/code/BaiduPCS-Go/BaiduPCS-Go}"
REMOTE_BASE="${REMOTE_BASE:-/BP-MoE-runs/blt1b_warmstart/archived_1k_20260603}"
PART_ROOT="${PART_ROOT:-/data2/BP-MoE-upload-parts/archive_1k_weights}"
PART_SIZE="${PART_SIZE:-20G}"
UPLOAD_PARALLEL="${UPLOAD_PARALLEL:-4}"
UPLOAD_LOAD="${UPLOAD_LOAD:-1}"
UPLOAD_RETRY="${UPLOAD_RETRY:-5}"

RUNS=(
  "/data2/BP-MoE-runs/blt1b_warmstart/dense_blt1b_1000step_lrmatch"
  "/data2/BP-MoE-runs/blt1b_warmstart/hidden_only_blt1b_1000step_lrmatch"
  "/data2/BP-MoE-runs/blt1b_warmstart/patchmoe_blt1b_warmstart_ep2_1000step"
)

mkdir_remote_path() {
  local path="$1"
  local cur=""
  IFS='/' read -ra parts <<<"${path#/}"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    cur="$cur/$part"
    "$BAIDUPCS_BIN" mkdir "$cur" >/dev/null 2>&1 || true
  done
}

upload_one() {
  local src="$1"
  local dst="$2"
  "$BAIDUPCS_BIN" upload -p "$UPLOAD_PARALLEL" -l "$UPLOAD_LOAD" --retry "$UPLOAD_RETRY" "$src" "$dst"
}

write_manifest_for_file() {
  local file="$1"
  local manifest="$2"
  {
    echo "file=$file"
    stat --printf='size_bytes=%s\nmtime=%y\n' "$file"
    echo "split_part_size=$PART_SIZE"
    echo "created_at=$(date --iso-8601=seconds)"
  } >"$manifest"
}

archive_run() {
  local run_dir="$1"
  local run_name
  run_name="$(basename "$run_dir")"
  local remote_run="$REMOTE_BASE/$run_name"
  local work_dir="$PART_ROOT/$run_name"

  if [[ ! -d "$run_dir" ]]; then
    echo "[$(date --iso-8601=seconds)] skip missing $run_dir"
    return 0
  fi

  echo "[$(date --iso-8601=seconds)] archiving $run_name"
  mkdir -p "$work_dir"
  mkdir_remote_path "$remote_run"

  local metadata_tar="$work_dir/${run_name}.metadata.tar.gz"
  tar \
    --exclude='./checkpoints/0000001000/*.distcp' \
    --exclude='./checkpoints/0000001000/consolidated' \
    -czf "$metadata_tar" \
    -C "$run_dir" .
  upload_one "$metadata_tar" "$remote_run"
  rm -f "$metadata_tar"

  local ckpt_dir="$run_dir/checkpoints/0000001000"
  local remote_ckpt="$remote_run/checkpoints/0000001000"
  mkdir_remote_path "$remote_ckpt"

  local file
  for file in "$ckpt_dir"/*.distcp; do
    [[ -e "$file" ]] || continue
    local base part_dir manifest remote_parts
    base="$(basename "$file")"
    part_dir="$work_dir/${base}.parts"
    manifest="$part_dir/${base}.manifest.txt"
    remote_parts="$remote_ckpt/${base}.parts"

    rm -rf "$part_dir"
    mkdir -p "$part_dir"
    write_manifest_for_file "$file" "$manifest"
    split -b "$PART_SIZE" -d -a 3 "$file" "$part_dir/${base}.part."
    mkdir_remote_path "$remote_parts"
    upload_one "$part_dir" "$remote_ckpt"
    rm -rf "$part_dir"
    rm -f "$file"
    echo "[$(date --iso-8601=seconds)] uploaded and removed $file"
  done

  rm -rf "$ckpt_dir/consolidated"
  rm -rf "$run_dir"
  echo "[$(date --iso-8601=seconds)] local run removed: $run_dir"
}

mkdir -p "$PART_ROOT"
mkdir_remote_path "$REMOTE_BASE"

for run_dir in "${RUNS[@]}"; do
  archive_run "$run_dir"
done

rmdir "$PART_ROOT" 2>/dev/null || true
echo "[$(date --iso-8601=seconds)] archive done"
