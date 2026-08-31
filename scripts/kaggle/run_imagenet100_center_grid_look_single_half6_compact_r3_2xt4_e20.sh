#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^(1|2|3|4|6)$ ]]; then
  echo "usage: $0 {1|2|3|4|6}" >&2
  exit 2
fi

LAYERS_PER_PROBE="$1"
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"

stem="deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share${LAYERS_PER_PROBE}l_half6_compact_r3"
name="${stem}_imagenet100_ddp_e20"
run_dir="${OUTPUT_ROOT}/${name}"
resume_args=()

if [[ -f "${run_dir}/summary.json" ]]; then
  echo "[skip] ${name} is already complete"
  exit 0
fi
if [[ -f "${run_dir}/last.pt" ]]; then
  resume_args=(--resume "${run_dir}/last.pt")
  echo "[resume] ${name}"
fi

echo "[run] ${EPOCHS} epochs | PE + Center Grid Look | one probe reused by ${LAYERS_PER_PROBE} layer(s), independent 4x12 grid per layer | batch ${BATCH_SIZE}/GPU"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m experiments.imagenet100.train_vit \
  --config "configs/imagenet100/${stem}_ddp_e20.json" \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}" "${resume_args[@]}"
