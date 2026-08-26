#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
NAME="deit_tiny_rot_hex_harmonic_pe_look_imagenet100_ddp_e20"
RUN_DIR="${OUTPUT_ROOT}/${NAME}"
RESUME_ARGS=()

if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "[skip] ${NAME} is already complete"
  exit 0
fi
if [[ -f "${RUN_DIR}/last.pt" ]]; then
  RESUME_ARGS=(--resume "${RUN_DIR}/last.pt")
fi

echo "[run] Rot-Hex dot cos/sin | PE + Look | batch ${BATCH_SIZE}/GPU"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/deit_tiny_rot_hex_harmonic_pe_look_ddp_e20.json \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}" "${RESUME_ARGS[@]}"
