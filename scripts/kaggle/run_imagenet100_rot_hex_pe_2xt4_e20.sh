#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NAME="deit_tiny_rot_hex_pe_imagenet100_ddp_e20"
RUN_DIR="${OUTPUT_ROOT}/${NAME}"
SUMMARY="${RUN_DIR}/summary.json"
CHECKPOINT="${RUN_DIR}/last.pt"
RESUME_ARGS=()

if [[ -f "${SUMMARY}" ]]; then
  echo "[skip] ${NAME} is already complete"
  exit 0
fi
if [[ -f "${CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume "${CHECKPOINT}")
  echo "[resume] ${NAME} from ${CHECKPOINT}"
fi

echo "[run] Rot-Hex Full-4 + learned PE | 24/12 | ${NPROC_PER_NODE} GPUs | batch ${BATCH_SIZE}/GPU"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/deit_tiny_rot_hex_pe_ddp_e20.json \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  "${RESUME_ARGS[@]}"

echo "[done] ${RUN_DIR}"
