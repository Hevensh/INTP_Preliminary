#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NAME="deit_tiny_gevit_p4_local_imagenet100_ddp_e20"
CONFIG="configs/imagenet100/deit_tiny_gevit_p4_local_ddp_e20.json"
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

echo "[run] GE-ViT p4 local | ${NPROC_PER_NODE} GPUs | batch ${BATCH_SIZE}/GPU | global batch $((NPROC_PER_NODE * BATCH_SIZE))"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m experiments.imagenet100.train_vit \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  "${RESUME_ARGS[@]}"

echo "[done] ${RUN_DIR}"
