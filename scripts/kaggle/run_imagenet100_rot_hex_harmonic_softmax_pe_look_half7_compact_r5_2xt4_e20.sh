#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"
NAME="deit_tiny_rot_hex_harmonic_softmax_pe_look_half7_compact_r5_imagenet100_ddp_e20"
RUN_DIR="${OUTPUT_ROOT}/${NAME}"
RESUME_ARGS=()

if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "[skip] ${NAME} is already complete"
  exit 0
fi
if [[ -f "${RUN_DIR}/last.pt" ]]; then
  RESUME_ARGS=(--resume "${RUN_DIR}/last.pt")
fi

echo "[run] 20 epochs | compact r5 half7/14 null-softmax | PE + Look | batch ${BATCH_SIZE}/GPU"
echo "[look] Image Look | probe r5, 7 angles, K24/K12 | field rings 5/10/15/20 | no differentiation"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/deit_tiny_rot_hex_harmonic_softmax_pe_look_half7_compact_r5_ddp_e20.json \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}" "${RESUME_ARGS[@]}"
