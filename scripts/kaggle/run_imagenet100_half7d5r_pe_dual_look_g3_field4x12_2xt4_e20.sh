#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NAME="half7d5r_pe_dual_look_g3_field4x12_imagenet100_ddp_e20"
RUN_DIR="${OUTPUT_ROOT}/${NAME}"
RESUME_ARGS=()
if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "[skip] ${NAME} is already complete"
  exit 0
fi
if [[ -f "${RUN_DIR}/last.pt" ]]; then
  RESUME_ARGS=(--resume "${RUN_DIR}/last.pt")
fi
echo "[run] ${EPOCHS:-20} epochs | half7d5r | PE + Image Look + Feature Look G3 | batch ${BATCH_SIZE:-256}/GPU"
echo "[look] both fields 4x12 | single probes | all tokens | no differentiation"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/half7d5r_pe_dual_look_g3_field4x12_ddp_e20.json \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE:-256}" --epochs "${EPOCHS:-20}" "${RESUME_ARGS[@]}"
