#!/usr/bin/env bash
set -euo pipefail
# Independent twenty-epoch run; never resume the five-epoch smoke checkpoint.
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NAME="deit_tiny_rot_hex_harmonic_softmax_pe_half3_compact_r3_imagenet100_ddp_e20"
RUN_DIR="${OUTPUT_ROOT}/${NAME}"
RESUME_ARGS=()
if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "[skip] ${NAME} is already complete"
  exit 0
fi
if [[ -f "${RUN_DIR}/last.pt" ]]; then
  RESUME_ARGS=(--resume "${RUN_DIR}/last.pt")
fi
echo "[run] 20 epochs | half3d3r + null-softmax | PE only | no differentiation | batch ${BATCH_SIZE}/GPU"
echo "[poses] 3 half-circle directions | global grid 6 | K24/K12 | 6 real poses + null"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/deit_tiny_rot_hex_harmonic_softmax_pe_half3_compact_r3_ddp_e20.json \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" "${RESUME_ARGS[@]}"
