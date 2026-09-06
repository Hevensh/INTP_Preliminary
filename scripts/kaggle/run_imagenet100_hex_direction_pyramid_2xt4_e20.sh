#!/usr/bin/env bash
set -euo pipefail
echo "[run] 20 epochs | Hex directions 6 | stages 96/192/256 | tokens 195/52/14 | batch ${BATCH_SIZE:-256}/GPU"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/hex_direction_pyramid_ddp_e20.json \
  --data-root "${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}" \
  --output-root "${OUTPUT_ROOT:-/kaggle/working/runs}" \
  --batch-size "${BATCH_SIZE:-256}"
