#!/usr/bin/env bash
set -euo pipefail
echo "[run] 5 epochs | raw six-direction Hex | 96/direction | 12 layers | local 19x6 | batch ${BATCH_SIZE:-256}/GPU"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/hex_direction_local96_ddp_e5.json \
  --data-root "${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}" \
  --output-root "${OUTPUT_ROOT:-/kaggle/working/runs}" \
  --batch-size "${BATCH_SIZE:-256}"
