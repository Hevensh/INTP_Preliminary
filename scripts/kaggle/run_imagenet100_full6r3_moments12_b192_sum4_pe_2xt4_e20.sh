#!/usr/bin/env bash
set -euo pipefail
echo "[run] full6d3r 0/60/120/180/240/300 | PE only, 192 bases, moments1+2, four-group sum | 20 epochs | batch ${BATCH_SIZE:-256}/GPU"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/rot_hex_full6r3_moments12_b192_sum4_pe_ddp_e20.json \
  --data-root "${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}" \
  --output-root "${OUTPUT_ROOT:-/kaggle/working/runs}" \
  --batch-size "${BATCH_SIZE:-256}"
