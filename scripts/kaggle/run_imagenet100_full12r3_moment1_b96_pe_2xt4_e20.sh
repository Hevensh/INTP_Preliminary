#!/usr/bin/env bash
set -euo pipefail
echo "[run] full12d3r | 96 bases | first moment ->192 | null-softmax + PE only | 20 epochs"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/rot_hex_full12r3_moment1_b96_pe_ddp_e20.json \
  --data-root "${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}" \
  --output-root "${OUTPUT_ROOT:-/kaggle/working/runs}" \
  --batch-size "${BATCH_SIZE:-256}"
