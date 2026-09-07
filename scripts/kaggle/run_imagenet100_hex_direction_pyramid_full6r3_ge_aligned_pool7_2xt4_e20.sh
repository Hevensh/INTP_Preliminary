#!/usr/bin/env bash
set -euo pipefail
echo "[run] full6d3r 0/60/120/180/240/300 | 144 raw prototypes, no input projection | GE-aligned GN / relative PE / scale / init / readout; retained-center + 6 neighbors max pooling | pyramid 144/288/336 | 20 epochs | batch ${BATCH_SIZE:-256}/GPU"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/hex_direction_pyramid_full6r3_ge_aligned_pool7_ddp_e20.json \
  --data-root "${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}" \
  --output-root "${OUTPUT_ROOT:-/kaggle/working/runs}" \
  --batch-size "${BATCH_SIZE:-256}"
