#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"

python -u -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/deit_tiny_smoke.json \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}"
