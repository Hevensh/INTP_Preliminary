#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-downloads/kaggle/imagenet100-val}"

"${PYTHON_BIN}" -u scripts/kaggle/download_imagenet100_val.py \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
