#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-/kaggle/working/runs}"
OUTPUT="${OUTPUT:-/kaggle/working/intp_run_artifacts_no_weights.zip}"

python scripts/kaggle/package_run_artifacts.py \
  --runs-root "${RUNS_ROOT}" \
  --output "${OUTPUT}"
