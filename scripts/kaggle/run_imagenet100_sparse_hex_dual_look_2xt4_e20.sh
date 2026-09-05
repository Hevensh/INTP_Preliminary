#!/usr/bin/env bash
set -euo pipefail
# Last argument is Feature Look sharing span G, not center spacing.
# Example: !bash scripts/kaggle/run_imagenet100_sparse_hex_dual_look_2xt4_e20.sh 3
export G="${1:-3}"
export EPOCHS=20
export FEATURE_MODE=independent
export SPARSE_HEX_LOOK=1
echo "[layout] sparse Hex centers | K12 inner6 / K24 outer12 | Feature all18 | no Look interpolation"
echo "[attention] single SDPA | temporary dense bias | no separate FP32 branch"
bash scripts/kaggle/run_imagenet100_dual_look_multiprobe_2xt4.sh 4 4
