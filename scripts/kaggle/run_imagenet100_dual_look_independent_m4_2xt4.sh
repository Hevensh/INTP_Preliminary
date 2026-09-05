#!/usr/bin/env bash
set -euo pipefail
# Image M4 / Feature M4, independent W per direction; same G3/diff357/e20.
export FEATURE_MODE=independent
bash scripts/kaggle/run_imagenet100_dual_look_multiprobe_2xt4.sh 4 4
