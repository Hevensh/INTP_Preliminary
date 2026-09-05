#!/usr/bin/env bash
set -euo pipefail
# Same differentiated G3 baseline; new independent probe count ablation only.
# Usage: bash scripts/kaggle/run_imagenet100_dual_look_multiprobe_2xt4.sh 1 4
# or:    bash scripts/kaggle/run_imagenet100_dual_look_multiprobe_2xt4.sh 4 4
IMAGE_PROBES="${1:-4}"
FEATURE_PROBES="${2:-4}"
G="${G:-3}"
EPOCHS="${EPOCHS:-20}"
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if (( EPOCHS < 7 )); then
  echo "[error] this differentiated experiment needs >=7 epochs (splits at 3/5/7)" >&2
  exit 2
fi
for value in "$IMAGE_PROBES" "$FEATURE_PROBES" "$G"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "probe counts and G must be positive integers" >&2; exit 2; }
done
NAME="deit_tiny_pe_dual_look_rotW_im${IMAGE_PROBES}_fm${FEATURE_PROBES}_G${G}_diff357_half6_r3_e${EPOCHS}"
RESUME=()
if [[ -f "${OUTPUT_ROOT}/${NAME}/summary.json" ]]; then
  echo "[skip] ${NAME} already complete"; exit 0
fi
if [[ -f "${OUTPUT_ROOT}/${NAME}/last.pt" ]]; then
  RESUME=(--resume "${OUTPUT_ROOT}/${NAME}/last.pt")
fi
echo "[run] Image M=${IMAGE_PROBES} | Feature M=${FEATURE_PROBES}, rotating W | G=${G} | ${EPOCHS} epochs | batch ${BATCH_SIZE}/GPU"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/deit_tiny_rot_hex_pe_dual_look_diff357_half6_compact_r3_ddp_e20.json \
  --experiment-name "$NAME" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
  --center-look-layers-per-probe "$G" \
  --image-look-probes "$IMAGE_PROBES" --feature-look-probes "$FEATURE_PROBES" \
  --feature-look-rotating-probes "${RESUME[@]}"
