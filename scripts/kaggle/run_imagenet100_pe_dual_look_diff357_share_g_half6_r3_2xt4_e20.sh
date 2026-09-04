#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"
BASE_CONFIG="configs/imagenet100/deit_tiny_rot_hex_pe_dual_look_diff357_half6_compact_r3_ddp_e20.json"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 G [G ...]"
  echo "G must be one of: 1 2 3 4 6 12"
  exit 2
fi

run_share() {
  local layers_per_probe="$1"
  case "${layers_per_probe}" in
    1|2|3|4|6|12) ;;
    *)
      echo "[error] unsupported G=${layers_per_probe}; choose from 1 2 3 4 6 12" >&2
      return 2
      ;;
  esac

  local name="deit_tiny_rot_hex_pe_dual_look_diff357_share${layers_per_probe}l_half6_compact_r3_imagenet100_ddp_e${EPOCHS}"
  local run_dir="${OUTPUT_ROOT}/${name}"
  local resume_args=()

  if [[ -f "${run_dir}/summary.json" ]]; then
    echo "[skip] ${name} is already complete"
    return
  fi
  if [[ -f "${run_dir}/last.pt" ]]; then
    resume_args=(--resume "${run_dir}/last.pt")
    echo "[resume] ${name}"
  fi

  echo "[run] ${EPOCHS} epochs | dual Look + differentiation 3/5/7 | G=${layers_per_probe} | batch ${BATCH_SIZE}/GPU"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m experiments.imagenet100.train_vit \
    --config "${BASE_CONFIG}" \
    --experiment-name "${name}" \
    --center-look-layers-per-probe "${layers_per_probe}" \
    --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
    --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}" "${resume_args[@]}"
}

for layers_per_probe in "$@"; do
  run_share "${layers_per_probe}"
done

echo "[suite done] outputs: ${OUTPUT_ROOT}"
