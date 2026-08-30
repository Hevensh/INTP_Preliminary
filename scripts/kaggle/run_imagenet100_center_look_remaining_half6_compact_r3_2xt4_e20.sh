#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"

run_experiment() {
  local config="$1"
  local name="$2"
  local description="$3"
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

  echo "[run] ${EPOCHS} epochs | ${description} | batch ${BATCH_SIZE}/GPU"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m experiments.imagenet100.train_vit \
    --config "${config}" \
    --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
    --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}" \
    "${resume_args[@]}"
}

run_experiment \
  configs/imagenet100/deit_tiny_rot_hex_harmonic_softmax_pe_center_look_half6_compact_r3_ddp_e20.json \
  deit_tiny_rot_hex_harmonic_softmax_pe_center_look_half6_compact_r3_imagenet100_ddp_e20 \
  "Hex half6d3r + null-softmax | PE + Center Look"

run_experiment \
  configs/imagenet100/deit_tiny_rot_hex_harmonic_softmax_center_look_half6_compact_r3_ddp_e20.json \
  deit_tiny_rot_hex_harmonic_softmax_center_look_half6_compact_r3_imagenet100_ddp_e20 \
  "Hex half6d3r + null-softmax | Center Look only"

echo "[suite done] outputs: ${OUTPUT_ROOT}"
