#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RUN_EQUI_GMR="${RUN_EQUI_GMR:-1}"
RUN_ARC="${RUN_ARC:-1}"

run_experiment() {
  local config="$1"
  local name="$2"
  local run_dir="${OUTPUT_ROOT}/${name}"
  local summary="${run_dir}/summary.json"
  local checkpoint="${run_dir}/last.pt"
  local resume_args=()

  if [[ -f "${summary}" ]]; then
    echo "[skip] ${name} is already complete"
    return
  fi
  if [[ -f "${checkpoint}" ]]; then
    resume_args=(--resume "${checkpoint}")
    echo "[resume] ${name} from ${checkpoint}"
  fi

  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m experiments.imagenet100.train_vit \
    --config "${config}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --batch-size "${BATCH_SIZE}" \
    "${resume_args[@]}"
}

echo "[suite] ${NPROC_PER_NODE} GPUs | batch ${BATCH_SIZE}/GPU | global batch $((NPROC_PER_NODE * BATCH_SIZE))"

if [[ "${RUN_EQUI_GMR}" == "1" ]]; then
  run_experiment \
    configs/imagenet100/deit_tiny_equi_gmr_pe_ddp_e20.json \
    deit_tiny_equi_gmr_pe_imagenet100_ddp_e20
fi

if [[ "${RUN_ARC}" == "1" ]]; then
  run_experiment \
    configs/imagenet100/deit_tiny_arc_adaptive_pe_ddp_e20.json \
    deit_tiny_arc_adaptive_pe_imagenet100_ddp_e20
fi

echo "[suite done] outputs: ${OUTPUT_ROOT}"
