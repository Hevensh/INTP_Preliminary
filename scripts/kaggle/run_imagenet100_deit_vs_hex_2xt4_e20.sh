#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/ambityga/imagenet100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_HEX="${RUN_HEX:-1}"

run_experiment() {
  local config="$1"
  local name="$2"
  local run_dir="${OUTPUT_ROOT}/${name}"
  local summary="${run_dir}/summary.json"
  local checkpoint="${run_dir}/last.pt"
  local resume_args=()

  if [[ -f "${summary}" ]]; then
    echo "{\"event\":\"skip_complete\",\"experiment_name\":\"${name}\",\"summary\":\"${summary}\"}"
    return
  fi
  if [[ -f "${checkpoint}" ]]; then
    resume_args=(--resume "${checkpoint}")
    echo "{\"event\":\"resume\",\"experiment_name\":\"${name}\",\"checkpoint\":\"${checkpoint}\"}"
  fi

  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m experiments.imagenet100.train_vit \
    --config "${config}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --batch-size "${BATCH_SIZE}" \
    "${resume_args[@]}"
}

echo "{\"event\":\"suite_start\",\"gpus\":${NPROC_PER_NODE},\"per_device_batch_size\":${BATCH_SIZE},\"global_batch_size\":$((NPROC_PER_NODE * BATCH_SIZE))}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_experiment \
    configs/imagenet100/deit_tiny_ddp_e20.json \
    deit_tiny_imagenet100_ddp_e20
fi

if [[ "${RUN_HEX}" == "1" ]]; then
  run_experiment \
    configs/imagenet100/deit_tiny_hex_patch_ddp_e20.json \
    deit_tiny_hex_patch_imagenet100_ddp_e20
fi

echo "{\"event\":\"suite_complete\",\"output_root\":\"${OUTPUT_ROOT}\"}"
