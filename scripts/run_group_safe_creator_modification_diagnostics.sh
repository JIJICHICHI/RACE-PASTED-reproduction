#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

latest_formal_run() {
  local seed="$1"
  local parent="results/pasted_race/fourclass_baseline_official_seed${seed}_lr29e-6"
  local run
  run="$(find "${parent}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [[ -z "${run}" || ! -f "${run}/best_model.pth" || ! -f "${run}/test_predictions.json" ]]; then
    echo "Missing completed formal Strong RACE run for seed ${seed}" >&2
    return 1
  fi
  printf '%s' "${run}"
}

run42="$(latest_formal_run 42)"
run2026="$(latest_formal_run 2026)"
run3407="$(latest_formal_run 3407)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/hub

conda run --no-capture-output -n race python \
  scripts/diagnose_group_safe_creator_modification.py \
  --run "42=${run42}" \
  --run "2026=${run2026}" \
  --run "3407=${run3407}" \
  --output-dir results/diagnostics/group_safe_creator_modification \
  --device cuda \
  --batch-size 32 \
  --num-workers 4
