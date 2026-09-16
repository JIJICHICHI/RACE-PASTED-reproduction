#!/usr/bin/env bash
set -euo pipefail

# HISTORICAL ONLY: fixed seed-42 trace-head initialization control.
# The canonical experiment is train_pasted_race_trace_end_to_end_multiseed.sh.
if [[ "${ALLOW_HISTORICAL_FIXED_TRACE:-0}" != "1" ]]; then
  echo "Historical fixed-trace control. Use scripts/train_pasted_race_trace_end_to_end_multiseed.sh instead." >&2
  echo "Set ALLOW_HISTORICAL_FIXED_TRACE=1 only to reproduce the archived control." >&2
  exit 2
fi

cd "$(dirname "$0")/.."

export HF_HOME="/home/dx/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/dx/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/home/dx/.cache/huggingface/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

baseline_checkpoint() {
  case "$1" in
    42) echo "results/pasted_race/fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26/best_model.pth" ;;
    2026) echo "results/pasted_race/fourclass_baseline_official_seed2026_lr29e-6/2026-09-14_22-03-37/best_model.pth" ;;
    3407) echo "results/pasted_race/fourclass_baseline_official_seed3407_lr29e-6/2026-09-13_17-28-59/best_model.pth" ;;
    *) echo "Unsupported seed: $1" >&2; return 1 ;;
  esac
}

run_one() {
  local variant="$1"
  local seed="$2"
  local config="configs/pasted_race/PASTED_RACE_fourclass_${variant}_official.json"
  local output="results/pasted_race/fourclass_${variant}_official_seed${seed}"
  local baseline
  baseline="$(baseline_checkpoint "$seed")"

  if [[ -f "$output/metrics.json" ]]; then
    echo "Skipping completed run: $output"
    return
  fi

  conda run --no-capture-output -n race python train_pasted_race_fourclass.py \
    --config "$config" \
    --seed "$seed" \
    --baseline_checkpoint "$baseline" \
    --output_dir "$output"
}

for variant in single dual; do
  for seed in 42 2026 3407; do
    run_one "$variant" "$seed"
  done
done
