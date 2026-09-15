#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=/home/dx/.cache/huggingface
export HF_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

baseline_for_seed() {
  case "$1" in
    42) echo "results/pasted_race/fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26/best_model.pth" ;;
    2026) echo "results/pasted_race/fourclass_baseline_official_seed2026_lr29e-6/2026-09-14_22-03-37/best_model.pth" ;;
    3407) echo "results/pasted_race/fourclass_baseline_official_seed3407_lr29e-6/2026-09-13_17-28-59/best_model.pth" ;;
    *) return 1 ;;
  esac
}

for mode in edu edu_root edu_root_interaction; do
  for seed in 42 2026 3407; do
    output="results/pasted_race/creator_retention_p3_${mode}_seed${seed}"
    if [[ -s "$output/metrics.json" && -s "$output/best_model.pt" ]]; then
      echo "Skipping completed $mode seed=$seed"
      continue
    fi
    conda run --no-capture-output -n race python train_creator_retention.py \
      --config configs/pasted_race/PASTED_RACE_creator_retention_p3.json \
      --input_mode "$mode" \
      --seed "$seed" \
      --baseline_checkpoint "$(baseline_for_seed "$seed")" \
      --output_dir "$output"
  done
done
