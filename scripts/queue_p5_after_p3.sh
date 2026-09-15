#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=/home/dx/.cache/huggingface
export HF_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

for mode in edu edu_root edu_root_interaction; do
  for seed in 42 2026 3407; do
    target="results/pasted_race/creator_retention_p3_${mode}_seed${seed}/metrics.json"
    while [[ ! -s "$target" ]]; do
      sleep 30
    done
  done
done

selected_mode="$(conda run -n race python utils/select_creator_p3_mode.py \
  --results_dir results/pasted_race \
  --output results/pasted_race/creator_retention_p3_selection.json)"

conda run --no-capture-output -n race python utils/merge_creator_retention_labels.py \
  --base_dir data/pasted_race_fourclass_dual \
  --edited_dir data/pasted_race_creator_retention_signal \
  --output_dir data/pasted_race_creator_editor

conda run --no-capture-output -n race python train_pasted_race_fourclass.py \
  --config configs/pasted_race/PASTED_RACE_fourclass_creator_editor_official.json \
  --seed 42 \
  --baseline_checkpoint "results/pasted_race/fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26/best_model.pth" \
  --lexical_checkpoint "results/pasted_race/lexical_official_seed42/best_model.pt" \
  --humanization_checkpoint "results/pasted_race/humanization_lexical_official_seed42/best_model.pt" \
  --creator_checkpoint "results/pasted_race/creator_retention_p3_${selected_mode}_seed42/best_model.pt" \
  --creator_input_mode "$selected_mode" \
  --output_dir /tmp/p5_creator_editor_effective_smoke \
  --epochs 1 \
  --max_train_samples 200 \
  --max_eval_samples 100

baseline_for_seed() {
  case "$1" in
    42) echo "results/pasted_race/fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26/best_model.pth" ;;
    2026) echo "results/pasted_race/fourclass_baseline_official_seed2026_lr29e-6/2026-09-14_22-03-37/best_model.pth" ;;
    3407) echo "results/pasted_race/fourclass_baseline_official_seed3407_lr29e-6/2026-09-13_17-28-59/best_model.pth" ;;
  esac
}

for seed in 42 2026 3407; do
  output="results/pasted_race/fourclass_creator_editor_official_seed${seed}"
  if [[ -s "$output/metrics.json" && -s "$output/best_model.pt" ]]; then
    continue
  fi
  conda run --no-capture-output -n race python train_pasted_race_fourclass.py \
    --config configs/pasted_race/PASTED_RACE_fourclass_creator_editor_official.json \
    --seed "$seed" \
    --baseline_checkpoint "$(baseline_for_seed "$seed")" \
    --lexical_checkpoint "results/pasted_race/lexical_official_seed${seed}/best_model.pt" \
    --humanization_checkpoint "results/pasted_race/humanization_lexical_official_seed${seed}/best_model.pt" \
    --creator_checkpoint "results/pasted_race/creator_retention_p3_${selected_mode}_seed${seed}/best_model.pt" \
    --creator_input_mode "$selected_mode" \
    --output_dir "$output"
done
