#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=/home/dx/.cache/huggingface
export HF_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

for seed in 42 2026 3407; do
  while [[ ! -s "results/pasted_race/fourclass_creator_editor_official_seed${seed}/metrics.json" ]]; do
    sleep 30
  done
done

selected_mode="$(python -c 'import json; print(json.load(open("results/pasted_race/creator_retention_p3_selection.json"))["selected_mode"])')"

baseline_for_seed() {
  case "$1" in
    42) echo "results/pasted_race/fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26/best_model.pth" ;;
    2026) echo "results/pasted_race/fourclass_baseline_official_seed2026_lr29e-6/2026-09-14_22-03-37/best_model.pth" ;;
    3407) echo "results/pasted_race/fourclass_baseline_official_seed3407_lr29e-6/2026-09-13_17-28-59/best_model.pth" ;;
  esac
}

for variant in creator_only creator_no_fusion all_lambda_zero; do
  for seed in 42 2026 3407; do
    output="results/pasted_race/fourclass_p6_${variant}_seed${seed}"
    if [[ -s "$output/metrics.json" ]]; then
      continue
    fi
    extra_args=()
    if [[ "$variant" == "all_lambda_zero" ]]; then
      extra_args+=(
        --lexical_checkpoint "results/pasted_race/lexical_official_seed${seed}/best_model.pt"
        --humanization_checkpoint "results/pasted_race/humanization_lexical_official_seed${seed}/best_model.pt"
      )
    fi
    conda run --no-capture-output -n race python train_pasted_race_fourclass.py \
      --config configs/pasted_race/PASTED_RACE_fourclass_creator_editor_official.json \
      --ablation_variant "$variant" \
      --seed "$seed" \
      --baseline_checkpoint "$(baseline_for_seed "$seed")" \
      --creator_checkpoint "results/pasted_race/creator_retention_p3_${selected_mode}_seed${seed}/best_model.pt" \
      --creator_input_mode "$selected_mode" \
      --output_dir "$output" \
      "${extra_args[@]}"
    test -s "$output/metrics.json"
    rm -f "$output/best_model.pt"
  done
done

conda run --no-capture-output -n race python \
  utils/aggregate_creator_editor_experiments.py \
  --results_dir results/pasted_race \
  --output_dir reports/creator_editor_p5_p6_final

git add -f reports/creator_editor_p5_p6_final
if ! git diff --cached --quiet; then
  git commit -m "Add final Creator Editor experiment results"
  git push origin main
fi
