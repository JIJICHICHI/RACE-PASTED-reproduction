#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=/home/dx/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

baseline_checkpoint() {
  case "$1" in
    42) echo "results/pasted_race/fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26/best_model.pth" ;;
    2026) echo "results/pasted_race/fourclass_baseline_official_seed2026_lr29e-6/2026-09-14_22-03-37/best_model.pth" ;;
    3407) echo "results/pasted_race/fourclass_baseline_official_seed3407_lr29e-6/2026-09-13_17-28-59/best_model.pth" ;;
    *) echo "Unsupported补训 seed: $1" >&2; return 1 ;;
  esac
}

run_trace() {
  local direction="$1"
  local seed="$2"
  local config output
  case "$direction" in
    polishing)
      config="configs/pasted_race/PASTED_RACE_lexical_official.json"
      output="results/pasted_race/lexical_official_seed${seed}"
      ;;
    humanization)
      config="configs/pasted_race/PASTED_RACE_humanization_lexical_official.json"
      output="results/pasted_race/humanization_lexical_official_seed${seed}"
      ;;
    *) echo "Unsupported direction: $direction" >&2; return 1 ;;
  esac

  if [[ -s "$output/metrics.json" && -s "$output/best_model.pt" ]]; then
    echo "Skipping completed trace run: $output"
    return
  fi
  conda run --no-capture-output -n race python train_pasted_race.py \
    --config "$config" \
    --seed "$seed" \
    --output_dir "$output"
  jq -e '.test.mse and .test.pearson' "$output/metrics.json" >/dev/null
  test -s "$output/best_model.pt"
}

run_joint() {
  local variant="$1"
  local seed="$2"
  local config output baseline lexical humanization
  config="configs/pasted_race/PASTED_RACE_fourclass_${variant}_official.json"
  output="results/pasted_race/fourclass_${variant}_official_e2e_seed${seed}"
  baseline="$(baseline_checkpoint "$seed")"
  lexical="results/pasted_race/lexical_official_seed${seed}/best_model.pt"
  humanization="results/pasted_race/humanization_lexical_official_seed${seed}/best_model.pt"

  if [[ -s "$output/metrics.json" && -s "$output/best_model.pt" ]]; then
    echo "Skipping completed paired joint run: $output"
    return
  fi
  if [[ "$variant" == "single" ]]; then
    conda run --no-capture-output -n race python train_pasted_race_fourclass.py \
      --config "$config" \
      --seed "$seed" \
      --baseline_checkpoint "$baseline" \
      --lexical_checkpoint "$lexical" \
      --output_dir "$output"
  else
    conda run --no-capture-output -n race python train_pasted_race_fourclass.py \
      --config "$config" \
      --seed "$seed" \
      --baseline_checkpoint "$baseline" \
      --lexical_checkpoint "$lexical" \
      --humanization_checkpoint "$humanization" \
      --output_dir "$output"
  fi
  jq -e '.test.f1_macro and .test.auroc_macro and .test.tpr_at_1_fpr_macro' \
    "$output/metrics.json" >/dev/null
  test -s "$output/best_model.pt"
}

for direction in polishing humanization; do
  for seed in 42 2026 3407; do
    run_trace "$direction" "$seed"
  done
done

for variant in single dual; do
  for seed in 42 2026 3407; do
    run_joint "$variant" "$seed"
  done
done
