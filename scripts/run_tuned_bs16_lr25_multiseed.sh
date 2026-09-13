#!/usr/bin/env bash
set -euo pipefail

cd /home/dx/RACE

source /home/dx/miniconda3/etc/profile.d/conda.sh
conda activate race

export HF_HOME=/home/dx/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers

mkdir -p logs configs/tuning_multiseed/generated results/tuning_multiseed/bs16_lr25e-6

BASE_CONFIG="configs/tuning_multiseed/RACE_bs16_lr25e-6.json"

for seed in 3407 42 2025 0 1; do
  out_dir="results/tuning_multiseed/bs16_lr25e-6/seed_${seed}"
  cfg="configs/tuning_multiseed/generated/RACE_bs16_lr25e-6_seed_${seed}.json"
  log_file="logs/tuned_bs16_lr25_seed_${seed}.log"

  if find -L "$out_dir" -name test_metrics.json -print -quit | grep -q .; then
    echo "==== tuned bs16_lr25e-6 seed ${seed}: already complete, skipping ====" | tee -a "$log_file"
    continue
  fi

  mkdir -p "$out_dir"
  /home/dx/miniconda3/bin/jq --argjson seed "$seed" --arg out "$out_dir" \
    '.seed = $seed | .output_dir = $out' \
    "$BASE_CONFIG" > "$cfg"

  echo "==== tuned bs16_lr25e-6 seed ${seed} ====" | tee -a "$log_file"
  python -u train.py --config "$cfg" 2>&1 | tee -a "$log_file"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "Seed ${seed} failed with status ${status}" | tee -a "$log_file"
    exit "$status"
  fi
done
