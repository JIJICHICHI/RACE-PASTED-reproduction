#!/usr/bin/env bash
set -uo pipefail
cd /home/dx/RACE_LE_RACE || exit 1
mkdir -p logs/far_race
export HF_HOME=/home/dx/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers

run_one() {
  local name="$1"
  local config="$2"
  local log="logs/far_race/${name}.log"
  echo "[$name] start $(date)" | tee "$log"
  conda run --no-capture-output -n race python train.py --config "$config" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "[$name] exit $status $(date)" | tee -a "$log"
  return "$status"
}

run_one beta010_tpr_earlystop configs/far_race/FAR_RACE_factor_contrast_beta010_tpr_earlystop.json || exit $?
run_one beta005_tpr_earlystop configs/far_race/FAR_RACE_factor_contrast_beta005_tpr_earlystop.json || exit $?
