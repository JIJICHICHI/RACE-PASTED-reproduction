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

run_one diag_final_supcon_only_beta005 configs/far_race/FAR_diag_final_supcon_only_beta005.json || exit $?
run_one diag_creator_supcon_only_beta005 configs/far_race/FAR_diag_creator_supcon_only_beta005.json || exit $?
run_one diag_editor_supcon_only_beta005 configs/far_race/FAR_diag_editor_supcon_only_beta005.json || exit $?
