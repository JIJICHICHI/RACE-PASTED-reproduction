#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/home/dx/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

conda run --no-capture-output -n race python train_pasted_race_fourclass.py \
  --config configs/pasted_race/PASTED_RACE_fourclass_joint.json "$@"
