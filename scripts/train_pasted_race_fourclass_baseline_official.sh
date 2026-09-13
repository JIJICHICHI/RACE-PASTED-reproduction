#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/home/dx/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/dx/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/home/dx/.cache/huggingface/transformers"

conda run --no-capture-output -n race python train.py \
  --config configs/pasted_race/PASTED_RACE_fourclass_baseline_official_seed3407.json
