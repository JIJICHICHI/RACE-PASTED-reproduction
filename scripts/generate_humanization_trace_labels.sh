#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/home/dx/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

conda run --no-capture-output -n race python utils/humanization_trace_label_builder.py \
  --inputs data/hart_split/train_graph.jsonl data/hart_split/val_graph.jsonl data/hart_split/test_graph.jsonl \
  --manifest data/pasted_race/split_manifest.json \
  --output_dir data/pasted_race_humanization \
  --backbone_model_path FacebookAI/roberta-base \
  --device cuda
