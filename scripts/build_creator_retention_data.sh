#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run --no-capture-output -n race python utils/creator_retention_label_builder.py \
  --inputs \
    data/pasted_race_fourclass_dual/train_graph.jsonl \
    data/pasted_race_fourclass_dual/val_graph.jsonl \
    data/pasted_race_fourclass_dual/test_graph.jsonl \
  --output_dir data/pasted_race_creator_editor \
  --model_name allenai/scibert_scivocab_uncased \
  --device cuda \
  --max_length 512 \
  --batch_size 8
