#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=/home/dx/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers

# SciBERT currently publishes legacy PyTorch weights. The pinned PASTED
# environment (torch 2.1 / transformers 4.36) can load them, whereas the RACE
# environment's transformers 5.x intentionally rejects torch<2.6 .bin loads.
conda run --no-capture-output -n pasted python utils/creator_retention_label_builder.py \
  --inputs \
    data/pasted_race_fourclass_dual/train_graph.jsonl \
    data/pasted_race_fourclass_dual/val_graph.jsonl \
    data/pasted_race_fourclass_dual/test_graph.jsonl \
  --output_dir data/pasted_race_creator_retention_signal \
  --model_name allenai/scibert_scivocab_uncased \
  --device cuda \
  --max_length 512 \
  --batch_size 8 \
  --edited_pairs_only

conda run --no-capture-output -n race python utils/analyze_creator_retention_signal.py \
  --input_dir data/pasted_race_creator_retention_signal \
  --output_dir results/pasted_race/creator_retention_signal
