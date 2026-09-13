#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python utils/evidence_label_builder.py \
  --input_train data/hart_split/train_graph.jsonl \
  --input_val data/hart_split/val_graph.jsonl \
  --input_test data/hart_split/test_graph.jsonl \
  --output_dir data/hart_split_evidence \
  --backbone_model_path roberta-base \
  --max_seq_length 512 \
  --semantic_sim_threshold 0.85 \
  --edit_ratio_threshold 0.25 \
  --min_sent_tokens 5 \
  --spacy_model en_core_web_sm
