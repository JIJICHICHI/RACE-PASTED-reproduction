#!/usr/bin/env bash
set -euo pipefail

conda run --no-capture-output -n race python utils/build_pasted_race_fourclass.py \
  --source_inputs data/hart_split/train_graph.jsonl data/hart_split/val_graph.jsonl data/hart_split/test_graph.jsonl \
  --lexical_inputs data/pasted_race/train_graph.jsonl data/pasted_race/val_graph.jsonl data/pasted_race/test_graph.jsonl \
  --humanization_inputs data/pasted_race_humanization/train_graph.jsonl data/pasted_race_humanization/val_graph.jsonl data/pasted_race_humanization/test_graph.jsonl \
  --manifest data/pasted_race/split_manifest.json \
  --output_dir data/pasted_race_fourclass_dual
