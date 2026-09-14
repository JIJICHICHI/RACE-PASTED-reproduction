#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run --no-capture-output -n race python train_pasted_race_fourclass.py \
  --config configs/pasted_race/PASTED_RACE_fourclass_creator_editor_joint.json
