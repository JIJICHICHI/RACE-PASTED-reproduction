#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs/pasted_race
log_path="logs/pasted_race/p5_p6_resume_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_path") 2>&1

echo "[$(date '+%F %T')] Persistent P5/P6 queue started"
echo "log_path=$log_path"

bash scripts/queue_p5_after_p3.sh
bash scripts/queue_p6_after_p5.sh

conda run --no-capture-output -n race python \
  utils/aggregate_creator_editor_experiments.py \
  --results_dir results/pasted_race \
  --output_dir reports/creator_editor_p5_p6_final

echo "[$(date '+%F %T')] Persistent P5/P6 queue completed"
