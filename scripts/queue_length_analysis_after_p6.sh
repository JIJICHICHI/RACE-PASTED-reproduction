#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=/home/dx/.cache/huggingface
export HF_HUB_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

output_dir="reports/length_analysis_group_safe"
if [[ -s "$output_dir/COMPLETED" ]]; then
  echo "Length analysis already complete: $output_dir/report.md"
  exit 0
fi

for variant in creator_only creator_no_fusion all_lambda_zero; do
  for seed in 42 2026 3407; do
    metric="results/pasted_race/fourclass_p6_${variant}_seed${seed}/metrics.json"
    while [[ ! -s "$metric" ]]; do
      echo "Waiting for $metric"
      sleep 30
    done
  done
done

# Let the P6 launcher finish aggregation/publication before beginning its
# post-hoc consumer. The bracketed pattern does not match this grep itself.
while pgrep -f '[q]ueue_p6_after_p5.sh' >/dev/null; do
  echo "All P6 cells exist; waiting for the P6 launcher to exit"
  sleep 30
done

conda run --no-capture-output -n race python \
  utils/analyze_length_bucket_performance.py \
  --data_path data/pasted_race_fourclass/test_graph.jsonl \
  --results_dir results/pasted_race \
  --output_dir "$output_dir" \
  --tokenizer FacebookAI/roberta-base \
  --local_files_only
