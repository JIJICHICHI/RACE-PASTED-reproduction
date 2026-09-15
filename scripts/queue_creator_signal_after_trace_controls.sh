#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

required=(
  results/pasted_race/fourclass_single_official_seed42/metrics.json
  results/pasted_race/fourclass_single_official_seed2026/metrics.json
  results/pasted_race/fourclass_single_official_seed3407/metrics.json
  results/pasted_race/fourclass_dual_official_seed42/metrics.json
  results/pasted_race/fourclass_dual_official_seed2026/metrics.json
  results/pasted_race/fourclass_dual_official_seed3407/metrics.json
)

while true; do
  missing=0
  for result in "${required[@]}"; do
    if [[ ! -s "$result" ]]; then
      missing=$((missing + 1))
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  echo "$(date --iso-8601=seconds) waiting for trace controls: $missing/6 incomplete"
  sleep 30
done

for result in "${required[@]}"; do
  jq -e '.test.f1_macro and .test.auroc_macro and .test.tpr_at_1_fpr_macro' "$result" >/dev/null
done

echo "$(date --iso-8601=seconds) all trace controls complete; starting Creator Retention P2"
bash scripts/run_creator_retention_signal_analysis.sh
echo "$(date --iso-8601=seconds) Creator Retention P2 complete"
