"""Diagnostics for LE-RACE weak labels and evidence predictions."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def summarize_evidence_labels(jsonl_paths: list[str], output_path: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in jsonl_paths:
        for item in load_jsonl(path):
            groups["overall"].append(item)
            groups[f"class:{item.get('label', 'unknown')}"] .append(item)
            groups[f"domain:{item.get('source_dataset', 'unknown')}"] .append(item)

    def summarize(items: list[dict[str, Any]]) -> dict[str, float]:
        total_edu = sum(len(x.get("edu_evidence_masks", [])) for x in items)
        total_mask = sum(sum(x.get("edu_evidence_masks", [])) for x in items)
        total_pos = sum(sum(x.get("edu_evidence_labels", [])) for x in items)
        return {
            "items": len(items),
            "coverage": sum(1 for x in items if x.get("has_trace_ref")) / max(1, len(items)),
            "positive_ratio": total_pos / max(1, total_mask),
            "masked_ratio": 1.0 - total_mask / max(1, total_edu),
            "avg_full_edu": total_edu / max(1, len(items)),
        }

    stats = {name: summarize(items) for name, items in groups.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def summarize_evidence_predictions(prediction_path: str, output_path: str) -> dict[str, Any]:
    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    groups: dict[str, list[float]] = defaultdict(list)
    for pred in predictions:
        scores = pred.get("evidence_probs") or []
        if not scores:
            continue
        label = str(pred.get("label", "unknown"))
        groups[label].append(sum(scores) / len(scores))
        groups["overall"].append(sum(scores) / len(scores))
    stats = {}
    for key, values in groups.items():
        mean = sum(values) / max(1, len(values))
        var = sum((x - mean) ** 2 for x in values) / max(1, len(values))
        stats[key] = {"count": len(values), "mean": mean, "std": var ** 0.5}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_jsonl", nargs="*")
    parser.add_argument("--predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.label_jsonl:
        summarize_evidence_labels(args.label_jsonl, args.output)
    elif args.predictions:
        summarize_evidence_predictions(args.predictions, args.output)
    else:
        raise SystemExit("Provide --label_jsonl or --predictions")


if __name__ == "__main__":
    main()
