#!/usr/bin/env python3
"""Reproduce the RACE Figure-4 length-bucket protocol on formal saved logits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import compute_classification_metrics


SEEDS = (42, 2026, 3407)
CLASS_NAMES = ("Human", "Polished", "Generated", "Humanized")
LABELS = {
    "human_written": 0,
    "human_ai_polished": 1,
    "ai_generated": 2,
    "ai_humanized": 3,
}
BUCKETS = (
    ("0–200", 0, 200),
    ("200–400", 200, 400),
    ("400–600", 400, 600),
    ("600–800", 600, 800),
    ("800+", 800, math.inf),
)
METHODS = {
    "Strong RACE": "baseline",
    "Single Trace": "fourclass_single_official_e2e_seed{seed}",
    "Dual / Editor": "fourclass_dual_official_e2e_seed{seed}",
    "Creator + Editor": "fourclass_creator_editor_official_seed{seed}",
    "Creator-only": "fourclass_p6_creator_only_seed{seed}",
    "Creator, no fusion": "fourclass_p6_creator_no_fusion_seed{seed}",
    "All lambdas = 0": "fourclass_p6_all_lambda_zero_seed{seed}",
}
BASELINE_RUNS = {
    42: "fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26",
    2026: "fourclass_baseline_official_seed2026_lr29e-6/2026-09-14_22-03-37",
    3407: "fourclass_baseline_official_seed3407_lr29e-6/2026-09-13_17-28-59",
}
PRIMARY_KEYS = ("accuracy", "f1_macro", "auroc_macro", "tpr_at_1_fpr_macro")
PAPER_FIGURE4 = {
    "RACE": [0.588, 0.800, 0.868, 0.970, 0.948],
    "CoCo": [0.576, 0.744, 0.791, 0.970, 0.960],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default=Path("data/pasted_race_fourclass/test_graph.jsonl"))
    parser.add_argument("--results_dir", type=Path, default=Path("results/pasted_race"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports/length_analysis_group_safe"))
    parser.add_argument("--tokenizer", default="FacebookAI/roberta-base")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_label(value: Any) -> int:
    if isinstance(value, str):
        if value not in LABELS:
            raise ValueError(f"Unknown label: {value}")
        return LABELS[value]
    result = int(value)
    if result not in range(4):
        raise ValueError(f"Label outside [0,3]: {value}")
    return result


def bucket_name(length: int) -> str:
    for name, lower, upper in BUCKETS:
        if lower <= length < upper:
            return name
    raise AssertionError(length)


def load_manifest(path: Path, tokenizer_name: str, local_only: bool) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    if len(rows) != 3200:
        raise ValueError(f"Expected canonical 3200-item test manifest, found {len(rows)}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_only, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    texts = [str(row["article"]) for row in rows]
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
    by_id: dict[str, dict[str, Any]] = {}
    for row, token_ids in zip(rows, encoded):
        item_id = str(row["item_id"])
        if item_id in by_id:
            raise ValueError(f"Duplicate manifest item_id: {item_id}")
        record = {
            "item_id": item_id,
            "group_id": str(row["group_id"]),
            "label": normalize_label(row["label"]),
            "token_length": len(token_ids),
            "bucket": bucket_name(len(token_ids)),
        }
        by_id[item_id] = record
    return list(by_id.values()), by_id


def run_dir(results_dir: Path, method: str, seed: int) -> Path:
    spec = METHODS[method]
    return results_dir / (BASELINE_RUNS[seed] if spec == "baseline" else spec.format(seed=seed))


def load_predictions(path: Path) -> list[dict[str, Any]]:
    json_path = path / "test_predictions.json"
    jsonl_path = path / "test_predictions.jsonl"
    if json_path.is_file():
        with json_path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    elif jsonl_path.is_file():
        rows = read_jsonl(jsonl_path)
    else:
        raise FileNotFoundError(f"No formal test predictions in {path}")
    return rows


def saved_test_metrics(path: Path) -> dict[str, Any]:
    baseline_path = path / "test_metrics.json"
    if baseline_path.is_file():
        with baseline_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    metrics_path = path / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if "test" not in payload:
        raise KeyError(f"Missing test metrics in {metrics_path}")
    return payload["test"]


def clean_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if logits.ndim != 2 or logits.shape[1] != 4 or not np.isfinite(logits).all():
        raise ValueError(f"Invalid logits shape/content: {logits.shape}")
    support = Counter(labels.tolist())
    if set(support) != set(range(4)):
        raise ValueError(f"Length bucket lacks four-class support: {dict(support)}")
    raw = compute_classification_metrics(
        torch.tensor(logits, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        num_classes=4,
    )
    return {key: float(value) for key, value in raw.items() if key != "classification_report"}


def aligned_arrays(
    prediction_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]], manifest_by_id: dict[str, dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    predictions: dict[str, tuple[np.ndarray, int | None]] = {}
    for row in prediction_rows:
        item_id = str(row.get("item_id", row.get("id", "")))
        if not item_id or item_id in predictions:
            raise ValueError(f"Missing or duplicate prediction ID: {item_id!r}")
        saved_label = normalize_label(row["label"]) if "label" in row else None
        predictions[item_id] = (np.asarray(row["logits"], dtype=np.float64), saved_label)
    expected = set(manifest_by_id)
    if set(predictions) != expected:
        raise ValueError(
            f"Prediction IDs differ from manifest: missing={len(expected-set(predictions))}, extra={len(set(predictions)-expected)}"
        )
    logits, labels, buckets = [], [], []
    for row in manifest_rows:
        item_id = row["item_id"]
        item_logits, saved_label = predictions[item_id]
        if saved_label is not None and saved_label != row["label"]:
            raise ValueError(f"Label mismatch for {item_id}: {saved_label} != {row['label']}")
        logits.append(item_logits)
        labels.append(row["label"])
        buckets.append(row["bucket"])
    return np.stack(logits), np.asarray(labels, dtype=np.int64), buckets


def validate_whole_test(recomputed: dict[str, float], saved: dict[str, Any], path: Path) -> None:
    for key in PRIMARY_KEYS:
        if key not in saved:
            raise KeyError(f"Saved metric {key} missing in {path}")
        if not math.isclose(recomputed[key], float(saved[key]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Whole-test metric mismatch in {path}: {key}={recomputed[key]} saved={saved[key]}")


def aggregate(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    metric_keys = sorted(per_seed[0]["metrics"])
    for method in METHODS:
        for bucket, _, _ in BUCKETS:
            rows = [row for row in per_seed if row["method"] == method and row["bucket"] == bucket]
            if len(rows) != len(SEEDS):
                raise ValueError(f"Incomplete seed matrix for {method}/{bucket}")
            record: dict[str, Any] = {
                "method": method,
                "bucket": bucket,
                "documents": rows[0]["documents"],
                "class_counts": rows[0]["class_counts"],
            }
            for key in metric_keys:
                values = [row["metrics"][key] for row in rows]
                record[f"{key}_mean"] = mean(values)
                record[f"{key}_std"] = stdev(values)
            output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key not in {"metrics", "class_counts"}}
        if "class_counts" in row:
            for class_id, count in row["class_counts"].items():
                flat[f"count_class_{class_id}"] = count
        if "metrics" in row:
            flat.update(row["metrics"])
        flat_rows.append(flat)
    keys = sorted({key for row in flat_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)


def format_cell(value: float, deviation: float) -> str:
    return f"{100*value:.2f}±{100*deviation:.2f}"


def write_report(path: Path, manifest_sha: str, manifest_rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lookup = {(row["method"], row["bucket"]): row for row in summary}
    counts = Counter(row["bucket"] for row in manifest_rows)
    class_counts = {bucket: Counter(row["label"] for row in manifest_rows if row["bucket"] == bucket) for bucket, _, _ in BUCKETS}
    lines = [
        "# Group-Safe Text-Length Analysis",
        "",
        "## Protocol",
        "",
        "- Test set: canonical 3,200-document group-safe manifest; no resplitting or retraining.",
        f"- Manifest SHA-256: `{manifest_sha}`.",
        "- Length: full final-text token count from `FacebookAI/roberta-base`, without special tokens or truncation.",
        "- Buckets: `[0,200)`, `[200,400)`, `[400,600)`, `[600,800)`, `[800,+∞)`; display labels match RACE Figure 4.",
        "- Primary metric: four-class Macro TPR@1%FPR, reported as three-seed mean ± sample standard deviation (%).",
        "- The paper does not disclose its token counter. Its Figure 4 RACE/CoCo values below are historical reference, not same-split baselines.",
        "",
        "## Bucket audit",
        "",
        "| Bucket | Documents | Human | Polished | Generated | Humanized |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bucket, _, _ in BUCKETS:
        c = class_counts[bucket]
        lines.append(f"| {bucket} | {counts[bucket]} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
    lines.extend(["", "## Macro TPR@1%FPR", "", "| Method | " + " | ".join(x[0] for x in BUCKETS) + " |", "|---|" + "---:|" * len(BUCKETS)])
    for method in METHODS:
        cells = [format_cell(lookup[(method, b)]["tpr_at_1_fpr_macro_mean"], lookup[(method, b)]["tpr_at_1_fpr_macro_std"]) for b, _, _ in BUCKETS]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    for title, metric in (("Macro-F1", "f1_macro"), ("Macro-AUROC", "auroc_macro")):
        lines.extend(["", f"## {title}", "", "| Method | " + " | ".join(x[0] for x in BUCKETS) + " |", "|---|" + "---:|" * len(BUCKETS)])
        for method in METHODS:
            cells = [format_cell(lookup[(method, b)][f"{metric}_mean"], lookup[(method, b)][f"{metric}_std"]) for b, _, _ in BUCKETS]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## RACE paper Figure 4 reference",
        "",
        "| Published method | " + " | ".join(x[0] for x in BUCKETS) + " |",
        "|---|" + "---:|" * len(BUCKETS),
    ])
    for method, values in PAPER_FIGURE4.items():
        lines.append(f"| {method} | " + " | ".join(f"{100*x:.1f}" for x in values) + " |")
    lines.extend(["", "These published values use a different experiment provenance and are not used to calculate gains.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    lookup = {(row["method"], row["bucket"]): row for row in summary}
    labels = [x[0] for x in BUCKETS]
    panels = (
        ("Canonical models", ("Strong RACE", "Single Trace", "Dual / Editor", "Creator + Editor")),
        ("P6 ablations", ("Strong RACE", "Creator-only", "Creator, no fusion", "Creator + Editor", "All lambdas = 0")),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, (title, methods) in zip(axes, panels):
        for method in methods:
            y = [100 * lookup[(method, bucket)]["tpr_at_1_fpr_macro_mean"] for bucket in labels]
            err = [100 * lookup[(method, bucket)]["tpr_at_1_fpr_macro_std"] for bucket in labels]
            axis.errorbar(labels, y, yerr=err, marker="o", capsize=3, label=method)
        axis.set_title(title)
        axis.set_xlabel("RoBERTa token length")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Macro TPR @ 1% FPR (%)")
    fig.suptitle("Group-safe length robustness (mean ± sample std, 3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, manifest_by_id = load_manifest(args.data_path, args.tokenizer, args.local_files_only)
    manifest_sha = sha256(args.data_path)
    length_rows = sorted(manifest_rows, key=lambda row: row["item_id"])
    write_csv(args.output_dir / "token_lengths.csv", length_rows)

    per_seed: list[dict[str, Any]] = []
    provenance = []
    for method in METHODS:
        for seed in SEEDS:
            path = run_dir(args.results_dir, method, seed)
            logits, labels, bucket_labels = aligned_arrays(load_predictions(path), manifest_rows, manifest_by_id)
            whole = clean_metrics(logits, labels)
            validate_whole_test(whole, saved_test_metrics(path), path)
            provenance.append({"method": method, "seed": seed, "run_dir": str(path), "whole_test": {key: whole[key] for key in PRIMARY_KEYS}})
            bucket_array = np.asarray(bucket_labels)
            for bucket, _, _ in BUCKETS:
                mask = bucket_array == bucket
                metrics = clean_metrics(logits[mask], labels[mask])
                per_seed.append({
                    "method": method,
                    "seed": seed,
                    "bucket": bucket,
                    "documents": int(mask.sum()),
                    "class_counts": {str(i): int((labels[mask] == i).sum()) for i in range(4)},
                    "metrics": metrics,
                })

    summary = aggregate(per_seed)
    payload = {
        "protocol": {
            "data_path": str(args.data_path),
            "manifest_sha256": manifest_sha,
            "tokenizer": args.tokenizer,
            "add_special_tokens": False,
            "truncation": False,
            "buckets_half_open": [[name, lower, None if math.isinf(upper) else upper] for name, lower, upper in BUCKETS],
            "seeds": list(SEEDS),
            "paper_figure4_reference": PAPER_FIGURE4,
        },
        "provenance": provenance,
        "per_seed": per_seed,
        "summary": summary,
    }
    with (args.output_dir / "length_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)
    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed)
    write_csv(args.output_dir / "summary_metrics.csv", summary)
    write_report(args.output_dir / "report.md", manifest_sha, manifest_rows, summary)
    plot_summary(args.output_dir / "macro_tpr_by_length.png", summary)
    (args.output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    print(f"Length analysis complete: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
