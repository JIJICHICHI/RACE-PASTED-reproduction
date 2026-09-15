"""Analyze whether Creator Retention differs from EDU Editor Modification."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils.evidence_label_builder import load_jsonl


DIRECTION_SPEC = {
    "human_ai_polished": ("human_to_ai", "edu_lexical_scores", "edu_lexical_masks"),
    "ai_humanized": ("ai_to_human", "edu_humanization_scores", "edu_humanization_masks"),
}
DOMAINS = ("arxiv", "essay", "news", "writing")


def masked_mean(scores: list[float], masks: list[int]) -> float:
    values = [float(score) for score, mask in zip(scores, masks) if int(mask) == 1]
    if not values:
        raise ValueError("Edited item has no valid Editor Modification targets")
    return float(np.mean(values))


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def correlations(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    creator = np.asarray([row["creator_retention"] for row in rows], dtype=np.float64)
    editor = np.asarray([row["editor_modification"] for row in rows], dtype=np.float64)
    if len(rows) < 2 or creator.std() == 0 or editor.std() == 0:
        return {"count": len(rows), "pearson": None, "spearman": None}
    return {
        "count": len(rows),
        "pearson": float(pearsonr(creator, editor).statistic),
        "spearman": float(spearmanr(creator, editor).statistic),
    }


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "val", "test"):
        for item in load_jsonl(str(input_dir / f"{split}_graph.jsonl")):
            label = str(item.get("label"))
            if label not in DIRECTION_SPEC:
                raise ValueError(f"P2 input unexpectedly contains non-edited label: {label}")
            direction, score_key, mask_key = DIRECTION_SPEC[label]
            retention = float(item["creator_retention_score"])
            modification = masked_mean(item[score_key], item[mask_key])
            if not math.isfinite(retention) or not math.isfinite(modification):
                raise ValueError(f"Non-finite signal for {item['item_id']}")
            group_id = str(item["group_id"])
            domain = group_id.split("-", 1)[0].lower()
            if domain not in DOMAINS:
                raise ValueError(f"Unknown domain prefix in group_id={group_id}")
            rows.append(
                {
                    "split": split,
                    "item_id": str(item["item_id"]),
                    "group_id": group_id,
                    "domain": domain,
                    "label": label,
                    "direction": direction,
                    "reference_id": str(item["creator_retention_reference_id"]),
                    "creator_retention": retention,
                    "editor_modification": modification,
                }
            )
    return rows


def plot_overall(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"human_to_ai": "#3B82F6", "ai_to_human": "#F97316"}
    for column, direction in enumerate(("human_to_ai", "ai_to_human")):
        selected = [row for row in rows if row["direction"] == direction]
        for axis, key, title in (
            (axes[0, column], "creator_retention", "Creator Retention"),
            (axes[1, column], "editor_modification", "Editor Modification (mean EDU 1-BLEU4)"),
        ):
            axis.hist([row[key] for row in selected], bins=30, color=colors[direction], alpha=0.8)
            axis.set_title(f"{direction}: {title}")
            axis.set_xlabel(key)
            axis.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_domains(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for column, direction in enumerate(("human_to_ai", "ai_to_human")):
        selected = [row for row in rows if row["direction"] == direction]
        for axis, key, title in (
            (axes[0, column], "creator_retention", "Creator Retention"),
            (axes[1, column], "editor_modification", "Editor Modification"),
        ):
            values = [[row[key] for row in selected if row["domain"] == domain] for domain in DOMAINS]
            axis.boxplot(values, tick_labels=[name.title() for name in DOMAINS], showfliers=False)
            axis.set_title(f"{direction}: {title}")
            axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.input_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "edited_pair_signals.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_direction = defaultdict(list)
    by_direction_domain = defaultdict(list)
    for row in rows:
        by_direction[row["direction"]].append(row)
        by_direction_domain[(row["direction"], row["domain"])].append(row)
    report = {
        "definition": {
            "creator_retention": "unrescaled SciBERT contextual-token recall",
            "editor_modification": "document mean of valid EDU 1-BLEU4 targets",
            "scope": "edited same-group pairs only; no self-retention samples",
        },
        "rows": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "direction_counts": dict(Counter(row["direction"] for row in rows)),
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "directions": {},
    }
    for direction, selected in sorted(by_direction.items()):
        report["directions"][direction] = {
            "creator_retention": summarize([row["creator_retention"] for row in selected]),
            "editor_modification": summarize([row["editor_modification"] for row in selected]),
            "correlation": correlations(selected),
            "domains": {
                domain: {
                    "creator_retention": summarize(
                        [row["creator_retention"] for row in by_direction_domain[(direction, domain)]]
                    ),
                    "editor_modification": summarize(
                        [row["editor_modification"] for row in by_direction_domain[(direction, domain)]]
                    ),
                    "correlation": correlations(by_direction_domain[(direction, domain)]),
                }
                for domain in DOMAINS
            },
        }
    with (output_dir / "signal_analysis.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    plot_overall(rows, output_dir / "signal_distributions.png")
    plot_domains(rows, output_dir / "domain_distributions.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
