#!/usr/bin/env python3
"""Role-axis diagnostics from saved RACE predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


CLASS_NAMES = {
    0: "Human-Written",
    1: "LLM-Polished",
    2: "LLM-Generated",
    3: "Humanized",
}


def creator_label(labels: np.ndarray) -> np.ndarray:
    """0 = human creator, 1 = LLM creator."""
    return np.isin(labels, [2, 3]).astype(int)


def editor_label(labels: np.ndarray) -> np.ndarray:
    """0 = human editor, 1 = LLM editor."""
    return np.isin(labels, [1, 2]).astype(int)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    fprs, tprs, _ = roc_curve(y_true, y_score)
    ok = np.where(fprs <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(tprs[ok[-1]])


def binary_metrics(y_true: np.ndarray, y_score_pos: np.ndarray) -> dict:
    y_pred = (y_score_pos >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int((1 - y_true).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tpr_at_1_fpr": tpr_at_fpr(y_true, y_score_pos, 0.01),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_score_pos))
    except ValueError:
        out["auroc"] = 0.0
    return out


def subset_pair_metrics(
    labels: np.ndarray,
    scores_pos: np.ndarray,
    class_neg: int,
    class_pos: int,
    task: str,
    fixed_factor: str,
    tested_factor: str,
) -> dict:
    mask = np.isin(labels, [class_neg, class_pos])
    y_true = (labels[mask] == class_pos).astype(int)
    metrics = binary_metrics(y_true, scores_pos[mask])
    metrics.update(
        {
            "task": task,
            "fixed_factor": fixed_factor,
            "tested_factor": tested_factor,
            "negative_class": CLASS_NAMES[class_neg],
            "positive_class": CLASS_NAMES[class_pos],
        }
    )
    return metrics


def format_pct(x: float) -> float:
    return round(100.0 * float(x), 4)


def pct_table(rows: Iterable[dict], keys: Iterable[str]) -> list[dict]:
    out = []
    for row in rows:
        item = {}
        for key in keys:
            val = row[key]
            if isinstance(val, float):
                item[key] = format_pct(val)
            else:
                item[key] = val
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("r", encoding="utf-8") as f:
        records = json.load(f)

    labels = np.asarray([int(r["label"]) for r in records])
    preds = np.asarray([int(r["pred"]) for r in records])
    probs = np.asarray([[r["p_H"], r["p_P"], r["p_G"], r["p_Z"]] for r in records], dtype=float)

    p_h_creator = probs[:, 0] + probs[:, 1]
    p_llm_creator = probs[:, 2] + probs[:, 3]
    p_h_editor = probs[:, 0] + probs[:, 3]
    p_llm_editor = probs[:, 1] + probs[:, 2]

    y_creator = creator_label(labels)
    y_editor = editor_label(labels)

    axis_metrics = {
        "creator": binary_metrics(y_creator, p_llm_creator),
        "editor": binary_metrics(y_editor, p_llm_editor),
    }

    conditional = [
        subset_pair_metrics(
            labels,
            p_llm_creator,
            0,
            3,
            "Human-Written vs Humanized",
            "editor = Human",
            "creator",
        ),
        subset_pair_metrics(
            labels,
            p_llm_creator,
            1,
            2,
            "LLM-Polished vs LLM-Generated",
            "editor = LLM",
            "creator",
        ),
        subset_pair_metrics(
            labels,
            p_llm_editor,
            0,
            1,
            "Human-Written vs LLM-Polished",
            "creator = Human",
            "editor",
        ),
        subset_pair_metrics(
            labels,
            p_llm_editor,
            3,
            2,
            "LLM-Generated vs Humanized",
            "creator = LLM",
            "editor",
        ),
    ]

    true_creator = creator_label(labels)
    pred_creator = creator_label(preds)
    true_editor = editor_label(labels)
    pred_editor = editor_label(preds)
    wrong = labels != preds
    creator_wrong = true_creator != pred_creator
    editor_wrong = true_editor != pred_editor
    total_wrong = int(wrong.sum())
    error_decomposition = {
        "total_errors": total_wrong,
        "creator_wrong_only": int(np.logical_and.reduce([wrong, creator_wrong, ~editor_wrong]).sum()),
        "editor_wrong_only": int(np.logical_and.reduce([wrong, ~creator_wrong, editor_wrong]).sum()),
        "both_wrong": int(np.logical_and.reduce([wrong, creator_wrong, editor_wrong]).sum()),
    }
    for key in ["creator_wrong_only", "editor_wrong_only", "both_wrong"]:
        error_decomposition[f"{key}_ratio"] = (
            float(error_decomposition[key] / total_wrong) if total_wrong else 0.0
        )

    confusion_error_pairs = {}
    for true_c in range(4):
        for pred_c in range(4):
            if true_c == pred_c:
                continue
            count = int(np.logical_and(labels == true_c, preds == pred_c).sum())
            if count:
                key = f"{CLASS_NAMES[true_c]} -> {CLASS_NAMES[pred_c]}"
                confusion_error_pairs[key] = count

    creator_margin = np.abs(p_h_creator - p_llm_creator)
    editor_margin = np.abs(p_h_editor - p_llm_editor)
    margin_by_class = {}
    for c, name in CLASS_NAMES.items():
        mask = labels == c
        margin_by_class[name] = {
            "n": int(mask.sum()),
            "creator_margin_mean": float(creator_margin[mask].mean()),
            "creator_margin_median": float(np.median(creator_margin[mask])),
            "editor_margin_mean": float(editor_margin[mask].mean()),
            "editor_margin_median": float(np.median(editor_margin[mask])),
        }

    result = {
        "source_predictions": str(args.predictions),
        "class_order": CLASS_NAMES,
        "axis_metrics": axis_metrics,
        "conditional_binary_tasks": conditional,
        "error_decomposition": error_decomposition,
        "confusion_error_pairs": confusion_error_pairs,
        "role_margin_by_class": margin_by_class,
    }

    with (args.output_dir / "role_axis_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Original RACE Role-Axis Diagnostics\n")
    lines.append(f"Source: `{args.predictions}`\n")
    lines.append("## Axis Metrics\n")
    lines.append("| Axis | Acc | Macro-F1 | AUROC | TPR@1%FPR |")
    lines.append("|---|---:|---:|---:|---:|")
    for axis in ["creator", "editor"]:
        m = axis_metrics[axis]
        lines.append(
            f"| {axis} | {format_pct(m['accuracy']):.2f} | {format_pct(m['macro_f1']):.2f} | "
            f"{format_pct(m['auroc']):.2f} | {format_pct(m['tpr_at_1_fpr']):.2f} |"
        )
    lines.append("\n## Conditional Binary Tasks\n")
    lines.append("| Task | Fixed Factor | Tested Factor | Acc | F1 | AUROC | TPR@1%FPR |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for m in conditional:
        lines.append(
            f"| {m['task']} | {m['fixed_factor']} | {m['tested_factor']} | "
            f"{format_pct(m['accuracy']):.2f} | {format_pct(m['macro_f1']):.2f} | "
            f"{format_pct(m['auroc']):.2f} | {format_pct(m['tpr_at_1_fpr']):.2f} |"
        )
    lines.append("\n## Error Decomposition\n")
    lines.append("| Error Type | Count | Ratio of Errors |")
    lines.append("|---|---:|---:|")
    for key in ["creator_wrong_only", "editor_wrong_only", "both_wrong"]:
        lines.append(
            f"| {key} | {error_decomposition[key]} | "
            f"{format_pct(error_decomposition[key + '_ratio']):.2f} |"
        )
    lines.append(f"\nTotal errors: {total_wrong}\n")
    lines.append("## Role Margin By Class\n")
    lines.append("| True Class | Creator Margin Mean | Editor Margin Mean | Creator Median | Editor Median |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in margin_by_class.items():
        lines.append(
            f"| {name} | {m['creator_margin_mean']:.4f} | {m['editor_margin_mean']:.4f} | "
            f"{m['creator_margin_median']:.4f} | {m['editor_margin_median']:.4f} |"
        )
    lines.append("\n## Non-zero Error Pairs\n")
    for key, count in sorted(confusion_error_pairs.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {key}: {count}")

    (args.output_dir / "role_axis_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
