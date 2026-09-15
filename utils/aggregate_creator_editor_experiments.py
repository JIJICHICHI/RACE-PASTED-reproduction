"""Aggregate the complete group-safe Creator/Editor experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 2026, 3407)
CLASS_NAMES = ("human", "polished", "generated", "humanized")
PRIMARY = ("accuracy", "f1_macro", "auroc_macro", "tpr_at_1_fpr_macro")
CLASSWISE = tuple(
    key
    for index in range(4)
    for key in (f"f1_class_{index}", f"tpr_at_1_fpr_class_{index}")
)
CREATOR = (
    "creator_retention_mse",
    "creator_retention_pearson",
    "creator_retention_spearman",
)
EDITOR = (
    "lexical_mse", "lexical_pearson", "lexical_spearman", "lexical_auroc",
    "lexical_tpr_at_1pct_fpr", "humanization_mse", "humanization_pearson",
    "humanization_spearman", "humanization_auroc",
    "humanization_tpr_at_1pct_fpr",
)
ALL_METRICS = PRIMARY + CLASSWISE + CREATOR + EDITOR

BASELINE_PATHS = {
    42: "fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26/test_metrics.json",
    2026: "fourclass_baseline_official_seed2026_lr29e-6/2026-09-14_22-03-37/test_metrics.json",
    3407: "fourclass_baseline_official_seed3407_lr29e-6/2026-09-13_17-28-59/test_metrics.json",
}


def experiment_specs(seed: int) -> list[tuple[str, str, bool]]:
    return [
        ("RACE", BASELINE_PATHS[seed], False),
        ("RACE+Editor", f"fourclass_dual_official_e2e_seed{seed}/metrics.json", True),
        ("RACE+Creator", f"fourclass_p6_creator_only_seed{seed}/metrics.json", True),
        ("RACE+Creator (no fusion)", f"fourclass_p6_creator_no_fusion_seed{seed}/metrics.json", True),
        ("RACE+Creator+Editor", f"fourclass_creator_editor_official_seed{seed}/metrics.json", True),
        ("Full structure, lambda=0", f"fourclass_p6_all_lambda_zero_seed{seed}/metrics.json", True),
    ]


def finite_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


def load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method, relative_path, nested_test in experiment_specs(seed):
            path = results_dir / relative_path
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing required result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("test") if nested_test else payload
            if not isinstance(metrics, dict):
                raise ValueError(f"Missing test metrics object: {path}")
            missing = [key for key in PRIMARY + CLASSWISE if key not in metrics]
            if missing:
                raise ValueError(f"Missing classification keys in {path}: {missing}")
            row: dict[str, Any] = {"method": method, "seed": seed, "source": str(path)}
            row.update({key: finite_or_none(metrics.get(key)) for key in ALL_METRICS})
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    output: list[dict[str, Any]] = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        if [row["seed"] for row in group] != list(SEEDS):
            raise ValueError(f"Unexpected seed coverage for {method}")
        record: dict[str, Any] = {"method": method, "n_seeds": len(group)}
        for key in ALL_METRICS:
            values = [row[key] for row in group if row[key] is not None]
            record[f"{key}_mean"] = statistics.mean(values) if values else None
            record[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Creator/Editor P5–P6 Final Results", "",
        "All cells use the fixed group-safe split and seeds 42/2026/3407. "
        "Standard deviations are sample standard deviations.", "",
        "## Primary metrics (mean ± std)", "",
        "| Method | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        values = []
        for key in PRIMARY:
            mean, std = row[f"{key}_mean"], row[f"{key}_std"]
            values.append("—" if mean is None else f"{mean:.6f} ± {std:.6f}")
        lines.append(f"| {row['method']} | " + " | ".join(values) + " |")
    lines += ["", "## Primary metrics by seed", "",
              "| Method | Seed | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['seed']} | " +
                     " | ".join(fmt(row[key]) for key in PRIMARY) + " |")
    lines += ["", "## Class-wise F1 and TPR@1%FPR (mean ± std)", ""]
    for row in summary:
        lines += [f"### {row['method']}", "", "| Class | F1 | TPR@1%FPR |", "|---|---:|---:|"]
        for index, class_name in enumerate(CLASS_NAMES):
            f1, tpr = f"f1_class_{index}", f"tpr_at_1_fpr_class_{index}"
            lines.append(f"| {class_name} | {row[f1 + '_mean']:.6f} ± {row[f1 + '_std']:.6f} "
                         f"| {row[tpr + '_mean']:.6f} ± {row[tpr + '_std']:.6f} |")
        lines.append("")
    lines += ["## Auxiliary metrics", "",
              "Complete Creator and direction-specific Editor MSE/correlation/AUROC/TPR fields "
              "are stored in `summary.csv`, `per_seed.csv`, and `summary.json`. A dash indicates "
              "that the corresponding branch is not present in that method.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/pasted_race"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports/creator_editor_p5_p6_final"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_dir)
    summary = summarize(rows)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"seeds": SEEDS, "per_seed": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "README.md", rows, summary)
    print(args.output_dir / "README.md")


if __name__ == "__main__":
    main()

