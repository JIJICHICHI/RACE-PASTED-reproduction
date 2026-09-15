"""Select the P3 input mode by three-seed mean validation MSE."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    modes = ("edu", "edu_root", "edu_root_interaction")
    seeds = (42, 2026, 3407)
    summary = {}
    for mode in modes:
        rows = []
        for seed in seeds:
            path = Path(args.results_dir) / f"creator_retention_p3_{mode}_seed{seed}" / "metrics.json"
            with path.open(encoding="utf-8") as input_file:
                metrics = json.load(input_file)
            rows.append(metrics["val"])
        summary[mode] = {
            "mean_val_mse": statistics.mean(row["mse"] for row in rows),
            "mean_val_spearman": statistics.mean(row["spearman"] for row in rows),
            "per_seed": {str(seed): row for seed, row in zip(seeds, rows)},
        }
    selected = min(
        modes,
        key=lambda mode: (
            summary[mode]["mean_val_mse"],
            -summary[mode]["mean_val_spearman"],
        ),
    )
    report = {"selection_rule": "min mean validation MSE; max mean Spearman tie-break", "selected_mode": selected, "modes": summary}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
    print(selected)


if __name__ == "__main__":
    main()
