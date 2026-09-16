#!/usr/bin/env python3
"""Group-safe Creator/Modification diagnostics for formal Strong RACE runs.

The script performs two deliberately separate diagnostics:

1. derive binary/conditional scores from the saved four-class logits;
2. fit controlled linear probes on frozen h_root representations.

It never creates a new data split. Train/validation/test membership is read
from the exact graph JSONL paths stored in each checkpoint's ``args.json``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from models.flexible_model import FlexibleBaselineModel, RACEModel
from utils.flexible_dataset import FlexibleGraphDataset


CLASS_NAMES = {
    0: "Human",
    1: "Polished",
    2: "Generated",
    3: "Humanized",
}
LABEL_MAP = {
    "human_written": 0,
    "human_ai_polished": 1,
    "ai_generated": 2,
    "ai_humanized": 3,
}

# (negative classes, positive classes, human-readable target)
TASKS = {
    "creator": ((0, 1), (2, 3), "AI-origin"),
    "modification": ((0, 2), (1, 3), "Modified"),
    "editor_actor": ((0, 3), (1, 2), "AI final actor"),
    "human_origin_modification": ((0,), (1,), "Polished"),
    "ai_origin_modification": ((2,), (3,), "Humanized"),
    "unmodified_creator": ((0,), (2,), "Generated"),
    "modified_creator": ((1,), (3,), "Humanized"),
}
AXIS_TASKS = ("creator", "modification", "editor_actor")
CONDITIONAL_TASKS = (
    "human_origin_modification",
    "ai_origin_modification",
    "unmodified_creator",
    "modified_creator",
)
METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "auroc",
    "tpr_at_1_fpr",
    "reverse_tpr_at_1_fpr",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            item = json.loads(line)
            item_id = item.get("item_id") or item.get("id")
            if not item_id:
                raise ValueError(f"Missing item_id in {path}:{line_no}")
            raw_label = item["label"]
            label = LABEL_MAP[raw_label] if isinstance(raw_label, str) else int(raw_label)
            group_id = item.get("group_id")
            if not group_id:
                raise ValueError(f"Missing group_id in {path}:{line_no}")
            records.append((str(item_id), label, str(group_id)))
    ids = [row[0] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate item IDs in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "ids": np.asarray(ids),
        "labels": np.asarray([row[1] for row in records], dtype=np.int64),
        "groups": np.asarray([row[2] for row in records]),
        "class_counts": {CLASS_NAMES[i]: int(sum(row[1] == i for row in records)) for i in range(4)},
    }


def audit_manifests(manifests: dict[str, dict]) -> dict:
    group_sets = {split: set(payload["groups"].tolist()) for split, payload in manifests.items()}
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        count = len(group_sets[left] & group_sets[right])
        overlaps[f"{left}_{right}"] = count
        if count:
            raise ValueError(f"Group leakage detected between {left} and {right}: {count}")
    return {
        "splits": {
            split: {
                "path": payload["path"],
                "sha256": payload["sha256"],
                "documents": int(len(payload["labels"])),
                "groups": int(len(set(payload["groups"].tolist()))),
                "class_counts": payload["class_counts"],
            }
            for split, payload in manifests.items()
        },
        "group_overlap": overlaps,
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fprs, tprs, _ = roc_curve(y_true, y_score)
    valid = np.flatnonzero(fprs <= target_fpr)
    return float(tprs[valid[-1]]) if len(valid) else 0.0


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    predictions = (y_score >= 0.5).astype(np.int64)
    return {
        "n": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "tpr_at_1_fpr": tpr_at_fpr(y_true, y_score),
        # Low-FPR recall is directional. Reporting the reverse orientation is
        # required for a fair comparison with the historical G-positive
        # Generated/Humanized diagnostic.
        "reverse_tpr_at_1_fpr": tpr_at_fpr(1 - y_true, 1.0 - y_score),
    }


def task_arrays(labels: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    negative, positive, _ = TASKS[task]
    mask = np.isin(labels, negative + positive)
    targets = np.isin(labels[mask], positive).astype(np.int64)
    return mask, targets


def task_probability(probabilities: np.ndarray, labels: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    negative, positive, _ = TASKS[task]
    mask, targets = task_arrays(labels, task)
    negative_score = probabilities[:, list(negative)].sum(axis=1)
    positive_score = probabilities[:, list(positive)].sum(axis=1)
    denominator = np.maximum(negative_score + positive_score, np.finfo(np.float64).tiny)
    scores = positive_score / denominator
    return targets, scores[mask]


def load_saved_test(run_dir: Path, test_manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "test_predictions.json"
    with path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    by_id = {str(row["id"]): row for row in predictions}
    expected = set(test_manifest["ids"].tolist())
    if set(by_id) != expected:
        missing = len(expected - set(by_id))
        extra = len(set(by_id) - expected)
        raise ValueError(f"Prediction/manifest ID mismatch: missing={missing}, extra={extra}")
    ordered = [by_id[item_id] for item_id in test_manifest["ids"].tolist()]
    logits = np.asarray([row["logits"] for row in ordered], dtype=np.float64)
    if not all("features" in row for row in ordered):
        raise ValueError(f"Saved predictions in {path} do not contain h_root features")
    features = np.asarray([row["features"] for row in ordered], dtype=np.float32)
    return logits, features


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    graph_config = checkpoint["config"]
    metadata = checkpoint["metadata"]
    model_cls = FlexibleBaselineModel if train_args.get("model_type", "gnn") == "baseline" else RACEModel
    model = model_cls(
        feature_dim=train_args["feature_dim"],
        gnn_hidden_dim=train_args["gnn_hidden_dim"],
        num_heads=train_args["num_heads"],
        num_classes=train_args["num_classes"],
        config=graph_config,
        metadata=metadata,
        output_features=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, graph_config


def extract_features(
    model,
    manifest_path: Path,
    graph_config: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = FlexibleGraphDataset(file_path=str(manifest_path), config=graph_config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=FlexibleGraphDataset.collate_fn,
        num_workers=num_workers,
    )
    chunks = []
    ids = []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"Extract {manifest_path.name}")):
            outputs = model(batch, batch_idx=batch_index)
            if not isinstance(outputs, dict) or "features" not in outputs:
                raise RuntimeError("Model did not return h_root features")
            chunks.append(outputs["features"].detach().cpu().numpy().astype(np.float32))
            ids.extend(str(item_id) for item_id in batch["id"])
    return np.concatenate(chunks, axis=0), np.asarray(ids)


def cached_features(
    cache_path: Path,
    model,
    manifest: dict,
    graph_config: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached["manifest_sha256"].item() != manifest["sha256"]:
            raise ValueError(f"Stale feature cache manifest hash: {cache_path}")
        if not np.array_equal(cached["ids"], manifest["ids"]):
            raise ValueError(f"Stale feature cache ID order: {cache_path}")
        return cached["features"]
    features, ids = extract_features(
        model,
        Path(manifest["path"]),
        graph_config,
        batch_size,
        num_workers,
        device,
    )
    if not np.array_equal(ids, manifest["ids"]):
        raise ValueError(f"Extracted feature order differs from manifest: {cache_path}")
    np.savez(
        cache_path,
        features=features,
        ids=ids,
        manifest_sha256=np.asarray(manifest["sha256"]),
    )
    return features


def fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    task: str,
    c_grid: list[float],
    random_state: int,
) -> dict:
    train_mask, y_train = task_arrays(train_labels, task)
    val_mask, y_val = task_arrays(val_labels, task)
    test_mask, y_test = task_arrays(test_labels, task)
    candidates = []
    fitted = {}
    for c_value in sorted(c_grid):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            ),
        )
        classifier.fit(train_features[train_mask], y_train)
        val_scores = classifier.predict_proba(val_features[val_mask])[:, 1]
        val_result = binary_metrics(y_val, val_scores)
        candidates.append({"c": c_value, "val": val_result})
        fitted[c_value] = classifier
    # AUROC is threshold-independent and less discrete than 1%-FPR TPR on the
    # small validation Humanized subset. Prefer the simpler C on an exact tie.
    best = max(candidates, key=lambda row: (row["val"]["auroc"], -row["c"]))
    classifier = fitted[best["c"]]
    test_scores = classifier.predict_proba(test_features[test_mask])[:, 1]
    return {
        "target": TASKS[task][2],
        "selected_c": float(best["c"]),
        "selection": "maximum validation AUROC; smaller C breaks an exact tie",
        "candidates": candidates,
        "val": best["val"],
        "test": binary_metrics(y_test, test_scores),
    }


def exact_cosine_geometry(features: np.ndarray, labels: np.ndarray, task: str) -> dict:
    mask, targets = task_arrays(labels, task)
    values = features[mask].astype(np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, np.finfo(np.float64).tiny)
    sums = [values[targets == group].sum(axis=0) for group in (0, 1)]
    counts = [int((targets == group).sum()) for group in (0, 1)]
    within_sum = 0.0
    within_pairs = 0
    within_by_class = []
    for group in (0, 1):
        pair_count = counts[group] * (counts[group] - 1) // 2
        dot_sum = (float(np.dot(sums[group], sums[group])) - counts[group]) / 2.0
        within_by_class.append(dot_sum / pair_count if pair_count else None)
        within_sum += dot_sum
        within_pairs += pair_count
    between = float(np.dot(sums[0], sums[1]) / (counts[0] * counts[1]))
    within = float(within_sum / within_pairs)
    return {
        "n": int(len(targets)),
        "class_counts": counts,
        "silhouette_cosine": float(silhouette_score(values, targets, metric="cosine")),
        "within_cosine": within,
        "within_cosine_by_class": within_by_class,
        "between_cosine": between,
        "cosine_separation": within - between,
    }


def resolve_run(seed_run: str) -> tuple[int, Path]:
    seed_text, separator, run_text = seed_run.partition("=")
    if not separator:
        raise ValueError(f"Expected SEED=RUN_DIR, got {seed_run!r}")
    return int(seed_text), Path(run_text).resolve()


def resolve_manifest(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def process_seed(
    seed: int,
    run_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    c_grid: list[float],
    project_root: Path,
) -> dict:
    with (run_dir / "args.json").open("r", encoding="utf-8") as handle:
        run_args = json.load(handle)
    if int(run_args["seed"]) != seed:
        raise ValueError(f"Seed mismatch for {run_dir}: expected {seed}, found {run_args['seed']}")
    if not np.isclose(float(run_args["lr"]), 2.9e-5):
        raise ValueError(f"Run {run_dir} is not the formal lr=2.9e-5 checkpoint")
    paths = {
        "train": resolve_manifest(project_root, run_args["data_path"]),
        "val": resolve_manifest(project_root, run_args["val_data_path"]),
        "test": resolve_manifest(project_root, run_args["test_data_path"]),
    }
    manifests = {split: read_manifest(path) for split, path in paths.items()}
    audit = audit_manifests(manifests)

    logits, test_features = load_saved_test(run_dir, manifests["test"])
    probabilities = softmax(logits)
    derived = {}
    for task in TASKS:
        targets, scores = task_probability(probabilities, manifests["test"]["labels"], task)
        derived[task] = binary_metrics(targets, scores)

    seed_dir = output_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pth"
    train_cache = seed_dir / "train_hroot.npz"
    val_cache = seed_dir / "val_hroot.npz"
    if train_cache.exists() and val_cache.exists():
        # cached_features validates both manifest hash and exact item order.
        train_features = cached_features(
            train_cache, None, manifests["train"], {}, batch_size, num_workers, device
        )
        val_features = cached_features(
            val_cache, None, manifests["val"], {}, batch_size, num_workers, device
        )
    else:
        model, graph_config = load_model(checkpoint, device)
        train_features = cached_features(
            train_cache, model, manifests["train"], graph_config,
            batch_size, num_workers, device,
        )
        val_features = cached_features(
            val_cache, model, manifests["val"], graph_config,
            batch_size, num_workers, device,
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probes = {
        task: fit_probe(
            train_features,
            manifests["train"]["labels"],
            val_features,
            manifests["val"]["labels"],
            test_features,
            manifests["test"]["labels"],
            task,
            c_grid,
            seed,
        )
        for task in TASKS
    }
    geometry = {
        task: exact_cosine_geometry(test_features, manifests["test"]["labels"], task)
        for task in AXIS_TASKS
    }
    result = {
        "seed": seed,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "training_contract": {
            "lr": float(run_args["lr"]),
            "batch_size": int(run_args["batch_size"]),
            "epochs": int(run_args["epochs"]),
            "checkpoint_metric": run_args["best_checkpoint_metric"],
        },
        "audit": audit,
        "feature_dim": int(test_features.shape[1]),
        "derived_from_fourclass_logits": derived,
        "frozen_hroot_probes": probes,
        "test_hroot_geometry": geometry,
    }
    with (seed_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def summarize(values: list[float]) -> dict:
    return {
        "mean": float(mean(values)),
        "sample_std": float(stdev(values)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


def aggregate(results: list[dict]) -> dict:
    reference_hashes = {
        split: results[0]["audit"]["splits"][split]["sha256"] for split in ("train", "val", "test")
    }
    for result in results[1:]:
        hashes = {split: result["audit"]["splits"][split]["sha256"] for split in reference_hashes}
        if hashes != reference_hashes:
            raise ValueError("Formal seeds do not use identical data manifests")
    summary = {"seeds": [result["seed"] for result in results], "manifest_sha256": reference_hashes}
    for source, source_key in (
        ("fourclass", "derived_from_fourclass_logits"),
        ("hroot_probe", "frozen_hroot_probes"),
    ):
        summary[source] = {}
        for task in TASKS:
            summary[source][task] = {}
            for metric in METRIC_KEYS:
                values = []
                for result in results:
                    payload = result[source_key][task]
                    metrics = payload["test"] if source == "hroot_probe" else payload
                    values.append(metrics[metric])
                summary[source][task][metric] = summarize(values)
    summary["geometry"] = {}
    for task in AXIS_TASKS:
        summary["geometry"][task] = {}
        for metric in ("silhouette_cosine", "within_cosine", "between_cosine", "cosine_separation"):
            summary["geometry"][task][metric] = summarize(
                [result["test_hroot_geometry"][task][metric] for result in results]
            )
    return summary


def pct(summary: dict) -> str:
    return f"{100 * summary['mean']:.2f}±{100 * summary['sample_std']:.2f}"


def scalar(summary: dict) -> str:
    return f"{summary['mean']:.4f}±{summary['sample_std']:.4f}"


def write_report(summary: dict, results: list[dict], path: Path) -> None:
    lines = ["# Group-Safe Creator/Modification Diagnostics\n"]
    lines.append(
        "Formal Strong RACE checkpoints; fixed group-safe manifests; seeds "
        + "/".join(str(seed) for seed in summary["seeds"])
        + "; backbone training lr=2.9e-5. Values are mean±sample standard deviation.\n"
    )
    first_audit = results[0]["audit"]
    lines.extend(["## Data Audit\n", "| Split | Documents | Groups | Class counts |", "|---|---:|---:|---|"])
    for split in ("train", "val", "test"):
        row = first_audit["splits"][split]
        counts = ", ".join(f"{key}={value}" for key, value in row["class_counts"].items())
        lines.append(f"| {split} | {row['documents']} | {row['groups']} | {counts} |")
    overlaps = first_audit["group_overlap"]
    lines.append(
        f"\nGroup overlap: train/val={overlaps['train_val']}, "
        f"train/test={overlaps['train_test']}, val/test={overlaps['val_test']}.\n"
    )

    task_labels = {
        "creator": "Creator: H/P vs G/Hu",
        "modification": "Modification: H/G vs P/Hu",
        "editor_actor": "Editor actor: H/Hu vs P/G",
        "human_origin_modification": "Human origin: H vs P",
        "ai_origin_modification": "AI origin: G vs Hu",
        "unmodified_creator": "Unmodified: H vs G",
        "modified_creator": "Modified: P vs Hu",
    }
    for title, source, tasks in (
        ("Four-Class Probability Diagnostics", "fourclass", tuple(TASKS)),
        ("Frozen h_root Linear Probes", "hroot_probe", tuple(TASKS)),
    ):
        lines.extend([
            f"## {title}\n",
            "| Task | Accuracy | Macro-F1 | AUROC | Positive TPR@1%FPR | Negative TPR@1%FPR |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for task in tasks:
            row = summary[source][task]
            lines.append(
                f"| {task_labels[task]} | {pct(row['accuracy'])} | {pct(row['macro_f1'])} | "
                f"{pct(row['auroc'])} | {pct(row['tpr_at_1_fpr'])} | "
                f"{pct(row['reverse_tpr_at_1_fpr'])} |"
            )
        lines.append("")

    lines.extend([
        "## Frozen h_root Test Geometry\n",
        "| Axis | Silhouette (cosine) | Within cosine | Between cosine | Separation |",
        "|---|---:|---:|---:|---:|",
    ])
    for task in AXIS_TASKS:
        row = summary["geometry"][task]
        lines.append(
            f"| {task_labels[task]} | {scalar(row['silhouette_cosine'])} | "
            f"{scalar(row['within_cosine'])} | {scalar(row['between_cosine'])} | "
            f"{scalar(row['cosine_separation'])} |"
        )

    lines.extend([
        "\n## Per-Seed AI-Origin Modification\n",
        "| Seed | Source | Accuracy | Macro-F1 | AUROC | Humanized TPR@1%FPR | Generated TPR@1%FPR | Selected C |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        for source, key in (("four-class", "derived_from_fourclass_logits"), ("h_root probe", "frozen_hroot_probes")):
            payload = result[key]["ai_origin_modification"]
            metrics = payload["test"] if source == "h_root probe" else payload
            selected_c = f"{payload['selected_c']:g}" if source == "h_root probe" else "—"
            lines.append(
                f"| {result['seed']} | {source} | {100*metrics['accuracy']:.2f} | "
                f"{100*metrics['macro_f1']:.2f} | {100*metrics['auroc']:.2f} | "
                f"{100*metrics['tpr_at_1_fpr']:.2f} | "
                f"{100*metrics['reverse_tpr_at_1_fpr']:.2f} | {selected_c} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="SEED=formal Strong RACE run directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resolved_runs = [resolve_run(item) for item in args.run]
    seeds = [seed for seed, _ in resolved_runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds: {seeds}")

    results = []
    for seed, run_dir in resolved_runs:
        print(f"\n=== Seed {seed}: {run_dir} ===", flush=True)
        results.append(
            process_seed(
                seed,
                run_dir,
                args.output_dir,
                device,
                args.batch_size,
                args.num_workers,
                args.c_grid,
                project_root,
            )
        )
    results.sort(key=lambda row: row["seed"])
    summary = aggregate(results)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_report(summary, results, args.output_dir / "report.md")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
