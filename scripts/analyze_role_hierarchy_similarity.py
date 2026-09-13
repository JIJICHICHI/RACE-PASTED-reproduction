#!/usr/bin/env python3
"""Analyze creator/editor hierarchy in frozen RACE document representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from models.flexible_model import FlexibleBaselineModel, RACEModel
from utils.flexible_dataset import FlexibleGraphDataset


CLASS_NAMES = {
    0: "Human-Written",
    1: "LLM-Polished",
    2: "LLM-Generated",
    3: "Humanized",
}


def creator_labels(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, [2, 3]).astype(int)


def editor_labels(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, [1, 2]).astype(int)


def ai_involvement_labels(labels: np.ndarray) -> np.ndarray:
    """0 = no AI final/origin involvement, 1 = AI involved."""
    return (labels != 0).astype(int)


def normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


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
    return model, train_args, graph_config


def extract_split(model, path: str, graph_config: dict, batch_size: int, num_workers: int, device: torch.device):
    dataset = FlexibleGraphDataset(file_path=path, config=graph_config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=FlexibleGraphDataset.collate_fn,
        num_workers=num_workers,
    )
    features, labels, ids, domains = [], [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Extract {Path(path).name}")):
            outputs = model(batch, batch_idx=batch_idx)
            if not isinstance(outputs, dict) or "features" not in outputs:
                raise RuntimeError("Model did not return features.")
            features.append(outputs["features"].detach().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            ids.extend(batch["id"])
            domains.extend([str(x).split("-")[0] for x in batch["id"]])
    return {
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "ids": np.asarray(ids, dtype=object),
        "domains": np.asarray(domains, dtype=object),
    }


def load_or_extract_features(args, device: torch.device):
    cache_path = args.cache_path
    if cache_path.exists() and not args.rebuild_cache:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "features": data["features"],
            "labels": data["labels"],
            "splits": data["splits"],
            "ids": data["ids"],
            "domains": data["domains"],
        }

    model, train_args, graph_config = load_model(args.checkpoint, device)
    paths = {
        "train": train_args["data_path"],
        "val": train_args["val_data_path"],
        "test": train_args["test_data_path"],
    }
    extracted = {}
    for split, path in paths.items():
        extracted[split] = extract_split(
            model, path, graph_config, args.batch_size, args.num_workers, device
        )
    features = np.concatenate([extracted[s]["features"] for s in paths], axis=0)
    labels = np.concatenate([extracted[s]["labels"] for s in paths], axis=0)
    splits = np.concatenate([
        np.asarray([s] * len(extracted[s]["labels"]), dtype=object) for s in paths
    ])
    ids = np.concatenate([extracted[s]["ids"] for s in paths], axis=0)
    domains = np.concatenate([extracted[s]["domains"] for s in paths], axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels, splits=splits, ids=ids, domains=domains)
    return {"features": features, "labels": labels, "splits": splits, "ids": ids, "domains": domains}


def class_similarity_matrix(z: np.ndarray, labels: np.ndarray) -> dict:
    matrix = np.zeros((4, 4), dtype=float)
    counts = np.zeros((4, 4), dtype=int)
    for a in range(4):
        za = z[labels == a]
        for b in range(4):
            zb = z[labels == b]
            if len(za) == 0 or len(zb) == 0:
                matrix[a, b] = np.nan
                continue
            sims = za @ zb.T
            if a == b:
                mask = ~np.eye(len(za), dtype=bool)
                vals = sims[mask]
            else:
                vals = sims.ravel()
            matrix[a, b] = float(vals.mean()) if len(vals) else np.nan
            counts[a, b] = int(len(vals))
    return {
        "labels": [CLASS_NAMES[i] for i in range(4)],
        "mean_cosine": matrix.tolist(),
        "pair_counts": counts.tolist(),
    }


def relation_pair_pools(labels: np.ndarray, seed: int, max_pairs: int):
    creator = creator_labels(labels)
    editor = editor_labels(labels)
    n = len(labels)
    pools = {"same_label": [], "same_creator": [], "same_editor": [], "shared_role": [], "no_shared_role": []}

    rng = np.random.default_rng(seed)
    target = max_pairs
    max_attempts = max(200000, target * 80)
    attempts = 0
    while attempts < max_attempts and any(len(v) < target for v in pools.values()):
        attempts += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i

        same_label = labels[i] == labels[j]
        same_creator = creator[i] == creator[j]
        same_editor = editor[i] == editor[j]
        if same_label:
            if len(pools["same_label"]) < target:
                pools["same_label"].append((i, j))
        elif same_creator:
            if len(pools["same_creator"]) < target:
                pools["same_creator"].append((i, j))
            if len(pools["shared_role"]) < target:
                pools["shared_role"].append((i, j))
        elif same_editor:
            if len(pools["same_editor"]) < target:
                pools["same_editor"].append((i, j))
            if len(pools["shared_role"]) < target:
                pools["shared_role"].append((i, j))
        else:
            if len(pools["no_shared_role"]) < target:
                pools["no_shared_role"].append((i, j))
    return {k: np.asarray(v, dtype=np.int64) for k, v in pools.items()}


def sample_pair_sims(z: np.ndarray, pairs: np.ndarray, rng: np.random.Generator, max_pairs: int) -> np.ndarray:
    if len(pairs) == 0:
        return np.asarray([], dtype=float)
    if len(pairs) > max_pairs:
        pairs = pairs[rng.choice(len(pairs), size=max_pairs, replace=False)]
    return np.sum(z[pairs[:, 0]] * z[pairs[:, 1]], axis=1)


def hierarchy_metrics(z: np.ndarray, labels: np.ndarray, seed: int, max_pairs: int, compare_samples: int) -> dict:
    rng = np.random.default_rng(seed)
    pools = relation_pair_pools(labels, seed, max_pairs)
    sims = {k: sample_pair_sims(z, v, rng, max_pairs) for k, v in pools.items()}
    out = {
        "pair_counts": {k: int(len(v)) for k, v in pools.items()},
        "sampled_pair_counts": {k: int(len(v)) for k, v in sims.items()},
        "mean_similarity": {k: float(v.mean()) if len(v) else None for k, v in sims.items()},
        "median_similarity": {k: float(np.median(v)) if len(v) else None for k, v in sims.items()},
    }

    def compare(left: np.ndarray, right: np.ndarray) -> float | None:
        if len(left) == 0 or len(right) == 0:
            return None
        k = min(compare_samples, len(left), len(right))
        left_s = left[rng.choice(len(left), size=k, replace=False)]
        right_s = right[rng.choice(len(right), size=k, replace=False)]
        return float((left_s > right_s).mean())

    out["pairwise_hierarchy_accuracy"] = {
        "same_label_gt_shared_role": compare(sims["same_label"], sims["shared_role"]),
        "shared_role_gt_no_shared_role": compare(sims["shared_role"], sims["no_shared_role"]),
        "same_creator_gt_no_shared_role": compare(sims["same_creator"], sims["no_shared_role"]),
        "same_editor_gt_no_shared_role": compare(sims["same_editor"], sims["no_shared_role"]),
    }
    c_mean = out["mean_similarity"]["same_creator"]
    e_mean = out["mean_similarity"]["same_editor"]
    out["creator_minus_editor_mean_similarity"] = None if c_mean is None or e_mean is None else float(c_mean - e_mean)
    return out


def clustering_metrics(z: np.ndarray, labels: np.ndarray, seed: int, max_samples: int) -> dict:
    rng = np.random.default_rng(seed)
    if len(z) > max_samples:
        idx = rng.choice(len(z), size=max_samples, replace=False)
        z_eval = z[idx]
        labels_eval = labels[idx]
    else:
        z_eval = z
        labels_eval = labels

    label_sets = {
        "final_4class": labels_eval,
        "creator": creator_labels(labels_eval),
        "editor": editor_labels(labels_eval),
        "ai_involvement": ai_involvement_labels(labels_eval),
    }
    out = {}
    for name, y in label_sets.items():
        if len(np.unique(y)) < 2:
            continue
        out[name] = {
            "n": int(len(y)),
            "silhouette_cosine": float(silhouette_score(z_eval, y, metric="cosine")),
            "davies_bouldin": float(davies_bouldin_score(z_eval, y)),
            "calinski_harabasz": float(calinski_harabasz_score(z_eval, y)),
        }
    return out


def analyze_subset(data: dict, split_name: str, args) -> dict:
    mask = np.ones(len(data["labels"]), dtype=bool) if split_name == "all" else data["splits"] == split_name
    features = data["features"][mask]
    labels = data["labels"][mask]
    z = normalize(features)
    return {
        "n": int(len(labels)),
        "class_counts": {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(4)},
        "class_similarity_matrix": class_similarity_matrix(z, labels),
        "hierarchy_metrics": hierarchy_metrics(z, labels, args.seed, args.max_pairs, args.compare_samples),
        "clustering_metrics": clustering_metrics(z, labels, args.seed, args.max_cluster_samples),
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = ["# Role Hierarchy Similarity Analysis\n"]
    lines.append(f"Checkpoint: `{result['checkpoint']}`\n")
    for split, payload in result["subsets"].items():
        lines.append(f"## {split}\n")
        lines.append(f"N = {payload['n']}\n")
        labels = payload["class_similarity_matrix"]["labels"]
        matrix = payload["class_similarity_matrix"]["mean_cosine"]
        lines.append("### Mean Cosine Similarity Matrix\n")
        lines.append("| Class | " + " | ".join(labels) + " |")
        lines.append("|---" + "|---:" * len(labels) + "|")
        for name, row in zip(labels, matrix):
            lines.append("| " + name + " | " + " | ".join(f"{x:.4f}" for x in row) + " |")
        hm = payload["hierarchy_metrics"]
        lines.append("\n### Pair Similarity Means\n")
        lines.append("| Pair Type | Mean | Median | Sampled Pairs |")
        lines.append("|---|---:|---:|---:|")
        for key in ["same_label", "same_creator", "same_editor", "shared_role", "no_shared_role"]:
            mean = hm["mean_similarity"][key]
            med = hm["median_similarity"][key]
            lines.append(
                f"| {key} | {mean:.4f} | {med:.4f} | {hm['sampled_pair_counts'][key]} |"
            )
        lines.append("\n### Hierarchy Accuracy\n")
        lines.append("| Comparison | Accuracy |")
        lines.append("|---|---:|")
        for key, value in hm["pairwise_hierarchy_accuracy"].items():
            lines.append(f"| {key} | {100*value:.2f} |")
        lines.append(
            f"\ncreator_minus_editor_mean_similarity = {hm['creator_minus_editor_mean_similarity']:.4f}\n"
        )
        lines.append("### Clustering Metrics\n")
        lines.append("| Labeling | Silhouette(cosine) | Davies-Bouldin | Calinski-Harabasz |")
        lines.append("|---|---:|---:|---:|")
        for key, m in payload["clustering_metrics"].items():
            lines.append(
                f"| {key} | {m['silhouette_cosine']:.4f} | {m['davies_bouldin']:.4f} | {m['calinski_harabasz']:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--compare-samples", type=int, default=100000)
    parser.add_argument("--max-cluster-samples", type=int, default=5000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_or_extract_features(args, device)

    result = {
        "checkpoint": str(args.checkpoint),
        "cache_path": str(args.cache_path),
        "class_order": CLASS_NAMES,
        "subsets": {
            "test": analyze_subset(data, "test", args),
            "all": analyze_subset(data, "all", args),
        },
    }
    with (args.output_dir / "role_hierarchy_similarity.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_markdown(result, args.output_dir / "role_hierarchy_similarity.md")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
