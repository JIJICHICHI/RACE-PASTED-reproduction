#!/usr/bin/env python3
"""Analyze role hierarchy using frozen pretrained text encoder embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


CLASS_NAMES = {0: "Human-Written", 1: "LLM-Polished", 2: "LLM-Generated", 3: "Humanized"}
LABEL_MAP = {"human_written": 0, "human_ai_polished": 1, "ai_generated": 2, "ai_humanized": 3}


def creator_labels(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, [2, 3]).astype(int)


def editor_labels(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, [1, 2]).astype(int)


def ai_involvement_labels(labels: np.ndarray) -> np.ndarray:
    return (labels != 0).astype(int)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class HartTextDataset(Dataset):
    def __init__(self, path: str, split: str):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                label = LABEL_MAP[row["label"].lower()]
                self.items.append({
                    "id": row.get("item_id"),
                    "text": row.get("article", ""),
                    "label": label,
                    "split": split,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_text(batch):
    return {
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "labels": np.asarray([x["label"] for x in batch], dtype=np.int64),
        "splits": np.asarray([x["split"] for x in batch], dtype=object),
    }


def pool_outputs(outputs, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
    raise ValueError("pooling must be cls or mean")


def extract_embeddings(args, device):
    if args.cache_path.exists() and not args.rebuild_cache:
        data = np.load(args.cache_path, allow_pickle=True)
        return {k: data[k] for k in ["features", "labels", "splits", "ids"]}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, use_safetensors=args.use_safetensors)
    model.to(device)
    model.eval()

    paths = {"train": args.train_path, "val": args.val_path, "test": args.test_path}
    features, labels, splits, ids = [], [], [], []
    with torch.no_grad():
        for split, path in paths.items():
            ds = HartTextDataset(path, split)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_text, num_workers=args.num_workers)
            for batch in tqdm(loader, desc=f"{args.model_name} {split}"):
                texts = [args.text_prefix + t for t in batch["texts"]]
                enc = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                pooled = pool_outputs(out, enc["attention_mask"], args.pooling)
                features.append(pooled.detach().cpu().numpy())
                labels.append(batch["labels"])
                splits.append(batch["splits"])
                ids.extend(batch["ids"])

    data = {
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "splits": np.concatenate(splits, axis=0),
        "ids": np.asarray(ids, dtype=object),
    }
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache_path, **data)
    return data


def class_similarity_matrix(z, labels):
    matrix = np.zeros((4, 4), dtype=float)
    counts = np.zeros((4, 4), dtype=int)
    for a in range(4):
        za = z[labels == a]
        for b in range(4):
            zb = z[labels == b]
            sims = za @ zb.T
            if a == b:
                vals = sims[~np.eye(len(za), dtype=bool)]
            else:
                vals = sims.ravel()
            matrix[a, b] = float(vals.mean())
            counts[a, b] = int(len(vals))
    return {"labels": [CLASS_NAMES[i] for i in range(4)], "mean_cosine": matrix.tolist(), "pair_counts": counts.tolist()}


def relation_pair_pools(labels, seed, max_pairs):
    creator = creator_labels(labels)
    editor = editor_labels(labels)
    n = len(labels)
    pools = {"same_label": [], "same_creator": [], "same_editor": [], "shared_role": [], "no_shared_role": []}
    rng = np.random.default_rng(seed)
    max_attempts = max(200000, max_pairs * 100)
    attempts = 0
    while attempts < max_attempts and any(len(v) < max_pairs for v in pools.values()):
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
            if len(pools["same_label"]) < max_pairs:
                pools["same_label"].append((i, j))
        elif same_creator:
            if len(pools["same_creator"]) < max_pairs:
                pools["same_creator"].append((i, j))
            if len(pools["shared_role"]) < max_pairs:
                pools["shared_role"].append((i, j))
        elif same_editor:
            if len(pools["same_editor"]) < max_pairs:
                pools["same_editor"].append((i, j))
            if len(pools["shared_role"]) < max_pairs:
                pools["shared_role"].append((i, j))
        else:
            if len(pools["no_shared_role"]) < max_pairs:
                pools["no_shared_role"].append((i, j))
    return {k: np.asarray(v, dtype=np.int64) for k, v in pools.items()}


def pair_sims(z, pairs):
    return np.sum(z[pairs[:, 0]] * z[pairs[:, 1]], axis=1) if len(pairs) else np.asarray([], dtype=float)


def hierarchy_metrics(z, labels, seed, max_pairs, compare_samples):
    rng = np.random.default_rng(seed)
    pools = relation_pair_pools(labels, seed, max_pairs)
    sims = {k: pair_sims(z, v) for k, v in pools.items()}

    def compare(left, right):
        k = min(compare_samples, len(left), len(right))
        l = left[rng.choice(len(left), size=k, replace=False)]
        r = right[rng.choice(len(right), size=k, replace=False)]
        return float((l > r).mean())

    out = {
        "sampled_pair_counts": {k: int(len(v)) for k, v in sims.items()},
        "mean_similarity": {k: float(v.mean()) for k, v in sims.items()},
        "median_similarity": {k: float(np.median(v)) for k, v in sims.items()},
        "pairwise_hierarchy_accuracy": {
            "same_label_gt_shared_role": compare(sims["same_label"], sims["shared_role"]),
            "shared_role_gt_no_shared_role": compare(sims["shared_role"], sims["no_shared_role"]),
            "same_creator_gt_no_shared_role": compare(sims["same_creator"], sims["no_shared_role"]),
            "same_editor_gt_no_shared_role": compare(sims["same_editor"], sims["no_shared_role"]),
        },
    }
    out["creator_minus_editor_mean_similarity"] = out["mean_similarity"]["same_creator"] - out["mean_similarity"]["same_editor"]
    return out


def clustering_metrics(z, labels, seed, max_samples):
    rng = np.random.default_rng(seed)
    if len(z) > max_samples:
        idx = rng.choice(len(z), size=max_samples, replace=False)
        z = z[idx]
        labels = labels[idx]
    label_sets = {
        "final_4class": labels,
        "creator": creator_labels(labels),
        "editor": editor_labels(labels),
        "ai_involvement": ai_involvement_labels(labels),
    }
    out = {}
    for name, y in label_sets.items():
        out[name] = {
            "n": int(len(y)),
            "silhouette_cosine": float(silhouette_score(z, y, metric="cosine")),
            "davies_bouldin": float(davies_bouldin_score(z, y)),
            "calinski_harabasz": float(calinski_harabasz_score(z, y)),
        }
    return out


def analyze_subset(data, split_name, args):
    mask = np.ones(len(data["labels"]), dtype=bool) if split_name == "all" else data["splits"] == split_name
    labels = data["labels"][mask]
    z = normalize(data["features"][mask])
    return {
        "n": int(len(labels)),
        "class_counts": {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(4)},
        "class_similarity_matrix": class_similarity_matrix(z, labels),
        "hierarchy_metrics": hierarchy_metrics(z, labels, args.seed, args.max_pairs, args.compare_samples),
        "clustering_metrics": clustering_metrics(z, labels, args.seed, args.max_cluster_samples),
    }


def write_markdown(result, path):
    lines = [f"# Frozen Pretrained Encoder Role Hierarchy Similarity\n", f"Model: `{result['model_name']}` | pooling: `{result['pooling']}`\n"]
    for split, payload in result["subsets"].items():
        lines.append(f"## {split}\n")
        lines.append(f"N = {payload['n']}\n")
        labels = payload["class_similarity_matrix"]["labels"]
        matrix = payload["class_similarity_matrix"]["mean_cosine"]
        lines.append("### Mean Cosine Similarity Matrix")
        lines.append("| Class | " + " | ".join(labels) + " |")
        lines.append("|---" + "|---:" * len(labels) + "|")
        for name, row in zip(labels, matrix):
            lines.append("| " + name + " | " + " | ".join(f"{x:.4f}" for x in row) + " |")
        hm = payload["hierarchy_metrics"]
        lines.append("\n### Pair Similarity Means")
        lines.append("| Pair Type | Mean | Median | Sampled Pairs |")
        lines.append("|---|---:|---:|---:|")
        for key in ["same_label", "same_creator", "same_editor", "shared_role", "no_shared_role"]:
            lines.append(f"| {key} | {hm['mean_similarity'][key]:.4f} | {hm['median_similarity'][key]:.4f} | {hm['sampled_pair_counts'][key]} |")
        lines.append("\n### Hierarchy Accuracy")
        lines.append("| Comparison | Accuracy |")
        lines.append("|---|---:|")
        for key, value in hm["pairwise_hierarchy_accuracy"].items():
            lines.append(f"| {key} | {100*value:.2f} |")
        lines.append(f"\ncreator_minus_editor_mean_similarity = {hm['creator_minus_editor_mean_similarity']:.4f}\n")
        lines.append("### Clustering Metrics")
        lines.append("| Labeling | Silhouette(cosine) | Davies-Bouldin | Calinski-Harabasz |")
        lines.append("|---|---:|---:|---:|")
        for key, m in payload["clustering_metrics"].items():
            lines.append(f"| {key} | {m['silhouette_cosine']:.4f} | {m['davies_bouldin']:.4f} | {m['calinski_harabasz']:.2f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--train-path", default="data/hart_split/train_graph.jsonl")
    parser.add_argument("--val-path", default="data/hart_split/val_graph.jsonl")
    parser.add_argument("--test-path", default="data/hart_split/test_graph.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--text-prefix", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--compare-samples", type=int, default=100000)
    parser.add_argument("--max-cluster-samples", type=int, default=5000)
    parser.add_argument("--subsets", nargs="+", default=["test", "all"], choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = extract_embeddings(args, device)
    result = {
        "model_name": args.model_name,
        "pooling": args.pooling,
        "cache_path": str(args.cache_path),
        "class_order": CLASS_NAMES,
        "subsets": {split: analyze_subset(data, split, args) for split in args.subsets},
    }
    with (args.output_dir / "pretrained_role_hierarchy_similarity.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_markdown(result, args.output_dir / "pretrained_role_hierarchy_similarity.md")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
