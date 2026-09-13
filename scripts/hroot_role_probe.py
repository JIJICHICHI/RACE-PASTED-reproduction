#!/usr/bin/env python3
"""Linear probes on frozen RACE h_root features for creator/editor roles."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from models.flexible_model import FlexibleBaselineModel, RACEModel
from utils.flexible_dataset import FlexibleGraphDataset


def role_labels(labels: np.ndarray, role: str) -> np.ndarray:
    if role == "creator":
        return np.isin(labels, [2, 3]).astype(int)
    if role == "editor":
        return np.isin(labels, [1, 2]).astype(int)
    raise ValueError(f"Unknown role: {role}")


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    fprs, tprs, _ = roc_curve(y_true, y_score)
    ok = np.where(fprs <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(tprs[ok[-1]])


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "tpr_at_1_fpr": tpr_at_fpr(y_true, y_score, 0.01),
    }


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


def extract_features(model, path: str, graph_config: dict, batch_size: int, num_workers: int, device: torch.device):
    dataset = FlexibleGraphDataset(file_path=path, config=graph_config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=FlexibleGraphDataset.collate_fn,
        num_workers=num_workers,
    )
    features = []
    labels = []
    ids = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Extract {Path(path).name}")):
            outputs = model(batch, batch_idx=batch_idx)
            if not isinstance(outputs, dict) or "features" not in outputs:
                raise RuntimeError("Model did not return h_root features.")
            features.append(outputs["features"].detach().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            ids.extend(batch["id"])
    return {
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "ids": ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, train_args, graph_config = load_model(args.checkpoint, device)
    paths = {
        "train": train_args["data_path"],
        "val": train_args["val_data_path"],
        "test": train_args["test_data_path"],
    }

    splits = {
        name: extract_features(model, path, graph_config, args.batch_size, args.num_workers, device)
        for name, path in paths.items()
    }

    results = {
        "checkpoint": str(args.checkpoint),
        "feature_dim": int(splits["train"]["features"].shape[1]),
        "split_sizes": {name: int(data["labels"].shape[0]) for name, data in splits.items()},
        "probes": {},
    }

    for role in ["creator", "editor"]:
        y_train = role_labels(splits["train"]["labels"], role)
        y_val = role_labels(splits["val"]["labels"], role)
        y_test = role_labels(splits["test"]["labels"], role)

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(splits["train"]["features"], y_train)
        val_scores = clf.predict_proba(splits["val"]["features"])[:, 1]
        test_scores = clf.predict_proba(splits["test"]["features"])[:, 1]

        results["probes"][role] = {
            "target": "LLM creator" if role == "creator" else "LLM editor",
            "train_positive": int(y_train.sum()),
            "val": binary_metrics(y_val, val_scores),
            "test": binary_metrics(y_test, test_scores),
        }

    with (args.output_dir / "hroot_role_probe.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = ["# h_root Linear Probe Diagnostics\n", f"Checkpoint: `{args.checkpoint}`\n"]
    lines.append("| Representation | Probe Target | Split | Acc | F1 | AUROC | TPR@1%FPR |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for role, payload in results["probes"].items():
        for split in ["val", "test"]:
            m = payload[split]
            lines.append(
                f"| h_root | {payload['target']} | {split} | "
                f"{100*m['accuracy']:.2f} | {100*m['macro_f1']:.2f} | "
                f"{100*m['auroc']:.2f} | {100*m['tpr_at_1_fpr']:.2f} |"
            )
    (args.output_dir / "hroot_role_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
