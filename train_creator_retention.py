"""Train document-level Creator Retention from edited final text only."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, Subset
from transformers import get_linear_schedule_with_warmup

from models.flexible_model import RACEModel
from train import get_metadata_from_file
from train_pasted_race import set_seed
from train_pasted_race_fourclass import _load_matching_state
from utils.flexible_dataset import FlexibleGraphDataset


LOGGER = logging.getLogger("creator_retention")


def limit_dataset(dataset, limit: int | None, seed: int, shuffle: bool):
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return Subset(dataset, indices[:limit])


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None if shuffle else SequentialSampler(dataset),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=FlexibleGraphDataset.collate_fn,
        generator=generator,
    )


def masked_mse(predictions, targets, masks, device):
    targets = targets.to(device)
    valid = masks.to(device) > 0
    if not bool(valid.any()):
        raise ValueError("Creator Retention batch contains no valid target")
    return F.mse_loss(predictions[valid], targets[valid])


def safe_corr(function, targets, predictions):
    if len(targets) < 2 or np.std(targets) == 0 or np.std(predictions) == 0:
        return float("nan")
    return float(function(targets, predictions).statistic)


@torch.no_grad()
def evaluate(model, loader, device, predictions_path: Path | None = None):
    model.eval()
    targets, predictions, records = [], [], []
    weighted_loss, valid_count = 0.0, 0
    for batch in loader:
        output = model(batch)
        scores = output["creator_retention_scores"]
        batch_targets = batch["creator_retention_scores"].to(device)
        masks = batch["creator_retention_masks"].to(device) > 0
        loss = masked_mse(scores, batch_targets, masks, device)
        count = int(masks.sum().item())
        weighted_loss += float(loss.item()) * count
        valid_count += count
        for item_id, score, target, valid in zip(
            batch["id"], scores, batch_targets, masks
        ):
            if bool(valid.item()):
                prediction = float(score.detach().cpu().item())
                gold = float(target.detach().cpu().item())
                predictions.append(prediction)
                targets.append(gold)
                records.append({"item_id": item_id, "prediction": prediction, "target": gold})
    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    metrics = {
        "documents": valid_count,
        "mse": weighted_loss / max(valid_count, 1),
        "pearson": safe_corr(pearsonr, target_array, prediction_array),
        "spearman": safe_corr(spearmanr, target_array, prediction_array),
    }
    if predictions_path is not None:
        with predictions_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_mode", choices=["edu", "edu_root", "edu_root_interaction"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline_checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)
    for key, value in {
        "seed": args.seed,
        "baseline_checkpoint": args.baseline_checkpoint,
        "output_dir": args.output_dir,
        "num_epochs": args.epochs,
    }.items():
        if value is not None:
            config[key] = value
    if args.input_mode is not None:
        config["graph"]["creator_retention_input_mode"] = args.input_mode

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as output_file:
        json.dump(config, output_file, ensure_ascii=False, indent=2)

    model_config = dict(config["graph"])
    model_config.update({
        "backbone_model_path": config["backbone_model_path"],
        "bert_dim": config.get("bert_dim", 768),
        "max_seq_length": config.get("max_seq_length", 512),
        "num_gnn_layers": config.get("num_gnn_layers", 2),
        "unfreeze_bert_layers": config.get("unfreeze_bert_layers", 1),
    })
    datasets = {
        split: FlexibleGraphDataset(config[f"{split}_file"], model_config)
        for split in ("train", "val", "test")
    }
    datasets["train"] = limit_dataset(datasets["train"], args.max_train_samples, seed, True)
    datasets["val"] = limit_dataset(datasets["val"], args.max_eval_samples, seed, False)
    datasets["test"] = limit_dataset(datasets["test"], args.max_eval_samples, seed, False)
    workers = int(config.get("num_workers", 0))
    loaders = {
        split: make_loader(
            datasets[split], int(config["batch_size"] if split == "train" else config["eval_batch_size"]),
            workers, split == "train", seed,
        )
        for split in ("train", "val", "test")
    }

    metadata = get_metadata_from_file(config["relation_table"], model_config)
    model = RACEModel(
        feature_dim=int(config.get("feature_dim", 128)),
        gnn_hidden_dim=int(config.get("gnn_hidden_dim", 512)),
        num_heads=int(config.get("num_heads", 4)),
        num_classes=4,
        config=model_config,
        metadata=metadata,
    )
    initialization_report = _load_matching_state(model, config["baseline_checkpoint"])
    for parameter in model.classifier.parameters():
        parameter.requires_grad = False
    model.to(device)
    initialization_report["creator_input_mode"] = model.creator_retention_input_mode
    with (output_dir / "initialization_report.json").open("w", encoding="utf-8") as output_file:
        json.dump(initialization_report, output_file, ensure_ascii=False, indent=2)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    total_steps = max(1, len(loaders["train"]) * int(config["num_epochs"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * float(config["warmup_ratio"])), total_steps
    )
    checkpoint_path = output_dir / "best_model.pt"
    best_mse, best_spearman, patience = float("inf"), -float("inf"), 0
    history = []
    for epoch in range(1, int(config["num_epochs"]) + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch)["creator_retention_scores"]
            loss = masked_mse(scores, batch["creator_retention_scores"], batch["creator_retention_masks"], device)
            loss.backward()
            clip_grad_norm_(trainable, float(config["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
            if step % int(config.get("logging_steps", 25)) == 0:
                LOGGER.info("epoch=%d step=%d/%d loss=%.6f", epoch, step, len(loaders["train"]), running / step)
        val_metrics = evaluate(model, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": running / max(len(loaders["train"]), 1), "val": val_metrics}
        history.append(record)
        LOGGER.info("epoch metrics=%s", json.dumps(record))
        mse, spearman = float(val_metrics["mse"]), float(val_metrics["spearman"])
        improved = mse < best_mse - 1e-12 or (abs(mse - best_mse) <= 1e-12 and spearman > best_spearman)
        if improved:
            best_mse, best_spearman, patience = mse, spearman, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "config": config, "model_config": model_config, "metadata": metadata, "val_metrics": val_metrics}, checkpoint_path)
        else:
            patience += 1
            if patience >= int(config["early_stopping_patience"]):
                LOGGER.info("early stopping after epoch %d", epoch)
                break
    with (output_dir / "history.json").open("w", encoding="utf-8") as output_file:
        json.dump(history, output_file, ensure_ascii=False, indent=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "val": evaluate(model, loaders["val"], device, output_dir / "val_predictions.jsonl"),
        "test": evaluate(model, loaders["test"], device, output_dir / "test_predictions.jsonl"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
