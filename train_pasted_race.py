"""Train and evaluate the minimal PASTED-style lexical trace branch on RACE."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score, roc_curve
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, Subset
from transformers import get_linear_schedule_with_warmup

from models.flexible_model import RACEModel
from train import get_metadata_from_file
from utils.flexible_dataset import FlexibleGraphDataset


LOGGER = logging.getLogger("pasted_race")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(dataset, limit: int | None, seed: int, shuffle: bool):
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:limit])


def masked_mse(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    device: torch.device,
    allow_empty: bool = False,
) -> torch.Tensor:
    """Strict masked mean MSE over valid EDUs in a batch."""
    squared_error_sum = torch.zeros((), device=device)
    valid_count = torch.zeros((), device=device)
    for prediction, target, mask in zip(predictions, targets, masks):
        length = min(prediction.numel(), target.numel(), mask.numel())
        if length == 0:
            continue
        target = target[:length].to(device)
        mask = mask[:length].to(device)
        squared_error_sum += (((prediction[:length] - target) ** 2) * mask).sum()
        valid_count += mask.sum()
    if valid_count.item() == 0:
        if allow_empty:
            return squared_error_sum
        raise ValueError("Batch contains no valid EDU lexical targets")
    return squared_error_sum / valid_count


def _safe_correlation(function, targets: np.ndarray, predictions: np.ndarray) -> float:
    if len(targets) < 2 or np.std(targets) == 0 or np.std(predictions) == 0:
        return float("nan")
    return float(function(targets, predictions).statistic)


def calculate_metrics(targets: list[float], predictions: list[float]) -> dict[str, Any]:
    y_true = np.asarray(targets, dtype=np.float64)
    y_score = np.asarray(predictions, dtype=np.float64)
    binary = (y_true > 1e-8).astype(np.int64)
    metrics: dict[str, Any] = {
        "valid_edus": int(len(y_true)),
        "positive_edus": int(binary.sum()),
        "mse": float(mean_squared_error(y_true, y_score)) if len(y_true) else float("nan"),
        "pearson": _safe_correlation(pearsonr, y_true, y_score),
        "spearman": _safe_correlation(spearmanr, y_true, y_score),
        "auroc": float("nan"),
        "tpr_at_1pct_fpr": float("nan"),
    }
    if len(np.unique(binary)) == 2:
        metrics["auroc"] = float(roc_auc_score(binary, y_score))
        fpr, tpr, _ = roc_curve(binary, y_score)
        eligible = tpr[fpr < 0.01]
        metrics["tpr_at_1pct_fpr"] = float(eligible[-1]) if len(eligible) else 0.0
    return metrics


def calculate_detection_metrics(
    labels: list[int], scores: list[float], prefix: str = "document"
) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) != 2:
        return {
            f"{prefix}_auroc": float("nan"),
            f"{prefix}_tpr_at_1pct_fpr": float("nan"),
        }
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = tpr[fpr < 0.01]
    return {
        f"{prefix}_auroc": float(roc_auc_score(y_true, y_score)),
        f"{prefix}_tpr_at_1pct_fpr": float(eligible[-1]) if len(eligible) else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: RACEModel,
    loader: DataLoader,
    device: torch.device,
    save_predictions: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    flat_targets: list[float] = []
    flat_predictions: list[float] = []
    document_labels: list[int] = []
    document_scores: list[float] = []
    classifier_scores: list[float] = []
    classifier_predictions: list[int] = []
    records: list[dict[str, Any]] = []
    loss_sum = 0.0
    batches = 0
    for batch in loader:
        output = model(batch)
        scores = output["lexical_trace_scores"]
        class_probabilities = torch.softmax(output["logits"], dim=-1)
        loss = masked_mse(
            scores, batch["edu_lexical_scores"], batch["edu_lexical_masks"], device
        )
        loss_sum += float(loss.item())
        batches += 1
        for item_id, label, class_probability, logits, prediction, target, mask in zip(
            batch["id"], batch["labels"], class_probabilities, output["logits"], scores,
            batch["edu_lexical_scores"], batch["edu_lexical_masks"]
        ):
            length = min(prediction.numel(), target.numel(), mask.numel())
            prediction_values = prediction[:length].detach().cpu().tolist()
            target_values = target[:length].tolist()
            mask_values = mask[:length].tolist()
            for pred, gold, valid in zip(prediction_values, target_values, mask_values):
                if valid:
                    flat_predictions.append(float(pred))
                    flat_targets.append(float(gold))
            valid_predictions = [
                pred for pred, valid in zip(prediction_values, mask_values) if valid
            ]
            if valid_predictions:
                document_labels.append(int(label.item()))
                document_scores.append(float(np.mean(valid_predictions)))
                classifier_scores.append(float(class_probability[1].item()))
                classifier_predictions.append(int(torch.argmax(class_probability).item()))
            records.append(
                {
                    "item_id": item_id,
                    "label": int(label.item()),
                    "logits": logits.detach().cpu().tolist(),
                    "class_1_probability": float(class_probability[1].item()),
                    "prediction": prediction_values,
                    "target": target_values,
                    "mask": [int(x) for x in mask_values],
                }
            )
    metrics = calculate_metrics(flat_targets, flat_predictions)
    metrics.update(calculate_detection_metrics(document_labels, document_scores))
    if model.use_lexical_fusion:
        metrics.update(
            calculate_detection_metrics(
                document_labels, classifier_scores, prefix="classifier"
            )
        )
        metrics["classifier_accuracy"] = float(
            accuracy_score(document_labels, classifier_predictions)
        )
        metrics["classifier_f1"] = float(
            f1_score(document_labels, classifier_predictions)
        )
    metrics["loss"] = loss_sum / max(batches, 1)
    metrics["documents"] = len(records)
    if save_predictions is not None:
        with save_predictions.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(42)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.data_dir is not None:
        config["train_file"] = os.path.join(args.data_dir, "train_graph.jsonl")
        config["val_file"] = os.path.join(args.data_dir, "val_graph.jsonl")
        config["test_file"] = os.path.join(args.data_dir, "test_graph.jsonl")
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if config.get("device", "cuda").startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(config.get("device", "cuda"))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as output_file:
        json.dump(config, output_file, ensure_ascii=False, indent=2)

    model_config = dict(config["graph"])
    model_config.update(
        {
            "backbone_model_path": config["backbone_model_path"],
            "bert_dim": config.get("bert_dim", 768),
            "max_seq_length": config.get("max_seq_length", 512),
            "num_gnn_layers": config.get("num_gnn_layers", 2),
            "unfreeze_bert_layers": config.get("unfreeze_bert_layers", 1),
        }
    )
    train_dataset = FlexibleGraphDataset(config["train_file"], model_config)
    val_dataset = FlexibleGraphDataset(config["val_file"], model_config)
    test_dataset = FlexibleGraphDataset(config["test_file"], model_config)
    train_dataset = limit_dataset(train_dataset, args.max_train_samples, seed, True)
    val_dataset = limit_dataset(val_dataset, args.max_eval_samples, seed, False)
    test_dataset = limit_dataset(test_dataset, args.max_eval_samples, seed, False)

    workers = int(config.get("num_workers", 0))
    train_loader = make_loader(train_dataset, int(config["batch_size"]), True, workers)
    val_loader = make_loader(val_dataset, int(config["eval_batch_size"]), False, workers)
    test_loader = make_loader(test_dataset, int(config["eval_batch_size"]), False, workers)

    metadata = get_metadata_from_file(config["relation_table"], model_config)
    model = RACEModel(
        feature_dim=int(config.get("feature_dim", 128)),
        gnn_hidden_dim=int(config.get("gnn_hidden_dim", 512)),
        num_heads=int(config.get("num_heads", 4)),
        num_classes=2,
        config=model_config,
        metadata=metadata,
    ).to(device)
    joint_classification = bool(config.get("joint_classification", False))
    if not joint_classification:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = False

    initialization_checkpoint = config.get("initialization_checkpoint")
    if initialization_checkpoint and not args.eval_only:
        initialization = torch.load(
            initialization_checkpoint, map_location=device, weights_only=False
        )
        excluded_prefixes = ("classifier.", "lexical_fusion_proj.")
        transferable_state = {
            key: value
            for key, value in initialization["model_state_dict"].items()
            if not key.startswith(excluded_prefixes)
        }
        incompatible = model.load_state_dict(transferable_state, strict=False)
        LOGGER.info(
            "initialized lexical backbone from %s (missing=%d unexpected=%d)",
            initialization_checkpoint,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    checkpoint_path = output_dir / "best_model.pt"
    if args.eval_only:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        artifact_suffix = "_subset" if args.max_eval_samples is not None else ""
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            output_dir / f"val_predictions{artifact_suffix}.jsonl",
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            output_dir / f"test_predictions{artifact_suffix}.jsonl",
        )
        final_metrics = {
            "best_epoch": checkpoint["epoch"],
            "val": val_metrics,
            "test": test_metrics,
        }
        with (output_dir / f"metrics{artifact_suffix}.json").open(
            "w", encoding="utf-8"
        ) as output_file:
            json.dump(final_metrics, output_file, ensure_ascii=False, indent=2)
        print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
        return

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=float(config.get("learning_rate", 2.5e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    total_steps = max(1, len(train_loader) * int(config["num_epochs"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config.get("warmup_ratio", 0.1))),
        num_training_steps=total_steps,
    )

    best_auc = -float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(int(config["num_epochs"])):
        model.train()
        running_loss = 0.0
        running_trace_loss = 0.0
        running_classification_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, epoch=epoch)
            trace_loss = masked_mse(
                output["lexical_trace_scores"],
                batch["edu_lexical_scores"],
                batch["edu_lexical_masks"],
                device,
            )
            classification_active = joint_classification and epoch >= int(
                config.get("classification_warmup_epochs", 1)
            )
            if classification_active:
                classification_loss = torch.nn.functional.cross_entropy(
                    output["logits"], batch["labels"].to(device)
                )
                loss = classification_loss + float(
                    config.get("lexical_loss_weight", 0.2)
                ) * trace_loss
            else:
                classification_loss = torch.zeros((), device=device)
                loss = trace_loss
            loss.backward()
            clip_grad_norm_(trainable, float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            running_trace_loss += float(trace_loss.item())
            running_classification_loss += float(classification_loss.item())
            if step % int(config.get("logging_steps", 25)) == 0:
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.6f ce=%.6f trace=%.6f",
                    epoch + 1,
                    step,
                    len(train_loader),
                    running_loss / step,
                    running_classification_loss / step,
                    running_trace_loss / step,
                )

        val_metrics = evaluate(model, val_loader, device)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "train_classification_loss": running_classification_loss
            / max(len(train_loader), 1),
            "train_trace_loss": running_trace_loss / max(len(train_loader), 1),
            "val": val_metrics,
        }
        history.append(epoch_record)
        LOGGER.info("epoch=%d metrics=%s", epoch + 1, json.dumps(epoch_record))

        auc = (
            val_metrics["classifier_auroc"]
            if joint_classification
            else val_metrics["auroc"]
        )
        score = -val_metrics["mse"] if np.isnan(auc) else auc
        if score > best_auc:
            best_auc = score
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "model_config": model_config,
                    "metadata": metadata,
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= int(config.get("early_stopping_patience", 3)):
                LOGGER.info("early stopping after epoch %d", epoch + 1)
                break

    with (output_dir / "history.json").open("w", encoding="utf-8") as output_file:
        json.dump(history, output_file, ensure_ascii=False, indent=2)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(model, val_loader, device, output_dir / "val_predictions.jsonl")
    test_metrics = evaluate(model, test_loader, device, output_dir / "test_predictions.jsonl")
    final_metrics = {
        "best_epoch": checkpoint["epoch"],
        "val": val_metrics,
        "test": test_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(final_metrics, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
