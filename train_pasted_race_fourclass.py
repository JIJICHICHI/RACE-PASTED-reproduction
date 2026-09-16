"""Train residual PASTED-RACE on the group-safe four-class HART task."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, Subset
from transformers import get_linear_schedule_with_warmup

from models.flexible_model import RACEModel
from models.supcon_loss import SupConLoss
from train import get_metadata_from_file
from train_pasted_race import calculate_metrics, masked_mse, set_seed
from utils.flexible_dataset import FlexibleGraphDataset
from utils.custom_sampler import StratifiedBatchSampler
from utils.metrics import compute_classification_metrics


LOGGER = logging.getLogger("pasted_race_fourclass")
CLASS_NAMES = ["human", "polished", "generated", "humanized"]


def masked_scalar_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Masked mean MSE for one scalar Creator Retention target per document."""
    targets = targets.to(device)
    masks = masks.to(device)
    valid = masks > 0
    if not bool(valid.any()):
        return predictions.sum() * 0.0
    return F.mse_loss(predictions[valid], targets[valid])


def limit_dataset(dataset, limit: int | None, seed: int, shuffle: bool):
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    if shuffle:
        generator = np.random.default_rng(seed)
        generator.shuffle(indices)
    return Subset(dataset, indices[:limit])


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    stratified: bool,
    seed: int = 42,
) -> DataLoader:
    if stratified:
        batch_sampler = StratifiedBatchSampler(
            dataset, batch_size=batch_size, shuffle=True
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=FlexibleGraphDataset.collate_fn,
        )
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=FlexibleGraphDataset.collate_fn,
    )


def _load_matching_state(model: RACEModel, checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    transferred = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    incompatible = model.load_state_dict(transferred, strict=False)
    return {
        "checkpoint": checkpoint_path,
        "transferred": len(transferred),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }


def _load_lexical_head(
    model: RACEModel,
    checkpoint_path: str,
    target_prefix: str = "lexical_trace_head.",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    source_keys = [key for key in source if key.startswith("lexical_trace_head.")]
    if not source_keys:
        raise ValueError("Lexical checkpoint contains no lexical_trace_head parameters")
    mapped_keys = []
    for source_key in source_keys:
        suffix = source_key.removeprefix("lexical_trace_head.")
        target_key = f"{target_prefix}{suffix}"
        if target_key not in target or target[target_key].shape != source[source_key].shape:
            raise ValueError(f"Incompatible lexical parameter: {source_key} -> {target_key}")
        target[target_key] = source[source_key]
        mapped_keys.append(target_key)
    model.load_state_dict(target)
    return {"checkpoint": checkpoint_path, "transferred_keys": mapped_keys}


def _load_creator_head(model: RACEModel, checkpoint_path: str) -> dict[str, Any]:
    """Load only the P3 Creator Retention head into a joint model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    prefix = "creator_retention_head."
    source_keys = [key for key in source if key.startswith(prefix)]
    if not source_keys:
        raise ValueError("Creator checkpoint contains no creator_retention_head parameters")
    for key in source_keys:
        if key not in target or target[key].shape != source[key].shape:
            raise ValueError(f"Incompatible Creator parameter: {key}")
        target[key] = source[key]
    model.load_state_dict(target)
    return {"checkpoint": checkpoint_path, "transferred_keys": source_keys}


def _to_builtin(value):
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@torch.no_grad()
def evaluate(
    model: RACEModel,
    loader: DataLoader,
    device: torch.device,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    lexical_targets: list[float] = []
    lexical_predictions: list[float] = []
    humanization_targets: list[float] = []
    humanization_predictions: list[float] = []
    creator_targets: list[float] = []
    creator_predictions: list[float] = []
    records: list[dict[str, Any]] = []
    ce_sum = 0.0
    trace_sum = 0.0
    humanization_sum = 0.0
    creator_sum = 0.0
    batches = 0

    for batch in loader:
        output = model(batch)
        logits = output["logits"]
        labels = batch["labels"].to(device)
        lexical_scores = output.get("lexical_trace_scores")
        if lexical_scores is not None:
            trace_loss = masked_mse(
                lexical_scores,
                batch["edu_lexical_scores"],
                batch["edu_lexical_masks"],
                device,
                allow_empty=True,
            )
        else:
            lexical_scores = [
                logits.new_empty((0,)) for _ in range(len(batch["labels"]))
            ]
            trace_loss = torch.zeros((), device=device)
        humanization_scores = output.get("humanization_trace_scores")
        if humanization_scores is not None:
            humanization_loss = masked_mse(
                humanization_scores,
                batch["edu_humanization_scores"],
                batch["edu_humanization_masks"],
                device,
                allow_empty=True,
            )
        else:
            humanization_scores = [
                logits.new_empty((0,)) for _ in range(len(batch["labels"]))
            ]
            humanization_loss = torch.zeros((), device=device)
        creator_scores = output.get("creator_retention_scores")
        if creator_scores is not None:
            creator_loss = masked_scalar_mse(
                creator_scores,
                batch["creator_retention_scores"],
                batch["creator_retention_masks"],
                device,
            )
        else:
            creator_scores = logits.new_zeros((len(batch["labels"]),))
            creator_loss = torch.zeros((), device=device)
        ce_sum += float(F.cross_entropy(logits, labels).item())
        trace_sum += float(trace_loss.item())
        humanization_sum += float(humanization_loss.item())
        creator_sum += float(creator_loss.item())
        batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

        probabilities = torch.softmax(logits, dim=-1)
        for (
            item_id, label, probability, item_logits, score, target, mask,
            humanization_score, humanization_target, humanization_mask,
            creator_score, creator_target, creator_mask,
        ) in zip(
            batch["id"], batch["labels"], probabilities, logits,
            lexical_scores, batch["edu_lexical_scores"],
            batch["edu_lexical_masks"],
            humanization_scores, batch["edu_humanization_scores"],
            batch["edu_humanization_masks"],
            creator_scores, batch["creator_retention_scores"],
            batch["creator_retention_masks"],
        ):
            length = min(score.numel(), target.numel(), mask.numel())
            score_values = score[:length].detach().cpu().tolist()
            target_values = target[:length].tolist()
            mask_values = mask[:length].tolist()
            for prediction, gold, valid in zip(score_values, target_values, mask_values):
                if valid:
                    lexical_predictions.append(float(prediction))
                    lexical_targets.append(float(gold))
            humanization_length = min(
                humanization_score.numel(),
                humanization_target.numel(),
                humanization_mask.numel(),
            )
            humanization_score_values = (
                humanization_score[:humanization_length].detach().cpu().tolist()
            )
            humanization_target_values = (
                humanization_target[:humanization_length].tolist()
            )
            humanization_mask_values = humanization_mask[:humanization_length].tolist()
            for prediction, gold, valid in zip(
                humanization_score_values,
                humanization_target_values,
                humanization_mask_values,
            ):
                if valid:
                    humanization_predictions.append(float(prediction))
                    humanization_targets.append(float(gold))
            if float(creator_mask.item()) > 0:
                creator_predictions.append(float(creator_score.detach().cpu().item()))
                creator_targets.append(float(creator_target.item()))
            records.append(
                {
                    "item_id": item_id,
                    "label": int(label.item()),
                    "prediction": int(torch.argmax(probability).item()),
                    "probabilities": probability.detach().cpu().tolist(),
                    "logits": item_logits.detach().cpu().tolist(),
                    "lexical_scores": score_values,
                    "lexical_targets": target_values,
                    "lexical_masks": [int(value) for value in mask_values],
                    "humanization_scores": humanization_score_values,
                    "humanization_targets": humanization_target_values,
                    "humanization_masks": [
                        int(value) for value in humanization_mask_values
                    ],
                    "creator_retention_score": float(
                        creator_score.detach().cpu().item()
                    ),
                    "creator_retention_target": float(creator_target.item()),
                    "creator_retention_mask": int(creator_mask.item()),
                }
            )

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_classification_metrics(
        logits, labels, input_type="logits", num_classes=4
    )
    probabilities = torch.softmax(logits, dim=-1).numpy()
    labels_np = labels.numpy()
    for class_id, class_name in enumerate(CLASS_NAMES):
        binary = (labels_np == class_id).astype(np.int64)
        metrics[f"auroc_{class_name}"] = (
            float(roc_auc_score(binary, probabilities[:, class_id]))
            if len(np.unique(binary)) == 2
            else float("nan")
        )
    lexical_metrics = calculate_metrics(lexical_targets, lexical_predictions)
    metrics.update({f"lexical_{key}": value for key, value in lexical_metrics.items()})
    if humanization_targets:
        humanization_metrics = calculate_metrics(
            humanization_targets, humanization_predictions
        )
        metrics.update(
            {f"humanization_{key}": value for key, value in humanization_metrics.items()}
        )
    if creator_targets:
        creator_metrics = calculate_metrics(creator_targets, creator_predictions)
        metrics.update(
            {f"creator_retention_{key}": value for key, value in creator_metrics.items()}
        )
    metrics["ce_loss"] = ce_sum / max(batches, 1)
    metrics["lexical_loss"] = trace_sum / max(batches, 1)
    metrics["humanization_loss"] = humanization_sum / max(batches, 1)
    metrics["creator_retention_loss"] = creator_sum / max(batches, 1)
    metrics["documents"] = int(len(labels))
    if model.lexical_residual_gamma is not None:
        metrics["lexical_residual_gamma"] = float(
            model.lexical_residual_gamma.detach().cpu().item()
        )
    if model.humanization_residual_gamma is not None:
        metrics["humanization_residual_gamma"] = float(
            model.humanization_residual_gamma.detach().cpu().item()
        )
    if model.creator_retention_residual_gamma is not None:
        metrics["creator_retention_residual_gamma"] = float(
            model.creator_retention_residual_gamma.detach().cpu().item()
        )
    metrics = _to_builtin(metrics)

    if predictions_path is not None:
        with predictions_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def set_calibration_trainability(
    model: RACEModel, joint_trainability: dict[str, bool], calibration: bool
) -> None:
    for name, parameter in model.named_parameters():
        if calibration:
            parameter.requires_grad = name.startswith(
                (
                    "lexical_trace_head.",
                    "humanization_trace_head.",
                    "creator_retention_head.",
                )
            )
        else:
            parameter.requires_grad = joint_trainability[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--baseline_checkpoint")
    parser.add_argument("--lexical_checkpoint")
    parser.add_argument("--humanization_checkpoint")
    parser.add_argument("--creator_checkpoint")
    parser.add_argument(
        "--creator_input_mode",
        choices=["edu", "edu_root", "edu_root_interaction"],
    )
    parser.add_argument(
        "--ablation_variant",
        choices=["creator_only", "creator_no_fusion", "all_lambda_zero"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if args.data_dir:
        for split in ("train", "val", "test"):
            config[f"{split}_file"] = str(Path(args.data_dir) / f"{split}_graph.jsonl")
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.baseline_checkpoint:
        config["baseline_checkpoint"] = args.baseline_checkpoint
    if args.lexical_checkpoint:
        config["lexical_checkpoint"] = args.lexical_checkpoint
    if args.humanization_checkpoint:
        config["humanization_checkpoint"] = args.humanization_checkpoint
    if args.creator_checkpoint:
        config["creator_checkpoint"] = args.creator_checkpoint
    if args.creator_input_mode:
        config["graph"]["creator_retention_input_mode"] = args.creator_input_mode
    if args.ablation_variant in {"creator_only", "creator_no_fusion"}:
        config["graph"].update(
            {
                "use_lexical_trace": False,
                "use_lexical_fusion": False,
                "use_humanization_trace": False,
                "use_humanization_fusion": False,
                "use_creator_retention": True,
                "use_creator_retention_fusion": args.ablation_variant == "creator_only",
            }
        )
        config["lexical_checkpoint"] = None
        config["humanization_checkpoint"] = None
        config["lexical_loss_weight"] = 0.0
        config["humanization_loss_weight"] = 0.0
        config["creator_retention_loss_weight"] = 0.2
    elif args.ablation_variant == "all_lambda_zero":
        config["lexical_loss_weight"] = 0.0
        config["humanization_loss_weight"] = 0.0
        config["creator_retention_loss_weight"] = 0.0
    if args.ablation_variant:
        config["ablation_variant"] = args.ablation_variant
    if args.epochs is not None:
        config["num_epochs"] = args.epochs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(int(config["seed"]))
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
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
    datasets = {
        split: FlexibleGraphDataset(config[f"{split}_file"], model_config)
        for split in ("train", "val", "test")
    }
    datasets["train"] = limit_dataset(
        datasets["train"], args.max_train_samples, int(config["seed"]), True
    )
    datasets["val"] = limit_dataset(datasets["val"], args.max_eval_samples, 0, False)
    datasets["test"] = limit_dataset(datasets["test"], args.max_eval_samples, 0, False)
    workers = int(config.get("num_workers", 0))
    run_seed = int(config["seed"])
    loaders = {
        "train": make_loader(
            datasets["train"], int(config["batch_size"]), workers, True, run_seed
        ),
        "val": make_loader(datasets["val"], int(config["eval_batch_size"]), workers, False),
        "test": make_loader(datasets["test"], int(config["eval_batch_size"]), workers, False),
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
    baseline_report = _load_matching_state(model, config["baseline_checkpoint"])
    lexical_report = None
    if config.get("lexical_checkpoint"):
        lexical_report = _load_lexical_head(model, config["lexical_checkpoint"])
    elif model.use_lexical_trace:
        raise ValueError("use_lexical_trace=true requires lexical_checkpoint")
    humanization_report = None
    if config.get("humanization_checkpoint"):
        humanization_report = _load_lexical_head(
            model,
            config["humanization_checkpoint"],
            target_prefix="humanization_trace_head.",
        )
    creator_report = None
    if config.get("creator_checkpoint"):
        creator_report = _load_creator_head(model, config["creator_checkpoint"])
    model.to(device)
    if (
        model.lexical_residual_gamma is not None
        and model.lexical_residual_gamma.item() != 0.0
    ):
        raise ValueError("Four-class residual gamma must initialize to exactly zero")
    if (
        model.humanization_residual_gamma is not None
        and model.humanization_residual_gamma.item() != 0.0
    ):
        raise ValueError("Humanization residual gamma must initialize to exactly zero")
    if (
        model.creator_retention_residual_gamma is not None
        and model.creator_retention_residual_gamma.item() != 0.0
    ):
        raise ValueError("Creator Retention residual gamma must initialize to exactly zero")
    with (output_dir / "initialization_report.json").open("w", encoding="utf-8") as report_file:
        json.dump(
            {
                "baseline": baseline_report,
                "lexical": lexical_report,
                "humanization": humanization_report,
                "creator": creator_report,
            },
            report_file, ensure_ascii=False, indent=2,
        )

    checkpoint_path = output_dir / "best_model.pt"
    if args.eval_only:
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
        return

    baseline_metrics = {
        "val": evaluate(model, loaders["val"], device),
        "test": evaluate(
            model, loaders["test"], device, output_dir / "baseline_test_predictions.jsonl"
        ),
    }
    with (output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(baseline_metrics, output_file, ensure_ascii=False, indent=2)
    LOGGER.info("same-split baseline metrics=%s", json.dumps(baseline_metrics))

    joint_trainability = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if joint_trainability[name]
    ]
    optimizer = AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    total_steps = max(1, len(loaders["train"]) * int(config["num_epochs"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    use_supcon = bool(config.get("use_supcon", False))
    supcon_criterion = SupConLoss(
        temperature=float(config.get("supcon_temperature", 0.07))
    )

    best_f1 = -float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    calibration_epochs = int(config["lexical_calibration_epochs"])
    for epoch in range(int(config["num_epochs"])):
        calibration = epoch < calibration_epochs
        set_calibration_trainability(model, joint_trainability, calibration)
        model.train()
        running = {
            "loss": 0.0,
            "ce": 0.0,
            "trace": 0.0,
            "humanization": 0.0,
            "creator_retention": 0.0,
            "supcon": 0.0,
        }
        for step, batch in enumerate(loaders["train"], start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, epoch=epoch)
            trace_loss = masked_mse(
                output["lexical_trace_scores"], batch["edu_lexical_scores"],
                batch["edu_lexical_masks"], device, allow_empty=True,
            ) if "lexical_trace_scores" in output else torch.zeros((), device=device)
            humanization_loss = masked_mse(
                output["humanization_trace_scores"],
                batch["edu_humanization_scores"],
                batch["edu_humanization_masks"],
                device,
                allow_empty=True,
            ) if "humanization_trace_scores" in output else torch.zeros((), device=device)
            creator_loss = masked_scalar_mse(
                output["creator_retention_scores"],
                batch["creator_retention_scores"],
                batch["creator_retention_masks"],
                device,
            ) if "creator_retention_scores" in output else torch.zeros((), device=device)
            if calibration:
                ce_loss = torch.zeros((), device=device)
                supcon_loss = torch.zeros((), device=device)
                loss = trace_loss + humanization_loss + creator_loss
            else:
                ce_loss = F.cross_entropy(output["logits"], batch["labels"].to(device))
                if use_supcon:
                    normalized_features = F.normalize(output["features"], p=2, dim=1)
                    supcon_loss = supcon_criterion(
                        normalized_features, batch["labels"].to(device)
                    )
                else:
                    supcon_loss = torch.zeros((), device=device)
                loss = (
                    ce_loss
                    + supcon_loss
                    + float(config.get("lexical_loss_weight", 0.2)) * trace_loss
                    + float(config.get("humanization_loss_weight", 0.2))
                    * humanization_loss
                    + float(config.get("creator_retention_loss_weight", 0.2))
                    * creator_loss
                )
            loss.backward()
            clip_grad_norm_(trainable, float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            running["loss"] += float(loss.item())
            running["ce"] += float(ce_loss.item())
            running["trace"] += float(trace_loss.item())
            running["humanization"] += float(humanization_loss.item())
            running["creator_retention"] += float(creator_loss.item())
            running["supcon"] += float(supcon_loss.item())
            if step % int(config.get("logging_steps", 25)) == 0:
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.6f ce=%.6f supcon=%.6f trace=%.6f humanization=%.6f creator=%.6f gamma_p=%.6f gamma_h=%.6f gamma_c=%.6f",
                    epoch + 1, step, len(loaders["train"]),
                    running["loss"] / step, running["ce"] / step,
                    running["supcon"] / step,
                    running["trace"] / step,
                    running["humanization"] / step,
                    running["creator_retention"] / step,
                    float(model.lexical_residual_gamma.detach().cpu().item())
                    if model.lexical_residual_gamma is not None else 0.0,
                    float(model.humanization_residual_gamma.detach().cpu().item())
                    if model.humanization_residual_gamma is not None else 0.0,
                    float(model.creator_retention_residual_gamma.detach().cpu().item())
                    if model.creator_retention_residual_gamma is not None else 0.0,
                )

        val_metrics = evaluate(model, loaders["val"], device)
        record = {
            "epoch": epoch + 1,
            "calibration": calibration,
            "train": {key: value / max(len(loaders["train"]), 1) for key, value in running.items()},
            "val": val_metrics,
        }
        history.append(record)
        LOGGER.info("epoch metrics=%s", json.dumps(record))
        current_f1 = float(val_metrics["f1_macro"])
        if calibration:
            continue
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "config": config,
                    "model_config": model_config,
                    "metadata": metadata,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= int(config["early_stopping_patience"]):
                LOGGER.info("early stopping after epoch %d", epoch + 1)
                break

    with (output_dir / "history.json").open("w", encoding="utf-8") as output_file:
        json.dump(history, output_file, ensure_ascii=False, indent=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "baseline": baseline_metrics,
        "val": evaluate(model, loaders["val"], device, output_dir / "val_predictions.jsonl"),
        "test": evaluate(model, loaders["test"], device, output_dir / "test_predictions.jsonl"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
