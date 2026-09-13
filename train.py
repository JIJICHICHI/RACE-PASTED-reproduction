import argparse
import datetime
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup

# Local imports
from models.flexible_model import (
    RACEModel,
    FlexibleBaselineModel,
)
from models.supcon_loss import SupConLoss, WeightedSupConLoss
from utils.custom_sampler import StratifiedBatchSampler
from utils.flexible_dataset import FlexibleGraphDataset
from utils.metrics import compute_classification_metrics
from utils.misc import (
    MetricLogger,
    SmoothedValue,
    init_distributed_mode,
    is_main_process,
    save_on_master,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EarlyStopping:
    def __init__(self, patience=10, metric="f1_macro", logger=None):
        """
        Early stops the training if validation metric doesn't improve after a given patience.
        """
        self.patience = patience
        self.metric = metric
        self.logger = logger
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # Mode can be 'min' or 'max'
        if self.metric == "eval_loss":
            self.mode = "min"
            self.best_score = float("inf")
        else:
            self.mode = "max"
            self.best_score = float("-inf")

    def __call__(self, current_score):
        should_stop = False
        if self.mode == "min":
            if current_score < self.best_score:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if current_score > self.best_score:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.logger:
                self.logger.info(
                    f"Early stopping triggered after {self.patience} epochs of no improvement."
                )
            should_stop = True

        return should_stop


def get_metadata_from_file(relation_file_path, config):
    """
    Builds metadata by reading a pre-defined list of relations from a file.

    Depending on the `edge_type_mode` in the config, it generates either:
    - 'tuple' mode (default): A list of all possible (src, rel, dst) triplets.
    - 'relation_only' mode: A list of unique relation name strings.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building metadata from relation file: {relation_file_path}")

    try:
        with open(relation_file_path, "r") as f:
            relations_with_suffix = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Relation file not found at: {relation_file_path}")
        raise

    base_relations = set(rel.split("_")[0] for rel in relations_with_suffix)
    all_relation_names = set()
    use_nuclearity = config.get("rst_use_nuclearity", False)
    use_distinct_reverse = config.get("use_distinct_reverse_edges", True)

    for base_rel in base_relations:
        if use_nuclearity:
            all_relation_names.add(f"{base_rel}_N")
            all_relation_names.add(f"{base_rel}_S")
        else:
            all_relation_names.add(base_rel)

    if use_distinct_reverse:
        reversed_names = {f"rev_{name}" for name in all_relation_names}
        all_relation_names.update(reversed_names)

    edge_types = set()
    node_types = ["edu_node", "relation_node"]
    for rel_name in all_relation_names:
        for target_type in node_types:
            edge_types.add(("relation_node", rel_name, target_type))
            edge_types.add((target_type, rel_name, "relation_node"))

    metadata = {"edge_types": sorted(list(edge_types))}
    logger.info(f"Generated {len(metadata['edge_types'])} unique edge types.")
    return metadata


def main(args):
    init_distributed_mode(args)
    logging.basicConfig(level=logging.INFO if is_main_process() else logging.WARN)
    logger = logging.getLogger(__name__)

    if is_main_process() and args.output_dir:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = os.path.join(args.output_dir, timestamp)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    graph_config = json.loads(args.graph_config)
    graph_config.update(
        {
            "backbone_model_path": args.backbone_model_path,
            "bert_dim": args.bert_dim,
            "max_seq_length": args.max_seq_length,
            "unfreeze_bert_layers": args.unfreeze_bert_layers,
            "num_gnn_layers": args.num_gnn_layers,
        }
    )
    logger.info(f"Graph configuration: {graph_config}")

    writer = (
        SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
        if is_main_process() and args.output_dir
        else None
    )

    dataset_train = FlexibleGraphDataset(file_path=args.data_path, config=graph_config)
    dataset_val = FlexibleGraphDataset(
        file_path=args.val_data_path, config=graph_config
    )

    # --- Sampler and DataLoader Initialization ---
    if getattr(args, "use_stratified_sampler", False) and not args.distributed:
        logger.info("Using Stratified Sampler for training data.")
        batch_sampler_train = StratifiedBatchSampler(
            dataset_train, batch_size=args.batch_size, shuffle=True
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=FlexibleGraphDataset.collate_fn,
            num_workers=args.num_workers,
        )
    else:
        logger.info("Using standard random sampler for training data.")
        sampler_train = (
            DistributedSampler(dataset_train)
            if args.distributed
            else RandomSampler(dataset_train)
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=sampler_train,
            collate_fn=FlexibleGraphDataset.collate_fn,
            num_workers=args.num_workers,
        )

    # Validation dataloader
    sampler_val = (
        DistributedSampler(dataset_val, shuffle=False)
        if args.distributed
        else SequentialSampler(dataset_val)
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        collate_fn=FlexibleGraphDataset.collate_fn,
        num_workers=args.num_workers,
    )

    # Build metadata from the pre-defined relation file
    metadata = {}
    if is_main_process():
        if not hasattr(args, "relation_table_path"):
            raise ValueError(
                "Configuration error: 'relation_table_path' must be specified in the config file."
            )

        metadata = get_metadata_from_file(args.relation_table_path, graph_config)
        if args.output_dir:
            with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

    if args.distributed:
        torch.distributed.barrier()
        with open(os.path.join(args.output_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
    if not metadata:
        raise ValueError("Metadata could not be loaded or is empty.")

    # --- Loss Function and Model Initialization ---
    loss_config = getattr(args, "loss", {})
    use_supcon = bool(loss_config.get("use_supcon", False))
    use_factor_contrast = bool(loss_config.get("use_factor_contrast", False))
    use_soft_hierarchy_contrast = bool(loss_config.get("use_soft_hierarchy_contrast", False))
    output_features = use_supcon or use_factor_contrast or use_soft_hierarchy_contrast

    model_type = getattr(args, "model_type", "gnn")
    if model_type == "baseline":
        logger.info("Initializing FlexibleBaselineModel (Backbone + Classifier).")
        model = FlexibleBaselineModel(
            feature_dim=args.feature_dim,
            gnn_hidden_dim=args.gnn_hidden_dim,
            num_heads=args.num_heads,
            num_classes=args.num_classes,
            config=graph_config,
            metadata=metadata,
            output_features=output_features,
        )
    else:
        logger.info("Initializing RACEModel (RST-based GNN).")
        model = RACEModel(
            feature_dim=args.feature_dim,
            gnn_hidden_dim=args.gnn_hidden_dim,
            num_heads=args.num_heads,
            num_classes=args.num_classes,
            config=graph_config,
            metadata=metadata,
            output_features=output_features,
        )

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)."
    )

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Criterion Initialization ---
    if use_supcon or use_factor_contrast or use_soft_hierarchy_contrast:
        logger.info("Using Supervised Contrastive Loss (SupCon) for representation losses.")
        criterion = SupConLoss(
            temperature=loss_config.get("temperature", 0.07)
        )
    else:
        logger.info("Using standard Cross Entropy Loss.")
        criterion = torch.nn.CrossEntropyLoss()

    # --- LR Scheduler Initialization ---
    lr_scheduler_config = getattr(args, "lr_scheduler", None)
    scheduler = None
    if lr_scheduler_config:
        num_training_steps = len(dataloader_train) * args.epochs
        scheduler_type = lr_scheduler_config.get("type")
        logger.info(f"Using '{scheduler_type}' learning rate scheduler.")

        if scheduler_type == "linear":
            warmup_ratio = lr_scheduler_config.get("warmup_ratio", 0.1)
            num_warmup_steps = int(num_training_steps * warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    else:
        logger.info("Not using any learning rate scheduler.")

    # --- Early Stopping Initialization ---
    early_stopper = None
    if hasattr(args, "early_stopping_patience") and args.early_stopping_patience > 0:
        metric_to_monitor = getattr(args, "early_stopping_metric", "f1_macro")
        early_stopper = EarlyStopping(
            patience=args.early_stopping_patience,
            metric=metric_to_monitor,
            logger=logger,
        )
        logger.info(
            f"Early stopping enabled with patience {args.early_stopping_patience} on metric '{metric_to_monitor}'."
        )

    logger.info("Start training")
    start_time = time.time()
    best_checkpoint_metric = getattr(args, "checkpoint_metric", getattr(args, "best_checkpoint_metric", "f1_macro"))
    checkpoint_tie_breaker = getattr(args, "checkpoint_tie_breaker", None)
    best_checkpoint_score = float("-inf")
    best_tie_score = float("-inf")
    logger.info(f"Saving best checkpoint based on validation metric '{best_checkpoint_metric}'.")

    for epoch in range(args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            dataloader_train,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            args,
        )
        if writer and is_main_process():
            for k, v in train_stats.items():
                writer.add_scalar(f"Train/{k}", v, epoch)

        val_stats = evaluate(model, dataloader_val, device, epoch, args)
        if writer and is_main_process():
            for k, v in val_stats.items():
                if k != "classification_report":
                    writer.add_scalar(f"Val/{k}", v, epoch)

        # --- Checkpoint Saving ---
        current_f1 = val_stats.get("f1_macro", 0.0)
        current_checkpoint_score = val_stats.get(best_checkpoint_metric)
        if current_checkpoint_score is None:
            logger.warning(
                f"Validation metric '{best_checkpoint_metric}' not found; falling back to f1_macro."
            )
            current_checkpoint_score = current_f1
        current_tie_score = val_stats.get(checkpoint_tie_breaker, 0.0) if checkpoint_tie_breaker else 0.0
        is_better = current_checkpoint_score > best_checkpoint_score
        if checkpoint_tie_breaker and abs(current_checkpoint_score - best_checkpoint_score) <= 1e-6:
            is_better = current_tie_score > best_tie_score
        if args.output_dir and is_main_process() and is_better:
            best_checkpoint_score = current_checkpoint_score
            best_tie_score = current_tie_score
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            save_on_master(
                {
                    "model_state_dict": model_without_ddp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "epoch": epoch,
                    "args": vars(args),
                    "config": graph_config,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            logger.info(
                f"Saved best model to {checkpoint_path} with {best_checkpoint_metric}: {best_checkpoint_score:.4f}"
            )

        # --- Early Stopping Check ---
        if early_stopper and is_main_process():
            metric_map = {
                "eval_loss": val_stats.get("loss", 0.0),
                "f1_macro": current_f1,
            }
            current_metric_val = metric_map.get(early_stopper.metric)
            if current_metric_val is None:
                current_metric_val = val_stats.get(early_stopper.metric)
            if current_metric_val is None:
                logger.warning(
                    f"Early stopping metric '{early_stopper.metric}' not found; falling back to f1_macro."
                )
                current_metric_val = current_f1

            if early_stopper(current_metric_val):
                break

    total_time = time.time() - start_time
    logger.info(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")
    if writer and is_main_process():
        writer.close()

    # ===== Final evaluation on the test set =====
    if hasattr(args, "test_data_path") and args.test_data_path:
        logger.info("Starting final evaluation on the test set...")

        checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            logger.warning("best_model.pth not found, skipping final evaluation.")
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded best model from {checkpoint_path} (epoch {checkpoint['epoch']})"
            )

            if args.distributed:
                torch.distributed.barrier()

            dataset_test = FlexibleGraphDataset(
                file_path=args.test_data_path, config=graph_config
            )
            sampler_test = (
                DistributedSampler(dataset_test, shuffle=False)
                if args.distributed
                else SequentialSampler(dataset_test)
            )
            dataloader_test = DataLoader(
                dataset_test,
                batch_size=args.batch_size,
                sampler=sampler_test,
                collate_fn=FlexibleGraphDataset.collate_fn,
                num_workers=args.num_workers,
            )

            model_without_ddp.output_features = True

            results = evaluate(
                model_without_ddp,
                dataloader_test,
                device,
                epoch=None,
                args=args,
                return_logits=True,
                return_features=True,
            )

            if is_main_process() and results is not None:
                test_metrics, test_logits, test_ids, test_features = results
                if test_logits is not None and test_ids is not None:
                    predictions = []
                    for i, group_id in enumerate(test_ids):
                        pred_entry = {
                            "id": group_id,
                            "logits": test_logits[i].tolist(),
                            "prediction": torch.argmax(test_logits[i]).item(),
                        }
                        if test_features is not None:
                             pred_entry["features"] = test_features[i].tolist()
                        predictions.append(pred_entry)

                    predictions_path = os.path.join(
                        args.output_dir, "test_predictions.json"
                    )
                    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
                    with open(predictions_path, "w") as f:
                        json.dump(predictions, f, indent=2)
                    with open(metrics_path, "w") as f:
                        json.dump(test_metrics, f, indent=2)
                    logger.info(
                        f"Saved test predictions to {predictions_path}, and metrics to {metrics_path}"
                    )



def masked_bce_loss(evidence_logits, evidence_labels, evidence_masks, device):
    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)
    for logits, labels, masks in zip(evidence_logits, evidence_labels, evidence_masks):
        labels = labels.to(device)[: logits.numel()]
        masks = masks.to(device)[: logits.numel()]
        if logits.numel() == 0 or masks.sum().item() == 0:
            continue
        loss_vec = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        total_loss = total_loss + (loss_vec * masks).sum()
        total_count = total_count + masks.sum()
    if total_count.item() == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / total_count.clamp_min(1e-6)


def mil_loss(evidence_probs, labels, mixed_class_ids, margin, topk_ratio):
    device = labels.device
    losses = []
    for probs, label in zip(evidence_probs, labels):
        if int(label.item()) not in mixed_class_ids or probs.numel() == 0:
            continue
        k = max(1, int(np.ceil(float(topk_ratio) * probs.numel())))
        topk_mean = probs.topk(k=min(k, probs.numel())).values.mean()
        losses.append(F.relu(torch.tensor(float(margin), device=device) - topk_mean))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def sparse_loss(evidence_probs):
    losses = [probs.mean() for probs in evidence_probs if probs.numel() > 0]
    if not losses:
        device = evidence_probs[0].device if evidence_probs else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def derive_creator_editor_labels(labels):
    creator_labels = (labels >= 2).long()
    editor_labels = ((labels == 1) | (labels == 2)).long()
    return creator_labels, editor_labels


def build_soft_hierarchy_pair_weights(labels, shared_role_weight=0.1):
    creator_labels, editor_labels = derive_creator_editor_labels(labels)
    same_label = labels.view(-1, 1).eq(labels.view(1, -1))
    same_creator = creator_labels.view(-1, 1).eq(creator_labels.view(1, -1))
    same_editor = editor_labels.view(-1, 1).eq(editor_labels.view(1, -1))
    shared_role = (same_creator | same_editor) & (~same_label)

    weights = torch.zeros((labels.size(0), labels.size(0)), device=labels.device)
    weights = weights.masked_fill(shared_role, float(shared_role_weight))
    weights = weights.masked_fill(same_label, 1.0)
    weights.fill_diagonal_(0.0)
    return weights

def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, device, epoch, args
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_con", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_ce", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_evidence", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_mil", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_sparse", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_creator", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_editor", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_factor", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_factor_y", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_factor_c", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_factor_e", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_factor_scaled", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_soft_hierarchy", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_soft_scaled", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = f"Epoch: [{epoch}/{args.epochs}] (Train)"

    loss_config = getattr(args, "loss", {})
    use_supcon = bool(loss_config.get("use_supcon", False))
    use_factor_contrast = bool(loss_config.get("use_factor_contrast", False))
    use_soft_hierarchy_contrast = bool(loss_config.get("use_soft_hierarchy_contrast", False))
    factor_beta = float(loss_config.get("factor_beta", loss_config.get("beta", 0.1)))
    factor_warmup_epochs = int(loss_config.get("factor_warmup_epochs", 0))
    soft_hierarchy_beta = float(loss_config.get("soft_hierarchy_beta", loss_config.get("beta", 0.01)))
    soft_hierarchy_warmup_epochs = int(loss_config.get("soft_hierarchy_warmup_epochs", 0))
    shared_role_weight = float(loss_config.get("shared_role_weight", 0.1))
    lambda_y = float(loss_config.get("lambda_y", 1.0))
    lambda_creator = float(loss_config.get("lambda_creator", 0.5))
    lambda_editor = float(loss_config.get("lambda_editor", 0.8))
    ce_weight = float(loss_config.get("ce_weight", 1.0))
    use_evidence_loss = bool(loss_config.get("use_evidence_loss", False))
    use_mil_loss = bool(loss_config.get("use_mil_loss", False))
    use_sparse_loss = bool(loss_config.get("use_sparse_loss", False))
    evidence_weight = float(loss_config.get("evidence_weight", 0.1))
    mil_weight = float(loss_config.get("mil_weight", 0.1))
    sparse_weight = float(loss_config.get("sparse_weight", 0.01))
    mil_margin = float(loss_config.get("mil_margin", 0.3))
    mil_topk_ratio = float(loss_config.get("mil_topk_ratio", 0.15))
    use_creator_loss = bool(loss_config.get("use_creator_loss", False))
    use_editor_loss = bool(loss_config.get("use_editor_loss", False))
    creator_weight = float(loss_config.get("creator_weight", 0.3))
    editor_weight = float(loss_config.get("editor_weight", 0.3))
    ce_criterion = torch.nn.CrossEntropyLoss()

    progress_bar = tqdm(
        dataloader, desc=header, disable=not is_main_process(), unit="batch"
    )

    for i, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        labels = batch["labels"].to(device)
        outputs = model(batch, batch_idx=i, epoch=epoch)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        loss_ce = ce_criterion(logits, labels)
        loss = ce_weight * loss_ce
        loss_contrastive = torch.tensor(0.0, device=device)
        loss_evidence = torch.tensor(0.0, device=device)
        loss_mil_value = torch.tensor(0.0, device=device)
        loss_sparse_value = torch.tensor(0.0, device=device)
        loss_creator = torch.tensor(0.0, device=device)
        loss_editor = torch.tensor(0.0, device=device)
        loss_factor = torch.tensor(0.0, device=device)
        loss_factor_y = torch.tensor(0.0, device=device)
        loss_factor_c = torch.tensor(0.0, device=device)
        loss_factor_e = torch.tensor(0.0, device=device)
        loss_factor_scaled = torch.tensor(0.0, device=device)
        loss_soft_hierarchy = torch.tensor(0.0, device=device)
        loss_soft_scaled = torch.tensor(0.0, device=device)

        if use_supcon:
            if not isinstance(outputs, dict) or "features" not in outputs:
                raise ValueError("SupCon requires model outputs with a 'features' field.")
            features = F.normalize(outputs["features"], p=2, dim=1)
            loss_contrastive = criterion(features, labels)
            loss = loss + loss_contrastive

        le_active = isinstance(outputs, dict) and "evidence_logits" in outputs and not outputs.get("in_warmup", False)
        if le_active and use_evidence_loss:
            loss_evidence = masked_bce_loss(
                outputs["evidence_logits"],
                batch.get("edu_evidence_labels", []),
                batch.get("edu_evidence_masks", []),
                device,
            )
            loss = loss + evidence_weight * loss_evidence
        if le_active and use_mil_loss:
            loss_mil_value = mil_loss(
                outputs["evidence_probs"], labels, {1, 3}, mil_margin, mil_topk_ratio
            )
            loss = loss + mil_weight * loss_mil_value
        if le_active and use_sparse_loss:
            loss_sparse_value = sparse_loss(outputs["evidence_probs"])
            loss = loss + sparse_weight * loss_sparse_value

        if use_factor_contrast and epoch >= factor_warmup_epochs:
            if not isinstance(outputs, dict):
                raise ValueError("Factor contrast requires model outputs with z_y/z_c/z_e.")
            missing = [key for key in ("z_y", "z_c", "z_e") if key not in outputs]
            if missing:
                raise ValueError(f"Factor contrast missing model outputs: {missing}")
            creator_labels, editor_labels = derive_creator_editor_labels(labels)
            loss_factor_y = criterion(outputs["z_y"], labels)
            loss_factor_c = criterion(outputs["z_c"], creator_labels)
            loss_factor_e = criterion(outputs["z_e"], editor_labels)
            loss_factor = (
                lambda_y * loss_factor_y
                + lambda_creator * loss_factor_c
                + lambda_editor * loss_factor_e
            )
            loss_factor_scaled = factor_beta * loss_factor
            loss = loss + loss_factor_scaled

        if use_soft_hierarchy_contrast and epoch >= soft_hierarchy_warmup_epochs:
            if not isinstance(outputs, dict) or "features" not in outputs:
                raise ValueError("Soft hierarchy contrast requires model outputs with a 'features' field.")
            features = F.normalize(outputs["features"], p=2, dim=1)
            pair_weights = build_soft_hierarchy_pair_weights(labels, shared_role_weight)
            weighted_criterion = WeightedSupConLoss(temperature=loss_config.get("temperature", 0.07))
            loss_soft_hierarchy = weighted_criterion(features, pair_weights)
            loss_soft_scaled = soft_hierarchy_beta * loss_soft_hierarchy
            loss = loss + loss_soft_scaled

        if isinstance(outputs, dict) and (use_creator_loss or use_editor_loss):
            creator_labels, editor_labels = derive_creator_editor_labels(labels)
            if use_creator_loss:
                if "creator_logits" not in outputs:
                    raise ValueError("Creator loss requires model outputs with 'creator_logits'.")
                loss_creator = ce_criterion(outputs["creator_logits"], creator_labels)
                loss = loss + creator_weight * loss_creator
            if use_editor_loss:
                if "editor_logits" not in outputs:
                    raise ValueError("Editor loss requires model outputs with 'editor_logits'.")
                loss_editor = ce_criterion(outputs["editor_logits"], editor_labels)
                loss = loss + editor_weight * loss_editor

        loss.backward()

        clip_grad_norm = getattr(args, "clip_grad_norm", None)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_con=loss_contrastive.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_evidence=loss_evidence.item())
        metric_logger.update(loss_mil=loss_mil_value.item())
        metric_logger.update(loss_sparse=loss_sparse_value.item())
        metric_logger.update(loss_creator=loss_creator.item())
        metric_logger.update(loss_editor=loss_editor.item())
        metric_logger.update(loss_factor=loss_factor.item())
        metric_logger.update(loss_factor_y=loss_factor_y.item())
        metric_logger.update(loss_factor_c=loss_factor_c.item())
        metric_logger.update(loss_factor_e=loss_factor_e.item())
        metric_logger.update(loss_factor_scaled=loss_factor_scaled.item())
        metric_logger.update(loss_soft_hierarchy=loss_soft_hierarchy.item())
        metric_logger.update(loss_soft_scaled=loss_soft_scaled.item())

        if is_main_process():
            progress_bar.set_postfix(
                {
                    "loss": f"{metric_logger.loss.median:.4f}",
                    "avg_loss": f"{metric_logger.loss.global_avg:.4f}",
                    "loss_con": f"{metric_logger.loss_con.median:.4f}",
                    "loss_ce": f"{metric_logger.loss_ce.median:.4f}",
                    "loss_ev": f"{metric_logger.loss_evidence.median:.4f}",
                    "loss_mil": f"{metric_logger.loss_mil.median:.4f}",
                    "loss_creator": f"{metric_logger.loss_creator.median:.4f}",
                    "loss_editor": f"{metric_logger.loss_editor.median:.4f}",
                    "loss_factor": f"{metric_logger.loss_factor.median:.4f}",
                    "factor_x_beta": f"{metric_logger.loss_factor_scaled.median:.4f}",
                    "loss_soft": f"{metric_logger.loss_soft_hierarchy.median:.4f}",
                    "soft_x_beta": f"{metric_logger.loss_soft_scaled.median:.4f}",
                    "lr": f"{metric_logger.lr.value:.6f}",
                }
            )

    metric_logger.synchronize_between_processes()
    logging.getLogger(__name__).info(
        f"Averaged stats for epoch {epoch} (Train): {metric_logger}"
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, dataloader, device, epoch, args, return_logits=False, return_features=False):
    model.eval()
    all_preds, all_labels, all_probs, all_logits, all_ids, all_features = [], [], [], [], [], []
    val_loss = 0.0
    header = (
        f"Epoch: [{epoch}/{args.epochs}] (Val)" if epoch is not None else "Evaluating"
    )

    ce_criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=header)):
            labels = batch["labels"].to(device)
            outputs = model(batch, batch_idx=i, epoch=epoch)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
                if return_features and "features" in outputs:
                    all_features.append(outputs["features"].cpu())
            else:
                logits = outputs

            loss = ce_criterion(logits, labels)
            val_loss += loss.item()

            all_logits.append(logits.cpu())
            all_probs.append(F.softmax(logits, dim=1).cpu())
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
            if return_logits:
                all_ids.extend(batch["id"])

    if args.distributed:
        payload_to_gather = {
            "preds": all_preds,
            "labels": all_labels,
            "probs": all_probs,
            "logits": all_logits,
        }
        if return_logits:
            payload_to_gather["ids"] = all_ids
        if return_features:
            payload_to_gather["features"] = all_features

        gathered_outputs = [None for _ in range(args.world_size)]
        torch.distributed.all_gather_object(gathered_outputs, payload_to_gather)

        if is_main_process():
            all_preds = [p for output in gathered_outputs for p in output["preds"]]
            all_labels = [l for output in gathered_outputs for l in output["labels"]]
            all_probs = [p for output in gathered_outputs for p in output["probs"]]
            all_logits = [l for output in gathered_outputs for l in output["logits"]]
            if return_logits:
                all_ids = [i for output in gathered_outputs for i in output["ids"]]
            if return_features:
                all_features = [
                    f
                    for output in gathered_outputs
                    if "features" in output
                    for f in output["features"]
                ]

    if not is_main_process() and args.distributed:
        if return_logits:
            return None, None, None, None
        return None

    all_preds, all_labels, all_probs, all_logits = (
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_probs),
        torch.cat(all_logits),
    )

    final_features = None
    if return_features and all_features:
        final_features = torch.cat(all_features)

    val_metrics = compute_classification_metrics(all_logits, all_labels, input_type="logits")
    val_loss /= len(dataloader)

    logger = logging.getLogger(__name__)
    log_header = (
        f"Validation Results - Epoch: {epoch}" if epoch is not None else "Test Results"
    )
    logger.info(f"{log_header}, Loss: {val_loss:.4f}")
    if is_main_process():
        logger.info(f"Metrics: {val_metrics['classification_report']}")

    final_metrics = {"loss": val_loss, **val_metrics}

    if return_logits:
        return final_metrics, all_logits, all_ids, final_features
    return final_metrics

def get_args_parser():
    """
    Parses command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser("RACE Training Script", add_help=False)

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file for training.",
    )
    # This is needed for torch.distributed.launch
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    cli_args = parser.parse_args()

    # Load configuration from the specified JSON file
    with open(cli_args.config, "r") as f:
        config_dict = json.load(f)

    # Convert the configuration dictionary to a Namespace object
    args = argparse.Namespace(**config_dict)

    # Add distributed training arguments from the command line
    args.local_rank = cli_args.local_rank

    # The script expects 'graph_config' to be a JSON string, but the config file has a 'graph' object.
    if hasattr(args, "graph"):
        args.graph_config = json.dumps(args.graph)

    main(args)
