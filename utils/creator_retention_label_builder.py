"""Attach group-safe Creator Retention labels to four-class RACE records.

Creator Retention is unrescaled BERTScore Recall from the original creator
document (reference) to the final document (candidate). Source documents use
the exact self-retention target 1.0; paired source text is never added as a
detector input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils.evidence_label_builder import load_jsonl, write_jsonl


SOURCE_LABELS = {"human_written", "ai_generated"}
PAIR_SPEC = {
    "human_ai_polished": ("human_written", "lexical_reference_id", "human_to_ai"),
    "ai_humanized": ("ai_generated", "humanization_reference_id", "ai_to_human"),
}
ALL_LABELS = SOURCE_LABELS | set(PAIR_SPEC)


def _valid_token_mask(encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = encoded["attention_mask"].bool()
    special = encoded.get("special_tokens_mask")
    return mask if special is None else mask & ~special.bool()


@torch.no_grad()
def bertscore_recall_batch(
    references: list[str],
    candidates: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> list[float]:
    """Compute uniform-weight, unrescaled BERTScore Recall for text pairs."""
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have equal length")
    common = {
        "padding": True,
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
        "return_special_tokens_mask": True,
    }
    ref = tokenizer(references, **common)
    cand = tokenizer(candidates, **common)
    ref_mask = _valid_token_mask(ref).to(device)
    cand_mask = _valid_token_mask(cand).to(device)
    ref_inputs = {
        key: value.to(device)
        for key, value in ref.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    cand_inputs = {
        key: value.to(device)
        for key, value in cand.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    ref_h = F.normalize(model(**ref_inputs).last_hidden_state.float(), p=2, dim=-1)
    cand_h = F.normalize(model(**cand_inputs).last_hidden_state.float(), p=2, dim=-1)
    similarities = torch.bmm(ref_h, cand_h.transpose(1, 2))
    similarities = similarities.masked_fill(~cand_mask[:, None, :], -1e4)
    best_for_reference = similarities.max(dim=-1).values
    recall = (best_for_reference * ref_mask).sum(dim=-1) / ref_mask.sum(dim=-1).clamp_min(1)
    return recall.clamp(0.0, 1.0).cpu().tolist()


def resolve_pairs(items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Resolve every edited item to exactly one same-group creator document."""
    by_id = {str(item["item_id"]): item for item in items}
    by_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in items:
        by_group[str(item["group_id"])][str(item["label"])].append(item)

    pairs = []
    for item in items:
        label = str(item.get("label"))
        if label not in PAIR_SPEC:
            continue
        source_label, reference_field, direction = PAIR_SPEC[label]
        reference_id = item.get(reference_field)
        reference = by_id.get(str(reference_id)) if reference_id else None
        if reference is None:
            candidates = by_group[str(item["group_id"])].get(source_label, [])
            if len(candidates) != 1:
                raise ValueError(
                    f"{item['item_id']} has {len(candidates)} candidate {source_label} references"
                )
            reference = candidates[0]
        if str(reference.get("group_id")) != str(item.get("group_id")):
            raise ValueError(f"Cross-group reference for {item['item_id']}")
        if reference.get("label") != source_label:
            raise ValueError(f"Wrong creator label for {item['item_id']}")
        pairs.append((reference, item, direction))
    return pairs


def build_split(
    items: list[dict[str, Any]],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    edited_pairs_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = resolve_pairs(items)
    scores_by_id: dict[str, tuple[float, str, str]] = {}
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        scores = bertscore_recall_batch(
            [str(reference["article"]) for reference, _, _ in chunk],
            [str(target["article"]) for _, target, _ in chunk],
            tokenizer,
            model,
            device,
            max_length,
        )
        for (reference, target, direction), score in zip(chunk, scores):
            if not math.isfinite(score):
                raise ValueError(f"Non-finite retention target for {target['item_id']}")
            scores_by_id[str(target["item_id"])] = (
                float(score), str(reference["item_id"]), direction
            )
        if start == 0 or start + len(chunk) == len(pairs) or (start // batch_size) % 50 == 0:
            print(f"Creator Retention: {start + len(chunk)}/{len(pairs)} edited pairs", flush=True)

    output = []
    values_by_label: dict[str, list[float]] = defaultdict(list)
    for item in items:
        label = str(item.get("label"))
        if label not in ALL_LABELS:
            continue
        if edited_pairs_only and label in SOURCE_LABELS:
            continue
        new_item = dict(item)
        if label in SOURCE_LABELS:
            score, reference_id, direction = 1.0, str(item["item_id"]), "self"
        else:
            score, reference_id, direction = scores_by_id[str(item["item_id"])]
        new_item["creator_retention_score"] = score
        new_item["creator_retention_mask"] = 1
        new_item["creator_retention_reference_id"] = reference_id
        new_item["creator_retention_direction"] = direction
        output.append(new_item)
        values_by_label[label].append(score)

    stats = {
        "documents": len(output),
        "groups": len({str(item["group_id"]) for item in output}),
        "class_counts": dict(Counter(item["label"] for item in output)),
        "edited_pairs": len(pairs),
        "score_by_label": {
            label: {
                "count": len(values),
                "mean": float(sum(values) / max(len(values), 1)),
                "min": float(min(values)) if values else None,
                "max": float(max(values)) if values else None,
            }
            for label, values in sorted(values_by_label.items())
        },
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_groups", type=int)
    parser.add_argument(
        "--edited_pairs_only",
        action="store_true",
        help="Write only Polished/Humanized edited targets; omit self-retention sources.",
    )
    args = parser.parse_args()

    if len(args.inputs) != 3:
        raise ValueError("Expected exactly three group-safe split files")
    split_names = ("train", "val", "test")
    split_items = {name: load_jsonl(path) for name, path in zip(split_names, args.inputs)}
    if args.max_groups is not None:
        for name, items in split_items.items():
            selected = sorted({str(item["group_id"]) for item in items})[: args.max_groups]
            selected_set = set(selected)
            split_items[name] = [item for item in items if str(item["group_id"]) in selected_set]

    split_groups = {
        name: {str(item["group_id"]) for item in items}
        for name, items in split_items.items()
    }
    if any(
        split_groups[left] & split_groups[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("group_id leakage detected")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    os.makedirs(args.output_dir, exist_ok=True)
    stats: dict[str, Any] = {
        "config": vars(args),
        "score": "uniform-weight unrescaled BERTScore Recall",
        "splits": {},
        "group_overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
    }
    for split in split_names:
        built, split_stats = build_split(
            split_items[split],
            tokenizer,
            model,
            device,
            args.max_length,
            args.batch_size,
            edited_pairs_only=args.edited_pairs_only,
        )
        write_jsonl(built, str(Path(args.output_dir) / f"{split}_graph.jsonl"))
        stats["splits"][split] = split_stats
    with (Path(args.output_dir) / "creator_retention_stats.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(stats, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
