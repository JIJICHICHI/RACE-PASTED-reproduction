"""Build group-safe four-class HART splits with partial lexical supervision."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from utils.edu_utils import traverse_rst_edus
from utils.evidence_label_builder import load_jsonl, write_jsonl
from utils.lexical_trace_label_builder import deduplicate_items


LEXICAL_LABELS = {"human_written", "human_ai_polished"}
HUMANIZATION_LABELS = {"ai_generated", "ai_humanized"}
ALL_LABELS = LEXICAL_LABELS | HUMANIZATION_LABELS


def load_lexical_targets(paths: list[str]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for path in paths:
        for item in load_jsonl(path):
            if item.get("label") not in LEXICAL_LABELS:
                continue
            item_id = str(item["item_id"])
            targets[item_id] = {
                "edu_lexical_scores": item["edu_lexical_scores"],
                "edu_lexical_masks": item["edu_lexical_masks"],
                "lexical_reference_id": item.get("lexical_reference_id"),
            }
    return targets


def load_humanization_targets(paths: list[str]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for path in paths:
        for item in load_jsonl(path):
            if item.get("label") not in HUMANIZATION_LABELS:
                continue
            item_id = str(item["item_id"])
            targets[item_id] = {
                "edu_humanization_scores": item["edu_lexical_scores"],
                "edu_humanization_masks": item["edu_lexical_masks"],
                "humanization_reference_id": item.get("lexical_reference_id"),
            }
    return targets


def build_fourclass_splits(
    source_items: list[dict[str, Any]],
    manifest: dict[str, list[str]],
    lexical_targets: dict[str, dict[str, Any]],
    humanization_targets: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    humanization_targets = humanization_targets or {}
    assignment = {
        str(group_id): split
        for split, group_ids in manifest.items()
        for group_id in group_ids
    }
    output = {"train": [], "val": [], "test": []}
    unassigned: list[str] = []
    missing_lexical: list[str] = []

    for item in source_items:
        if item.get("label") not in ALL_LABELS:
            continue
        item_id = str(item["item_id"])
        split = assignment.get(str(item.get("group_id")))
        if split is None:
            unassigned.append(item_id)
            continue

        new_item = dict(item)
        if item["label"] in LEXICAL_LABELS:
            target = lexical_targets.get(item_id)
            if target is None:
                missing_lexical.append(item_id)
                continue
            new_item.update(target)
        else:
            edu_count = len(traverse_rst_edus(item.get("rst_structure")))
            new_item["edu_lexical_scores"] = [0.0] * edu_count
            new_item["edu_lexical_masks"] = [0] * edu_count
            new_item["lexical_reference_id"] = None

        edu_count = len(traverse_rst_edus(item.get("rst_structure")))
        humanization_target = humanization_targets.get(item_id)
        if humanization_target is not None:
            new_item.update(humanization_target)
        else:
            new_item["edu_humanization_scores"] = [0.0] * edu_count
            new_item["edu_humanization_masks"] = [0] * edu_count
            new_item["humanization_reference_id"] = None
        if item["label"] == "ai_humanized" and humanization_targets and humanization_target is None:
            missing_lexical.append(f"humanization:{item_id}")
            continue
        output[split].append(new_item)

    if unassigned:
        raise ValueError(f"Manifest does not assign {len(unassigned)} items")
    if missing_lexical:
        raise ValueError(f"Missing lexical targets for {len(missing_lexical)} items")

    split_groups = {
        split: {str(item["group_id"]) for item in items}
        for split, items in output.items()
    }
    overlap_counts = {
        "train_val": len(split_groups["train"] & split_groups["val"]),
        "train_test": len(split_groups["train"] & split_groups["test"]),
        "val_test": len(split_groups["val"] & split_groups["test"]),
    }
    if any(overlap_counts.values()):
        raise AssertionError(f"group_id leakage detected: {overlap_counts}")

    stats: dict[str, Any] = {
        "documents": sum(len(items) for items in output.values()),
        "manifest_groups": len(assignment),
        "group_overlap_counts": overlap_counts,
        "splits": {},
    }
    for split, items in output.items():
        valid_by_label = Counter()
        humanization_valid_by_label = Counter()
        total_by_label = Counter()
        for item in items:
            label = item["label"]
            valid_by_label[label] += sum(item["edu_lexical_masks"])
            humanization_valid_by_label[label] += sum(
                item["edu_humanization_masks"]
            )
            total_by_label[label] += len(item["edu_lexical_masks"])
        stats["splits"][split] = {
            "documents": len(items),
            "groups": len(split_groups[split]),
            "class_counts": dict(Counter(item["label"] for item in items)),
            "valid_lexical_edus": dict(valid_by_label),
            "valid_humanization_edus": dict(humanization_valid_by_label),
            "total_edus": dict(total_by_label),
        }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_inputs", nargs="+", required=True)
    parser.add_argument("--lexical_inputs", nargs="+", required=True)
    parser.add_argument("--humanization_inputs", nargs="*", default=[])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    source_items = deduplicate_items(
        [item for path in args.source_inputs for item in load_jsonl(path)]
    )
    lexical_targets = load_lexical_targets(args.lexical_inputs)
    humanization_targets = load_humanization_targets(args.humanization_inputs)
    with open(args.manifest, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    splits, stats = build_fourclass_splits(
        source_items, manifest, lexical_targets, humanization_targets
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for split, items in splits.items():
        write_jsonl(items, os.path.join(args.output_dir, f"{split}_graph.jsonl"))
    with open(
        os.path.join(args.output_dir, "fourclass_stats.json"), "w", encoding="utf-8"
    ) as stats_file:
        json.dump(stats, stats_file, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
