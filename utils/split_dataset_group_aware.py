"""Create a group-aware train/val/test split for flattened HART items."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any


def load_items(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_items_by_group_id(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item.get("group_id", item.get("item_id", "unknown"))].append(item)
    return groups


def infer_group_strata(group_items: list[dict[str, Any]]) -> tuple[str, str]:
    first = group_items[0]
    domain = first.get("source_dataset") or first.get("group_id", "unknown").split("-")[0]
    has_humanized = any(x.get("label") == "ai_humanized" for x in group_items)
    return str(domain), str(int(has_humanized))


def split_groups(groups: dict[str, list[dict[str, Any]]], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for gid, items in groups.items():
        by_strata[infer_group_strata(items)].append(gid)

    train_ids, val_ids, test_ids = set(), set(), set()
    for ids in by_strata.values():
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train:n_train + n_val])
        test_ids.update(ids[n_train + n_val:])

    def expand(id_set):
        return [item for gid in id_set for item in groups[gid]]
    return expand(train_ids), expand(val_ids), expand(test_ids)


def write_jsonl(items: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_split_stats(splits: dict[str, list[dict[str, Any]]], path: str) -> None:
    lines = ["# Group-aware Split Stats", ""]
    for split, items in splits.items():
        by_label = defaultdict(int)
        by_domain = defaultdict(int)
        for item in items:
            by_label[item.get("label", "unknown")] += 1
            by_domain[item.get("source_dataset", "unknown")] += 1
        lines.append(f"## {split} ({len(items)})")
        lines.append("Labels: " + json.dumps(dict(by_label), ensure_ascii=False, sort_keys=True))
        lines.append("Domains: " + json.dumps(dict(by_domain), ensure_ascii=False, sort_keys=True))
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/flattened_articles.jsonl")
    parser.add_argument("--output_dir", default="data/hart_group_split")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    items = load_items(args.input)
    groups = group_items_by_group_id(items)
    train, val, test = split_groups(groups, args.train_ratio, args.val_ratio, args.seed)
    splits = {"train": train, "val": val, "test": test}
    for split, split_items in splits.items():
        write_jsonl(split_items, os.path.join(args.output_dir, f"{split}_graph.jsonl"))
    write_split_stats(splits, os.path.join(args.output_dir, "stats.md"))


if __name__ == "__main__":
    main()
