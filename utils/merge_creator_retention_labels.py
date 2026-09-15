"""Join P2 Creator Retention labels onto the group-safe dual-trace dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SOURCE_LABELS = {"human_written", "ai_generated"}
EDITED_LABELS = {"human_ai_polished", "ai_humanized"}


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--edited_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    split_groups = {}
    for split in ("train", "val", "test"):
        base = load_jsonl(Path(args.base_dir) / f"{split}_graph.jsonl")
        edited = load_jsonl(Path(args.edited_dir) / f"{split}_graph.jsonl")
        edited_by_id = {str(item["item_id"]): item for item in edited}
        output = []
        for item in base:
            record = dict(item)
            label = str(record["label"])
            if label in SOURCE_LABELS:
                record.update({
                    "creator_retention_score": 1.0,
                    "creator_retention_mask": 1,
                    "creator_retention_reference_id": str(record["item_id"]),
                    "creator_retention_direction": "self",
                })
            elif label in EDITED_LABELS:
                source = edited_by_id.get(str(record["item_id"]))
                if source is None:
                    raise ValueError(f"Missing edited Creator target: {record['item_id']}")
                for key in (
                    "creator_retention_score", "creator_retention_mask",
                    "creator_retention_reference_id", "creator_retention_direction",
                ):
                    record[key] = source[key]
            else:
                raise ValueError(f"Unexpected label: {label}")
            output.append(record)
        if len(edited_by_id) != sum(str(item["label"]) in EDITED_LABELS for item in base):
            raise ValueError(f"Edited pair count mismatch in {split}")
        with (output_dir / f"{split}_graph.jsonl").open("w", encoding="utf-8") as output_file:
            for record in output:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        split_groups[split] = {str(item["group_id"]) for item in output}
        all_stats[split] = {
            "documents": len(output),
            "groups": len(split_groups[split]),
            "class_counts": dict(Counter(str(item["label"]) for item in output)),
            "valid_creator_targets": sum(int(item["creator_retention_mask"]) for item in output),
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_groups[left] & split_groups[right]:
            raise AssertionError(f"group leakage: {left}/{right}")
    with (output_dir / "creator_editor_stats.json").open("w", encoding="utf-8") as output_file:
        json.dump(all_stats, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
