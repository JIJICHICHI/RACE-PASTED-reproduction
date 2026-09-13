"""Build group-safe AI-generated to Humanized EDU lexical-trace data."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from utils.evidence_label_builder import load_jsonl, write_jsonl
from utils.lexical_trace_label_builder import (
    _human_zero_targets,
    build_sentence_scores,
    deduplicate_items,
    load_spacy,
    project_sentence_scores_to_edus,
)


REFERENCE_LABEL = "ai_generated"
TARGET_LABEL = "ai_humanized"


def build_unique_pairs(items: list[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Return groups with exactly one Generated and one Humanized document."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in items:
        label = item.get("label")
        group_id = item.get("group_id")
        if label in {REFERENCE_LABEL, TARGET_LABEL} and group_id:
            grouped[str(group_id)][str(label)].append(item)

    pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    ambiguous: list[str] = []
    for group_id, variants in grouped.items():
        references = variants.get(REFERENCE_LABEL, [])
        targets = variants.get(TARGET_LABEL, [])
        if not targets:
            continue
        if len(references) != 1 or len(targets) != 1:
            ambiguous.append(group_id)
            continue
        pairs[group_id] = (references[0], targets[0])
    if ambiguous:
        raise ValueError(
            f"Found {len(ambiguous)} Humanized groups without a unique Generated pair"
        )
    return pairs


def assign_pairs(
    pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    manifest: dict[str, list[str]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Assign pairs with the existing group-safe four-class manifest."""
    assignment = {
        str(group_id): split
        for split, group_ids in manifest.items()
        for group_id in group_ids
    }
    output = {"train": [], "val": [], "test": []}
    missing: list[str] = []
    for group_id, pair in pairs.items():
        split = assignment.get(group_id)
        if split not in output:
            missing.append(group_id)
            continue
        output[split].append(pair)
    if missing:
        raise ValueError(f"Manifest does not assign {len(missing)} paired groups")
    return output


@torch.no_grad()
def build_split(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    nlp,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    min_similarity: float,
    min_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach zero/reference and continuous Humanized targets to one split."""
    output: list[dict[str, Any]] = []
    all_similarities: list[float] = []
    for index, (reference, target) in enumerate(pairs, start=1):
        reference_item = dict(reference)
        reference_scores, reference_masks = _human_zero_targets(reference)
        reference_item["edu_lexical_scores"] = reference_scores
        reference_item["edu_lexical_masks"] = reference_masks
        reference_item["lexical_reference_id"] = None
        reference_item["lexical_trace_direction"] = "ai_to_human"

        target_scores, target_masks, similarities = build_pieces(
            reference,
            target,
            nlp,
            model,
            tokenizer,
            device,
            max_length,
            min_similarity,
            min_tokens,
        )
        target_item = dict(target)
        target_item["edu_lexical_scores"] = target_scores
        target_item["edu_lexical_masks"] = target_masks
        target_item["lexical_reference_id"] = reference.get("item_id")
        target_item["lexical_trace_direction"] = "ai_to_human"

        output.extend((reference_item, target_item))
        all_similarities.extend(similarities)
        if index % 50 == 0:
            print(f"Processed {index}/{len(pairs)} pairs", flush=True)

    valid = sum(sum(item["edu_lexical_masks"]) for item in output)
    total = sum(len(item["edu_lexical_masks"]) for item in output)
    target_sum = sum(
        sum(score * mask for score, mask in zip(item["edu_lexical_scores"], item["edu_lexical_masks"]))
        for item in output
    )
    stats = {
        "documents": len(output),
        "groups": len(pairs),
        "class_counts": dict(Counter(item["label"] for item in output)),
        "valid_edus": valid,
        "total_edus": total,
        "masked_ratio": float(1.0 - valid / max(total, 1)),
        "mean_valid_target": float(target_sum / max(valid, 1)),
        "mean_alignment_similarity": float(
            sum(all_similarities) / max(len(all_similarities), 1)
        ),
    }
    return output, stats


def build_pieces(
    reference,
    target,
    nlp,
    model,
    tokenizer,
    device,
    max_length,
    min_similarity,
    min_tokens,
):
    sentences, sentence_scores, sentence_masks, similarities = build_sentence_scores(
        reference,
        target,
        nlp,
        model,
        tokenizer,
        device,
        max_length,
        min_similarity,
        min_tokens,
    )
    edu_scores, edu_masks = project_sentence_scores_to_edus(
        target, sentences, sentence_scores, sentence_masks
    )
    return edu_scores, edu_masks, similarities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone_model_path", default="FacebookAI/roberta-base")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_similarity", type=float, default=0.50)
    parser.add_argument("--min_tokens", type=int, default=5)
    args = parser.parse_args()

    merged = deduplicate_items(
        [item for path in args.inputs for item in load_jsonl(path)]
    )
    pairs = build_unique_pairs(merged)
    with open(args.manifest, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    split_pairs = assign_pairs(pairs, manifest)

    split_groups = {
        split: {str(pair[0]["group_id"]) for pair in values}
        for split, values in split_pairs.items()
    }
    if any(
        split_groups[left] & split_groups[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("group_id leakage detected")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nlp = load_spacy(args.spacy_model)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)
    model = AutoModel.from_pretrained(args.backbone_model_path).to(device).eval()

    stats: dict[str, Any] = {
        "config": vars(args),
        "deduplicated_documents": len(merged),
        "complete_groups": len(pairs),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        built, split_stats = build_split(
            split_pairs[split], nlp, model, tokenizer, device,
            args.max_length, args.min_similarity, args.min_tokens
        )
        write_jsonl(built, os.path.join(args.output_dir, f"{split}_graph.jsonl"))
        stats["splits"][split] = split_stats

    with open(os.path.join(args.output_dir, "lexical_trace_stats.json"), "w", encoding="utf-8") as output:
        json.dump(stats, output, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "split_manifest.json"), "w", encoding="utf-8") as output:
        json.dump({split: sorted(groups) for split, groups in split_groups.items()}, output, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
