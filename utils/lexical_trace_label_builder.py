"""Build group-safe EDU-level lexical trace regression data for PASTED-RACE."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any

import torch
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoModel, AutoTokenizer

from utils.edu_utils import traverse_rst_edus
from utils.evidence_label_builder import (
    embed_texts,
    get_sentences,
    load_jsonl,
    normalize_tokens,
    write_jsonl,
)


ALLOWED_LABELS = {"human_written", "human_ai_polished"}


def deduplicate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate merged source splits by stable item ID."""
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get("item_id") or item.get("id") or "")
        if not item_id:
            raise ValueError("Every item must have item_id or id")
        if item_id in unique and unique[item_id] != item:
            raise ValueError(f"Conflicting duplicate item: {item_id}")
        unique[item_id] = item
    return list(unique.values())


def build_complete_pairs(
    items: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep source groups containing both human and AI-polished variants."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("label") in ALLOWED_LABELS and item.get("group_id"):
            grouped[str(item["group_id"])].append(item)

    complete: dict[str, list[dict[str, Any]]] = {}
    for group_id, group_items in grouped.items():
        labels = {item.get("label") for item in group_items}
        if ALLOWED_LABELS.issubset(labels):
            complete[group_id] = group_items
    return complete


def _group_domain(group_id: str) -> str:
    return group_id.split("-", 1)[0].lower()


def split_groups(
    groups: dict[str, list[dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Make a deterministic domain-stratified group split."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and train_ratio + val_ratio < 1")

    rng = random.Random(seed)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for group_id in groups:
        by_domain[_group_domain(group_id)].append(group_id)

    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for domain_ids in by_domain.values():
        domain_ids = sorted(domain_ids)
        rng.shuffle(domain_ids)
        n_groups = len(domain_ids)
        n_train = int(round(n_groups * train_ratio))
        n_val = int(round(n_groups * val_ratio))
        n_train = min(n_train, n_groups)
        n_val = min(n_val, n_groups - n_train)
        split_ids["train"].extend(domain_ids[:n_train])
        split_ids["val"].extend(domain_ids[n_train : n_train + n_val])
        split_ids["test"].extend(domain_ids[n_train + n_val :])

    result: dict[str, list[dict[str, Any]]] = {}
    for split_name, group_ids in split_ids.items():
        rng.shuffle(group_ids)
        result[split_name] = [item for group_id in group_ids for item in groups[group_id]]
    return result


def bleu4_diversity(reference: str, target: str, nlp) -> float:
    """Return PASTED-style lexical diversity: 1 minus unsmoothed BLEU-4."""
    reference_tokens = normalize_tokens(reference, nlp)
    target_tokens = normalize_tokens(target, nlp)
    if not reference_tokens or not target_tokens:
        return 0.0
    return float(1.0 - sentence_bleu([reference_tokens], target_tokens))


@torch.no_grad()
def build_sentence_scores(
    reference_item: dict[str, Any],
    target_item: dict[str, Any],
    nlp,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    min_similarity: float,
    min_tokens: int,
) -> tuple[list[dict[str, Any]], list[float], list[int], list[float]]:
    """Align target sentences to the nearest reference and calculate lexical scores."""
    reference_sentences = get_sentences(reference_item, nlp)
    target_sentences = get_sentences(target_item, nlp)
    if not reference_sentences or not target_sentences:
        return target_sentences, [0.0] * len(target_sentences), [0] * len(target_sentences), []

    reference_embeddings = embed_texts(
        [sentence["text"] for sentence in reference_sentences],
        model,
        tokenizer,
        device,
        max_len=max_length,
    )
    target_embeddings = embed_texts(
        [sentence["text"] for sentence in target_sentences],
        model,
        tokenizer,
        device,
        max_len=max_length,
    )
    similarities = target_embeddings @ reference_embeddings.T

    scores: list[float] = []
    masks: list[int] = []
    best_similarities: list[float] = []
    for target_idx, target_sentence in enumerate(target_sentences):
        reference_idx = int(torch.argmax(similarities[target_idx]).item())
        similarity = float(similarities[target_idx, reference_idx].item())
        best_similarities.append(similarity)
        enough_tokens = len(target_sentence["tokens"]) >= min_tokens
        reliable = enough_tokens and similarity >= min_similarity
        if reliable:
            score = bleu4_diversity(
                reference_sentences[reference_idx]["text"],
                target_sentence["text"],
                nlp,
            )
            scores.append(score)
            masks.append(1)
        else:
            scores.append(0.0)
            masks.append(0)
    return target_sentences, scores, masks, best_similarities


def project_sentence_scores_to_edus(
    item: dict[str, Any],
    sentences: list[dict[str, Any]],
    sentence_scores: list[float],
    sentence_masks: list[int],
) -> tuple[list[float], list[int]]:
    """Project continuous sentence scores to EDUs by weighted character overlap."""
    edus = traverse_rst_edus(item.get("rst_structure"))
    edu_scores: list[float] = []
    edu_masks: list[int] = []
    for edu in edus:
        edu_start = int(edu.get("start") or 0)
        edu_end = int(edu.get("end") or 0)
        weighted_score = 0.0
        total_overlap = 0
        for sentence, score, mask in zip(sentences, sentence_scores, sentence_masks):
            if not mask:
                continue
            overlap = max(
                0,
                min(edu_end, int(sentence["end"]))
                - max(edu_start, int(sentence["start"])),
            )
            if overlap > 0:
                weighted_score += overlap * score
                total_overlap += overlap
        if total_overlap:
            edu_scores.append(float(weighted_score / total_overlap))
            edu_masks.append(1)
        else:
            edu_scores.append(0.0)
            edu_masks.append(0)
    return edu_scores, edu_masks


def _human_zero_targets(item: dict[str, Any]) -> tuple[list[float], list[int]]:
    edus = traverse_rst_edus(item.get("rst_structure"))
    scores = [0.0] * len(edus)
    masks = [1 if str(edu.get("text") or "").strip() else 0 for edu in edus]
    return scores, masks


def build_split(
    items: list[dict[str, Any]],
    nlp,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    min_similarity: float,
    min_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach regression targets to one already group-safe split."""
    human_by_group = {
        str(item["group_id"]): item
        for item in items
        if item.get("label") == "human_written"
    }
    output: list[dict[str, Any]] = []
    all_similarities: list[float] = []
    missing_references = 0
    for index, item in enumerate(items, start=1):
        new_item = dict(item)
        if item.get("label") == "human_written":
            edu_scores, edu_masks = _human_zero_targets(item)
            reference_id = None
        else:
            reference = human_by_group.get(str(item.get("group_id")))
            if reference is None:
                missing_references += 1
                continue
            sentences, sentence_scores, sentence_masks, similarities = build_sentence_scores(
                reference,
                item,
                nlp,
                model,
                tokenizer,
                device,
                max_length,
                min_similarity,
                min_tokens,
            )
            edu_scores, edu_masks = project_sentence_scores_to_edus(
                item, sentences, sentence_scores, sentence_masks
            )
            all_similarities.extend(similarities)
            reference_id = reference.get("item_id")

        new_item["edu_lexical_scores"] = edu_scores
        new_item["edu_lexical_masks"] = edu_masks
        new_item["lexical_reference_id"] = reference_id
        output.append(new_item)
        if index % 100 == 0:
            print(f"Processed {index}/{len(items)}", flush=True)

    class_counts = Counter(item["label"] for item in output)
    valid = sum(sum(item["edu_lexical_masks"]) for item in output)
    total = sum(len(item["edu_lexical_masks"]) for item in output)
    target_sum = sum(
        sum(score * mask for score, mask in zip(item["edu_lexical_scores"], item["edu_lexical_masks"]))
        for item in output
    )
    stats = {
        "documents": len(output),
        "groups": len({item["group_id"] for item in output}),
        "class_counts": dict(class_counts),
        "valid_edus": valid,
        "total_edus": total,
        "masked_ratio": float(1.0 - valid / max(total, 1)),
        "mean_valid_target": float(target_sum / max(valid, 1)),
        "mean_alignment_similarity": float(
            sum(all_similarities) / max(len(all_similarities), 1)
        ),
        "missing_references": missing_references,
    }
    return output, stats


def load_spacy(model_name: str):
    import spacy

    return spacy.load(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone_model_path", default="roberta-base")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_similarity", type=float, default=0.50)
    parser.add_argument("--min_tokens", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_groups", type=int)
    args = parser.parse_args()

    merged = deduplicate_items(
        [item for path in args.inputs for item in load_jsonl(path)]
    )
    groups = build_complete_pairs(merged)
    if args.max_groups is not None:
        selected_ids = sorted(groups)[: args.max_groups]
        groups = {group_id: groups[group_id] for group_id in selected_ids}
    split_items = split_groups(groups, args.train_ratio, args.val_ratio, args.seed)

    split_group_sets = {
        name: {str(item["group_id"]) for item in items}
        for name, items in split_items.items()
    }
    if any(
        split_group_sets[left] & split_group_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("group_id leakage detected after split")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nlp = load_spacy(args.spacy_model)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)
    model = AutoModel.from_pretrained(args.backbone_model_path).to(device).eval()

    stats: dict[str, Any] = {
        "config": vars(args),
        "deduplicated_documents": len(merged),
        "complete_groups": len(groups),
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        built, split_stats = build_split(
            split_items[split_name],
            nlp,
            model,
            tokenizer,
            device,
            args.max_length,
            args.min_similarity,
            args.min_tokens,
        )
        write_jsonl(built, os.path.join(args.output_dir, f"{split_name}_graph.jsonl"))
        stats["splits"][split_name] = split_stats

    with open(
        os.path.join(args.output_dir, "lexical_trace_stats.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(stats, output, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.output_dir, "split_manifest.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            {name: sorted(group_ids) for name, group_ids in split_group_sets.items()},
            output,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
