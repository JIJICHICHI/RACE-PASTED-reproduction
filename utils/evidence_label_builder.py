"""Build EDU-level weak evidence labels for LE-RACE."""

from __future__ import annotations

import argparse
import json
import math
import os
import string
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils.edu_utils import assign_sentence_label_to_edu, traverse_rst_edus


PURE_LABELS = {"human_written", "ai_generated"}
MIXED_LABELS = {"human_ai_polished", "ai_humanized"}
DOC_EMBED_CACHE: dict[str, torch.Tensor] = {}
SENTENCE_CACHE: dict[str, list[dict[str, Any]]] = {}
SENT_EMBED_CACHE: dict[str, torch.Tensor] = {}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line {line_no} in {path}")
    return items


def write_jsonl(items: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_group_index(
    all_items: list[dict[str, Any]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in all_items:
        group_id = item.get("group_id")
        label = item.get("label")
        if group_id and label:
            index[group_id][label].append(item)
    return index


def normalize_tokens(text: str, nlp) -> list[str]:
    punct = set(string.punctuation)
    return [
        tok.text.lower()
        for tok in nlp(text)
        if tok.text and tok.text not in punct and not tok.is_space
    ]


def lcs_len(tokens_a: list[str], tokens_b: list[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    prev = [0] * (len(tokens_b) + 1)
    for tok_a in tokens_a:
        cur = [0]
        for j, tok_b in enumerate(tokens_b, start=1):
            if tok_a == tok_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def token_lcs_edit_ratio(ref_text: str, tgt_text: str, nlp) -> float:
    ref_tokens = normalize_tokens(ref_text, nlp)
    tgt_tokens = normalize_tokens(tgt_text, nlp)
    denom = max(len(ref_tokens), len(tgt_tokens))
    if denom == 0:
        return 0.0
    return 1.0 - (lcs_len(ref_tokens, tgt_tokens) / denom)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


@torch.no_grad()
def embed_texts(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_len: int = 512,
    batch_size: int = 16,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        outputs.append(mean_pool(hidden, encoded["attention_mask"]).cpu())
    if not outputs:
        return torch.empty((0, model.config.hidden_size))
    return F.normalize(torch.cat(outputs, dim=0), p=2, dim=1)


@torch.no_grad()
def embed_long_text(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_len: int = 512,
    stride: int = 256,
) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
    if not encoded:
        return torch.zeros(model.config.hidden_size)
    chunks: list[str] = []
    for start in range(0, len(encoded), stride):
        chunk_ids = encoded[start : start + max_len - 2]
        if not chunk_ids:
            continue
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if start + max_len - 2 >= len(encoded):
            break
    embeds = embed_texts(chunks, model, tokenizer, device, max_len=max_len)
    if embeds.numel() == 0:
        return torch.zeros(model.config.hidden_size)
    return F.normalize(embeds.mean(dim=0), p=2, dim=0)


def sentence_split(text: str, nlp) -> list[dict[str, Any]]:
    doc = nlp(text)
    sentences: list[dict[str, Any]] = []
    for sent in doc.sents:
        sentences.append(
            {
                "text": sent.text,
                "start": sent.start_char,
                "end": sent.end_char,
                "tokens": [tok.text for tok in sent if not tok.is_space],
            }
        )
    return sentences


def item_cache_key(item: dict[str, Any]) -> str:
    return str(item.get("item_id") or item.get("id") or id(item))


def get_sentences(item: dict[str, Any], nlp) -> list[dict[str, Any]]:
    key = item_cache_key(item)
    if key not in SENTENCE_CACHE:
        SENTENCE_CACHE[key] = sentence_split(item.get("article", ""), nlp)
    return SENTENCE_CACHE[key]


def get_sentence_embeddings(
    item: dict[str, Any],
    sentences: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    max_len: int,
) -> torch.Tensor:
    key = item_cache_key(item)
    if key not in SENT_EMBED_CACHE:
        SENT_EMBED_CACHE[key] = embed_texts(
            [s["text"] for s in sentences], model, tokenizer, device, max_len
        )
    return SENT_EMBED_CACHE[key]


def get_doc_embedding(
    item: dict[str, Any], model, tokenizer, device: torch.device, max_len: int
) -> torch.Tensor:
    key = item_cache_key(item)
    if key not in DOC_EMBED_CACHE:
        DOC_EMBED_CACHE[key] = embed_long_text(
            item.get("article", ""), model, tokenizer, device, max_len
        )
    return DOC_EMBED_CACHE[key]


def compute_sentence_alignment(
    ref_sents: list[dict[str, Any]],
    tgt_sents: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    max_len: int,
) -> list[tuple[int, int, float]]:
    if not ref_sents or not tgt_sents:
        return []
    ref_emb = embed_texts([s["text"] for s in ref_sents], model, tokenizer, device, max_len)
    tgt_emb = embed_texts([s["text"] for s in tgt_sents], model, tokenizer, device, max_len)
    sims = tgt_emb @ ref_emb.T
    alignments: list[tuple[int, int, float]] = []
    for tgt_idx in range(sims.size(0)):
        ref_idx = int(torch.argmax(sims[tgt_idx]).item())
        alignments.append((tgt_idx, ref_idx, float(sims[tgt_idx, ref_idx].item())))
    return alignments


def find_reference(
    item: dict[str, Any],
    group_index: dict[str, dict[str, list[dict[str, Any]]]],
    model=None,
    tokenizer=None,
    device: torch.device | None = None,
    max_len: int = 512,
) -> dict[str, Any] | None:
    group_id = item.get("group_id")
    label = item.get("label")
    if not group_id or label not in MIXED_LABELS:
        return None
    group = group_index.get(group_id, {})

    if label == "human_ai_polished":
        candidates = group.get("human_written", [])
        return candidates[0] if candidates else None

    candidates = group.get("ai_generated", [])
    if not candidates:
        return None

    model_name = item.get("model_name")
    if model_name:
        for candidate in candidates:
            if candidate.get("model_name") == model_name:
                return candidate

    if model is None or tokenizer is None or device is None:
        return candidates[0]

    target_emb = get_doc_embedding(item, model, tokenizer, device, max_len)
    best_item = candidates[0]
    best_sim = -math.inf
    for candidate in candidates:
        cand_emb = get_doc_embedding(candidate, model, tokenizer, device, max_len)
        sim = float(torch.dot(target_emb, cand_emb).item())
        if sim > best_sim:
            best_item = candidate
            best_sim = sim
    return best_item


def label_target_sentences(
    ref_item: dict[str, Any] | None,
    tgt_item: dict[str, Any],
    config: dict[str, Any],
    nlp,
    model,
    tokenizer,
    device: torch.device,
) -> tuple[list[int], list[int], dict[str, Any]]:
    min_sent_tokens = int(config["min_sent_tokens"])
    semantic_sim_threshold = float(config["semantic_sim_threshold"])
    edit_ratio_threshold = float(config["edit_ratio_threshold"])
    max_len = int(config["max_seq_length"])

    tgt_sents = get_sentences(tgt_item, nlp)
    label = tgt_item.get("label")
    stats = {
        "num_sentences": len(tgt_sents),
        "has_trace_ref": False,
        "positive_sentences": 0,
        "masked_sentences": 0,
    }

    if label in PURE_LABELS:
        sent_labels = [0] * len(tgt_sents)
        sent_masks = [1 if len(s["tokens"]) >= min_sent_tokens else 0 for s in tgt_sents]
        stats["has_trace_ref"] = True
        stats["masked_sentences"] = sum(sent_masks)
        return sent_labels, sent_masks, stats

    if ref_item is None:
        return [0] * len(tgt_sents), [0] * len(tgt_sents), stats

    ref_sents = get_sentences(ref_item, nlp)
    if ref_sents and tgt_sents:
        ref_emb = get_sentence_embeddings(ref_item, ref_sents, model, tokenizer, device, max_len)
        tgt_emb = get_sentence_embeddings(tgt_item, tgt_sents, model, tokenizer, device, max_len)
        sims = tgt_emb @ ref_emb.T
        alignments = []
        for tgt_idx in range(sims.size(0)):
            ref_idx = int(torch.argmax(sims[tgt_idx]).item())
            alignments.append((tgt_idx, ref_idx, float(sims[tgt_idx, ref_idx].item())))
    else:
        alignments = []
    sent_labels = [0] * len(tgt_sents)
    sent_masks = [0] * len(tgt_sents)

    for tgt_idx, ref_idx, semantic_sim in alignments:
        if len(tgt_sents[tgt_idx]["tokens"]) < min_sent_tokens:
            continue
        edit_ratio = token_lcs_edit_ratio(
            ref_sents[ref_idx]["text"], tgt_sents[tgt_idx]["text"], nlp
        )
        if semantic_sim >= semantic_sim_threshold and edit_ratio >= edit_ratio_threshold:
            sent_labels[tgt_idx] = 1
            sent_masks[tgt_idx] = 1
        elif semantic_sim >= 0.90 and edit_ratio <= 0.10:
            sent_labels[tgt_idx] = 0
            sent_masks[tgt_idx] = 1

    stats["has_trace_ref"] = True
    stats["positive_sentences"] = sum(sent_labels)
    stats["masked_sentences"] = sum(sent_masks)
    return sent_labels, sent_masks, stats


def project_sentence_labels_to_edus(
    item: dict[str, Any],
    sent_labels: list[int],
    sent_masks: list[int],
    nlp,
) -> tuple[list[int], list[int]]:
    edus = traverse_rst_edus(item.get("rst_structure"))
    sentences = get_sentences(item, nlp)
    sent_spans = [(s["start"], s["end"]) for s in sentences]
    edu_labels: list[int] = []
    edu_masks: list[int] = []
    for edu in edus:
        label, mask = assign_sentence_label_to_edu(
            (edu.get("start"), edu.get("end")), sent_spans, sent_labels, sent_masks
        )
        edu_labels.append(label)
        edu_masks.append(mask)
    return edu_labels, edu_masks


def summarize_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups["overall"].append(item)
        groups[f"class:{item.get('label', 'unknown')}"].append(item)
        groups[f"domain:{item.get('source_dataset', 'unknown')}"].append(item)

    def _summarize(subset: list[dict[str, Any]]) -> dict[str, float]:
        total_edu = sum(len(x.get("edu_evidence_masks", [])) for x in subset)
        total_mask = sum(sum(x.get("edu_evidence_masks", [])) for x in subset)
        total_pos = sum(sum(x.get("edu_evidence_labels", [])) for x in subset)
        return {
            "items": len(subset),
            "coverage": float(sum(1 for x in subset if x.get("has_trace_ref")) / max(1, len(subset))),
            "positive_ratio": float(total_pos / max(1, total_mask)),
            "masked_ratio": float(1.0 - total_mask / max(1, total_edu)),
            "avg_full_edu": float(total_edu / max(1, len(subset))),
        }

    return {name: _summarize(subset) for name, subset in groups.items()}


def build_evidence_labels_for_split(
    split_items: list[dict[str, Any]],
    group_index: dict[str, dict[str, list[dict[str, Any]]]],
    config: dict[str, Any],
    nlp,
    model,
    tokenizer,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_items: list[dict[str, Any]] = []
    for idx, item in enumerate(split_items, start=1):
        ref_item = find_reference(
            item, group_index, model=model, tokenizer=tokenizer, device=device,
            max_len=int(config["max_seq_length"])
        )
        sent_labels, sent_masks, sent_stats = label_target_sentences(
            ref_item, item, config, nlp, model, tokenizer, device
        )
        edu_labels, edu_masks = project_sentence_labels_to_edus(
            item, sent_labels, sent_masks, nlp
        )

        new_item = dict(item)
        new_item["edu_evidence_labels"] = edu_labels
        new_item["edu_evidence_masks"] = edu_masks
        new_item["has_trace_ref"] = bool(sent_stats.get("has_trace_ref", False))
        new_item["evidence_ref_id"] = ref_item.get("item_id") if ref_item else None
        new_item["evidence_ref_label"] = ref_item.get("label") if ref_item else None
        valid_masks = sum(edu_masks)
        new_item["evidence_positive_ratio"] = float(sum(edu_labels) / max(1, valid_masks))
        new_item["evidence_masked_ratio"] = float(1.0 - valid_masks / max(1, len(edu_masks)))
        output_items.append(new_item)

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(split_items)} items", flush=True)

    return output_items, summarize_items(output_items)


def load_spacy(model_name: str):
    try:
        import spacy
        return spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load spaCy model '{model_name}'. Install it with: "
            f"python -m spacy download {model_name}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LE-RACE EDU evidence labels.")
    parser.add_argument("--input_train", default="data/hart_split/train_graph.jsonl")
    parser.add_argument("--input_val", default="data/hart_split/val_graph.jsonl")
    parser.add_argument("--input_test", default="data/hart_split/test_graph.jsonl")
    parser.add_argument("--output_dir", default="data/hart_split_evidence")
    parser.add_argument("--backbone_model_path", default="roberta-base")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--semantic_sim_threshold", type=float, default=0.85)
    parser.add_argument("--edit_ratio_threshold", type=float, default=0.25)
    parser.add_argument("--min_sent_tokens", type=int, default=5)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = vars(args)
    train_items = load_jsonl(args.input_train)
    val_items = load_jsonl(args.input_val)
    test_items = load_jsonl(args.input_test)
    all_items = train_items + val_items + test_items
    group_index = build_group_index(all_items)

    device = torch.device(args.device)
    nlp = load_spacy(args.spacy_model)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)
    model = AutoModel.from_pretrained(args.backbone_model_path).to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    stats: dict[str, Any] = {"config": config, "splits": {}}
    for split_name, split_items in [
        ("train", train_items),
        ("val", val_items),
        ("test", test_items),
    ]:
        print(f"Building evidence labels for {split_name}: {len(split_items)} items", flush=True)
        output_items, split_stats = build_evidence_labels_for_split(
            split_items, group_index, config, nlp, model, tokenizer, device
        )
        write_jsonl(output_items, os.path.join(args.output_dir, f"{split_name}_graph.jsonl"))
        stats["splits"][split_name] = split_stats
        print(f"Finished {split_name}: {split_stats.get('overall', {})}", flush=True)

    with open(os.path.join(args.output_dir, "evidence_label_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

