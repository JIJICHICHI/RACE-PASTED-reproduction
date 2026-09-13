"""Utilities for aligning RST EDUs, sentence spans, and token spans."""

from __future__ import annotations

from typing import Any


def traverse_rst_edus(rst_node: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return EDU nodes in the same traversal order as FlexibleGraphDataset."""
    edus: list[dict[str, Any]] = []

    def _visit(node: dict[str, Any] | None) -> None:
        if not node:
            return
        if node.get("relation") == "elementary":
            edus.append(
                {
                    "id": len(edus),
                    "text": node.get("text", ""),
                    "start": node.get("start"),
                    "end": node.get("end"),
                    "nuclearity": node.get("nuclearity"),
                    "relation": node.get("relation"),
                }
            )
            return
        _visit(node.get("left"))
        _visit(node.get("right"))

    _visit(rst_node)
    return edus


def overlap_len(
    span_a: tuple[int | None, int | None],
    span_b: tuple[int | None, int | None],
) -> int:
    """Return character overlap length between two half-open spans."""
    if None in span_a or None in span_b:
        return 0
    a_start, a_end = int(span_a[0]), int(span_a[1])
    b_start, b_end = int(span_b[0]), int(span_b[1])
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def map_char_spans_to_token_spans(
    char_spans: list[tuple[int | None, int | None]],
    offset_mapping: list[tuple[int, int]],
) -> list[list[int]]:
    """Map character spans to token spans using tokenizer offsets."""
    token_spans: list[list[int]] = []
    for char_start, char_end in char_spans:
        if char_start is None or char_end is None:
            continue
        token_start_index = -1
        token_end_index = -1
        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                continue
            if token_start_index == -1 and start < char_end and end > char_start:
                token_start_index = idx
            if token_start_index != -1 and start < char_end and end > char_start:
                token_end_index = idx
        if token_start_index != -1:
            token_spans.append([token_start_index, token_end_index + 1])
    return token_spans


def assign_sentence_label_to_edu(
    edu_span: tuple[int | None, int | None],
    sent_spans: list[tuple[int, int]],
    sent_labels: list[int],
    sent_masks: list[int],
    min_overlap_gap: int = 5,
) -> tuple[int, int]:
    """Project sentence-level weak labels onto one EDU by maximum span overlap."""
    overlaps = [overlap_len(edu_span, sent_span) for sent_span in sent_spans]
    if not overlaps or max(overlaps) == 0:
        return 0, 0

    ranked = sorted(enumerate(overlaps), key=lambda item: item[1], reverse=True)
    best_idx, best_overlap = ranked[0]
    second_overlap = ranked[1][1] if len(ranked) > 1 else 0
    if best_overlap - second_overlap < min_overlap_gap:
        return 0, 0

    if best_idx >= len(sent_labels) or best_idx >= len(sent_masks):
        return 0, 0
    return int(sent_labels[best_idx]), int(sent_masks[best_idx])

