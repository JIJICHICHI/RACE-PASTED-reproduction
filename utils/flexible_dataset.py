import json
import torch
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from collections import defaultdict


class FlexibleGraphDataset(Dataset):
    """
    A flexible dataset that loads data from a JSONL file, processes it into a graph-compatible
    format based on RST (Rhetorical Structure Theory), and serves it for training or evaluation.
    """

    def __init__(self, file_path: str, config: dict):
        """
        Initializes the dataset by loading and processing the data.
        """
        self.config = config
        self.raw_data = self._load_data(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("backbone_model_path", None)
        )
        self.processed_data = self._process_data()

    def _load_data(self, file_path: str) -> list:
        """Loads data from a JSONL file."""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {file_path}")
                    continue
        return data

    def _rst_traversal(
        self, rst_node, parent_id, nodes, parent_child_edges, descendant_map
    ):
        """
        Recursively traverses the RST tree to extract nodes, edges, and descendant info.
        Returns the ID of the current node and a list of its descendant EDU node IDs.
        """
        if not rst_node:
            return None, []

        node_id = len(nodes)
        nodes.append(None)  # Add a placeholder, to be filled after recursive calls

        if parent_id is not None:
            parent_child_edges.append((parent_id, node_id))

        is_edu = rst_node.get("relation") == "elementary"

        if is_edu:
            node_info = {
                "id": node_id,
                "type": "edu_node",
                "text": rst_node.get("text", ""),
                "relation": rst_node.get("relation"),
                "nuclearity": rst_node.get("nuclearity"),
                "start": rst_node.get("start"),
                "end": rst_node.get("end"),
            }
            nodes[node_id] = node_info
            descendant_map[node_id] = [node_id]
            return node_id, [node_id]

        left_child_id, left_edu_ids = self._rst_traversal(
            rst_node.get("left"), node_id, nodes, parent_child_edges, descendant_map
        )
        right_child_id, right_edu_ids = self._rst_traversal(
            rst_node.get("right"), node_id, nodes, parent_child_edges, descendant_map
        )

        node_info = {
            "id": node_id,
            "type": "relation_node",
            "text": rst_node.get("text", ""),
            "relation": rst_node.get("relation"),
            "nuclearity": rst_node.get("nuclearity"),
            "left_child_id": left_child_id,
            "right_child_id": right_child_id,
        }
        nodes[node_id] = node_info

        my_descendant_edu_ids = left_edu_ids + right_edu_ids
        descendant_map[node_id] = my_descendant_edu_ids
        return node_id, my_descendant_edu_ids

    def _process_data(self) -> list:
        """
        Processes the raw data to create graph-compatible items based on RST structures.
        """
        processed_items = []
        for item in self.raw_data:
            # Switch to the new contextual processing method
            processed_item = self._process_contextual_rst_item(item)
            if processed_item:
                processed_items.append(processed_item)
        return processed_items

    def _process_contextual_rst_item(self, item):
        """
        Processes a single raw data item based on its RST structure, creating contextual embeddings.
        The full text is tokenized once, and each EDU's representation is derived from its
        corresponding token span in the full text's hidden states.
        """
        rst_structure = item.get("rst_structure")
        full_text = item.get("article")
        if not rst_structure or not full_text:
            return None

        # 1. Traverse RST to get node structure and EDU character spans
        nodes, parent_child_edges, descendant_map = [], [], {}
        self._rst_traversal(
            rst_structure, None, nodes, parent_child_edges, descendant_map
        )

        edu_nodes = [node for node in nodes if node["type"] == "edu_node"]
        edu_char_spans = [
            (node.get("start", 0), node.get("end", 0)) for node in edu_nodes
        ]

        # 2. Tokenize the full text and get character-to-token offset mapping
        tokenized_full_text = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.get("max_seq_length", 512),
            return_offsets_mapping=True,
        )
        input_ids = tokenized_full_text["input_ids"]
        offset_mapping = tokenized_full_text["offset_mapping"]

        # 3. Map EDU character spans to token spans
        edu_token_spans = []
        for char_start, char_end in edu_char_spans:
            token_start_index = -1
            token_end_index = -1

            for i, (start, end) in enumerate(offset_mapping):
                if (
                    start == end
                ):  # Skip special tokens like [CLS], [SEP] which have a (0,0) offset
                    continue

                # Find the first token that is part of the EDU
                if token_start_index == -1 and start < char_end and end > char_start:
                    token_start_index = i

                # Find the last token that is part of the EDU
                if token_start_index != -1 and start < char_end and end > char_start:
                    token_end_index = i

            if token_start_index != -1:
                # Add 1 to end index to make it an exclusive boundary for slicing
                edu_token_spans.append([token_start_index, token_end_index + 1])

        # If no valid spans were found but EDUs exist, it's a data mismatch. Handle gracefully.
        if len(edu_token_spans) != len(edu_nodes):
            # This can happen if EDUs are outside the truncated max_length of the tokenizer
            # The downstream code (graph_builder, model) is designed to handle this by
            # only using the number of found spans as the limit for EDU nodes.
            pass

        # [NEW] Find the root node(s)
        if not parent_child_edges and nodes:
            # If there are no edges, the single node is the root
            root_node_ids = [nodes[0]["id"]] if nodes else []
        else:
            parent_ids = {edge[0] for edge in parent_child_edges}
            child_ids = {edge[1] for edge in parent_child_edges}
            root_node_ids = list(parent_ids - child_ids)

        label_str = item.get("label", "")
        default_label_map = {
            "human_written": 0,
            "human_ai_polished": 1,
            "ai_generated": 2,
            "ai_humanized": 3,
        }
        label_map = self.config.get("label_map", default_label_map)
        cat_label = label_map.get(label_str.lower(), -1)

        # [NEW] Domain Label for DANN
        group_id = item.get("group_id", "")
        domain_str = group_id.split("-")[0] if group_id else ""
        domain_map = {
            "arxiv": 0,
            "news": 1,
            "writing": 2,
            "essay": 3,
        }
        domain_label = domain_map.get(domain_str.lower(), -1)

        n_valid_edu = len(edu_token_spans)
        raw_evidence_labels = item.get("edu_evidence_labels")
        raw_evidence_masks = item.get("edu_evidence_masks")
        if raw_evidence_labels is not None and raw_evidence_masks is not None:
            evidence_labels = [int(x) for x in raw_evidence_labels[:n_valid_edu]]
            evidence_masks = [int(x) for x in raw_evidence_masks[:n_valid_edu]]
            if len(evidence_labels) < n_valid_edu:
                evidence_labels.extend([0] * (n_valid_edu - len(evidence_labels)))
            if len(evidence_masks) < n_valid_edu:
                evidence_masks.extend([0] * (n_valid_edu - len(evidence_masks)))
        else:
            evidence_labels = [0] * n_valid_edu
            evidence_masks = [0] * n_valid_edu

        raw_lexical_scores = item.get("edu_lexical_scores")
        raw_lexical_masks = item.get("edu_lexical_masks")
        if raw_lexical_scores is not None and raw_lexical_masks is not None:
            lexical_scores = [float(x) for x in raw_lexical_scores[:n_valid_edu]]
            lexical_masks = [int(x) for x in raw_lexical_masks[:n_valid_edu]]
            if len(lexical_scores) < n_valid_edu:
                lexical_scores.extend([0.0] * (n_valid_edu - len(lexical_scores)))
            if len(lexical_masks) < n_valid_edu:
                lexical_masks.extend([0] * (n_valid_edu - len(lexical_masks)))
        else:
            lexical_scores = [0.0] * n_valid_edu
            lexical_masks = [0] * n_valid_edu

        raw_humanization_scores = item.get("edu_humanization_scores")
        raw_humanization_masks = item.get("edu_humanization_masks")
        if raw_humanization_scores is not None and raw_humanization_masks is not None:
            humanization_scores = [
                float(x) for x in raw_humanization_scores[:n_valid_edu]
            ]
            humanization_masks = [
                int(x) for x in raw_humanization_masks[:n_valid_edu]
            ]
            if len(humanization_scores) < n_valid_edu:
                humanization_scores.extend(
                    [0.0] * (n_valid_edu - len(humanization_scores))
                )
            if len(humanization_masks) < n_valid_edu:
                humanization_masks.extend(
                    [0] * (n_valid_edu - len(humanization_masks))
                )
        else:
            humanization_scores = [0.0] * n_valid_edu
            humanization_masks = [0] * n_valid_edu

        return {
            "id": item.get("item_id"),  # Add the item's ID for tracking
            "nodes": nodes,  # Keep full nodes for graph building, builder must be robust
            "parent_child_edges": parent_child_edges,
            "descendant_map": descendant_map,
            "root_node_ids": root_node_ids,
            "full_text_input_ids": torch.tensor(input_ids, dtype=torch.long),
            "full_text_attention_mask": torch.tensor(
                tokenized_full_text["attention_mask"], dtype=torch.long
            ),
            "edu_token_spans": edu_token_spans,
            "n_valid_edu": torch.tensor(n_valid_edu, dtype=torch.long),
            "edu_evidence_labels": torch.tensor(evidence_labels, dtype=torch.float),
            "edu_evidence_masks": torch.tensor(evidence_masks, dtype=torch.float),
            "edu_lexical_scores": torch.tensor(lexical_scores, dtype=torch.float),
            "edu_lexical_masks": torch.tensor(lexical_masks, dtype=torch.float),
            "edu_humanization_scores": torch.tensor(
                humanization_scores, dtype=torch.float
            ),
            "edu_humanization_masks": torch.tensor(
                humanization_masks, dtype=torch.float
            ),
            "has_trace_ref": bool(item.get("has_trace_ref", False)),
            "evidence_ref_id": item.get("evidence_ref_id"),
            "lexical_reference_id": item.get("lexical_reference_id"),
            "label": torch.tensor(cat_label, dtype=torch.long),
            "domain_label": torch.tensor(domain_label, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> dict:
        return self.processed_data[idx]

    @staticmethod
    def collate_fn(batch: list) -> dict:
        if not batch:
            return {}

        # This collate_fn is now more complex due to the contextual embedding strategy.
        # We need to pad the full_text_input_ids and attention_masks.

        elem = batch[0]
        collated_batch = defaultdict(list)
        labels = []
        domain_labels = []
        n_valid_edu = []
        edu_evidence_labels = []
        edu_evidence_masks = []
        edu_lexical_scores = []
        edu_lexical_masks = []
        edu_humanization_scores = []
        edu_humanization_masks = []
        has_trace_ref = []

        # Find the max length in the batch for padding
        max_len = max(len(item["full_text_input_ids"]) for item in batch)

        for item in batch:
            for key, value in item.items():
                if key == "label":
                    labels.append(value)
                elif key == "domain_label":
                    domain_labels.append(value)
                elif key == "n_valid_edu":
                    n_valid_edu.append(value)
                elif key == "edu_evidence_labels":
                    edu_evidence_labels.append(value)
                elif key == "edu_evidence_masks":
                    edu_evidence_masks.append(value)
                elif key == "edu_lexical_scores":
                    edu_lexical_scores.append(value)
                elif key == "edu_lexical_masks":
                    edu_lexical_masks.append(value)
                elif key == "edu_humanization_scores":
                    edu_humanization_scores.append(value)
                elif key == "edu_humanization_masks":
                    edu_humanization_masks.append(value)
                elif key == "has_trace_ref":
                    has_trace_ref.append(value)
                elif key in ["full_text_input_ids", "full_text_attention_mask"]:
                    # Pad to max_len
                    pad_len = max_len - len(value)
                    # Use 0 for padding, which is the standard pad_token_id for BERT-like models
                    padded_tensor = torch.nn.functional.pad(
                        value, (0, pad_len), mode="constant", value=0
                    )
                    collated_batch[key].append(padded_tensor)
                else:
                    collated_batch[key].append(value)

        collated_batch["labels"] = torch.stack(labels)
        if domain_labels:
            collated_batch["domain_labels"] = torch.stack(domain_labels)
        if n_valid_edu:
            collated_batch["n_valid_edu"] = torch.stack(n_valid_edu)
        if edu_evidence_labels:
            collated_batch["edu_evidence_labels"] = edu_evidence_labels
        if edu_evidence_masks:
            collated_batch["edu_evidence_masks"] = edu_evidence_masks
        if edu_lexical_scores:
            collated_batch["edu_lexical_scores"] = edu_lexical_scores
        if edu_lexical_masks:
            collated_batch["edu_lexical_masks"] = edu_lexical_masks
        if edu_humanization_scores:
            collated_batch["edu_humanization_scores"] = edu_humanization_scores
        if edu_humanization_masks:
            collated_batch["edu_humanization_masks"] = edu_humanization_masks
        if has_trace_ref:
            collated_batch["has_trace_ref"] = has_trace_ref
        # Stack the padded tensors to create a single batch tensor
        collated_batch["full_text_input_ids"] = torch.stack(
            collated_batch["full_text_input_ids"]
        )
        collated_batch["full_text_attention_mask"] = torch.stack(
            collated_batch["full_text_attention_mask"]
        )

        return dict(collated_batch)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("\n--- Testing Contextual RST Mode ---")
    RST_DATA_PATH = "./data/flattened_articles.jsonl"
    BATCH_SIZE = 2
    rst_config = {"backbone_model_path": "bert-base-uncased", "max_seq_length": 512}

    try:
        rst_dataset = FlexibleGraphDataset(file_path=RST_DATA_PATH, config=rst_config)
        print(f"Contextual RST Dataset created with {len(rst_dataset)} samples.")

        if len(rst_dataset) > 0:
            rst_dataloader = DataLoader(
                rst_dataset,
                batch_size=BATCH_SIZE,
                collate_fn=FlexibleGraphDataset.collate_fn,
            )
            rst_batch_data = next(iter(rst_dataloader))

            print("\n--- Contextual RST Batch Data Structure ---")
            for key, value in rst_batch_data.items():
                if isinstance(value, list):
                    print(f"'{key}': list of {len(value)} items")
                elif isinstance(value, torch.Tensor):
                    print(f"'{key}': tensor with shape {value.shape}")
                else:
                    print(f"'{key}': {type(value)}")
            print("-" * 20)

            if rst_batch_data.get("nodes"):
                first_item_nodes = rst_batch_data["nodes"][0]
                first_item_edu_count = sum(
                    1 for n in first_item_nodes if n["type"] == "edu_node"
                )
                first_item_spans = rst_batch_data["edu_token_spans"][0]

                print(f"First item total nodes: {len(first_item_nodes)}")
                print(f"First item EDU nodes count: {first_item_edu_count}")
                print(f"First item found EDU token spans: {len(first_item_spans)}")
                if first_item_spans:
                    print(f"Example EDU token span: {first_item_spans[0]}")

                # Check consistency
                if first_item_edu_count != len(first_item_spans):
                    print(
                        f"\nWARNING: Mismatch between EDU count ({first_item_edu_count}) and found spans ({len(first_item_spans)}). This might be due to tokenizer max_length truncation."
                    )

        else:
            print("RST dataset is empty, skipping dataloader test.")
    except Exception as e:
        import traceback

        print(f"Error during contextual RST mode test: {e}")
        traceback.print_exc()
