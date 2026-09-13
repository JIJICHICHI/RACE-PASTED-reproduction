import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import RGCNConv, GCNConv, global_mean_pool
from transformers.models.auto.modeling_auto import AutoModel

from utils.flexible_graph_builder import FlexibleGraphBuilder
from .modules import ClassificationHead

class ProjectionHead(nn.Module):
    """Two-layer normalized projection head for factor-level contrastive spaces."""

    def __init__(self, hidden_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class RACEModel(nn.Module):
    """
    RACE (Rhetorical Analysis for Creator-Editor Modeling):
    An RST-based Graph Neural Network model for fine-grained LLM-generated text detection.

    This model converts a document's RST (Rhetorical Structure Theory) tree into a heterogeneous
    graph, then applies relational GNN layers (RGCN) to learn structure-aware representations,
    and finally classifies the document based on the root node's representation.

    Architecture:
        1. Backbone (RoBERTa) encodes the document and produces contextual EDU features.
        2. RST-based graph is constructed with EDU nodes and Relation nodes.
        3. RGCN layers perform relational message passing over the graph.
        4. Root pooling extracts the document-level representation.
        5. MLP classifier produces the final prediction.
    """

    def __init__(
        self,
        feature_dim: int,
        gnn_hidden_dim: int,
        num_heads: int,
        num_classes: int,
        config: dict,
        metadata: dict,
        output_features: bool = False,
    ):
        super().__init__()
        # --- 1. Module Setup and Configuration ---
        self.config = config
        self.metadata = metadata
        self.output_features = output_features

        # GNN and feature dimensions.
        self.bert_dim = config.get("bert_dim", 768)
        self.feature_dim = feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_layers = config.get("num_gnn_layers", 2)

        # --- 2. Backbone Model (RoBERTa) ---
        self.bert_model = AutoModel.from_pretrained(config["backbone_model_path"])
        print(f"Loading backbone model from {config['backbone_model_path']}")

        # Freezing logic for the backbone model's layers based on the config.
        unfreeze_layers = config.get("unfreeze_bert_layers", 0)
        if unfreeze_layers > 0 and hasattr(self.bert_model, "encoder"):
            for param in self.bert_model.parameters():
                param.requires_grad = False
            for layer in self.bert_model.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif unfreeze_layers == -1:  # Unfreeze all layers
            self.bert_model.requires_grad_(True)
        else:  # Freeze all layers
            self.bert_model.requires_grad_(False)

        # --- 3. Relation Mapping (Critical for RGCN) ---
        # Maps edge type tuples like ('edu_node', 'Elaboration', 'relation_node') to unique integer IDs.
        # This is required for the `edge_type` tensor used by native relational GNN layers.
        self.edge_types_list = list(metadata["edge_types"])
        self.edge_type_to_idx = {
            etype: i for i, etype in enumerate(self.edge_types_list)
        }
        self.num_relations = len(self.edge_types_list)
        self.metadata["edge_type_to_idx"] = self.edge_type_to_idx
        print(
            f"Initialized RACE model with {self.num_relations} relation types."
        )
        self.graph_builder = FlexibleGraphBuilder(config, metadata)

        # --- 4. Node Type Embeddings ---
        # Creates an embedding layer to distinguish original node types (0 for EDU, 1 for Relation).
        # This allows the model to learn type-specific information even after node features are merged.
        self.node_type_embedding = nn.Embedding(2, self.bert_dim)

        # --- 5. Unified Projection Layer ---
        # Projects the initial node features (BERT output + type embedding) into the GNN's working dimension.
        # The LayerNorm is crucial for stabilizing training by normalizing feature distributions.
        self.node_proj = nn.Sequential(
            nn.Linear(self.bert_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(config.get("proj_dropout", 0.1)),
        )

        # --- 6. Native Relational GNN Layers ---
        # Uses efficient, native PyG layers designed for relational graphs.
        self.convs = nn.ModuleList()
        gnn_layer_type = self.config.get("gnn_layer_type", "RGCN")
        for i in range(self.num_layers):
            in_dim = self.feature_dim if i == 0 else self.gnn_hidden_dim
            if gnn_layer_type == "GCN":
                self.convs.append(
                    GCNConv(
                        in_channels=in_dim,
                        out_channels=gnn_hidden_dim,
                    )
                )
            else:  # Default to RGCN
                self.convs.append(
                    RGCNConv(
                        in_channels=in_dim,
                        out_channels=gnn_hidden_dim,
                        num_relations=self.num_relations,
                        num_bases=config.get("rgcn_num_bases", None),
                    )
                )

        # --- 7. Classifier ---
        classifier_type = self.config.get("classifier_type", "linear")
        if classifier_type == "mlp":
            self.classifier = ClassificationHead(
                input_dim=self.gnn_hidden_dim,
                hidden_dim=self.gnn_hidden_dim,
                num_classes=num_classes,
                dropout_prob=self.config.get("classifier_dropout", 0.1),
            )
        else:
            self.classifier = nn.Linear(self.gnn_hidden_dim, num_classes)

        # --- 8. Optional FCE-RACE Factorized Creator-Editor Branches ---
        self.use_factorized_ce = bool(self.config.get("use_factorized_ce", False))
        self.factorized_residual_gamma = float(
            self.config.get("factorized_residual_gamma", 0.0)
        )
        if self.use_factorized_ce:
            self.creator_head = ClassificationHead(
                input_dim=self.gnn_hidden_dim,
                hidden_dim=self.gnn_hidden_dim,
                num_classes=2,
                dropout_prob=self.config.get("creator_dropout", 0.1),
            )
            self.editor_proj = nn.Sequential(
                nn.Linear(self.bert_dim, self.gnn_hidden_dim),
                nn.LayerNorm(self.gnn_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.get("editor_dropout", 0.1)),
            )
            self.editor_queries = nn.Parameter(torch.empty(2, self.gnn_hidden_dim))
            nn.init.xavier_uniform_(self.editor_queries)
            editor_repr_dim = 3 * self.gnn_hidden_dim
            self.editor_use_variation_features = bool(
                self.config.get("editor_use_variation_features", True)
            )
            if self.editor_use_variation_features:
                self.editor_variation_proj = nn.Sequential(
                    nn.Linear(3, self.gnn_hidden_dim),
                    nn.LayerNorm(self.gnn_hidden_dim),
                    nn.GELU(),
                )
                editor_repr_dim += self.gnn_hidden_dim
            else:
                self.editor_variation_proj = None
            self.editor_head = ClassificationHead(
                input_dim=editor_repr_dim,
                hidden_dim=self.gnn_hidden_dim,
                num_classes=2,
                dropout_prob=self.config.get("editor_head_dropout", 0.1),
            )
        else:
            self.creator_head = None
            self.editor_proj = None
            self.editor_queries = None
            self.editor_variation_proj = None
            self.editor_head = None

        # --- 9. Optional FAR-RACE Factor Contrast Projection Heads ---
        self.use_factor_contrast = bool(self.config.get("use_factor_contrast", False))
        if self.use_factor_contrast:
            factor_proj_dim = int(self.config.get("factor_proj_dim", 128))
            self.proj_y = ProjectionHead(self.gnn_hidden_dim, factor_proj_dim)
            self.proj_c = ProjectionHead(self.gnn_hidden_dim, factor_proj_dim)
            self.proj_e = ProjectionHead(self.gnn_hidden_dim, factor_proj_dim)
        else:
            self.proj_y = None
            self.proj_c = None
            self.proj_e = None

        # --- 10. Optional LE-RACE Local Evidence Branch ---
        self.use_local_evidence = bool(self.config.get("use_local_evidence", False))
        self.evidence_pooling = self.config.get("evidence_pooling", "soft")
        self.evidence_topk_ratio = float(self.config.get("evidence_topk_ratio", 0.15))
        self.evidence_warmup_epochs = int(self.config.get("evidence_warmup_epochs", 2))
        self.disable_evidence_pooling = bool(self.config.get("disable_evidence_pooling", False))
        if self.use_local_evidence:
            self.trace_scorer = nn.Sequential(
                nn.Linear(3 * self.gnn_hidden_dim, self.gnn_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.get("evidence_dropout", 0.1)),
                nn.Linear(self.gnn_hidden_dim, 1),
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(3 * self.gnn_hidden_dim, self.gnn_hidden_dim),
                nn.LayerNorm(self.gnn_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.get("fusion_dropout", 0.1)),
            )
        else:
            self.trace_scorer = None
            self.fusion_proj = None

        # --- 11. Optional PASTED-RACE EDU Lexical Regression Branch ---
        self.use_lexical_trace = bool(self.config.get("use_lexical_trace", False))
        self.use_lexical_fusion = bool(self.config.get("use_lexical_fusion", False))
        if self.use_lexical_fusion and not self.use_lexical_trace:
            raise ValueError("use_lexical_fusion requires use_lexical_trace=true")
        self.lexical_attention_temperature = float(
            self.config.get("lexical_attention_temperature", 1.0)
        )
        self.lexical_fusion_warmup_epochs = int(
            self.config.get("lexical_fusion_warmup_epochs", 1)
        )
        self.lexical_fusion_mode = self.config.get(
            "lexical_fusion_mode", "replace"
        )
        if self.lexical_fusion_mode not in {"replace", "residual"}:
            raise ValueError("lexical_fusion_mode must be 'replace' or 'residual'")
        if self.use_lexical_trace:
            self.lexical_trace_head = nn.Sequential(
                nn.Linear(3 * self.gnn_hidden_dim, self.gnn_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.get("lexical_trace_dropout", 0.1)),
                nn.Linear(self.gnn_hidden_dim, 1),
            )
        else:
            self.lexical_trace_head = None
        if self.use_lexical_fusion:
            self.lexical_fusion_proj = nn.Sequential(
                nn.Linear(3 * self.gnn_hidden_dim, self.gnn_hidden_dim),
                nn.LayerNorm(self.gnn_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.get("lexical_fusion_dropout", 0.1)),
            )
            if self.lexical_fusion_mode == "residual":
                self.lexical_residual_gamma = nn.Parameter(
                    torch.tensor(
                        float(self.config.get("lexical_residual_gamma_init", 0.0))
                    )
                )
            else:
                self.register_parameter("lexical_residual_gamma", None)
        else:
            self.lexical_fusion_proj = None
            self.register_parameter("lexical_residual_gamma", None)

        self.use_humanization_trace = bool(
            self.config.get("use_humanization_trace", False)
        )
        self.use_humanization_fusion = bool(
            self.config.get("use_humanization_fusion", False)
        )
        if self.use_humanization_fusion and not self.use_humanization_trace:
            raise ValueError(
                "use_humanization_fusion requires use_humanization_trace=true"
            )
        self.humanization_attention_temperature = float(
            self.config.get(
                "humanization_attention_temperature",
                self.lexical_attention_temperature,
            )
        )
        if self.use_humanization_trace:
            self.humanization_trace_head = nn.Sequential(
                nn.Linear(3 * self.gnn_hidden_dim, self.gnn_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.get("humanization_trace_dropout", 0.1)),
                nn.Linear(self.gnn_hidden_dim, 1),
            )
        else:
            self.humanization_trace_head = None
        if self.use_humanization_fusion:
            self.humanization_fusion_proj = nn.Sequential(
                nn.Linear(3 * self.gnn_hidden_dim, self.gnn_hidden_dim),
                nn.LayerNorm(self.gnn_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.get("humanization_fusion_dropout", 0.1)),
            )
            self.humanization_residual_gamma = nn.Parameter(
                torch.tensor(
                    float(self.config.get("humanization_residual_gamma_init", 0.0))
                )
            )
        else:
            self.humanization_fusion_proj = None
            self.register_parameter("humanization_residual_gamma", None)

        # --- 12. Weight Initialization ---
        self.node_proj.apply(self._init_weights)
        self.convs.apply(self._init_weights)
        self._init_classifier(self.classifier)
        if self.use_factorized_ce:
            self._init_classifier(self.creator_head)
            self.editor_proj.apply(self._init_weights)
            if self.editor_variation_proj is not None:
                self.editor_variation_proj.apply(self._init_weights)
            self._init_classifier(self.editor_head)
        if self.use_factor_contrast:
            self.proj_y.apply(self._init_weights)
            self.proj_c.apply(self._init_weights)
            self.proj_e.apply(self._init_weights)
        if self.use_local_evidence:
            self.trace_scorer.apply(self._init_weights)
            self.fusion_proj.apply(self._init_weights)
        if self.use_lexical_trace:
            self.lexical_trace_head.apply(self._init_weights)
        if self.use_lexical_fusion:
            self.lexical_fusion_proj.apply(self._init_weights)
        if self.use_humanization_trace:
            self.humanization_trace_head.apply(self._init_weights)
        if self.use_humanization_fusion:
            self.humanization_fusion_proj.apply(self._init_weights)

    def _init_weights(self, m):
        """Initializes weights for Linear and LayerNorm layers using Xavier uniform."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _init_classifier(self, module):
        """Initializes classifier weights with small normal distribution for stable initial predictions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "children"):
            for child in module.children():
                self._init_classifier(child)

    def _compute_local_evidence(self, x, graph_batch, graph_list, root_node_indices):
        z_local_list = []
        evidence_logits_list = []
        evidence_probs_list = []

        for item_idx, graph_item in enumerate(graph_list):
            offset = graph_batch.ptr[item_idx].item()
            root_h = x[root_node_indices[item_idx]]
            if not getattr(graph_item, "_edu_indices", None):
                z_local_list.append(torch.zeros_like(root_h))
                evidence_logits_list.append(x.new_empty((0,)))
                evidence_probs_list.append(x.new_empty((0,)))
                continue

            edu_abs_indices = torch.tensor(
                graph_item._edu_indices, dtype=torch.long, device=x.device
            ) + offset
            edu_h = x[edu_abs_indices]
            if edu_h.size(0) == 0:
                z_local_list.append(torch.zeros_like(root_h))
                evidence_logits_list.append(x.new_empty((0,)))
                evidence_probs_list.append(x.new_empty((0,)))
                continue

            root_expand = root_h.unsqueeze(0).expand_as(edu_h)
            score_input = torch.cat([edu_h, root_expand, edu_h * root_expand], dim=-1)
            evidence_logits = self.trace_scorer(score_input).squeeze(-1)
            evidence_probs = torch.sigmoid(evidence_logits)

            if self.evidence_pooling == "topk":
                k = max(1, math.ceil(self.evidence_topk_ratio * edu_h.size(0)))
                top_idx = evidence_probs.topk(k=min(k, edu_h.size(0))).indices
                z_local = edu_h[top_idx].mean(dim=0)
            else:
                weights = evidence_probs / evidence_probs.sum().clamp_min(1e-8)
                z_local = (weights.unsqueeze(-1) * edu_h).sum(dim=0)

            z_local_list.append(z_local)
            evidence_logits_list.append(evidence_logits)
            evidence_probs_list.append(evidence_probs)

        return (
            torch.stack(z_local_list, dim=0),
            evidence_logits_list,
            evidence_probs_list,
        )

    def _compute_lexical_trace(
        self,
        x,
        graph_batch,
        graph_list,
        root_node_indices,
        head=None,
        temperature=None,
    ):
        """Predict EDU scores and pool a document lexical representation."""
        head = self.lexical_trace_head if head is None else head
        temperature = (
            self.lexical_attention_temperature
            if temperature is None
            else float(temperature)
        )
        trace_scores = []
        trace_weights = []
        lexical_representations = []
        for item_idx, graph_item in enumerate(graph_list):
            offset = graph_batch.ptr[item_idx].item()
            root_h = x[root_node_indices[item_idx]]
            edu_indices = getattr(graph_item, "_edu_indices", None)
            if not edu_indices:
                trace_scores.append(x.new_empty((0,)))
                trace_weights.append(x.new_empty((0,)))
                lexical_representations.append(torch.zeros_like(root_h))
                continue
            edu_abs_indices = torch.tensor(
                edu_indices, dtype=torch.long, device=x.device
            ) + offset
            edu_h = x[edu_abs_indices]
            root_expand = root_h.unsqueeze(0).expand_as(edu_h)
            trace_input = torch.cat(
                [edu_h, root_expand, edu_h * root_expand], dim=-1
            )
            scores = head(trace_input).squeeze(-1)
            weights = torch.softmax(
                scores / max(temperature, 1e-6), dim=0
            )
            trace_scores.append(scores)
            trace_weights.append(weights)
            lexical_representations.append((weights.unsqueeze(-1) * edu_h).sum(dim=0))
        return (
            trace_scores,
            trace_weights,
            torch.stack(lexical_representations, dim=0),
        )

    def _compute_editor_branch(self, edu_features_per_item):
        editor_reprs = []
        editor_attention = []
        device = self.editor_queries.device

        for edu_features in edu_features_per_item:
            if edu_features.numel() == 0:
                s_i = torch.zeros((1, self.gnn_hidden_dim), device=device)
            else:
                s_i = self.editor_proj(edu_features.to(device))

            scale = math.sqrt(float(self.gnn_hidden_dim))
            attn_scores = torch.matmul(self.editor_queries, s_i.transpose(0, 1)) / scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            pooled = torch.matmul(attn_weights, s_i)
            z_human_editor = pooled[0]
            z_llm_editor = pooled[1]
            parts = [z_human_editor, z_llm_editor, z_human_editor - z_llm_editor]

            if self.editor_use_variation_features:
                if s_i.size(0) > 1:
                    adjacent_cos = F.cosine_similarity(s_i[:-1], s_i[1:], dim=-1)
                    adjacent_delta = 1.0 - adjacent_cos.mean()
                    adjacent_std = adjacent_cos.std(unbiased=False)
                    edu_var = s_i.var(dim=0, unbiased=False).mean()
                    variation = torch.stack([adjacent_delta, adjacent_std, edu_var])
                else:
                    variation = torch.zeros(3, device=device)
                parts.append(self.editor_variation_proj(variation.unsqueeze(0)).squeeze(0))

            editor_reprs.append(torch.cat(parts, dim=-1))
            editor_attention.append(attn_weights)

        return torch.stack(editor_reprs, dim=0), editor_attention

    def _combine_factorized_logits(self, creator_logits, editor_logits, residual_logits=None):
        creator_h = creator_logits[:, 0]
        creator_l = creator_logits[:, 1]
        editor_h = editor_logits[:, 0]
        editor_l = editor_logits[:, 1]
        logits = torch.stack(
            [
                creator_h + editor_h,
                creator_h + editor_l,
                creator_l + editor_l,
                creator_l + editor_h,
            ],
            dim=-1,
        )
        if residual_logits is not None and self.factorized_residual_gamma != 0.0:
            logits = logits + self.factorized_residual_gamma * residual_logits
        return logits

    def forward(self, batch: dict, batch_idx: int = -1, epoch: int | None = None):
        """
        Forward pass of the RACE model.

        Args:
            batch (dict): A batch dictionary from the DataLoader containing:
                - 'full_text_input_ids': Token IDs for the full document text.
                - 'full_text_attention_mask': Attention mask for the full document text.
                - 'edu_token_spans': List of (start, end) token spans for each EDU.
                - 'nodes': List of node info dicts for each item.
                - 'parent_child_edges': List of (parent_id, child_id) edges for each item.
                - 'root_node_ids': List of root node IDs for each item.
                - 'descendant_map': Map from relation node ID to its descendant EDU IDs.
            batch_idx (int): Index of the current batch (used for debugging).

        Returns:
            If output_features is True: dict with 'features' and 'logits'.
            Otherwise: logits tensor of shape [batch_size, num_classes].
        """
        device = self.node_proj[0].weight.device

        # --- Part 1: Initial Feature and Graph Preparation ---
        # Get contextualized token embeddings from the backbone model.
        input_ids = batch["full_text_input_ids"].to(device)
        attention_mask = batch["full_text_attention_mask"].to(device)
        bert_outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = bert_outputs.last_hidden_state
        batch_size = last_hidden_state.size(0)

        # Build a pre-homogenized `Data` object for each item in the batch.
        graph_list = [
            self.graph_builder.build(
                {
                    "nodes": batch["nodes"][i],
                    "parent_child_edges": batch["parent_child_edges"][i],
                    "root_node_ids": batch["root_node_ids"][i],
                },
                max_edu_nodes=len(batch["edu_token_spans"][i]),
            )
            for i in range(batch_size)
        ]

        graph_batch = Batch.from_data_list(graph_list).to(device)

        # Calculate initial EDU features by averaging token embeddings for each span.
        edu_features_per_item = [
            (
                torch.stack(
                    [
                        (
                            last_hidden_state[i][start:end].mean(dim=0)
                            if start < end
                            else torch.zeros(self.bert_dim, device=device)
                        )
                        for start, end in batch["edu_token_spans"][i]
                    ]
                )
                if batch["edu_token_spans"][i]
                else torch.empty((0, self.bert_dim), device=device)
            )
            for i in range(batch_size)
        ]

        # ==============================================================================
        # Part 2: GNN Input Preparation
        # ==============================================================================

        # 1. Create a placeholder for the unified feature tensor.
        unified_features = torch.zeros(
            (graph_batch.num_nodes, self.bert_dim), device=device
        )
        root_node_indices = []

        # 2. Loop through each item to calculate its features and place them in the unified tensor.
        for i in range(batch_size):
            graph_item = graph_list[i]
            edu_feats = edu_features_per_item[i]

            # Calculate initial features for relation nodes (average of descendant EDUs).
            num_rels = len(graph_item._relation_indices)
            relation_features = torch.zeros((num_rels, self.bert_dim), device=device)
            if num_rels > 0:
                desc_map = batch["descendant_map"][i]
                rel_nodes = [
                    n for n in batch["nodes"][i] if n["type"] == "relation_node"
                ]
                for rel_idx, rel_node_info in enumerate(rel_nodes):
                    desc_edu_indices = [
                        graph_item._edu_mapping[str(eid)]
                        for eid in desc_map.get(rel_node_info["id"], [])
                        if str(eid) in graph_item._edu_mapping
                    ]
                    if desc_edu_indices and edu_feats.numel() > 0:
                        relation_features[rel_idx] = edu_feats[desc_edu_indices].mean(
                            dim=0
                        )

            # 3. Use `ptr` (batch offsets) to find the correct slice in the batch tensor and fill features.
            offset = graph_batch.ptr[i].item()
            # Place EDU features and add type embedding 0.
            edu_indices = torch.tensor(graph_item._edu_indices, device=device) + offset
            unified_features[edu_indices] = edu_feats + self.node_type_embedding(
                torch.zeros(len(edu_indices), dtype=torch.long, device=device)
            )
            # Place relation features and add type embedding 1.
            if num_rels > 0:
                rel_indices = (
                    torch.tensor(graph_item._relation_indices, device=device) + offset
                )
                unified_features[rel_indices] = (
                    relation_features
                    + self.node_type_embedding(
                        torch.ones(len(rel_indices), dtype=torch.long, device=device)
                    )
                )

            # Record the absolute index of the root node in the batch.
            if graph_item._root_idx != -1:
                root_node_indices.append(graph_item._root_idx + offset)

        # 4. Project features and get graph structure directly from the `Data` batch object.
        x = self.node_proj(unified_features)
        edge_index, edge_type, batch_vector = (
            graph_batch.edge_index,
            graph_batch.edge_type,
            graph_batch.batch,
        )

        # ==============================================================================
        # Part 3: GNN Propagation, Pooling & Classification
        # ==============================================================================
        # 1. Run the GNN layers.
        gnn_layer_type = self.config.get("gnn_layer_type", "RGCN")
        for i, conv in enumerate(self.convs):
            if gnn_layer_type == "GCN":
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.elu(x)  # Apply activation between layers.

        # 2. Pool node features to get a single vector for each graph.
        pooling_strategy = self.config.get("pooling_strategy", "root")
        if pooling_strategy == "root" and root_node_indices:
            # Use precise root indices if available.
            pooled_features = x[root_node_indices]
        else:
            # Fallback to global mean pooling over all nodes in each graph.
            pooled_features = global_mean_pool(x, batch_vector)

        # Safety check for empty graphs in a batch.
        if pooled_features.size(0) != batch_size:
            raise ValueError(
                f"Pooled features size {pooled_features.size(0)} does not match batch size {batch_size}."
            )

        features = pooled_features
        local_outputs = {}
        if self.use_local_evidence:
            in_warmup = epoch is not None and epoch < self.evidence_warmup_epochs
            z_local, evidence_logits, evidence_probs = self._compute_local_evidence(
                x, graph_batch, graph_list, root_node_indices
            )
            if not in_warmup and not self.disable_evidence_pooling:
                fusion_input = torch.cat(
                    [pooled_features, z_local, pooled_features * z_local], dim=-1
                )
                features = self.fusion_proj(fusion_input)
            local_outputs = {
                "root_features": pooled_features,
                "z_local": z_local,
                "evidence_logits": evidence_logits,
                "evidence_probs": evidence_probs,
                "in_warmup": in_warmup,
            }

        lexical_outputs = {}
        if self.use_lexical_trace:
            lexical_scores, lexical_weights, z_lexical = self._compute_lexical_trace(
                x, graph_batch, graph_list, root_node_indices
            )
            lexical_in_warmup = (
                epoch is not None and epoch < self.lexical_fusion_warmup_epochs
            )
            if self.use_lexical_fusion and not lexical_in_warmup:
                lexical_fusion_input = torch.cat(
                    [pooled_features, z_lexical, pooled_features * z_lexical], dim=-1
                )
                lexical_fused = self.lexical_fusion_proj(lexical_fusion_input)
                if self.lexical_fusion_mode == "residual":
                    features = (
                        pooled_features
                        + self.lexical_residual_gamma * lexical_fused
                    )
                else:
                    features = lexical_fused
            lexical_outputs = {
                "lexical_trace_scores": lexical_scores,
                "lexical_trace_weights": lexical_weights,
                "z_lexical": z_lexical,
                "lexical_in_warmup": lexical_in_warmup,
                "lexical_residual_gamma": self.lexical_residual_gamma,
            }

        humanization_outputs = {}
        if self.use_humanization_trace:
            humanization_scores, humanization_weights, z_humanization = (
                self._compute_lexical_trace(
                    x,
                    graph_batch,
                    graph_list,
                    root_node_indices,
                    head=self.humanization_trace_head,
                    temperature=self.humanization_attention_temperature,
                )
            )
            humanization_in_warmup = (
                epoch is not None and epoch < self.lexical_fusion_warmup_epochs
            )
            if self.use_humanization_fusion and not humanization_in_warmup:
                humanization_fusion_input = torch.cat(
                    [
                        pooled_features,
                        z_humanization,
                        pooled_features * z_humanization,
                    ],
                    dim=-1,
                )
                humanization_fused = self.humanization_fusion_proj(
                    humanization_fusion_input
                )
                features = (
                    features
                    + self.humanization_residual_gamma * humanization_fused
                )
            humanization_outputs = {
                "humanization_trace_scores": humanization_scores,
                "humanization_trace_weights": humanization_weights,
                "z_humanization": z_humanization,
                "humanization_in_warmup": humanization_in_warmup,
                "humanization_residual_gamma": self.humanization_residual_gamma,
            }

        # 3. Final classification.
        factorized_outputs = {}
        if self.use_factorized_ce:
            creator_logits = self.creator_head(pooled_features)
            editor_features, editor_attention = self._compute_editor_branch(
                edu_features_per_item
            )
            editor_logits = self.editor_head(editor_features)
            residual_logits = self.classifier(features)
            logits = self._combine_factorized_logits(
                creator_logits, editor_logits, residual_logits
            )
            features = pooled_features
            factorized_outputs = {
                "creator_logits": creator_logits,
                "editor_logits": editor_logits,
                "editor_features": editor_features,
                "editor_attention": editor_attention,
                "residual_logits": residual_logits,
            }
        else:
            logits = self.classifier(features)

        contrast_outputs = {}
        if self.use_factor_contrast:
            contrast_source = pooled_features
            contrast_outputs = {
                "z_y": self.proj_y(contrast_source),
                "z_c": self.proj_c(contrast_source),
                "z_e": self.proj_e(contrast_source),
            }

        if (
            self.output_features
            or self.use_local_evidence
            or self.use_factorized_ce
            or self.use_factor_contrast
            or self.use_lexical_trace
            or self.use_humanization_trace
        ):
            output = {"features": features, "logits": logits}
            output.update(local_outputs)
            output.update(factorized_outputs)
            output.update(contrast_outputs)
            output.update(lexical_outputs)
            output.update(humanization_outputs)
            return output
        else:
            return logits


class FlexibleBaselineModel(nn.Module):
    """
    Baseline model for ablation study: only includes the backbone model and a classifier.
    No GNN components or graph building. Uses the [CLS] token representation directly.
    """

    def __init__(
        self,
        feature_dim: int,
        gnn_hidden_dim: int,
        num_heads: int,
        num_classes: int,
        config: dict,
        metadata: dict = None,
        output_features: bool = False,
    ):
        super().__init__()
        self.config = config
        self.output_features = output_features
        self.bert_dim = config.get("bert_dim", 768)

        # 1. Backbone Model
        self.bert_model = AutoModel.from_pretrained(config["backbone_model_path"])
        print(f"Loading baseline backbone model from {config['backbone_model_path']}")

        # Freezing logic (identical to RACEModel)
        unfreeze_layers = config.get("unfreeze_bert_layers", 0)
        if unfreeze_layers > 0 and hasattr(self.bert_model, "encoder"):
            for param in self.bert_model.parameters():
                param.requires_grad = False
            for layer in self.bert_model.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif unfreeze_layers == -1:
            self.bert_model.requires_grad_(True)
        else:
            self.bert_model.requires_grad_(False)

        # 2. Classifier
        classifier_type = self.config.get("classifier_type", "linear")
        if classifier_type == "mlp":
            self.classifier = ClassificationHead(
                input_dim=self.bert_dim,
                hidden_dim=self.bert_dim,
                num_classes=num_classes,
                dropout_prob=self.config.get("classifier_dropout", 0.1),
            )
        else:
            self.classifier = nn.Linear(self.bert_dim, num_classes)

        # 3. Weight Initialization
        self._init_classifier(self.classifier)

    def _init_classifier(self, module):
        """Initializes classifier weights with small normal distribution for stable initial predictions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "children"):
            for child in module.children():
                self._init_classifier(child)

    def _compute_local_evidence(self, x, graph_batch, graph_list, root_node_indices):
        z_local_list = []
        evidence_logits_list = []
        evidence_probs_list = []

        for item_idx, graph_item in enumerate(graph_list):
            offset = graph_batch.ptr[item_idx].item()
            root_h = x[root_node_indices[item_idx]]
            if not getattr(graph_item, "_edu_indices", None):
                z_local_list.append(torch.zeros_like(root_h))
                evidence_logits_list.append(x.new_empty((0,)))
                evidence_probs_list.append(x.new_empty((0,)))
                continue

            edu_abs_indices = torch.tensor(
                graph_item._edu_indices, dtype=torch.long, device=x.device
            ) + offset
            edu_h = x[edu_abs_indices]
            if edu_h.size(0) == 0:
                z_local_list.append(torch.zeros_like(root_h))
                evidence_logits_list.append(x.new_empty((0,)))
                evidence_probs_list.append(x.new_empty((0,)))
                continue

            root_expand = root_h.unsqueeze(0).expand_as(edu_h)
            score_input = torch.cat([edu_h, root_expand, edu_h * root_expand], dim=-1)
            evidence_logits = self.trace_scorer(score_input).squeeze(-1)
            evidence_probs = torch.sigmoid(evidence_logits)

            if self.evidence_pooling == "topk":
                k = max(1, math.ceil(self.evidence_topk_ratio * edu_h.size(0)))
                top_idx = evidence_probs.topk(k=min(k, edu_h.size(0))).indices
                z_local = edu_h[top_idx].mean(dim=0)
            else:
                weights = evidence_probs / evidence_probs.sum().clamp_min(1e-8)
                z_local = (weights.unsqueeze(-1) * edu_h).sum(dim=0)

            z_local_list.append(z_local)
            evidence_logits_list.append(evidence_logits)
            evidence_probs_list.append(evidence_probs)

        return (
            torch.stack(z_local_list, dim=0),
            evidence_logits_list,
            evidence_probs_list,
        )

    def forward(self, batch: dict, batch_idx: int = -1, epoch: int | None = None):
        """
        Forward pass of the baseline model.
        Uses only the [CLS] token from the backbone for classification, without any GNN.

        Args:
            batch (dict): A batch dictionary from the DataLoader.
            batch_idx (int): Index of the current batch.

        Returns:
            If output_features is True: dict with 'features' and 'logits'.
            Otherwise: logits tensor of shape [batch_size, num_classes].
        """
        # Determine device from the classifier layer.
        device = next(self.classifier.parameters()).device

        input_ids = batch["full_text_input_ids"].to(device)
        attention_mask = batch["full_text_attention_mask"].to(device)

        bert_outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Standard BERT pooling: take the [CLS] token (index 0)
        pooled_output = bert_outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)

        if self.output_features:
            return {"features": pooled_output, "logits": logits}
        else:
            return logits
