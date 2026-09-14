# Implementation Guide - LE-RACE
> Status: PENDING_REVIEW
> 生成时间：2026-07-06 | 策略：强 baseline 改造 | 状态：PENDING_REVIEW  
> 关联实验设计：`docs/idea_report.md` Part 3

## AI-Humanization Lexical Extension

`utils/build_humanization_trace_data.py` filters complete same-group
`ai_generated`/`ai_humanized` pairs, assigns them with the existing PASTED
split manifest, gives the generated reference zero EDU targets, and reuses the
semantic sentence alignment plus `1-BLEU4` projection for Humanized targets.
It must assert exactly one reference and target per retained group and zero
group overlap across output splits.

`FlexibleGraphDataset` accepts an optional configuration-level `label_map`, so
the lexical-only trainer can map `ai_generated -> 0` and `ai_humanized -> 1`
without relabeling stored examples. The new config writes to a distinct output
directory and uses the existing `train_pasted_race.py`; the original
Human-to-Polished defaults remain unchanged.

---

## 1 原始项目信息

### 1.1 Baseline Project

本实现基于用户本地 RACE 开源项目改造：

```text
C:/Users/18785/Downloads/RACE-main/RACE-main
```

原始 RACE 项目已经包含：

- HART 数据解析与 flatten pipeline。
- RST parsing pipeline。
- RST graph construction。
- RoBERTa + RGCN + root pooling 模型。
- SupCon + CE 训练。
- AUROC、TPR@1%FPR、F1 等指标。

> 实现策略是不重写整个项目，而是在原有代码上做可开关式增强。默认配置仍可运行原始 RACE；打开 LE-RACE 配置后才启用 EDU-level local evidence branch。

### 1.2 Implementation Goal

阶段 D 的目标是实现 Part 3 中的实验设计，核心包括：

1. RACE 复现与 metric sanity check。
2. LE-RACE 主模型。
3. EDU-level weak evidence label 生成。
4. evidence loss、MIL loss、sparsity loss。
5. LE-RACE 消融配置。
6. group-aware split。
7. weak label 与 evidence score 诊断。

第一版不实现完整 LEAR-RACE。LEAR-RACE 只在 LE-RACE 主实验有效后作为第二阶段增强。

---

## 2 改写范围总览

### 2.1 修改后目录树

在原始 RACE 项目中建议形成如下结构：

```text
RACE-main/
  configs/
    RACE.json
    LE_RACE.json
    ablations/
      LE_RACE_no_evidence_loss.json
      LE_RACE_no_mil.json
      LE_RACE_no_evidence_pooling.json
      LE_RACE_no_sparse.json
      LE_RACE_mil_only.json
      LE_RACE_topk_pooling.json
  models/
    flexible_model.py
    modules.py
    supcon_loss.py
  scripts/
    train_single.sh
    eval_single.sh
    run_multiple_seeds.sh
    generate_edu_evidence_labels.sh
    split_group_aware.sh
    run_lerace_ablation.sh
  utils/
    flexible_dataset.py
    flexible_graph_builder.py
    metrics.py
    edu_utils.py
    evidence_label_builder.py
    split_dataset_group_aware.py
    diagnostics.py
    relation_table.txt
  train.py
  test.py
  requirements.txt
```

> 新增文件集中放在 `utils/`、`configs/ablations/` 和 `scripts/`。原始模型文件只做最小侵入式增强。

### 2.2 文件功能表

| 文件 | 操作 | 功能 | 被谁调用 |
|---|---|---|---|
| `configs/LE_RACE.json` | 新增 | LE-RACE 主实验配置 | `train.py` |
| `configs/ablations/*.json` | 新增 | 各消融实验配置 | `train.py` |
| `models/flexible_model.py` | 修改 | 在 RACEModel 内增加 local evidence scorer、pooling、fusion | `train.py`, `test.py` |
| `utils/flexible_dataset.py` | 修改 | 加载 EDU weak labels/masks，维护 valid EDU 对齐 | `train.py`, `test.py` |
| `utils/edu_utils.py` | 新增 | RST EDU 遍历、span 映射、token span 计算等共享工具 | dataset 与 label builder |
| `utils/evidence_label_builder.py` | 新增 | 离线生成 EDU-level weak evidence labels | script |
| `utils/split_dataset_group_aware.py` | 新增 | 生成 group-aware train/val/test split | script |
| `utils/metrics.py` | 修改 | 修复 double-softmax 风险，补充 mixed-class TPR 与 checkpoint metric | `train.py`, `test.py` |
| `utils/diagnostics.py` | 新增 | 输出 weak label 统计、evidence score 分布、top EDU 诊断 | train/eval 后处理 |
| `train.py` | 修改 | 支持 LE-RACE loss、warm-up、checkpoint selection、诊断输出 | main training |
| `test.py` | 修改 | 保存 evidence probs、top evidence EDU、分层指标 | evaluation |
| `requirements.txt` | 修改 | 可加 `spacy`，不得加入 torch/torchvision/torchaudio | install |

---

## 3 数据流设计

### 3.1 原始 RACE 数据流

```text
HART raw json
  -> data_unification_pipeline.py
  -> rst_parser_pipeline.py
  -> flatten_data_pipeline.py
  -> split_dataset.py
  -> train_graph.jsonl / val_graph.jsonl / test_graph.jsonl
  -> FlexibleGraphDataset
  -> RACEModel
```

### 3.2 LE-RACE 数据流

LE-RACE 在原始数据流中插入 weak evidence label 生成步骤：

```text
train_graph.jsonl / val_graph.jsonl / test_graph.jsonl
  -> evidence_label_builder.py
  -> train_graph_evidence.jsonl / val_graph_evidence.jsonl / test_graph_evidence.jsonl
  -> FlexibleGraphDataset
  -> RACEModel(use_local_evidence=true)
  -> CE + SupCon + evidence BCE + MIL + sparse
```

### 3.3 新增样本字段

每个 JSONL item 需新增以下字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `edu_evidence_labels` | `list[int]` | 每个 EDU 的 weak evidence label，0/1 |
| `edu_evidence_masks` | `list[int]` | 每个 EDU 是否参与 evidence BCE，0/1 |
| `has_trace_ref` | `bool` | 是否有可靠 reference 可生成 BCE 标签 |
| `evidence_ref_id` | `str/null` | reference 样本 ID |
| `evidence_ref_label` | `str/null` | reference 类别 |
| `evidence_positive_ratio` | `float` | 该样本有效 EDU 中正标签比例，诊断用 |
| `evidence_masked_ratio` | `float` | 该样本 EDU 被 mask 比例，诊断用 |

> `n_valid_edu` 不建议离线固定写死为唯一真值，因为它依赖 tokenizer、`max_seq_length` 和 backbone。实现中应在 `FlexibleGraphDataset._process_contextual_rst_item()` 内按当前 tokenizer 实时计算，并用它截断 evidence labels/masks。

### 3.4 EDU Label 对齐规则

原始 RST 可能产生 `N_full` 个 EDU，但 RoBERTa 512 token 截断后只有 `N_valid = len(edu_token_spans)` 个 EDU 有 token span。模型只会保留前 `N_valid` 个 EDU 节点。

因此 dataset 必须执行：

```python
labels = labels[:n_valid_edu]
masks = masks[:n_valid_edu]
```

并检查：

```python
assert len(labels) == len(edu_token_spans)
assert len(masks) == len(edu_token_spans)
```

如果原始样本没有 weak label 字段：

- pure RACE 配置下允许缺失；
- LE-RACE 配置下应填充全 0 labels 和全 0 masks，并记录 warning；
- 对 `LLM-Polished` / `Humanized`，即使 BCE mask 全 0，MIL 仍可计算。

---

## 4 文件级实现说明

### 4.1 `utils/edu_utils.py`

**文件职责**：提供 RST EDU 遍历、句子/EDU span overlap、token span 映射等共享函数，避免 dataset 与 weak label 脚本各写一套逻辑。

#### `traverse_rst_edus(rst_node: dict) -> list[dict]`

**参数**：

- `rst_node`：单个样本的 `rst_structure`。

**返回**：

返回按 RST traversal 顺序排列的 EDU 列表，每个元素包含：

```python
{
    "id": int,
    "text": str,
    "start": int,
    "end": int,
    "nuclearity": str | None,
    "relation": str | None,
}
```

**实现逻辑**：

1. 递归遍历 RST tree。
2. 判断 `relation == "elementary"` 的节点为 EDU。
3. 保留原始 `id/start/end/text`。
4. 输出顺序必须与 `FlexibleGraphDataset._rst_traversal()` 中 EDU 节点顺序一致。

> 这是 LE-RACE 最重要的对齐函数之一。若这里的 EDU 顺序与 dataset 中节点顺序不同，weak label 会错位。

#### `map_char_spans_to_token_spans(char_spans: list[tuple[int, int]], offset_mapping: list[tuple[int, int]]) -> list[list[int]]`

**参数**：

- `char_spans`：EDU char span。
- `offset_mapping`：tokenizer 输出的 offset mapping。

**返回**：

返回有效 EDU token spans，格式与现有 dataset 保持一致：

```python
[[token_start, token_end_exclusive], ...]
```

**实现逻辑**：

1. 跳过 `(0, 0)` special tokens。
2. 对每个 EDU 找到与 char span 有 overlap 的第一个和最后一个 token。
3. 若找不到 token，说明该 EDU 被截断或 span 异常，不加入结果。

> 此函数应复用现有 `FlexibleGraphDataset._process_contextual_rst_item()` 里的 char-to-token 逻辑，保证 `n_valid_edu` 计算一致。

#### `overlap_len(span_a: tuple[int, int], span_b: tuple[int, int]) -> int`

**返回**：

两个 char span 的重叠长度。

**实现逻辑**：

```python
max(0, min(a_end, b_end) - max(a_start, b_start))
```

#### `assign_sentence_label_to_edu(edu_span: tuple[int, int], sent_spans: list[tuple[int, int]], sent_labels: list[int], sent_masks: list[int], min_overlap_gap: int = 5) -> tuple[int, int]`

**返回**：

`(edu_label, edu_mask)`。

**实现逻辑**：

1. 计算 EDU 与所有 sentence spans 的 overlap。
2. 若最大 overlap 为 0，返回 `(0, 0)`。
3. 若 EDU 跨句，且最大 overlap 与第二大 overlap 差距小于 `min_overlap_gap`，返回 `(0, 0)`。
4. 否则继承 overlap 最大句子的 label/mask。

> 该规则用于处理 EDU 跨句边界，防止不确定 span 被硬标注。

---

### 4.2 `utils/evidence_label_builder.py`

**文件职责**：离线生成 EDU-level weak evidence labels。

#### CLI 参数

建议使用 argparse：

```text
--input_train data/hart_split/train_graph.jsonl
--input_val data/hart_split/val_graph.jsonl
--input_test data/hart_split/test_graph.jsonl
--output_dir data/hart_split_evidence
--backbone_model_path Facebook-AI/roberta-base
--max_seq_length 512
--semantic_sim_threshold 0.85
--edit_ratio_threshold 0.25
--min_sent_tokens 5
--spacy_model en_core_web_sm
```

#### `load_jsonl(path: str) -> list[dict]`

读取 JSONL，返回 item list。遇到坏行应跳过并打印 warning。

#### `write_jsonl(items: list[dict], path: str) -> None`

写出 JSONL，保留原字段并追加 weak evidence 字段。

#### `build_group_index(all_items: list[dict]) -> dict[str, dict[str, list[dict]]]`

**参数**：

- `all_items = train + val + test`。

**返回**：

```python
{
  group_id: {
    "human_written": [item, ...],
    "human_ai_polished": [item, ...],
    "ai_generated": [item, ...],
    "ai_humanized": [item, ...],
  }
}
```

**实现逻辑**：

1. 用全量 train/val/test 建 index。
2. 不按 split 限制 reference 查找。
3. reference 文本只用于离线标注，不作为模型输入，因此不视为训练泄漏。

> 必须全量建 reference index，否则默认 random split 可能让 target 和 reference 分在不同 partition，导致 train 中 mixed class 大量找不到 reference。

#### `find_reference(item: dict, group_index: dict) -> dict | None`

**规则**：

- `human_ai_polished`：优先找同 `group_id` 的 `human_written`。
- `ai_humanized`：优先找同 `group_id` 且 `model_name` 一致或来源一致的 `ai_generated`。
- 如果同源模型无法确认，则在同 group 的 `ai_generated` 候选中选择 document-level semantic similarity 最高者。
- `human_written` 与 `ai_generated` 不需要 reference，作为 pure class negative evidence。

**长文本 similarity**：

对于长文档，不得只取 RoBERTa 前 512 tokens。实现 `embed_long_text()` 分段 mean pooling：

```python
def embed_long_text(text, model, tokenizer, max_len=512, stride=256) -> Tensor
```

返回多个 chunk embedding 的平均值，仅用于离线 reference selection。

#### `sentence_split(text: str, nlp) -> list[dict]`

**返回**：

```python
[
  {"text": str, "start": int, "end": int, "tokens": list[str]},
  ...
]
```

**实现要求**：

- 使用 spaCy `en_core_web_sm`。
- tokenization、LCS tokenization 都使用同一个 spaCy pipeline。
- 对 token 数小于 `min_sent_tokens=5` 的句子，后续 label 设为 `(0, 0)`。

#### `compute_sentence_alignment(ref_sents: list[dict], tgt_sents: list[dict]) -> list[tuple[int, int, float]]`

**返回**：

target sentence 到 reference sentence 的对齐：

```python
[(tgt_idx, ref_idx, semantic_sim), ...]
```

**实现逻辑**：

1. 计算每个 ref/tgt sentence embedding。
2. 对每个 target sentence 选择 semantic similarity 最高的 reference sentence。
3. 若 similarity 低于阈值，标为低置信。

#### `token_lcs_edit_ratio(ref_text: str, tgt_text: str, nlp) -> float`

**定义**：

使用 spaCy tokenizer，小写、去标点后计算 LCS：

```text
edit_ratio = 1 - LCS(ref_tokens, tgt_tokens) / max(len(ref_tokens), len(tgt_tokens))
```

**解释**：

- `edit_ratio` 越高，表层改动越大。
- `semantic_sim` 高且 `edit_ratio` 高，说明语义保持但表达变化明显，更像 editing-sensitive evidence。

#### `label_target_sentences(ref_item: dict | None, tgt_item: dict, config: dict) -> tuple[list[int], list[int], dict]`

**返回**：

`sent_labels, sent_masks, stats`。

**规则**：

- 若 target 是 pure class：
  - `sent_labels = 0`
  - `sent_masks = 1`
  - `has_trace_ref = True`
- 若 target 是 mixed class 且 `ref_item is None`：
  - `sent_labels = 0`
  - `sent_masks = 0`
  - `has_trace_ref = False`
  - MIL 后续仍可计算。
- 若 target 是 mixed class 且有 reference：
  - 若 `semantic_sim >= 0.85` 且 `edit_ratio >= edit_ratio_threshold`，标 `(1, 1)`。
  - 若 `semantic_sim >= 0.90` 且 `edit_ratio <= 0.10`，标 `(0, 1)`。
  - 其他情况标 `(0, 0)`。

> 注意：早先草案中出现过 `semantic_sim >= 0.85 且 edit_distance <= 0.10` 的表述，那更像“高置信没改写”。对于 positive editing evidence，应该是语义相似且表层差异较大，即 `edit_ratio` 高于阈值。

#### `project_sentence_labels_to_edus(item: dict, sent_labels: list[int], sent_masks: list[int], nlp) -> tuple[list[int], list[int]]`

**实现逻辑**：

1. 用 `traverse_rst_edus()` 得到 EDU char spans。
2. 用 `assign_sentence_label_to_edu()` 继承句子标签。
3. 输出长度等于全文 EDU 数的 `edu_evidence_labels` 和 `edu_evidence_masks`。

#### `build_evidence_labels_for_split(split_items: list[dict], group_index: dict, config: dict) -> tuple[list[dict], dict]`

**实现逻辑**：

1. 遍历 split items。
2. 找 reference。
3. 生成 sentence labels。
4. 投影到 EDU labels。
5. 写入 item 新字段。
6. 汇总 coverage、positive_ratio、masked_ratio。

#### 输出文件

```text
data/hart_split_evidence/
  train_graph.jsonl
  val_graph.jsonl
  test_graph.jsonl
  evidence_label_stats.json
```

---

### 4.3 `utils/flexible_dataset.py`

**文件职责变化**：在原有 RST graph dataset 基础上，加载 weak evidence labels/masks，并保证与 valid EDU 对齐。

#### 修改 `_process_contextual_rst_item(self, item)`

新增逻辑位置：已有 `edu_token_spans` 计算完成后。

**新增变量**：

```python
n_valid_edu = len(edu_token_spans)
raw_labels = item.get("edu_evidence_labels")
raw_masks = item.get("edu_evidence_masks")
```

**实现逻辑**：

1. 若 `raw_labels/raw_masks` 存在：
   - 转为 list[int]。
   - 截断到 `n_valid_edu`。
   - 若短于 `n_valid_edu`，尾部补 0 mask。
2. 若不存在：
   - labels 全 0。
   - masks 全 0。
3. pure class 如果无 weak label，可在 label builder 端解决；dataset 不强行推断。
4. 在返回 dict 中加入：

```python
"n_valid_edu": n_valid_edu,
"edu_evidence_labels": torch.tensor(labels, dtype=torch.float),
"edu_evidence_masks": torch.tensor(masks, dtype=torch.float),
"has_trace_ref": bool(item.get("has_trace_ref", False)),
"evidence_ref_id": item.get("evidence_ref_id"),
```

#### 修改 `collate_fn(batch: list) -> dict`

新增 collate 字段：

```python
edu_evidence_labels: list[Tensor[n_i]]
edu_evidence_masks: list[Tensor[n_i]]
has_trace_ref: list[bool]
n_valid_edu: Tensor[B]
```

不 padding labels/masks，保持 ragged list，原因是每篇文档 EDU 数不同，loss 可逐样本计算。

新增 `edu_node_slices`：

```python
edu_node_slices = []
offset = 0
for item in batch:
    n_edu = item["n_valid_edu"]
    edu_node_slices.append((offset, offset + n_edu))
    offset += item["num_total_nodes"]  # 如果 dataset 可得到；否则由 model forward 内部从 graph_list 计算
```

实际建议：`edu_node_slices` 在 model forward 内基于 `graph_batch.ptr` 和 `graph_item._edu_indices` 计算更可靠。dataset 只需提供 `n_valid_edu` 与 labels/masks。

> 设计理由：RACE 的 graph builder 会把 EDU nodes 放在 relation nodes 前面，但 `num_total_nodes` 是图构建后才最可靠。为了避免 dataset 和 graph builder 重复维护节点数量，最终 absolute EDU indices 应在 `RACEModel.forward()` 内计算。

---

### 4.4 `models/flexible_model.py`

**文件职责变化**：在 `RACEModel` 中加入可选 LE-RACE 分支。配置关闭时行为必须与原 RACE 一致。

#### 新增配置读取

在 `RACEModel.__init__()` 中读取：

```python
self.use_local_evidence = config.get("use_local_evidence", False)
self.evidence_pooling = config.get("evidence_pooling", "soft")
self.evidence_topk_ratio = config.get("evidence_topk_ratio", 0.15)
self.evidence_warmup_epochs = config.get("evidence_warmup_epochs", 2)
self.disable_evidence_pooling = config.get("disable_evidence_pooling", False)
```

#### 新增模块

仅当 `use_local_evidence=True` 时初始化：

```python
self.trace_scorer = nn.Sequential(
    nn.Linear(3 * self.gnn_hidden_dim, self.gnn_hidden_dim),
    nn.GELU(),
    nn.Dropout(config.get("evidence_dropout", 0.1)),
    nn.Linear(self.gnn_hidden_dim, 1),
)

self.fusion_proj = nn.Sequential(
    nn.Linear(3 * self.gnn_hidden_dim, self.gnn_hidden_dim),
    nn.LayerNorm(self.gnn_hidden_dim),
    nn.GELU(),
    nn.Dropout(config.get("fusion_dropout", 0.1)),
)
```

> `trace_scorer` 的输入是 `[h_i; h_root; h_i * h_root]`，所以维度为 `3 * gnn_hidden_dim`。`fusion_proj` 的输入是 `[h_root; z_local; h_root * z_local]`，同样是 `3 * gnn_hidden_dim`。

#### 修改 `forward(self, batch: dict, batch_idx: int = -1, epoch: int | None = None)`

原签名为：

```python
def forward(self, batch: dict, batch_idx: int = -1):
```

修改为：

```python
def forward(self, batch: dict, batch_idx: int = -1, epoch: int | None = None):
```

训练时 `train_one_epoch()` 传入当前 epoch。评估时 `epoch=None`，默认启用完整 LE-RACE 路径。

#### 保留原始 RACE 路径

如果：

```python
not self.use_local_evidence
```

则完全执行原始逻辑：

```python
pooled_features = x[root_node_indices]
logits = self.classifier(pooled_features)
```

返回格式保持兼容。

#### 新增 `_compute_local_evidence(...)`

建议在类内新增私有方法：

```python
def _compute_local_evidence(
    self,
    x: torch.Tensor,
    graph_batch,
    graph_list: list,
    root_node_indices: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
```

**参数**：

- `x`：RGCN 后所有节点表示，shape `[total_nodes, gnn_hidden_dim]`。
- `graph_batch`：PyG Batch，含 `ptr`。
- `graph_list`：单样本图列表，含 `_edu_indices`。
- `root_node_indices`：batch 内每个样本 root node 的 absolute index。

**返回**：

- `z_local_batch`：shape `[B, gnn_hidden_dim]`。
- `evidence_logits_list`：list of Tensor，每个 shape `[n_edu_i]`。
- `evidence_probs_list`：list of Tensor，每个 shape `[n_edu_i]`。

**实现逻辑**：

对 batch 内每个样本：

1. `offset = graph_batch.ptr[i].item()`。
2. `edu_abs_indices = tensor(graph_item._edu_indices) + offset`。
3. `edu_h = x[edu_abs_indices]`。
4. `root_h = x[root_node_indices[i]]`。
5. `root_expand = root_h.unsqueeze(0).expand_as(edu_h)`。
6. `score_input = cat([edu_h, root_expand, edu_h * root_expand], dim=-1)`。
7. `logits_i = trace_scorer(score_input).squeeze(-1)`。
8. `probs_i = sigmoid(logits_i)`。
9. 根据 pooling 策略计算 `z_local_i`。

空 EDU 保护：

若 `edu_h.size(0) == 0`，则：

```python
z_local_i = torch.zeros_like(root_h)
logits_i = torch.empty(0, device=x.device)
probs_i = torch.empty(0, device=x.device)
```

#### Evidence pooling 实现

soft pooling：

```python
weights = probs_i / (probs_i.sum() + 1e-8)
z_local_i = (weights.unsqueeze(-1) * edu_h).sum(dim=0)
```

top-k pooling：

```python
k = max(1, math.ceil(self.evidence_topk_ratio * n_edu))
top_idx = probs_i.topk(k).indices
z_local_i = edu_h[top_idx].mean(dim=0)
```

hybrid pooling 可第二版实现，第一版可只预留配置但不启用。

#### Fusion and logits

非 warm-up 阶段：

```python
h_root = x[root_node_indices]
fusion_input = torch.cat([h_root, z_local, h_root * z_local], dim=-1)
features = self.fusion_proj(fusion_input)
logits = self.classifier(features)
```

warm-up 阶段：

```python
features = h_root
logits = self.classifier(h_root)
```

判断 warm-up：

```python
in_warmup = (
    self.use_local_evidence
    and epoch is not None
    and epoch < self.evidence_warmup_epochs
)
```

#### forward 返回格式

当 `output_features=True` 或 `use_local_evidence=True` 时统一返回 dict：

```python
{
    "features": features,
    "root_features": h_root,
    "logits": logits,
    "evidence_logits": evidence_logits_list,
    "evidence_probs": evidence_probs_list,
    "z_local": z_local,
    "in_warmup": in_warmup,
}
```

否则保持原始返回：

```python
logits
```

> 设计理由：训练 loss 需要 evidence logits/probs；评估诊断需要 evidence probs；SupCon 需要 features。统一 dict 可减少 train/evaluate 中的分支混乱。

---

### 4.5 `train.py`

**文件职责变化**：支持 LE-RACE 多 loss、warm-up、checkpoint metric 和诊断输出。

#### 修改模型 forward 调用

原代码：

```python
outputs = model(batch, batch_idx=i)
```

修改为：

```python
outputs = model(batch, batch_idx=i, epoch=epoch)
```

#### 新增 loss 配置读取

从 `args.loss` 读取：

```python
use_supcon = loss_config.get("use_supcon", False)
use_evidence_loss = loss_config.get("use_evidence_loss", False)
use_mil_loss = loss_config.get("use_mil_loss", False)
use_sparse_loss = loss_config.get("use_sparse_loss", False)
evidence_weight = loss_config.get("evidence_weight", 0.1)
mil_weight = loss_config.get("mil_weight", 0.1)
sparse_weight = loss_config.get("sparse_weight", 0.01)
mil_margin = loss_config.get("mil_margin", 0.3)
mil_topk_ratio = loss_config.get("mil_topk_ratio", 0.15)
```

#### 新增 `masked_bce_loss(...)`

```python
def masked_bce_loss(
    evidence_logits: list[torch.Tensor],
    evidence_labels: list[torch.Tensor],
    evidence_masks: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
```

**返回**：

masked BCE scalar。

**实现逻辑**：

1. 遍历 batch 内样本。
2. 将 labels/masks 移到 device。
3. 若 `mask.sum() == 0`，跳过该样本。
4. 使用 `F.binary_cross_entropy_with_logits(logits, labels, reduction="none")`。
5. 乘以 mask 后求和。
6. 除以总有效 mask 数。
7. 若全 batch 无有效 mask，返回 `torch.tensor(0.0, device=device)`。

#### 新增 `mil_loss(...)`

```python
def mil_loss(
    evidence_probs: list[torch.Tensor],
    labels: torch.Tensor,
    mixed_class_ids: set[int],
    margin: float,
    topk_ratio: float,
) -> torch.Tensor:
```

**逻辑**：

仅对 class id in `{1, 3}` 的 LLM-Polished 与 Humanized 计算：

```python
k = max(1, math.ceil(topk_ratio * n_edu))
topk_mean = probs.topk(k).values.mean()
loss_i = F.relu(margin - topk_mean)
```

若 batch 中没有 mixed class，返回 0。

#### 新增 `sparse_loss(...)`

```python
def sparse_loss(evidence_probs: list[torch.Tensor]) -> torch.Tensor:
```

**建议第一版实现**：

对每个样本的 evidence prob 均值求平均：

```python
loss_i = probs.mean()
```

该项鼓励整体 evidence score 不要全高。权重 `rho` 默认很小。

#### 修改 `train_one_epoch(...)`

完整 loss 计算顺序：

1. forward。
2. 取 logits。
3. `L_ce`。
4. 若 use_supcon：
   - `features = F.normalize(outputs["features"], p=2, dim=1)`。
   - `L_supcon = criterion(features, labels)`。
5. 若 LE-RACE 且非 warm-up：
   - `L_evidence`。
   - `L_mil`。
   - `L_sparse`。
6. 总 loss：

```python
loss = L_ce + L_supcon + eta * L_evidence + mu * L_mil + rho * L_sparse
```

兼容原 RACE：

- 如果 `use_local_evidence=False`，训练逻辑应退回原 CE/SupCon。
- 如果某个 ablation 禁用某项 loss，对应权重或 config 开关为 false。

#### 修改 checkpoint selection

新增配置：

```json
"checkpoint_metric": "tpr_at_1_fpr_macro",
"checkpoint_tie_breaker": "auroc_macro"
```

训练中：

```python
current_main = val_stats.get(args.checkpoint_metric, 0.0)
current_tie = val_stats.get(args.checkpoint_tie_breaker, 0.0)
```

保存规则：

1. main metric 更高则保存。
2. main metric 相等或差距小于 `1e-6` 时，tie-breaker 更高则保存。

#### 修改 evaluate 调用

当前 `evaluate()` 里把 probs 传给 `compute_classification_metrics()`。应改为：

```python
all_logits.append(logits.cpu())
...
val_metrics = compute_classification_metrics(all_logits, all_labels, input_type="logits")
```

或者新增更明确函数 `compute_classification_metrics_from_logits()`。

---

### 4.6 `utils/metrics.py`

**文件职责变化**：修复 logits/probs 口径，补充 mixed-class 指标。

#### 修改 `compute_classification_metrics(...)`

建议签名：

```python
def compute_classification_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    input_type: str = "logits",
    num_classes: int | None = None,
) -> dict:
```

**参数**：

- `preds`：logits 或 probs。
- `labels`：class ids。
- `input_type`：`"logits"` 或 `"probs"`。
- `num_classes`：默认从 `preds.shape[1]` 推断。

**实现逻辑**：

```python
if input_type == "logits":
    probs = torch.softmax(preds, dim=1).cpu().numpy()
else:
    probs = preds.cpu().numpy()
```

不得对 probs 再 softmax。

#### 新增 mixed-class 指标

在已有 class-wise TPR 基础上新增：

```python
metrics["tpr_at_1_fpr_mixed_avg"] = mean([
    metrics["tpr_at_1_fpr_class_1"],
    metrics["tpr_at_1_fpr_class_3"],
])
```

同理可加：

```python
metrics["tpr_at_5_fpr_mixed_avg"]
```

> 该指标直接服务用户关注的 LLM-Polished 与 Humanized，不替代主指标，但用于 checkpoint analysis 和结果解释。

---

### 4.7 `utils/diagnostics.py`

**文件职责**：生成 weak label 与 evidence score 诊断文件。

#### `summarize_evidence_labels(jsonl_paths: list[str], output_path: str) -> dict`

**统计**：

- coverage。
- positive_ratio。
- masked_ratio。
- avg_valid_edu。
- avg_full_edu。
- truncation_ratio。
- 按 class 分组。
- 按 domain 分组。

**输出**：

```text
results/diagnostics/evidence_label_stats.json
```

#### `summarize_evidence_predictions(prediction_path: str, output_path: str) -> dict`

**输入**：

test prediction 文件，需包含：

- `id`
- `label`
- `prediction`
- `evidence_probs`
- `top_evidence_edu_texts`

**输出统计**：

- 每类 evidence score mean/std。
- 每类 top-k mean。
- mixed vs pure score gap。

#### `extract_top_evidence_edus(batch: dict, outputs: dict, top_k: int = 5) -> list[dict]`

**返回**：

每个样本 top evidence EDU：

```python
{
  "id": str,
  "label": int,
  "prediction": int,
  "top_edus": [
    {"rank": int, "edu_index": int, "score": float, "text": str}
  ]
}
```

> 诊断输出不参与训练，只用于判断 evidence scorer 是否学到了编辑痕迹。

---

### 4.8 `utils/split_dataset_group_aware.py`

**文件职责**：生成 group-aware split。

#### CLI 参数

```text
--input data/flattened_articles.jsonl
--output_dir data/hart_group_split
--train_ratio 0.70
--val_ratio 0.10
--test_ratio 0.20
--seed 42
```

#### `load_items(path: str) -> list[dict]`

读取 flattened graph jsonl。

#### `group_items_by_group_id(items: list[dict]) -> dict[str, list[dict]]`

同一 group_id 的所有 variants 归为一组。

#### `infer_group_strata(group_items: list[dict]) -> tuple[str, str]`

返回 group-level stratify key，例如：

```python
(domain, has_humanized)
```

若无法精确多标签 stratify，则至少按 domain stratify。

#### `split_groups(...) -> tuple[list[dict], list[dict], list[dict]]`

按 group 划分 train/val/test，再展开为 item list。

#### `write_split_stats(...) -> None`

输出：

```text
data/hart_group_split/stats.md
```

统计每个 split 下 class/domain/model_name 数量。

---

### 4.9 `configs/LE_RACE.json`

基于 `configs/RACE.json` 复制，新增字段：

```json
{
  "data_path": "./data/hart_split_evidence/train_graph.jsonl",
  "val_data_path": "./data/hart_split_evidence/val_graph.jsonl",
  "test_data_path": "./data/hart_split_evidence/test_graph.jsonl",
  "checkpoint_metric": "tpr_at_1_fpr_macro",
  "checkpoint_tie_breaker": "auroc_macro",
  "graph": {
    "use_local_evidence": true,
    "evidence_pooling": "soft",
    "evidence_topk_ratio": 0.15,
    "evidence_warmup_epochs": 2,
    "evidence_dropout": 0.1,
    "fusion_dropout": 0.1,
    "disable_evidence_pooling": false
  },
  "loss": {
    "use_supcon": true,
    "temperature": 0.07,
    "ce_weight": 1.0,
    "use_evidence_loss": true,
    "evidence_weight": 0.1,
    "use_mil_loss": true,
    "mil_weight": 0.1,
    "mil_margin": 0.3,
    "mil_topk_ratio": 0.15,
    "use_sparse_loss": true,
    "sparse_weight": 0.01
  }
}
```

> 注意：当前 `train.py` 会把 `args.graph` 合并进 `graph_config`，因此 `use_local_evidence` 等模型配置放在 `graph` 下最少改动。

### 4.10 消融配置

每个消融只改必要字段。

#### `LE_RACE_no_evidence_loss.json`

```json
"use_evidence_loss": false,
"evidence_weight": 0.0
```

#### `LE_RACE_no_mil.json`

```json
"use_mil_loss": false,
"mil_weight": 0.0
```

#### `LE_RACE_no_evidence_pooling.json`

```json
"disable_evidence_pooling": true
```

forward 中若启用该项：

```python
features = h_root
logits = classifier(h_root)
```

但仍输出 evidence logits/probs 并可计算 evidence loss。

#### `LE_RACE_no_sparse.json`

```json
"use_sparse_loss": false,
"sparse_weight": 0.0
```

#### `LE_RACE_mil_only.json`

```json
"use_evidence_loss": false,
"evidence_weight": 0.0,
"use_mil_loss": true
```

#### `LE_RACE_topk_pooling.json`

```json
"evidence_pooling": "topk",
"evidence_topk_ratio": 0.15
```

---

### 4.11 `test.py`

**文件职责变化**：评估时保存 evidence 诊断信息。

#### 修改输出 prediction entry

当模型输出包含 `evidence_probs` 时，保存：

```python
pred_entry["evidence_probs"] = evidence_probs[i].tolist()
pred_entry["top_evidence_indices"] = top_indices
pred_entry["top_evidence_scores"] = top_scores
pred_entry["top_evidence_texts"] = top_texts
```

需要从 batch 的 `nodes` 中取 EDU text。注意只取 valid EDU 范围。

#### 分层评估

新增可选函数：

```python
def evaluate_by_slices(predictions: list[dict], metadata: list[dict]) -> dict:
```

分层维度：

- domain。
- model_name。
- length bucket。
- truncation flag。

输出：

```text
results/slice_metrics.json
```

---

### 4.12 `requirements.txt`

原始 requirements 不包含 torch 系列，保持该规则。

编码约束：不要写入 torch / torchvision / torchaudio；也不要把 `torch-geometric` 写进 `requirements.txt`。

建议新增：

```text
spacy
```

不加入：

```text
torch
torchvision
torchaudio
torch-geometric
```

> PyTorch 与 PyG 仍按照 RACE README 手动安装，以匹配 CUDA。

---

## 5 Results 文件格式规范

### 5.1 Training Outputs

每次训练输出目录沿用 RACE：

```text
results/{timestamp}/
  args.json
  metadata.json
  best_model.pth
  test_metrics.json
  test_predictions.json
  logs/
```

LE-RACE 新增：

```text
results/{timestamp}/
  evidence_diagnostics.json
  top_evidence_edus.json
  slice_metrics.json
```

### 5.2 `test_metrics.json`

必须包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `auroc_macro` | float | Macro AUROC |
| `f1_macro` | float | Macro F1 |
| `tpr_at_1_fpr_macro` | float | Avg TPR@1%FPR |
| `tpr_at_1_fpr_class_0` | float | Human-Written TPR@1%FPR |
| `tpr_at_1_fpr_class_1` | float | LLM-Polished TPR@1%FPR |
| `tpr_at_1_fpr_class_2` | float | LLM-Generated TPR@1%FPR |
| `tpr_at_1_fpr_class_3` | float | Humanized TPR@1%FPR |
| `tpr_at_1_fpr_mixed_avg` | float | class 1 和 class 3 平均 |

### 5.3 `evidence_label_stats.json`

字段：

```json
{
  "overall": {
    "coverage": 0.0,
    "positive_ratio": 0.0,
    "masked_ratio": 0.0,
    "avg_valid_edu": 0.0,
    "truncation_ratio": 0.0
  },
  "by_class": {
    "human_written": {},
    "human_ai_polished": {},
    "ai_generated": {},
    "ai_humanized": {}
  }
}
```

### 5.4 `top_evidence_edus.json`

每条记录：

```json
{
  "id": "sample-id",
  "label": 1,
  "prediction": 1,
  "top_edus": [
    {
      "rank": 1,
      "edu_index": 7,
      "score": 0.83,
      "text": "..."
    }
  ]
}
```

---

## 6 实现顺序

### Step 0: 建立可复现基线

1. 修复或明确 metric logits/probs 口径。
2. 跑 RACE 1 seed smoke test。
3. 确认 `test_metrics.json` 中指标字段完整。

### Step 1: 实现 weak label 生成

1. 新增 `utils/edu_utils.py`。
2. 新增 `utils/evidence_label_builder.py`。
3. 新增 `scripts/generate_edu_evidence_labels.sh`。
4. 生成 `data/hart_split_evidence/`。
5. 输出并检查 `evidence_label_stats.json`。

### Step 2: Dataset 接入 weak labels

1. 修改 `utils/flexible_dataset.py`。
2. 确认 labels/masks 与 `edu_token_spans` 长度一致。
3. 用一个 batch 打印：
   - `len(edu_evidence_labels[i])`
   - `len(edu_token_spans[i])`
   - `n_valid_edu[i]`

### Step 3: 实现 LE-RACE forward

1. 修改 `models/flexible_model.py`。
2. 增加 trace scorer。
3. 增加 evidence pooling。
4. 增加 fusion proj。
5. 保留 RACE 原始路径。
6. 检查输出 shape：
   - `h_root`: `[B, 512]`
   - `z_local`: `[B, 512]`
   - `z_final`: `[B, 512]`
   - `logits`: `[B, 4]`
   - `evidence_probs[i]`: `[n_edu_i]`

### Step 4: 实现 LE-RACE loss

1. 修改 `train.py`。
2. 增加 masked BCE。
3. 增加 MIL loss。
4. 增加 sparse loss。
5. 增加 warm-up epoch 控制。
6. 确保 SupCon features 使用 `F.normalize()`。

### Step 5: 指标与 checkpoint

1. 修改 `utils/metrics.py`。
2. 增加 `input_type`。
3. 增加 mixed-class TPR。
4. 修改 checkpoint selection 为 `tpr_at_1_fpr_macro`。

### Step 6: 诊断与消融

1. 新增 `utils/diagnostics.py`。
2. 新增 configs/ablations。
3. 新增 batch scripts。
4. 跑 1 seed smoke test。

### Step 7: Group-aware split

1. 新增 `utils/split_dataset_group_aware.py`。
2. 新增 `scripts/split_group_aware.sh`。
3. 生成 group split。
4. 重新生成 evidence labels。
5. 跑 RACE vs LE-RACE。

---

## 7 实验覆盖校验

| Part 3 实验 | 对应实现 |
|---|---|
| Metric sanity check 与 RACE 复现 | `utils/metrics.py`, `train.py`, `configs/RACE.json` |
| LE-RACE 主实验 | `models/flexible_model.py`, `utils/flexible_dataset.py`, `configs/LE_RACE.json` |
| evidence loss 消融 | `train.py`, `LE_RACE_no_evidence_loss.json` |
| MIL 消融 | `train.py`, `LE_RACE_no_mil.json` |
| evidence pooling 消融 | `models/flexible_model.py`, `LE_RACE_no_evidence_pooling.json` |
| sparse 消融 | `train.py`, `LE_RACE_no_sparse.json` |
| MIL-only | `train.py`, `LE_RACE_mil_only.json` |
| pooling 策略消融 | `models/flexible_model.py`, `LE_RACE_topk_pooling.json` |
| weak label 质量诊断 | `utils/evidence_label_builder.py`, `utils/diagnostics.py` |
| group-aware split | `utils/split_dataset_group_aware.py` |
| domain/generator/length 分层 | `utils/diagnostics.py`, `test.py` |
| LEAR-RACE 增强 | 第一版暂不实现，预留后续 `role_query` 配置 |

---

## 8 逻辑一致性校验

### 8.1 Tensor Shape

| 变量 | shape | 来源 |
|---|---|---|
| `last_hidden_state` | `[B, L, 768]` | RoBERTa |
| `edu_features` | `[n_edu_i, 768]` | token span mean |
| `x` before RGCN | `[total_nodes, 128]` | node projection |
| `x` after RGCN | `[total_nodes, 512]` | RGCN |
| `h_root` | `[B, 512]` | root node indexing |
| `edu_h_i` | `[n_edu_i, 512]` | EDU absolute indices |
| `score_input_i` | `[n_edu_i, 1536]` | `[h_i; h_root; h_i*h_root]` |
| `evidence_logits_i` | `[n_edu_i]` | trace scorer |
| `z_local` | `[B, 512]` | evidence pooling |
| `fusion_input` | `[B, 1536]` | `[h_root; z_local; h_root*z_local]` |
| `z_final` | `[B, 512]` | fusion proj |
| `logits` | `[B, 4]` | classifier |

### 8.2 Loss Compatibility

- CE 使用 `logits [B,4]` 与 `labels [B]`。
- SupCon 使用 L2-normalized `features [B,512]` 与 `labels [B]`。
- Evidence BCE 使用 ragged list：`evidence_logits_i [n_i]` 与 `edu_evidence_labels_i [n_i]`。
- MIL 使用 `evidence_probs_i [n_i]` 与 document label。
- Sparse 使用 `evidence_probs_i [n_i]`。

### 8.3 Warm-Up Compatibility

warm-up 时：

- `features = h_root`。
- `logits = classifier(h_root)`。
- SupCon 与 CE 完全等价于 RACE。
- evidence loss/MIL/sparse 可跳过或权重为 0。

非 warm-up 时：

- `features = z_final`。
- `logits = classifier(z_final)`。
- evidence branch 参与分类与 loss。

---

## 9 编码前确认清单

开始阶段 E 编码前，需要用户确认：

1. 是否直接在 `C:/Users/18785/Downloads/RACE-main/RACE-main` 上改代码，还是先复制到 `C:/Users/18785/Documents/RACE` 下再改。
2. 是否安装 `spacy` 与 `en_core_web_sm` 用于 weak label 句子切分。
3. 是否第一版只实现 LE-RACE，不实现 LEAR-RACE。
4. 是否 checkpoint 主指标改为 `tpr_at_1_fpr_macro`。
5. 是否允许新增 `data/hart_split_evidence/` 与 `results/diagnostics/` 目录。
6. 是否保留原 `configs/RACE.json` 不改，只新增 `configs/LE_RACE.json` 和 ablation configs。

---

## 10 Validation Report

实验要求覆盖：通过。

- RACE 复现、LE-RACE 主实验、消融、weak label 诊断、group-aware split、分层分析均有对应文件和函数设计。

逻辑一致性：通过。

- `h_root`、`z_local`、`z_final` shape 一致。
- ragged EDU labels 与 ragged evidence logits 逐样本计算。
- valid EDU 截断规则与 graph builder 的 `max_edu_nodes=len(edu_token_spans)` 对齐。

完整性：通过。

- 每个新增/修改文件均有职责说明。
- 核心函数均有参数、返回值和实现逻辑。
- results 文件格式、实现顺序、编码前确认清单已列出。


---

## FCE-RACE Addendum — Factorized Creator-Editor Modeling

> Added: 2026-07-06 21:19

基于最新实验结论，LE-RACE 的 EDU evidence 直接增强四分类会伤害 Generated 的 one-vs-rest 排序。因此新增 FCE-RACE 作为下一版主线：不再把 EDU weak evidence 作为主监督，而是显式拆分 creator/editor 两个二分类轴。

### 设计目标

- `creator ∈ {Human, LLM}`：由 RST/RGCN/root pooling 的 creator branch 建模。
- `editor ∈ {Human, LLM}`：由 pre-RGCN EDU expression states 的 editor branch 建模。
- 四类 logits 由二维组合得到：
  - Human-Written = creator_H + editor_H
  - LLM-Polished = creator_H + editor_L
  - LLM-Generated = creator_L + editor_L
  - Humanized = creator_L + editor_H

### 修改文件

- `models/flexible_model.py`
  - 增加 `use_factorized_ce` 开关。
  - 增加 `creator_head`、`editor_proj`、双 editor query attention pooling、`editor_head`。
  - 增加 factorized logits 组合函数。
  - `features` 保持 root/creator representation，SupCon 不作用于 editor/fusion 表征。
- `train.py`
  - 自动由四分类 label 派生 creator/editor 标签。
  - 增加 `use_creator_loss`、`creator_weight`、`use_editor_loss`、`editor_weight`。
- `configs/FCE_RACE.json`
  - 使用原始 `data/hart_split/*.jsonl`，不依赖 EDU evidence label。
  - loss 为 `CE_4class + SupCon_root + α CE_creator + β CE_editor`。

### 标签派生

```text
creator_label:
  Human-Written, LLM-Polished -> Human
  LLM-Generated, Humanized    -> LLM

editor_label:
  Human-Written, Humanized    -> Human
  LLM-Generated, LLM-Polished -> LLM
```

### 第一版限制

- 不加入 gradient reversal/adversarial disentanglement。
- 不使用 EDU weak BCE/MIL/sparse loss。
- residual 四分类 head 默认 `factorized_residual_gamma=0`。

---

## PASTED-RACE Minimal Experiment Addendum

> Added: 2026-09-12 | Status: APPROVED_FOR_IMPLEMENTATION

### 实验边界

本实验仅复现与 PASTED 相同的场景：`human_written` 原文与其
`human_ai_polished`（AI 改写）版本。`ai_generated` 和 `ai_humanized`
不进入数据、训练或评估。任务是 EDU-level lexical trace regression，
不训练 RACE 四分类、不做 evidence pooling/fusion、不使用 BCE/MIL/sparse loss。

### 数据与防泄漏

- 合并现有 `data/hart_split/{train,val,test}_graph.jsonl`，按 `item_id` 去重。
- 仅保留同时具有 `human_written` 和 `human_ai_polished` 的完整 `group_id`。
- 以 `group_id` 为最小单位按 70/10/20 重新划分，任何 source group 不跨 split。
- `human_written` 的所有有效 EDU 连续标签为 0。
- `human_ai_polished` 与同组 human reference 做句子语义对齐；可靠匹配计算
  `1 - sentence_bleu(reference_tokens, target_tokens)`，再按字符 overlap 投影到 EDU。
- 对齐失败、过短句子或无字符 overlap 的 EDU 设置 `edu_lexical_mask=0`。

输出目录：

```text
data/pasted_race/
  train_graph.jsonl
  val_graph.jsonl
  test_graph.jsonl
  lexical_trace_stats.json
  split_manifest.json
```

### 文件与函数

- `utils/lexical_trace_label_builder.py`
  - `deduplicate_items(items) -> list[dict]`
  - `build_complete_pairs(items) -> dict[str, list[dict]]`
  - `split_groups(groups, train_ratio, val_ratio, seed) -> dict[str, list[dict]]`
  - `bleu4_diversity(reference, target, nlp) -> float`
  - `build_sentence_scores(reference_item, target_item, ...) -> tuple[list[float], list[int]]`
  - `project_sentence_scores_to_edus(item, scores, masks, nlp) -> tuple[list[float], list[int]]`
  - `build_split(items, ...) -> tuple[list[dict], dict]`
- `utils/flexible_dataset.py`
  - 加载并截断 `edu_lexical_scores`、`edu_lexical_masks`，collate 时保持 ragged list。
- `models/flexible_model.py`
  - 配置 `use_lexical_trace=true` 时初始化 `lexical_trace_head`。
  - `_compute_lexical_trace(x, graph_batch, graph_list, root_node_indices)` 返回
    `list[Tensor[n_i]]`。
  - 输入为 `[h_i; h_root; h_i*h_root]`，shape `[n_i, 1536]`；输出 `[n_i]`。
- `train_pasted_race.py`
  - `masked_trace_mse(...)`：严格按有效 EDU 数平均。
  - `evaluate_trace(...)`：计算 MSE、Pearson、Spearman、AUROC、TPR@1%FPR。
  - 仅用 trace loss 更新模型；验证 AUROC 选择 checkpoint，相关系数作 tie-breaker。
- `configs/pasted_race/PASTED_RACE_lexical.json`：实验超参数。
- `scripts/generate_pasted_race_labels.sh`、`scripts/train_pasted_race.sh`：运行入口。

### Tensor shape 与 loss

| 变量 | Shape | 含义 |
|---|---|---|
| `edu_h_i` | `[n_i, 512]` | RGCN 后 EDU 表示 |
| `h_root_i` | `[512]` | 文档 root 表示 |
| `trace_input_i` | `[n_i, 1536]` | 局部、全局及逐维交互 |
| `trace_scores_i` | `[n_i]` | EDU 连续预测 |
| `edu_lexical_scores_i` | `[n_i]` | `1-BLEU4` 目标 |
| `edu_lexical_masks_i` | `[n_i]` | 有效监督 mask |

损失：

```text
L_trace = sum(mask * (prediction - target)^2) / max(sum(mask), 1)
```

### 评估输出

```text
results/pasted_race/lexical/<timestamp>/
  args.json
  metadata.json
  best_model.pth
  val_metrics.json
  test_metrics.json
  test_predictions.jsonl
```

指标包括 `mse`、`pearson`、`spearman`、`auc`、`tpr_at_1_fpr`、
`num_documents` 与 `num_valid_edus`。二值检测标签采用 `target > 1e-6`。

### 实现校验

- ✅ 实验要求覆盖：数据生成、连续监督、训练、测试与逐 EDU 输出均有对应模块。
- ✅ 逻辑一致性：所有 ragged EDU tensor 逐文档对齐，loss 只统计 mask=1 的位置。
- ✅ 完整性：新增/修改文件、运行入口和结果格式均已列出。
## PASTED-RACE Joint Fusion Addendum

### Architecture change

`models/flexible_model.py::RACEModel` adds an optional
`use_lexical_fusion` path. `_compute_lexical_trace()` returns ragged EDU scores,
normalized EDU attention weights, and one pooled lexical representation per
document. `lexical_fusion_proj` maps
`[h_root; z_lex; h_root * z_lex]` back to `gnn_hidden_dim`; the existing binary
classifier consumes the fused representation. When the flag is false, all
existing RACE and lexical-only behavior remains unchanged.

### Joint optimization

`train_pasted_race.py` supports `joint_classification=true`, an initialization
checkpoint, `classification_warmup_epochs`, and `lexical_loss_weight`. During
warm-up it optimizes masked EDU MSE only. Later epochs optimize document CE plus
weighted masked EDU MSE. The classifier and new fusion projection are excluded
from lexical checkpoint loading so that they receive a clean initialization.

### Evaluation and artifacts

Joint evaluation reports classification AUROC, strict TPR@1%FPR, accuracy and
F1 alongside the existing lexical metrics. Predictions store document logits,
class-1 probability, ragged EDU scores, targets, and masks. Checkpoint selection
uses validation classification AUROC for the joint experiment.

### Shape contract

- `h_root`: `[B, H]`
- per-document `h_edu`: `[N_i, H]`
- per-document lexical scores/weights: `[N_i]`
- `z_lex`: `[B, H]`
- fusion input: `[B, 3H]`
- binary logits: `[B, 2]`

### Validation order

1. Static compilation and legacy lexical-only eval.
2. Joint forward/loss/checkpoint smoke test on the 20-pair smoke data.
3. Full group-safe seed-42 joint training and test comparison.

## Four-Class Residual PASTED-RACE Addendum

### `utils/build_pasted_race_fourclass.py`

Merge and deduplicate the three legacy HART files, then assign every document
using `data/pasted_race/split_manifest.json`. Copy lexical targets from the
already generated group-safe Human/Polished files. For Generated/Humanized,
write zero-valued arrays with all-zero masks matching the RST EDU count. Assert
that all 16,000 item IDs are assigned once and split group intersections are
empty.

### `models/flexible_model.py`

Add `lexical_fusion_mode=residual` and a scalar `lexical_residual_gamma`
initialized to zero. In residual mode, retain the original RACE root feature
and add gated lexical fusion instead of replacing it. Binary replacement fusion
remains available for backward compatibility.

### `train_pasted_race_fourclass.py`

Instantiate `RACEModel(num_classes=4)`. Load the newly retrained group-safe RACE
baseline checkpoint into all matching base/classifier keys, load only
`lexical_trace_head.*` from the lexical-only checkpoint, and leave the new
fusion projection/gamma initialized. The legacy RACE checkpoint is forbidden
for formal initialization because it has seen groups assigned to the rebuilt
test split.
Epoch 1 freezes the original RACE base and classifier and optimizes observed
Human/Polished lexical MSE. Later epochs unfreeze according to the original
RACE policy and optimize four-class CE plus weighted masked MSE.

Evaluation reuses `utils.metrics.compute_classification_metrics` with logits,
adds per-class one-vs-rest AUROC, and retains lexical regression diagnostics.
Checkpoint selection follows validation macro-F1 with strict per-class
TPR@1%FPR as reported diagnostics.

### Shape and masking contract

- Four-class logits: `[B, 4]`.
- Lexical outputs remain ragged `[N_i]`.
- Human/Polished lexical masks contain valid positions.
- Generated/Humanized lexical masks are all zero and never enter MSE.
- If a batch has no observed lexical target, lexical MSE returns a differentiable
  zero rather than raising.

## AI-to-Humanized Lexical Trace Extension

`utils/humanization_trace_label_builder.py` selects only groups containing one
`ai_generated` and one `ai_humanized` item, enforces one-to-one pairing, and
assigns them with `data/pasted_race/split_manifest.json`. It reuses the existing
semantic sentence alignment and unsmoothed `1-BLEU4` projection implementation.
Generated EDUs receive zero regression targets; Humanized EDUs receive aligned
continuous targets. Output is isolated under `data/pasted_race_humanization/`.

`FlexibleGraphDataset` accepts an optional configuration-level `label_map`, so
this independent trace experiment maps Generated/Humanized to 0/1 without
rewriting semantic class names in stored data. `train_pasted_race.py` remains
in lexical-only masked-MSE mode and writes a distinct checkpoint under
`results/pasted_race/humanization_lexical_seed42/`.

## Dual-Trace Four-Class Integration

`utils/build_pasted_race_fourclass.py` accepts both target datasets and writes
`edu_lexical_scores/masks` for Human/Polished plus
`edu_humanization_scores/masks` for Generated/Humanized. Every document has
both field pairs, with the unrelated direction masked out. Dual data is written
under `data/pasted_race_fourclass_dual/` with the existing manifest.

`FlexibleGraphDataset` truncates and collates both ragged target pairs.
`RACEModel` optionally adds `humanization_trace_head`,
`humanization_fusion_proj`, and `humanization_residual_gamma`. Both residuals
are computed from the unchanged root representation and summed. Disabling the
new option preserves existing behavior.

`train_pasted_race_fourclass.py` maps the second checkpoint's
`lexical_trace_head.*` parameters into `humanization_trace_head.*`. Calibration
updates both trace heads only; joint epochs optimize CE and the two separately
weighted masked means. Metrics and predictions retain distinct `lexical_*` and
`humanization_*` namespaces.

Shape and mask contract:

- both score lists are independently `[N_i]`;
- Human/Polished have polishing masks; Generated/Humanized have humanization
  masks only when a deterministic paired target exists;
- an empty direction contributes differentiable zero loss;
- four-class logits remain `[B, 4]`.

## Creator-Retention / Editor-Modification Addendum

`utils/creator_retention_label_builder.py` reads the already group-safe dual
four-class JSONL files. Within each split it resolves `Human -> Polished` and
`Generated -> Humanized` references by stored reference ID, computes unrescaled
SciBERT BERTScore Recall in batches, and writes the original records plus
`creator_retention_score`, `creator_retention_mask`,
`creator_retention_reference_id`, and `creator_retention_direction`. Human and
Generated receive exact self-retention 1.0. No source text is included as a
model input.

`utils/flexible_dataset.py` exposes the retention score and mask as `[B]`
tensors. `models/flexible_model.py` adds an optional
`creator_retention_head(h_root) -> [B]`, sigmoid-bounded scores, a residual
fusion projection, and a zero-initialized scalar gate. Existing polishing and
humanization trace heads jointly constitute the Editor Modification branch and
remain independently supervised.

`train_pasted_race_fourclass.py` conditionally computes masked document MSE,
logs Creator regression metrics and predictions, includes the new head during
the calibration epoch, and adds `creator_retention_loss_weight` in joint
training. Existing configs with `use_creator_retention=false` remain behaviorally
unchanged.

`configs/pasted_race/PASTED_RACE_fourclass_creator_editor_joint.json` defines
the full three-objective experiment. `scripts/build_creator_retention_data.sh`
generates its dataset; `scripts/train_pasted_race_creator_editor.sh` runs it.

Validation contract:

- classification logits `[B,4]` and retention scores `[B]`;
- retention targets are finite and in `[0,1]`, with one valid mask per document;
- no group overlap and every edited item resolves to a same-group source;
- all three residual gates equal zero immediately after checkpoint loading;
- legacy baseline and dual-trace configs pass unchanged forward smoke tests.
