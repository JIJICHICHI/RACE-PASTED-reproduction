# RACE: Rhetorical Analysis for Creator-Editor Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2604.04932-b31b1b.svg)](https://arxiv.org/abs/2604.04932)
[![ACL-2026](https://img.shields.io/badge/ACL-2026--Main-red.svg)](https://arxiv.org/abs/2604.04932)

This repository contains the implementation of **RACE** (_Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection_), which is accepted by **ACL 2026 Main Conference**.

> We are currently cleaning up the codebase, and there might be some omissions during this process. Please feel free to raise an issue if you meet any problems when using this code.

RACE leverages Rhetorical Structure Theory (RST) to construct logical graphs from documents, then applies Relational Graph Convolutional Networks (RGCN) to learn structure-aware representations for fine-grained AI-generated text detection.

## 1. Installation

### Requirements

- **Python**: 3.8+
- **PyTorch & PyTorch Geometric**:
  Please install these first, following the official guides for your CUDA version to ensure GPU support:
  - [PyTorch](https://pytorch.org/get-started/locally/)
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [isanlp_rst](https://github.com/tchewik/isanlp_rst)

### Install Dependencies

Install the remaining Python dependencies using `pip`:

```bash
pip install -r requirements.txt
```

**Note**: The script `scripts/run_multiple_seeds.sh` requires `jq` (a command-line JSON processor).

- Ubuntu/Debian: `sudo apt-get install jq`
- MacOS: `brew install jq`

## 2. Data Preparation

This codebase expects processed data in `.jsonl` format. By default, the configuration looks for data in a `data/hart_split/` directory.

### Data Format

Each line in the `.jsonl` file is a JSON object with the following fields:

| Field            | Type   | Description                                                                           |
| ---------------- | ------ | ------------------------------------------------------------------------------------- |
| `item_id`        | string | Unique identifier for the article                                                     |
| `group_id`       | string | Group identifier (e.g., same source document)                                         |
| `source_dataset` | string | Origin dataset name                                                                   |
| `article`        | string | Full text content of the document                                                     |
| `label`          | string | Category label (`human_written`, `ai_generated`, `human_ai_polished`, `ai_humanized`) |
| `model_name`     | string | Name of the AI model (if applicable)                                                  |
| `rst_structure`  | object | Pre-parsed RST tree structure                                                         |

A sample data file is provided in `data/example/sample_data.jsonl` for reference.

### Full Preprocessing Pipeline (Recommended)

1. Place the [HART dataset](https://github.com/baoguangsheng/truth-mirror) into the `data/` directory to form the `data/truth-mirror/` structure.
2. Run the one-command script to process it end-to-end and generate the training-ready `data/hart_split/` directory:

```bash
bash scripts/prepare_data.sh
```

This runs the following 4 steps automatically:

| Step           | Script                               | Input → Output                                                         |
| -------------- | ------------------------------------ | ---------------------------------------------------------------------- |
| 1. Unification | `utils/data_unification_pipeline.py` | Hart raw JSON → `data/master_grouped.jsonl`                            |
| 2. RST Parsing | `utils/rst_parser_pipeline.py`       | `master_grouped.jsonl` → `master_rst_parsed.jsonl`                     |
| 3. Flattening  | `utils/flatten_data_pipeline.py`     | `master_rst_parsed.jsonl` → `flattened_articles.jsonl`                 |
| 4. Splitting   | `utils/split_dataset.py`             | `flattened_articles.jsonl` → `hart_split/{train,val,test}_graph.jsonl` |

> **Note**: Step 2 (RST Parsing) requires the [`isanlp_rst`](https://github.com/tchewik/isanlp_rst) parser. Set the `RST_PARSER_PATH` in your `.env` file before running.

### Run Steps Individually

You can also run each step separately:

```bash
# Step 1: Parse and unify Hart dataset
python utils/data_unification_pipeline.py --output data/master_grouped.jsonl

# Step 2: RST parsing (requires isanlp_rst)
python utils/rst_parser_pipeline.py --input data/master_grouped.jsonl --output data/master_rst_parsed.jsonl

# Step 3: Flatten to per-article format
python utils/flatten_data_pipeline.py --input_file data/master_rst_parsed.jsonl --output_file data/flattened_articles.jsonl

# Step 4: Split into train/val/test
python utils/split_dataset.py
```

### Expected Output Structure

```
RACE/
├── data/
│   ├── truth-mirror/          # Raw dataset from HART
│   │   ├── benchmark/
│   │   │   ├── hart/
│   │   │   └── raid/
│   │   └── assets/
│   └── hart_split/            # Generated training-ready graphs
│       ├── train_graph.jsonl
│       ├── val_graph.jsonl
│       ├── test_graph.jsonl
│       └── stats.md           # Split statistics
└── ...
```

> **Note**: You can change the data paths in `configs/RACE.json` to point to any location on your system.

## 3. Usage

### PASTED-RACE lexical trace experiment

This minimal experiment keeps only paired `human_written` and
`human_ai_polished` documents. It rebuilds group-safe splits, assigns continuous
EDU targets using aligned-sentence `1 - BLEU-4`, and trains an EDU regression
head on top of the RACE encoder:

```bash
bash scripts/generate_pasted_race_labels.sh
bash scripts/train_pasted_race.sh
```

For a short training smoke test, append
`--data_dir data/pasted_race_smoke --output_dir results/pasted_race/smoke
--max_train_samples 16 --max_eval_samples 4 --epochs 1` to the second command.
Outputs are written to `results/pasted_race/lexical_seed42/`.

To initialize from that lexical checkpoint and train the integrated RACE binary
classifier with lexical attention fusion:

```bash
bash scripts/train_pasted_race_joint.sh
```

The joint run writes its checkpoint, metrics, history, and EDU/document
predictions to `results/pasted_race/joint_seed42/`.

For the no-leak four-class extension, rebuild all four classes, retrain the
original RACE architecture on that split, place the resulting checkpoint path
in `PASTED_RACE_fourclass_joint.json`, and run residual lexical fusion:

```bash
bash scripts/build_pasted_race_fourclass.sh
bash scripts/train_pasted_race_fourclass_baseline.sh
bash scripts/train_pasted_race_fourclass_joint.sh
```

### Training (Single GPU)

To train the model using the provided configuration:

```bash
bash scripts/train_single.sh configs/RACE.json
```

You can optionally specify a GPU ID (default is 0):

```bash
# Run on GPU 1
bash scripts/train_single.sh configs/RACE.json 1
```

### Evaluation

To evaluate a trained model checkpoint on the test set:

```bash
bash scripts/eval_single.sh configs/RACE.json path/to/your/checkpoint.pth
```

You can optionally specify a GPU ID (default is 0) as the third argument:

```bash
# Evaluate on GPU 1
bash scripts/eval_single.sh configs/RACE.json path/to/your/checkpoint.pth 1
```

### Multiple Seeds (Reproducibility)

To run the training loop multiple times with different random seeds:

```bash
# Run 5 experiments with random seeds
bash scripts/run_multiple_seeds.sh configs/RACE.json 5
```

## 4. Configuration

The core configuration is located in `configs/RACE.json`. Key settings include:

| Category               | Parameters                                        |
| ---------------------- | ------------------------------------------------- |
| **Data Paths**         | `data_path`, `val_data_path`, `test_data_path`    |
| **Model Backbone**     | `backbone_model_path` (e.g., `roberta-base`)      |
| **Training**           | `lr`, `batch_size`, `epochs`, `weight_decay`      |
| **GNN**                | `feature_dim`, `gnn_hidden_dim`, `num_gnn_layers` |
| **Graph Construction** | Under the `graph` key (see below)                 |
| **Loss**               | `loss.use_supcon` (`true` / `false`)              |

### Graph Configuration

The `graph` object controls RST-based graph construction:

```json
{
  "graph": {
    "graph_builder_type": "rst",
    "gnn_layer_type": "RGCN",
    "rst_use_nuclearity": false,
    "use_distinct_reverse_edges": false,
    "rgcn_num_bases": 10,
    "pooling_strategy": "root"
  }
}
```

## 5. Model Variants

| Model                     | Config `model_type` | Description                        |
| ------------------------- | ------------------- | ---------------------------------- |
| **RACEModel**             | `"gnn"` (default)   | Full RACE model with RST-based GNN |
| **FlexibleBaselineModel** | `"baseline"`        | Ablation: backbone only, no GNN    |

## 6. Directory Structure

```
RACE/
├── configs/          # Model and training configurations
├── data/             # Dataset directory (user-provided)
│   └── example/      # Sample data for reference
├── models/           # Core PyTorch model definitions
├── scripts/          # Shell scripts for training and evaluation
├── utils/            # Data loading, graph building, metrics, and data prep
├── train.py          # Main training entry point
├── test.py           # Main evaluation entry point
└── requirements.txt  # Python dependencies
```

## 7. PASTED-RACE experiments

### AI-to-Humanized lexical trace experiment

Generate aligned `Generated -> Humanized` EDU targets and train the independent
seed-42 trace model:

```bash
bash scripts/generate_humanization_trace_labels.sh
bash scripts/train_humanization_trace.sh
```

Artifacts are written to `data/pasted_race_humanization/` and
`results/pasted_race/humanization_lexical_seed42/`; the existing
Human-to-Polished checkpoint is not overwritten.

Build and train the dual-trace four-class model:

```bash
bash scripts/build_pasted_race_fourclass_dual.sh
bash scripts/train_pasted_race_fourclass_dual_joint.sh
```

### Creator Retention + Editor Modification

Generate document-level Creator Retention labels with unrescaled SciBERT
BERTScore Recall, then train the joint four-class model:

```bash
bash scripts/build_creator_retention_data.sh
bash scripts/train_pasted_race_creator_editor.sh
```

The paired creator text is used only for offline target generation. Model
inference consumes only the final document. The two existing EDU `1-BLEU4`
heads remain the direction-specific Editor Modification supervision.

## 8. Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{li2026finalactormodelingdual,
  title={Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection},
  author={Yang Li and Qiang Sheng and Zhengjia Wang and Yehan Yang and Danding Wang and Juan Cao},
  year={2026},
  eprint={2604.04932},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.04932},
}
```
