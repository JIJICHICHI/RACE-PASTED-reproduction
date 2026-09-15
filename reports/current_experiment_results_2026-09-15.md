# Current Experiment Results — 2026-09-15

All reported four-class experiments use the same group-safe split and the
official-optimized RACE outer loop (`lr=2.9e-5`, batch size 16, warmup 0.1,
maximum 20 epochs, patience 5, validation Macro-F1 selection). Values are mean
± sample standard deviation over seeds 42, 2026, and 3407.

## End-to-end seed-matched trace results

| Method | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |
|---|---:|---:|---:|---:|
| Strong RACE | 94.06±0.41% | 91.08±0.37% | 98.45±0.10% | 79.75±1.60% |
| Single trace | **94.57±0.60%** | **91.79±0.60%** | 98.44±0.05% | **82.58±1.16%** |
| Dual trace | 94.32±0.88% | 91.44±0.95% | **98.45±0.06%** | 82.41±0.52% |

Single trace improves mean Accuracy, Macro-F1, and low-FPR TPR. Its Macro-F1
exceeds the matching strong baseline for all three seeds, while Accuracy still
drops for seed 3407. Dual trace remains less stable at seed 3407.

## Creator Retention P2 signal analysis

The analysis contains 5,015 real edited pairs and no artificial self-retention
examples: 4,000 Human→Polished and 1,015 Generated→Humanized pairs.

| Direction | Creator Retention | Editor Modification | Pearson | Spearman |
|---|---:|---:|---:|---:|
| Human→Polished | 0.781899±0.099840 | 0.729806±0.245705 | -0.823426 | -0.872743 |
| Generated→Humanized | 0.779110±0.075592 | 0.813736±0.142824 | -0.731668 | -0.796033 |

The signals are strongly negatively correlated but are not exact inverses,
especially for Generated→Humanized.

## Creator Retention P3 standalone regression

The detector sees only the edited final text. The original creator text is
used only to compute the offline target. Checkpoints are selected by minimum
validation MSE, with validation Spearman as the tie-breaker.

| Creator input | Test MSE ↓ | Test Pearson ↑ | Test Spearman ↑ |
|---|---:|---:|---:|
| `h_i` | **0.004049±0.000218** | **0.733611±0.013816** | **0.723041±0.016967** |
| `[h_i; h_root]` | 0.004391±0.000290 | 0.721476±0.004053 | 0.710226±0.001689 |
| `[h_i; h_root; h_i*h_root]` | 0.004412±0.000056 | 0.708204±0.009905 | 0.698380±0.006117 |

The automatic P3 selection uses mean validation rather than test performance:

| Input | Mean validation MSE ↓ | Mean validation Spearman ↑ |
|---|---:|---:|
| `h_i` | **0.004445** | **0.716198** |
| `[h_i; h_root]` | 0.004761 | 0.700159 |
| `[h_i; h_root; h_i*h_root]` | 0.004606 | 0.706490 |

The selected P5 Creator input is therefore `h_i`.

## Current execution status

P5 Creator+Editor joint training is active. It uses matching-seed Strong RACE,
polishing trace, humanization trace, and selected P3 Creator checkpoints. P5
metrics are intentionally omitted until the three formal seeds finish.

Model checkpoints, large prediction files, datasets, and runtime logs are not
tracked by Git. Their provenance and exact local paths are recorded in
`docs/dev_log.md` and resolved run configurations.
