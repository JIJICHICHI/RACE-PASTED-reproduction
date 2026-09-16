# Group-Safe Creator/Modification Diagnostics

Formal Strong RACE checkpoints; fixed group-safe manifests; seeds 42/2026/3407; backbone training lr=2.9e-5. Values are mean±sample standard deviation.

## Data Audit

| Split | Documents | Groups | Class counts |
|---|---:|---:|---|
| train | 11200 | 2800 | Human=2800, Polished=2800, Generated=4897, Humanized=703 |
| val | 1600 | 400 | Human=400, Polished=400, Generated=682, Humanized=118 |
| test | 3200 | 800 | Human=800, Polished=800, Generated=1406, Humanized=194 |

Group overlap: train/val=0, train/test=0, val/test=0.

## Four-Class Probability Diagnostics

| Task | Accuracy | Macro-F1 | AUROC | Positive TPR@1%FPR | Negative TPR@1%FPR |
|---|---:|---:|---:|---:|---:|
| Creator: H/P vs G/Hu | 96.90±0.40 | 96.90±0.40 | 99.44±0.10 | 94.29±0.97 | 85.38±3.60 |
| Modification: H/G vs P/Hu | 94.42±0.40 | 93.49±0.42 | 97.45±0.16 | 74.98±6.40 | 39.23±14.43 |
| Editor actor: H/Hu vs P/G | 96.83±0.02 | 96.24±0.02 | 98.68±0.05 | 61.67±2.26 | 92.42±0.57 |
| Human origin: H vs P | 98.17±0.20 | 98.17±0.20 | 99.76±0.12 | 98.00±1.23 | 98.04±1.13 |
| AI origin: G vs Hu | 95.54±0.29 | 88.43±0.57 | 94.40±0.40 | 70.96±0.79 | 37.22±8.27 |
| Unmodified: H vs G | 99.59±0.05 | 99.56±0.05 | 99.93±0.02 | 99.93±0.00 | 99.71±0.07 |
| Modified: P vs Hu | 97.18±0.44 | 95.27±0.79 | 99.43±0.14 | 94.85±0.52 | 86.46±5.51 |

## Frozen h_root Linear Probes

| Task | Accuracy | Macro-F1 | AUROC | Positive TPR@1%FPR | Negative TPR@1%FPR |
|---|---:|---:|---:|---:|---:|
| Creator: H/P vs G/Hu | 97.03±0.06 | 97.03±0.06 | 99.35±0.09 | 94.00±0.97 | 86.50±2.91 |
| Modification: H/G vs P/Hu | 94.33±0.52 | 93.45±0.56 | 97.82±0.05 | 79.01±1.51 | 48.78±3.28 |
| Editor actor: H/Hu vs P/G | 97.00±0.16 | 96.46±0.18 | 98.46±0.14 | 52.80±6.93 | 92.59±0.38 |
| Human origin: H vs P | 98.33±0.25 | 98.33±0.25 | 99.72±0.12 | 98.00±1.11 | 97.38±1.39 |
| AI origin: G vs Hu | 94.40±1.07 | 86.91±2.01 | 93.46±0.79 | 72.51±1.66 | 17.02±11.47 |
| Unmodified: H vs G | 99.65±0.05 | 99.62±0.06 | 99.90±0.05 | 99.88±0.04 | 99.71±0.07 |
| Modified: P vs Hu | 97.52±0.31 | 95.97±0.55 | 99.19±0.33 | 91.92±0.79 | 62.04±21.09 |

## Frozen h_root Test Geometry

| Axis | Silhouette (cosine) | Within cosine | Between cosine | Separation |
|---|---:|---:|---:|---:|
| Creator: H/P vs G/Hu | 0.6287±0.0124 | 0.7548±0.0042 | 0.1693±0.0169 | 0.5855±0.0145 |
| Modification: H/G vs P/Hu | 0.0556±0.0333 | 0.4460±0.0198 | 0.4832±0.0086 | -0.0372±0.0253 |
| Editor actor: H/Hu vs P/G | 0.6216±0.0249 | 0.7321±0.0140 | 0.1016±0.0406 | 0.6304±0.0541 |

## Per-Seed AI-Origin Modification

| Seed | Source | Accuracy | Macro-F1 | AUROC | Humanized TPR@1%FPR | Generated TPR@1%FPR | Selected C |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | four-class | 95.38 | 88.13 | 94.25 | 71.13 | 33.78 | — |
| 42 | h_root probe | 93.25 | 84.83 | 94.32 | 73.20 | 29.80 | 0.01 |
| 2026 | four-class | 95.88 | 89.08 | 94.10 | 71.65 | 31.22 | — |
| 2026 | h_root probe | 95.38 | 88.85 | 92.77 | 73.71 | 13.66 | 0.01 |
| 3407 | four-class | 95.38 | 88.07 | 94.85 | 70.10 | 46.66 | — |
| 3407 | h_root probe | 94.56 | 87.04 | 93.30 | 70.62 | 7.61 | 0.01 |
