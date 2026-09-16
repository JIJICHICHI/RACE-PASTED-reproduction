# Group-Safe Text-Length Analysis

## Protocol

- Test set: canonical 3,200-document group-safe manifest; no resplitting or retraining.
- Manifest SHA-256: `41dc976bc542f68764d5c07b9d52ffc8601c342f55af28df4564966ae25a2c2a`.
- Length: full final-text token count from `FacebookAI/roberta-base`, without special tokens or truncation.
- Buckets: `[0,200)`, `[200,400)`, `[400,600)`, `[600,800)`, `[800,+∞)`; display labels match RACE Figure 4.
- Primary metric: four-class Macro TPR@1%FPR, reported as three-seed mean ± sample standard deviation (%).
- The paper does not disclose its token counter. Its Figure 4 RACE/CoCo values below are historical reference, not same-split baselines.

## Bucket audit

| Bucket | Documents | Human | Polished | Generated | Humanized |
|---|---:|---:|---:|---:|---:|
| 0–200 | 341 | 87 | 91 | 149 | 14 |
| 200–400 | 1372 | 329 | 349 | 591 | 103 |
| 400–600 | 949 | 248 | 214 | 432 | 55 |
| 600–800 | 353 | 87 | 99 | 151 | 16 |
| 800+ | 185 | 49 | 47 | 83 | 6 |

## Macro TPR@1%FPR

| Method | 0–200 | 200–400 | 400–600 | 600–800 | 800+ |
|---|---:|---:|---:|---:|---:|
| Strong RACE | 55.20±4.92 | 78.83±2.40 | 78.85±1.31 | 96.54±1.76 | 97.88±3.42 |
| Single Trace | 58.29±2.75 | 81.22±1.36 | 83.94±1.65 | 97.44±1.20 | 98.74±1.22 |
| Dual / Editor | 54.92±4.06 | 80.37±3.51 | 84.96±2.36 | 97.55±0.70 | 97.85±1.73 |
| Creator + Editor | 57.30±2.29 | 79.73±1.64 | 82.26±3.06 | 96.59±2.51 | 98.56±0.77 |
| Creator-only | 57.86±2.67 | 82.03±2.21 | 83.83±3.33 | 97.38±0.84 | 96.92±2.16 |
| Creator, no fusion | 57.33±3.67 | 76.25±8.70 | 82.95±2.17 | 96.04±1.75 | 97.96±2.63 |
| All lambdas = 0 | 57.63±0.83 | 80.81±1.69 | 83.75±2.24 | 96.47±1.46 | 98.63±0.68 |

## Macro-F1

| Method | 0–200 | 200–400 | 400–600 | 600–800 | 800+ |
|---|---:|---:|---:|---:|---:|
| Strong RACE | 80.37±2.33 | 90.87±1.01 | 92.14±0.26 | 96.61±1.32 | 97.74±1.78 |
| Single Trace | 83.05±3.75 | 91.53±0.10 | 92.63±0.60 | 96.43±1.47 | 97.55±1.89 |
| Dual / Editor | 82.01±3.01 | 91.10±0.28 | 92.22±1.00 | 97.24±1.50 | 97.69±2.85 |
| Creator + Editor | 81.94±1.19 | 90.73±0.96 | 91.19±1.67 | 95.48±1.42 | 96.80±0.98 |
| Creator-only | 83.67±2.60 | 91.54±0.51 | 92.58±1.15 | 96.26±1.71 | 97.10±1.43 |
| Creator, no fusion | 80.84±2.98 | 91.38±0.15 | 92.02±0.59 | 95.78±0.30 | 95.77±1.22 |
| All lambdas = 0 | 82.03±1.09 | 91.26±0.22 | 91.87±0.83 | 96.98±0.74 | 97.64±1.74 |

## Macro-AUROC

| Method | 0–200 | 200–400 | 400–600 | 600–800 | 800+ |
|---|---:|---:|---:|---:|---:|
| Strong RACE | 96.42±0.16 | 98.22±0.22 | 98.58±0.06 | 99.79±0.03 | 99.95±0.05 |
| Single Trace | 96.24±0.77 | 98.37±0.05 | 98.54±0.13 | 99.76±0.18 | 99.95±0.03 |
| Dual / Editor | 96.23±0.32 | 98.29±0.10 | 98.61±0.08 | 99.84±0.05 | 99.93±0.03 |
| Creator + Editor | 96.11±0.60 | 98.36±0.14 | 98.59±0.23 | 99.82±0.08 | 99.96±0.02 |
| Creator-only | 96.43±0.53 | 98.30±0.14 | 98.64±0.24 | 99.71±0.12 | 99.92±0.04 |
| Creator, no fusion | 96.43±0.28 | 98.29±0.24 | 98.41±0.29 | 99.87±0.05 | 99.91±0.11 |
| All lambdas = 0 | 96.62±0.29 | 98.36±0.20 | 98.74±0.19 | 99.86±0.04 | 99.95±0.02 |

## RACE paper Figure 4 reference

| Published method | 0–200 | 200–400 | 400–600 | 600–800 | 800+ |
|---|---:|---:|---:|---:|---:|
| RACE | 58.8 | 80.0 | 86.8 | 97.0 | 94.8 |
| CoCo | 57.6 | 74.4 | 79.1 | 97.0 | 96.0 |

These published values use a different experiment provenance and are not used to calculate gains.
