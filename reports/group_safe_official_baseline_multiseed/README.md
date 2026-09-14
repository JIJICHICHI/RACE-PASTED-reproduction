# Group-safe Official-style RACE Baseline: Three Seeds

All runs use the same group-overlap=0 four-class split, learning rate `2.9e-5`,
batch size 16, at most 20 epochs, linear warmup ratio 0.1, patience 5, and
validation macro-F1 checkpoint selection. Seed 3407 is the paper's official
seed; 42 and 2026 are stability seeds.

## Per-seed test results

| Seed | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |
|---:|---:|---:|---:|---:|
| 42 | 0.935938 | 0.906799 | 0.983910 | 0.779082 |
| 2026 | 0.942500 | 0.914046 | 0.983863 | 0.805350 |
| 3407 | 0.943438 | 0.911599 | 0.985618 | 0.808134 |
| Mean ± sample SD | 0.940625 ± 0.004086 | 0.910815 ± 0.003687 | 0.984464 ± 0.001000 | 0.797522 ± 0.016030 |

## Comparison with completed dual-trace runs

| Method | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |
|---|---:|---:|---:|---:|
| Strong baseline | 0.940625 ± 0.004086 | 0.910815 ± 0.003687 | 0.984464 ± 0.001000 | 0.797522 ± 0.016030 |
| Dual trace | 0.940625 ± 0.001432 | 0.912077 ± 0.001519 | 0.982034 ± 0.001563 | 0.801117 ± 0.010572 |

Dual trace has identical mean accuracy, macro-F1 +0.001263, macro TPR@1%FPR
+0.003595, and macro-AUROC -0.002430. It does not beat the strong baseline
consistently by seed: macro-F1 improves only for seed 42, accuracy improves
only for seed 42, and macro-AUROC is lower for all three seeds. The current
evidence therefore does **not** support a stable overall improvement claim.

The dual-trace runs also reuse fixed seed-42 baseline and lexical checkpoints,
so the seed-wise differences are not fully paired end-to-end retraining runs.

## Local result sources

- `results/pasted_race/fourclass_baseline_official_seed42_lr29e-6/2026-09-14_21-33-26/test_metrics.json`
- `results/pasted_race/fourclass_baseline_official_seed2026_lr29e-6/2026-09-14_22-03-37/test_metrics.json`
- `results/pasted_race/fourclass_baseline_official_seed3407_lr29e-6/2026-09-13_17-28-59/test_metrics.json`
