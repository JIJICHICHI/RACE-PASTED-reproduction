# Group-safe RACE baseline with official hyperparameters

This run keeps the locally rebuilt `group overlap = 0` English HART four-class
split fixed and changes only the baseline training hyperparameters to the
official checkpoint settings:

- seed: `3407`
- learning rate: `2.9e-5`
- batch size: `16`
- maximum epochs: `20`
- linear warm-up ratio: `0.1`
- early-stopping patience: `5`
- checkpoint selection: validation macro-F1

Training stopped after epoch 19. The selected checkpoint is epoch 14, whose
validation macro-F1 is `0.920429`.

## Test results

| Metric | Seed 42 / lr 2.5e-5 | Official seed 3407 / lr 2.9e-5 | Change |
|---|---:|---:|---:|
| Accuracy | 0.933125 | **0.943438** | +0.010313 |
| Macro-F1 | 0.902334 | **0.911599** | +0.009265 |
| Macro-AUROC | 0.983673 | **0.985618** | +0.001944 |
| Macro TPR@1%FPR | 0.774694 | **0.808134** | +0.033440 |

## TPR@1%FPR by class

| Class | Seed 42 baseline | Official-hyperparameter run | Change | Paper strict group-aware |
|---|---:|---:|---:|---:|
| Human | 0.987500 | **0.993750** | +0.006250 | 0.991300 |
| Polished | 0.720000 | **0.851250** | +0.131250 | 0.862500 |
| Generated | **0.659317** | 0.640114 | -0.019203 | 0.752000 |
| Humanized | 0.731959 | **0.747423** | +0.015464 | 0.670000 |
| Macro | 0.774694 | **0.808134** | +0.033440 | 0.819000 |

The official hyperparameters substantially improve this local group-safe
baseline, particularly the Polished low-FPR recall. The remaining macro
TPR@1%FPR gap to the paper is `1.09` percentage points and is concentrated in
Generated. Macro-AUROC is `98.56%`, which is higher than the paper's reported
strict group-aware `96.59%`; therefore the remaining discrepancy is not a
uniform degradation and should not be attributed solely to optimization.
The local split still differs from the paper's exact group-aware partition.

## Included files

- `args.json`: resolved training configuration.
- `test_metrics.json`: complete test metrics emitted by `train.py`.
- `validation_history.json`: per-epoch training loss and validation metrics.

The 603 MB checkpoint and 42 MB per-example prediction dump remain local and
are intentionally excluded from the source repository.
