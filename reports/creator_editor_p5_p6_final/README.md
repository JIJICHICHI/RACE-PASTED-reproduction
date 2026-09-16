# Creator/Editor P5–P6 Final Results

All cells use the fixed group-safe split and seeds 42/2026/3407. Standard deviations are sample standard deviations.

## Primary metrics (mean ± std)

| Method | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |
|---|---:|---:|---:|---:|
| RACE | 0.940625 ± 0.004086 | 0.910815 ± 0.003687 | 0.984464 ± 0.001000 | 0.797522 ± 0.016030 |
| RACE+Editor | 0.943229 ± 0.008819 | 0.914408 ± 0.009454 | 0.984484 ± 0.000601 | 0.824065 ± 0.005164 |
| RACE+Creator | 0.945833 ± 0.004944 | 0.917548 ± 0.006583 | 0.984624 ± 0.000795 | 0.826446 ± 0.017695 |
| RACE+Creator (no fusion) | 0.944375 ± 0.003549 | 0.912616 ± 0.003633 | 0.984062 ± 0.002393 | 0.808112 ± 0.023886 |
| RACE+Creator+Editor | 0.940833 ± 0.006602 | 0.907585 ± 0.011878 | 0.984779 ± 0.001207 | 0.810296 ± 0.021270 |
| Full structure, lambda=0 | 0.944271 ± 0.002345 | 0.913975 ± 0.003340 | 0.985662 ± 0.000796 | 0.827617 ± 0.006776 |

## Interpretation of the lambda-zero control

The lambda-zero cell retains the complete Creator/Editor branch and fusion
architecture but sets the Creator-retention, polishing-trace, and
humanization-trace loss weights to zero. It is therefore not the original RACE
model: the additional projections and fusion parameters are still optimized by
the four-class classification objective. They can act as extra learnable
feature channels even though they are not constrained to represent their named
continuous targets.

This control reaches `82.76±0.68%` Macro TPR@1%FPR, compared with
`79.75±1.60%` for RACE and `81.03±2.13%` for the fully supervised
Creator+Editor model. Consequently, the low-FPR improvement cannot be
attributed to Creator/Editor continuous supervision: the extra architecture,
parameterization, and altered optimization path are already sufficient to
explain it. Adding all continuous objectives does not add another gain and may
over-constrain the representation.

One plausible mechanism is gradient conflict. Classification, Creator
retention, polishing modification, and humanization modification can request
different updates to shared parameters. Their weighted gradient sum can move
the representation toward accurate continuous regression without improving
the extreme score ordering required at 1% FPR. This experiment alone does not
prove gradient conflict, however; label noise, loss scaling, checkpoint timing,
or redundant targets remain alternative explanations. Direct gradient-cosine
measurements would be required to establish that mechanism.

Creator-only still has the best Accuracy (`94.58%`) and Macro-F1 (`91.75%`),
while Editor improves low-FPR TPR relatively consistently. Thus the evidence
supports some value from individual branches, but does not support a claim that
joint Creator+Editor continuous supervision causes the best low-FPR result.

## Primary metrics by seed

| Method | Seed | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |
|---|---:|---:|---:|---:|---:|
| RACE | 42 | 0.935937 | 0.906799 | 0.983910 | 0.779082 |
| RACE+Editor | 42 | 0.949375 | 0.921729 | 0.984799 | 0.819827 |
| RACE+Creator | 42 | 0.951250 | 0.925149 | 0.984853 | 0.817838 |
| RACE+Creator (no fusion) | 42 | 0.945937 | 0.916550 | 0.985078 | 0.817355 |
| RACE+Creator+Editor | 42 | 0.947812 | 0.917767 | 0.985132 | 0.812446 |
| Full structure, lambda=0 | 42 | 0.946562 | 0.916693 | 0.986377 | 0.827716 |
| RACE | 2026 | 0.942500 | 0.914046 | 0.983863 | 0.805350 |
| RACE+Editor | 2026 | 0.947187 | 0.917761 | 0.983791 | 0.822551 |
| RACE+Creator | 2026 | 0.944688 | 0.913704 | 0.983740 | 0.814702 |
| RACE+Creator (no fusion) | 2026 | 0.946875 | 0.911908 | 0.981328 | 0.780985 |
| RACE+Creator+Editor | 2026 | 0.934688 | 0.894536 | 0.983436 | 0.788032 |
| Full structure, lambda=0 | 2026 | 0.944375 | 0.914985 | 0.984805 | 0.820793 |
| RACE | 3407 | 0.943438 | 0.911599 | 0.985618 | 0.808134 |
| RACE+Editor | 3407 | 0.933125 | 0.903735 | 0.984862 | 0.829817 |
| RACE+Creator | 3407 | 0.941562 | 0.913790 | 0.985279 | 0.846799 |
| RACE+Creator (no fusion) | 3407 | 0.940312 | 0.909389 | 0.985779 | 0.825995 |
| RACE+Creator+Editor | 3407 | 0.940000 | 0.910453 | 0.985770 | 0.830410 |
| Full structure, lambda=0 | 3407 | 0.941875 | 0.910246 | 0.985805 | 0.834343 |

## Class-wise F1 and TPR@1%FPR (mean ± std)

### RACE

| Class | F1 | TPR@1%FPR |
|---|---:|---:|
| human | 0.980881 ± 0.002098 | 0.992083 ± 0.002887 |
| polished | 0.923553 ± 0.006249 | 0.808750 ± 0.052545 |
| generated | 0.946023 ± 0.006026 | 0.640114 ± 0.009957 |
| humanized | 0.792802 ± 0.010478 | 0.749141 ± 0.002976 |

### RACE+Editor

| Class | F1 | TPR@1%FPR |
|---|---:|---:|
| human | 0.984326 ± 0.002379 | 0.995417 ± 0.001443 |
| polished | 0.925037 ± 0.015338 | 0.824583 ± 0.026877 |
| generated | 0.947119 ± 0.008943 | 0.720247 ± 0.051536 |
| humanized | 0.801152 ± 0.011874 | 0.756014 ± 0.010730 |

### RACE+Creator

| Class | F1 | TPR@1%FPR |
|---|---:|---:|
| human | 0.982999 ± 0.004568 | 0.995000 ± 0.000000 |
| polished | 0.931161 ± 0.006274 | 0.849583 ± 0.011204 |
| generated | 0.950800 ± 0.005084 | 0.694879 ± 0.070585 |
| humanized | 0.805231 ± 0.013452 | 0.766323 ± 0.016570 |

### RACE+Creator (no fusion)

| Class | F1 | TPR@1%FPR |
|---|---:|---:|
| human | 0.984503 ± 0.002902 | 0.994167 ± 0.000722 |
| polished | 0.931866 ± 0.006718 | 0.857917 ± 0.014050 |
| generated | 0.948039 ± 0.003757 | 0.627786 ± 0.109157 |
| humanized | 0.786055 ± 0.009150 | 0.752577 ± 0.008928 |

### RACE+Creator+Editor

| Class | F1 | TPR@1%FPR |
|---|---:|---:|
| human | 0.981940 ± 0.004447 | 0.993333 ± 0.001909 |
| polished | 0.925313 ± 0.011719 | 0.822917 ± 0.024822 |
| generated | 0.945759 ± 0.004114 | 0.672357 ± 0.079812 |
| humanized | 0.777328 ± 0.031276 | 0.752577 ± 0.000000 |

### Full structure, lambda=0

| Class | F1 | TPR@1%FPR |
|---|---:|---:|
| human | 0.985811 ± 0.001639 | 0.994583 ± 0.000722 |
| polished | 0.930864 ± 0.006287 | 0.849583 ± 0.020091 |
| generated | 0.947043 ± 0.001297 | 0.710289 ± 0.039666 |
| humanized | 0.792180 ± 0.006167 | 0.756014 ± 0.002976 |

## Auxiliary metrics

Complete Creator and direction-specific Editor MSE/correlation/AUROC/TPR fields are stored in `summary.csv`, `per_seed.csv`, and `summary.json`. A dash indicates that the corresponding branch is not present in that method.
