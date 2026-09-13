# PASTED-RACE Research Addendum

> Extended: focused Part 2/Part 3 addendum for the PASTED-RACE integration; the
> original project did not contain an `idea_report.md`.

## Part 2 — Method: Joint Lexical-Trace Fusion

The completed lexical-only baseline predicts a continuous rewrite score for
each valid EDU from `[h_i; h_root; h_i ⊙ h_root]`. The integrated model converts
these scores into a document-local attention distribution and pools the RGCN
EDU representations:

\[
\alpha_i=\operatorname{softmax}(r_i/\tau),\qquad
z_{lex}=\sum_i\alpha_i h_i.
\]

The binary RACE classifier receives
`[h_root; z_lex; h_root ⊙ z_lex]` through a fusion MLP. Training uses

\[
L=L_{CE}+\lambda_{lex}L_{MSE},
\]

where the first epoch retains lexical regression only, followed by joint
optimization with `lambda_lex=0.2`. Initialization comes from the completed
lexical-only seed-42 checkpoint; the new fusion layer and classifier are newly
initialized. Inference needs only the evaluated document and no human
reference.

## Part 3 — Experiment Design

The joint experiment uses exactly the existing group-safe 70/10/20
human-written/human-AI-polished splits. The primary document metrics are AUROC
and TPR at strict FPR below 1%; EDU MSE, Pearson, Spearman, AUROC, and TPR are
retained to detect whether classification training damages trace regression.
The lexical-only mean-EDU detector is the direct baseline. A later ablation may
remove lexical MSE or lexical fusion, but it is outside this first integration
run.

## Part 2 — Method: Four-Class Residual PASTED-RACE

The four-class extension preserves the original RACE root representation and
classifier. Lexical targets are partially observed: Human EDUs receive zero,
Polished EDUs receive aligned `1-BLEU4`, and Generated/Humanized EDUs are masked
out because PASTED-style human-to-AI rewrite intensity is undefined for them.

To avoid destroying a pretrained four-class decision boundary, lexical fusion
is residual:

\[
h_{final}=h_{root}+\gamma F([h_{root};z_{lex};h_{root}\odot z_{lex}]),
\qquad \gamma_0=0.
\]

The main objective uses all documents, while the auxiliary objective uses only
observed lexical targets:

\[
L=L_{CE}^{4class}+\lambda_{lex}L_{MSE}^{Human,Polished}.
\]

A newly retrained, group-safe RACE baseline checkpoint initializes the shared
encoder, RGCN, and four-class classifier. Reusing the legacy checkpoint would
leak its old training groups into the rebuilt test split. The lexical-only
checkpoint initializes only the lexical head; a frozen-head calibration stage
aligns it to the group-safe RACE feature space before joint fine-tuning.

## Part 3 — Four-Class Experiment Design

All 16,000 unique HART documents are reassigned with the existing 4,000-group
manifest, producing group-safe 70/10/20 splits. The direct baseline is the
original RACE architecture retrained on the rebuilt group-safe training split
and evaluated on this same rebuilt test split.
The main comparison reports accuracy, macro-F1, macro AUROC, per-class AUROC,
and strict per-class TPR@1%FPR. Particular attention is paid to whether Polished
improves without reducing Generated or Humanized low-FPR detection.

The first run uses one lexical calibration epoch, residual-gate initialization
at zero, and `lambda_lex=0.2`. Required ablations for later runs are no lexical
fusion and no lexical auxiliary loss; they are not run until the first full
four-class result is diagnosed.

## Part 2 — Method: AI-to-Humanized Lexical Trace

The reverse-direction experiment learns a separate scalar EDU trace rather
than forcing Human-to-Polished and AI-to-Humanized edits into one head. For an
AI-generated reference document and its same-group Humanized target, target
sentences are semantically aligned to reference sentences and assigned
`1-BLEU4`. AI-generated reference EDUs receive zero. The model architecture and
strict masked-mean MSE are otherwise identical to the completed PASTED lexical
experiment.

Keeping a separate head matters because the two operations have different
starting distributions and stylistic goals: polishing human prose is not the
same transformation as removing detectable AI style. The experiment therefore
tests whether the PASTED formulation transfers to humanization before any
dual-trace four-class fusion is attempted.

## Part 3 — AI-Humanization Experiment Design

The raw HART corpus contains 1,015 Humanized documents. Each has exactly one
same-group AI-generated reference, so reference selection is deterministic and
requires no model generation. The existing four-class group-safe manifest
produces 703/118/194 train/validation/test pairs. The model is trained with
seed 42, validation early stopping, and the same lexical-only architecture and
hyperparameters as Human-to-Polished. Report EDU MSE, Pearson, Spearman, AUROC,
TPR@1%FPR and document-level AUROC/TPR. Because the training set is smaller,
the train-validation trajectory is explicitly inspected for overfitting.

## Part 2 — Method: Dual-Trace Four-Class Fusion

The dual model retains two semantically distinct EDU regressors. The polishing
head predicts Human-to-Polished traces and the humanization head predicts
Generated-to-Humanized traces. Each head independently produces EDU attention
and pooled representations `z_p` and `z_h`:

\[
h_{final}=h_{root}+\gamma_pF_p([h_{root};z_p;h_{root}\odot z_p])
                    +\gamma_hF_h([h_{root};z_h;h_{root}\odot z_h]),
\qquad \gamma_p=\gamma_h=0\text{ initially}.
\]

Training uses disjoint masks and strict masked means:

\[
L=L_{CE}^{4class}+0.2L_{MSE}^{Human,Polished}
                    +0.2L_{MSE}^{Generated,Humanized}.
\]

Encoder, RGCN, and classifier weights come from the no-leak four-class
baseline. Each trace head is copied only from its matching lexical checkpoint;
the two fusion projections are new.

## Part 3 — Dual-Trace Experiment Design

Compare the group-safe baseline, polishing-only residual fusion, and dual-trace
residual fusion on the identical 16,000-document split. The main diagnostic is
whether the second branch recovers Humanized AUROC/TPR@1%FPR without erasing
the Polished gain. Report both heads' EDU metrics and both learned residual
gates separately. One calibration epoch updates only the two trace heads;
later epochs optimize four-class CE plus both masked MSE losses.
