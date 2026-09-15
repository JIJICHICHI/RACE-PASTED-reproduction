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

## Part 2 — Method: Creator Retention and Editor Modification

The public `gyc-nii/CAS-CS-and-dual-head-detector` repository currently
publishes the multi-task dataset but not detector code. Its reusable supervision
contract is a document-level SciBERT/BERT-Sci BERTScore regression target plus
token labels. We adapt the regression target, not an unavailable architecture.

For each group-safe RACE pair, Creator Retention is the unrescaled BERTScore
Recall whose reference is the original creator text and whose candidate is the
final text:

\[
CR_{H\rightarrow P}=R_{BERT}(H,P),\qquad
CR_{G\rightarrow Hu}=R_{BERT}(G,Hu).
\]

Recall averages, over creator/reference tokens, the maximum contextual cosine
similarity to any final-text token. Consequently it asks how much creator
content survives, rather than what fraction of the final text was human-written.
Unedited Human and Generated examples use the exact self-retention target 1.

The detector receives only the final document. A document head predicts
`sigmoid(f_CR(h_root))`; its paired source is used only to construct the offline
target. The existing two EDU `1-BLEU4` heads become direction-specific
subheads of the Editor Modification branch. Their question remains how much
the editor changed each EDU. The combined representation is initialized as:

\[
h_{final}=h_{root}+\gamma_pF_p+\gamma_hF_h+\gamma_cF_c,
\qquad \gamma_p=\gamma_h=\gamma_c=0,
\]

where `F_c` receives `h_root` together with its retention-gated version. The
objective is

\[
L=L_{CE}^{4class}+\lambda_cL_{MSE}^{CR}
 +\lambda_pL_{MSE}^{1-BLEU(H,P)}
 +\lambda_hL_{MSE}^{1-BLEU(G,Hu)}.
\]

This gives the branches non-duplicated semantics: Creator models source-content
retention at document level; Editor models local editing intensity at EDU level.

## Part 3 — Creator/Editor Experiment Design

Use the existing 4,000-group manifest and never split members of one source
group across train/validation/test. Compare (1) strong RACE, (2) Editor-only
dual trace, and (3) Creator Retention + Editor Modification under the same
split and optimization settings. Report four-class macro-F1/AUROC and strict
TPR@1%FPR, Creator MSE/Pearson/Spearman, both Editor regression metric sets,
and all three learned residual gates. Required ablations are Creator auxiliary
loss without Creator fusion, Creator fusion without Editor fusion, and the full
model. Retention labels must record model name, unrescaled setting, direction,
and reference ID so their provenance is auditable.

## Part 3 — Official-style Trace Stability Control

To remove training-recipe confounds from the earlier single/dual results, run
both trace variants with seeds 42, 2026, and 3407 under the exact strong RACE
outer-loop recipe: four-class CE plus SupCon (`temperature=0.07`), exact
stratified batches, learning rate `2.9e-5`, batch size 16 for train/evaluation,
20 maximum epochs, linear warmup 0.1, weight decay 0.01, gradient clipping 1.0,
patience 5, and validation macro-F1 checkpoint selection. Each run loads the
strong no-leak baseline checkpoint trained with the same seed.

The only non-baseline objectives are intrinsic to the compared methods:
single trace adds `0.2 L_polish`; dual trace adds
`0.2 L_polish + 0.2 L_humanize`. All epochs use the joint objective and trace
fusion is available from epoch 0, with zero-initialized gates preserving the
baseline logits at initialization. Compare per-seed differences and three-seed
mean ± sample standard deviation; do not call a gain stable unless it appears
across seeds rather than only in the aggregate mean.

## Part 3 — Staged Creator-Retention Signal Validation

After the official single/dual trace controls finish, validate Creator
Retention before training or fusing a Creator branch. P2 contains only genuine
same-group edited pairs: `Human -> Polished` and
`Generated -> Humanized`. Unedited self-pairs are excluded so an artificial
target of one cannot create a trivial class shortcut.

For each edited document, compare unrescaled SciBERT contextual-token recall
with the mean of its valid EDU `1-BLEU4` Editor Modification targets. Report
Pearson and Spearman correlations separately for the two editing directions,
as well as distributions overall and within Arxiv, Essay, News, and Writing.
Pair counts, missing targets, non-finite values, and group provenance are part
of the audit. A correlation close to negative one is evidence that the signal
is largely redundant; a materially weaker relationship supports proceeding.

P3 tests whether retention is predictable from final text alone and compares
`h_i`, `[h_i; h_root]`, and `[h_i; h_root; h_i * h_root]` using MSE, Pearson,
and Spearman. P2 non-redundancy and P3 learnability remain interpretation
criteria, but are no longer execution gates: the complete P3–P6 matrix is run
so weak or redundant signals can be documented rather than hidden by early
termination.

Every newly trained condition uses seeds 42, 2026, and 3407. P3 therefore has
nine runs. Its matching-seed strong RACE checkpoint initializes the structural
encoder/RGCN, but its objective is Creator MSE on edited final texts only and
the primary checkpoint criterion is minimum validation MSE. The strongest P3
input by three-seed validation performance initializes the Creator head for
P5.

P5 runs three full Creator+Editor models under the official RACE outer-loop
contract. P6 compares RACE, Creator-only, Editor-only, and Creator+Editor and
adds (i) Creator supervision without Creator fusion and (ii) full extra
structure with all continuous-loss lambdas set to zero. Existing P1, P4, and
P5 cells are reused, leaving nine new P6 runs rather than retraining equivalent
models. All classification conditions report per-seed results and mean ±
sample standard deviation.

## Part 3 — End-to-End Seed-Matched Trace Control

The first official-style stability table varied the strong RACE initialization
and joint-training randomness while fixing both trace-only heads to seed 42.
To measure the complete pipeline under one optimizer contract, independently
retrain polishing and humanization trace heads for all three seeds using the
identical group-safe direction-specific datasets and all applicable strong-RACE
settings, including learning rate `2.9e-5`, batch size 16, 20 epochs, warmup
0.1, weight decay 0.01, gradient clipping 1.0, and patience 5. The old seed-42
trace heads at `2.5e-5` remain historical controls only.

Then rerun single and dual joint models for all three seeds. For each
seed `s`, the RACE backbone/classifier, trace head or heads, stratified sampler,
dropout, and joint optimization all use `s`. The joint phase retains the exact
official RACE outer loop and method-specific MSE weights used in the earlier
control. Results are stored separately and compared against both the paired
strong baseline and the
fixed-trace-initialization table, preventing the two stability claims from
being conflated.
