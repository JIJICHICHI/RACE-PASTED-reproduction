# User Requirements

## Stage E Coding

- Base code: implement LE-RACE on top of the original RACE backup at `/home/dx/RACE_code_backups/RACE_code_20260617_222052`, not on the modified Tail/GZ/Trace experiment tree.
- Working directory: `/home/dx/RACE_LE_RACE`.
- Environment: reuse conda environment `race`.
- Data: reuse existing data through `/home/dx/RACE_LE_RACE/data -> /home/dx/RACE/data`.
- Scope: first implement LE-RACE only; do not implement LEAR-RACE in this pass.
- Metrics: keep the double-softmax fix as a metric sanity correction.
- Validation: run fast smoke tests automatically; do not start full long training unless requested.
- Git: no remote push unless explicitly requested.

## PASTED-RACE Minimal Experiment

- Scope: only `human_written -> human_ai_polished`, matching PASTED's AI rewriting setting.
- Exclude `ai_generated` and `ai_humanized` from this experiment.
- Target: continuous EDU lexical trace score derived from `1-BLEU4`.
- Objective: masked MSE only; no RACE four-class CE, agent head, MIL, sparse loss, or trace fusion.
- Split: must be group-safe by `group_id`.
- Execution: run smoke tests and the requested full single-seed experiment locally on the RTX 4090 D.
- Git: do not initialize or push a remote repository.

## PASTED-RACE Joint Integration

- Continue from the completed minimal lexical experiment and integrate its EDU
  trace predictions into the RACE document classifier.
- Keep the same binary scope: `human_written` versus `human_ai_polished`; do not
  add AI-generated, AI-humanized, human rewriting, or four-class training.
- Initialize the shared encoder/RGCN/lexical head from the best lexical seed-42
  checkpoint, while newly initializing classification/fusion parameters.
- Train with one lexical-only warm-up epoch followed by joint document CE and
  masked lexical MSE (`lambda_lexical=0.2`).
- Automatically run compatibility and GPU smoke tests, then the full requested
  seed-42 joint experiment; no Git push.

## Four-Class PASTED-RACE Integration

- Add the confirmed lexical trace branch to the original four-class RACE task:
  Human, Polished, Generated, and Humanized.
- Use PASTED lexical supervision only for Human (zero) and Polished
  (`1-BLEU4`); Generated and Humanized participate in four-class CE with lexical
  masks set to zero.
- Reassign all 16,000 documents through the existing group-safe 4,000-group
  manifest; do not evaluate on the leaking legacy split.
- First retrain the original RACE architecture on the rebuilt group-safe split;
  initialize encoder/RGCN/classifier from that no-leak baseline checkpoint and
  only the lexical head from the lexical-only checkpoint. The legacy checkpoint
  may be used for smoke checks only, not formal test results.
- Preserve the original classifier at initialization through zero-initialized
  residual lexical fusion.
- Run baseline evaluation on the rebuilt split, automatic smoke tests, and the
  full seed-42 four-class experiment on RTX 4090 D; no Git push.

## AI-Humanization Lexical Trace Experiment

- Train a second, independent PASTED-style lexical trace model for the reverse
  editing direction: `ai_generated -> ai_humanized`.
- Use the unique same-`group_id` AI-generated document as each Humanized
  document's reference; do not call an LLM or generate new text.
- Assign AI-generated EDUs a zero target and Humanized EDUs an aligned
  continuous `1-BLEU4` target.
- Reuse the existing group-safe manifest so this experiment remains directly
  compatible with the four-class train/validation/test split.
- Train and evaluate the seed-42 lexical-only model first; do not overwrite the
  Human-to-Polished checkpoint or silently merge the two trace directions.
- Run locally on the RTX 4090 D; no Git push.

## Dual-Trace Four-Class Integration

- Integrate both independently trained lexical checkpoints into the group-safe
  four-class RACE model.
- Keep separate EDU heads and targets for `Human -> Polished` and
  `Generated -> Humanized`; do not collapse the directions into one score.
- Apply polishing MSE only to Human/Polished and humanization MSE only to
  Generated/Humanized; all 16,000 documents still receive four-class CE.
- Initialize encoder/RGCN/classifier from the no-leak baseline and initialize
  each trace head only from its matching lexical checkpoint.
- Use two independently zero-initialized residual gates, preserving the exact
  baseline classifier at initialization.
- Run data audit, GPU smoke test, and full seed-42 training locally; no Git push.
- After the seed-42 result, run joint-fusion stability experiments with seeds
  2026 and 3407 in separate output directories. Keep the data split, baseline
  checkpoint, both lexical checkpoints, architecture, and hyperparameters fixed.

## P0 Strong RACE Baseline Multi-seed Control

- Run the original RACE architecture on the same group-safe four-class split
  with seeds `42`, `2026`, and `3407`.
- Use the official checkpoint optimization settings: learning rate `2.9e-5`,
  batch size `16`, 20 maximum epochs, linear warm-up ratio `0.1`, patience `5`,
  and validation macro-F1 checkpoint selection.
- Reuse the completed seed-3407 run and train only the missing seed-42 and
  seed-2026 runs.
- Compare the baseline three-seed mean and sample standard deviation against
  the completed dual-trace three-seed results before claiming stable gains.

## Creator-Retention / Editor-Modification Integration

- Reference `gyc-nii/CAS-CS-and-dual-head-detector` only for its published
  multi-task data contract; its detector implementation is currently not public.
- Define Creator Retention as document-level, unrescaled BERTScore Recall from
  the original creator text to the final text. Use same-group
  `Human -> Polished` and `Generated -> Humanized` pairs; self-retention for
  unedited Human and Generated documents is exactly `1.0`.
- Compute retention labels offline with SciBERT. Never expose the paired source
  text to the detector at inference time.
- Keep the existing PASTED EDU `1-BLEU4` regressors as the Editor Modification
  branch, with polishing and humanization directions separately masked.
- Train the group-safe four-class classifier jointly with document-level
  retention MSE and both direction-specific EDU modification MSE losses.
- Initialize the RACE backbone/RGCN/classifier from the no-leak baseline, the
  two editor heads from their matching lexical checkpoints, and every new
  residual gate at zero.

## Official-style Single/Dual-Trace Multi-seed Control

- Run both polishing-only single-trace and polishing+humanization dual-trace
  four-class models with seeds `42`, `2026`, and `3407`.
- Match the official RACE optimization contract: CE + four-class SupCon at
  temperature `0.07`, exact `StratifiedBatchSampler`, learning rate `2.9e-5`,
  train/eval batch size `16`, 20 maximum epochs, linear warmup ratio `0.1`,
  weight decay `0.01`, gradient clipping `1.0`, patience `5`, and validation
  macro-F1 checkpoint selection.
- Initialize every joint run from the completed strong group-safe baseline of
  the same seed. Keep zero-initialized residual trace gates.
- Use all 20 epochs for the joint objective; do not add a separate lexical-only
  calibration epoch or delayed fusion epoch in this strict control.
- Keep method-specific auxiliary weights at `0.2`. The polishing and
  humanization trace heads reuse their existing matching seed-42 lexical-only
  checkpoints because no other independently trained trace checkpoints exist.
- Store all six runs in isolated output directories and report mean ± sample
  standard deviation against the paired strong baseline.

## Post-Control Creator-Retention Experiment Queue

- Do not interrupt or overlap the running official-style single/dual-trace
  multi-seed control with another GPU experiment. Start this queue only after
  all six control runs and their paired-baseline comparison are complete.
- Treat P0 (fixed group-safe protocol) and P1 (strong RACE seeds
  `42/2026/3407`) as complete; do not rerun them. Treat the running official
  single/dual trace control as P4; do not duplicate it afterward.
- P2 first computes formal, unrescaled SciBERT BERTScore Recall only for real
  edited same-group pairs `Human -> Polished` and `Generated -> Humanized`.
  Do not include artificial self-retention examples in this first signal
  analysis.
- For each direction, compare Creator Retention with the corresponding Editor
  Modification target `1-BLEU4` using Pearson and Spearman correlations. Report
  overall distributions and domain-stratified distributions for Arxiv, Essay,
  News, and Writing, together with pair counts and missing-value audits.
- P3 trains Creator Retention as a standalone regression task before any new
  four-class fusion. Report MSE, Pearson, and Spearman and test three inputs:
  `h_i`, `[h_i; h_root]`, and
  `[h_i; h_root; h_i * h_root]`. The detector must see only the final text;
  the source/reference remains label-generation-only.
- Complete P3 through P6 even if P2 finds high Creator/Editor correlation or
  P3 finds weak standalone learnability. Treat those findings as diagnostics,
  not stop conditions, because the user explicitly requested the full Creator
  Retention experiment matrix.
- Use seeds `42/2026/3407` for every newly trained P3, P5, and P6 condition.
  For P3, initialize the encoder/RGCN from the matching-seed strong RACE
  checkpoint, train with Creator MSE only on edited final texts, and select by
  minimum validation MSE (validation Spearman is the tie-breaker).
- P3 contains three inputs for each seed: EDU-only `h_i`,
  `[h_i; h_root]`, and `[h_i; h_root; h_i * h_root]`. Report document-level
  MSE/Pearson/Spearman and mean ± sample standard deviation.
- P5 Creator+Editor joint runs use the same split and official optimization
  contract as the strong baseline: CE + SupCon, exact stratified batches,
  `lr=2.9e-5`, batch 16, warmup 0.1, maximum 20 epochs, patience 5, and
  validation Macro-F1 checkpoint selection. Initialize the Creator head from
  the matching-seed best P3 input variant, the RACE components from the
  matching-seed strong baseline, and Editor heads from their existing matching
  direction checkpoints. Keep all residual gates zero-initialized.
- P6 must include `RACE`, `RACE+Creator`, `RACE+Editor`, and
  `RACE+Creator+Editor`, plus a `lambda=0` structural control that keeps the
  extra network parameters but removes continuous supervision.
- Reuse completed P1 for `RACE`, completed official P4 dual trace for
  `RACE+Editor`, and P5 for `RACE+Creator+Editor`; do not retrain duplicate
  cells. Newly train `RACE+Creator`, Creator loss without Creator fusion, and
  full extra structure with every continuous-loss lambda set to zero, each at
  all three seeds.
- Execute only one GPU training process at a time. Store each seed/condition in
  an isolated directory, retain only required best checkpoints, and produce a
  final table with per-seed values and mean ± sample standard deviation for
  Accuracy, Macro-F1, Macro-AUROC, Macro TPR@1%FPR, four class-wise F1/TPR,
  Creator MSE/Pearson/Spearman, and applicable Editor metrics.

## End-to-End Seed-Matched Trace Retraining

- Before P3 Creator Retention training, complete strict end-to-end trace seed
  matching for seeds `42/2026/3407`. The old seed-42 trace-only checkpoint used
  learning rate `2.5e-5`, so it is historical control only and must not be
  reused in the fully official-optimized table.
- Retrain both independent directions at every seed: Human-to-Polished lexical
  trace and Generated-to-Humanized lexical trace. Keep the datasets and model
  architecture fixed, and align applicable optimizer settings with strong
  RACE: learning rate `2.9e-5`, train/eval batch 16, 20 maximum epochs, warmup
  0.1, weight decay 0.01, gradient clipping 1.0, and patience 5. The objective
  remains masked regression MSE; CE, SupCon, and Macro-F1 selection do not apply
  to trace-only regression.
- The trace-only DataLoader shuffle generator must use the actual run seed,
  rather than a hard-coded 42, so the new checkpoints genuinely cover the
  requested preprocessing/training randomness.
- Rerun single-trace joint models for all three seeds with the matching-seed
  polishing checkpoint, and rerun dual-trace joint models for all three seeds
  with both matching-seed direction checkpoints. Keep the same-seed
  strong RACE initialization and the already established official joint
  optimization contract unchanged.
- Write the reruns to new `paired` output directories; do not overwrite the
  earlier fixed-seed42-trace controls. Report both tables with explicit scope:
  fixed trace initialization versus end-to-end seed-matched initialization.
- Run all twelve new jobs sequentially on one GPU, then resume the Creator
  Retention P3–P6 queue.
