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
