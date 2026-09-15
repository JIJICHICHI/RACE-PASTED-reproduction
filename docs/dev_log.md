# Development Log - LE-RACE
> Created: 2026-07-06 | Last updated: 2026-07-06
> Related implementation guide: `docs/implementation.md`
> This file is append-only. Every code change should add a new log entry.

## Project Overview

| Item | Content |
|------|---------|
| Topic | LE-RACE: local evidence branch for RACE |
| Base code | Original RACE backup |
| Working tree | `/home/dx/RACE_LE_RACE` |
| Environment | conda `race` |
| Data | symlink to `/home/dx/RACE/data` |

## Implementation Progress

| Module | Files | Status | Time | Notes |
|--------|-------|--------|------|-------|
| Init | `docs/user_requirements.md`, `docs/dev_log.md` | Done | 2026-07-06 | Clean tree created from original backup |
| EDU utilities | `utils/edu_utils.py` | WIP | 2026-07-06 | Added traversal/span projection utilities; syntax check pending |
| Evidence labels | `utils/evidence_label_builder.py`, `scripts/generate_edu_evidence_labels.sh` | WIP | 2026-07-06 | Added first runnable weak-label builder; syntax check pending |
| Dataset | `utils/flexible_dataset.py` | WIP | 2026-07-06 | Added evidence labels/masks and n_valid_edu; syntax check passed |
| Model | `models/flexible_model.py` | WIP | 2026-07-06 | Added optional local evidence branch; syntax check passed |
| Training losses | `train.py` | WIP | 2026-07-06 | Added evidence BCE, MIL, sparse loss; syntax check passed |
| Metrics | `utils/metrics.py` | WIP | 2026-07-06 | Added input_type and mixed-class TPR; syntax check passed |
| Diagnostics | `utils/diagnostics.py` | WIP | 2026-07-06 | Added label/prediction summaries; syntax check passed |
| Configs/scripts | `configs/LE_RACE.json`, `configs/ablations/`, `scripts/` | WIP | 2026-07-06 | Added LE-RACE configs and label-generation script |

## Development Log

### 2026-07-06 - Initialize clean LE-RACE working tree

- Completed: copied original RACE backup into `/home/dx/RACE_LE_RACE`, linked existing data, copied `implementation.md` into `docs/implementation.md`.
- Issue: current `/home/dx/RACE` contains Tail/GZ/Trace experiment modifications and should not be used as the implementation base.
- Resolution: all LE-RACE coding will happen under `/home/dx/RACE_LE_RACE`.


### 2026-07-06 - Add EDU utilities and evidence label builder

- Completed: added `utils/edu_utils.py` for RST EDU traversal, span overlap, char-to-token mapping, and sentence-to-EDU label projection.
- Completed: added `utils/evidence_label_builder.py` and `scripts/generate_edu_evidence_labels.sh`.
- Issue: evidence generation depends on `spacy` and `en_core_web_sm`; the script raises a clear install message if unavailable.
- Resolution: keep the dependency explicit and validate before running full label generation.


### 2026-07-06 - Wire dataset, model, losses, metrics, configs

- Completed: modified `utils/flexible_dataset.py` to load ragged EDU evidence labels/masks while preserving original RACE behavior when fields are absent.
- Completed: modified `models/flexible_model.py` with optional `use_local_evidence` branch, evidence scorer, pooling, fusion projection, and warm-up behavior.
- Completed: modified `train.py` with evidence BCE, MIL, sparse loss, logits-based metrics, and checkpoint tie-breaker support.
- Completed: modified `utils/metrics.py` to support explicit logits/probs input and mixed-class TPR.
- Completed: added `utils/diagnostics.py`, `utils/split_dataset_group_aware.py`, `configs/LE_RACE.json`, and ablation configs.
- Validation: Python syntax checks passed for modified/new Python files.


### 2026-07-06 - Smoke tests

- Validation: `python -m py_compile` passed for `train.py`, `test.py`, `models/flexible_model.py`, `utils/flexible_dataset.py`, `utils/metrics.py`, `utils/edu_utils.py`, `utils/evidence_label_builder.py`, `utils/diagnostics.py`, and `utils/split_dataset_group_aware.py`.
- Validation: original RACE forward path works with `configs/RACE.json`; output logits shape `[2, 4]`, features shape `[2, 512]`.
- Validation: LE-RACE local evidence branch works on a small batch; output logits `[2, 4]`, `z_local` `[2, 512]`, per-document evidence probabilities match valid EDU counts.
- Validation: LE-RACE evidence BCE, MIL, and sparse losses are callable on the smoke batch.
- Issue: `spacy` is installed, but `en_core_web_sm` is missing. Evidence label generation needs that model or an approved fallback.


### 2026-07-06 - Install spaCy model and run LE-RACE training smoke test

- Completed: installed and verified `en_core_web_sm` in conda env `race`.
- Validation: evidence label builder smoke test succeeded on a tiny split under `/tmp/le_race_smoke/out`.
- Validation: generated records include `edu_evidence_labels`, `edu_evidence_masks`, `has_trace_ref`, `evidence_ref_id`, `evidence_positive_ratio`, and `evidence_masked_ratio`.
- Validation: 1-epoch tiny LE-RACE training smoke test completed and wrote `best_model.pth`, `test_metrics.json`, and `test_predictions.json` under `/tmp/le_race_smoke/results`.
- Note: tiny validation/test metrics are not meaningful because the smoke split contains too few classes; this run only verifies code execution.

## Known Issues

- Tiny smoke tests pass. Full evidence label generation has not been started yet; it may take nontrivial time on the full HART split.

## Run Instructions

### Environment

```bash
conda activate race
cd /home/dx/RACE_LE_RACE
```

- Activates the existing RACE environment and enters the clean LE-RACE working tree.

### Generate EDU Evidence Labels

```bash
bash scripts/generate_edu_evidence_labels.sh
```

- Parameters are defined inside the script: input split files under `data/hart_split/`, output under `data/hart_split_evidence/`, backbone `roberta-base`, and spaCy model `en_core_web_sm`.
- Output files: `train_graph.jsonl`, `val_graph.jsonl`, `test_graph.jsonl`, and `evidence_label_stats.json`.



### 2026-07-06 21:19 — 新增 FCE-RACE factorized creator/editor 实现
- **完成内容**：在 `models/flexible_model.py` 中新增 `use_factorized_ce` 分支，包含 creator head、pre-RGCN EDU expression editor branch、双 query attention pooling、local variation features 和二维 factorized logits 组合。
- **训练逻辑**：在 `train.py` 中新增 creator/editor 标签派生与辅助 CE loss，SupCon 仍作用在 root/creator representation。
- **配置**：新增 `configs/FCE_RACE.json`，使用原始 `data/hart_split`，不依赖 EDU weak evidence labels。
- **原因**：LE-RACE full 和消融显示 EDU evidence/fusion 会伤害 Generated 的低误报排序，因此主线转向 creator/editor 显式因子化。
- **运行说明更新**：新增 FCE-RACE 训练命令：`conda run --no-capture-output -n race python train.py --config configs/FCE_RACE.json`。


### 2026-07-07 11:06 — Original RACE creator/editor 诊断实验
- **完成内容**：新增 `scripts/diagnose_role_axes.py`，基于 Original RACE 的 `test_predictions.json` 直接计算 creator/editor 轴级指标、四个条件二分类任务、四分类错误分解和 role margin。
- **完成内容**：新增 `scripts/hroot_role_probe.py`，冻结 Original RACE checkpoint，抽取 train/val/test 的 `h_root` features，并训练 creator/editor 两个 logistic-regression probe。
- **输出结果**：`results/diagnostics/original_race_role_axes/role_axis_diagnostics.{json,md}` 和 `results/diagnostics/original_race_hroot_probe/hroot_role_probe.{json,md}`。
- **主要发现**：Original RACE 的 creator 轴低误报性能明显强于 editor 轴；在固定 `creator=LLM` 的 `LLM-Generated vs Humanized` 子任务上，editor TPR@1%FPR 仅为 30.92%，h_root editor probe 的 test TPR@1%FPR 也只有 43.72%。
- **运行说明更新**：新增 Original RACE role-axis 诊断和 h_root probe 命令，均不重新训练 RACE 主模型。

### Original RACE Role-Axis Diagnostics
```bash
conda run -n race python scripts/diagnose_role_axes.py   --predictions /home/dx/RACE/results/gz_trace/original_rerun_bs16_lr25e-6/2026-07-01_23-18-19/test_predictions.json   --output-dir results/diagnostics/original_race_role_axes
```
- **运行后会发生什么**：读取 Original RACE 测试集预测，不重新训练模型，派生 creator/editor 概率并计算轴级、条件二分类、错误分解和 margin 诊断。
- **输出什么**：`role_axis_diagnostics.json` 和 `role_axis_diagnostics.md`。

### Original RACE h_root Linear Probe
```bash
conda run --no-capture-output -n race python scripts/hroot_role_probe.py   --checkpoint /home/dx/RACE/results/gz_trace/original_rerun_bs16_lr25e-6/2026-07-01_23-18-19/best_model.pth   --output-dir results/diagnostics/original_race_hroot_probe   --batch-size 32   --num-workers 4
```
- **运行后会发生什么**：冻结 Original RACE，抽取 train/val/test 的 `h_root` features，然后训练 creator/editor 两个轻量 logistic-regression probe。
- **输出什么**：`hroot_role_probe.json` 和 `hroot_role_probe.md`。


### 2026-07-07 21:10 — HART creator/editor 层级相似性验证
- **完成内容**：新增 `scripts/analyze_role_hierarchy_similarity.py`，冻结 Original RACE checkpoint 抽取 `h_root` features，验证 HART 表示空间是否满足 `same-label > shared-role > no-shared-role` 的层级相似性。
- **实现修正**：首次版本对 all split 枚举所有 pair，16000 条样本会产生一亿级 pair，速度不合理；已改为每种 pair 关系采样最多 200000 对，class similarity matrix 仍按类别 block 精确计算。
- **输出结果**：`results/diagnostics/original_race_role_hierarchy/role_hierarchy_similarity.{json,md}`，缓存特征文件为 `hroot_features_all_splits.npz`。
- **主要发现**：test split 上 `same_label=0.9019`、`shared_role=0.4047`、`no_shared_role=-0.2393`，`same_label > shared_role` 准确率 95.79%，`shared_role > no_shared_role` 准确率 93.01%；all split 上对应准确率为 98.11% 和 95.62%。
- **解释**：HART/RACE 表示空间确实存在 creator/editor 层级相似性，creator 共享关系略强于 editor 共享关系，可作为 FAR-RACE 多层级对比学习的 preliminary evidence。


### 2026-07-07 21:35 — Frozen pretrained encoder 层级相似性控制实验
- **完成内容**：新增 `scripts/analyze_pretrained_hierarchy_similarity.py`，直接读取 HART 原文，用 frozen pretrained encoder 表示验证 creator/editor 层级关系，不经过 HART 四分类训练。
- **已完成模型**：`roberta-base` CLS、`roberta-base` mean pooling、`microsoft/deberta-v3-base` mean pooling、`intfloat/e5-base-v2` mean pooling（E5 使用 `passage: ` 前缀）。
- **环境处理**：为 DeBERTa tokenizer 安装 `sentencepiece` 与 `tiktoken`；DeBERTa 使用 `--use-safetensors` 绕开当前 torch 版本对 `.bin` 权重的安全限制。
- **输出目录**：`results/diagnostics/pretrained_roberta_base_cls/`、`results/diagnostics/pretrained_roberta_base_mean/`、`results/diagnostics/pretrained_deberta_v3_base_mean_testonly/`、`results/diagnostics/pretrained_e5_base_v2_mean_testonly/`。
- **主要发现**：frozen pretrained 表示中存在弱但一致的层级趋势，尤其 DeBERTa test split 上 `same_label=0.9448`、`shared_role=0.9375`、`no_shared_role=0.9160`，`shared_role > no_shared_role` accuracy 为 66.91%；相比 Original RACE h_root 的 93.01%，说明数据本身有弱层级结构，而 RACE 训练显著放大了这种结构。


### 2026-07-07 22:34 — FAR-RACE factor contrast loss 实现
- **完成内容**：参考 FAID 的“分类 CE + 多层级对比学习”训练思想，在原始 RACE 派生工作树 `/home/dx/RACE_LE_RACE` 中新增 FAR-RACE factor contrast loss。
- **模型改动**：`models/flexible_model.py` 新增 `ProjectionHead`，并在 `RACEModel` 上增加 `proj_y/proj_c/proj_e` 三个 L2-normalized projection heads，分别从 `h_root` 投影到 final-label、creator、editor 三个对比空间。
- **训练改动**：`train.py` 新增 `use_factor_contrast` 路径，训练目标为 `CE(final_label) + beta * (lambda_y * SupCon(z_y, y) + lambda_creator * SupCon(z_c, creator) + lambda_editor * SupCon(z_e, editor))`；未加入 creator/editor role classification CE。
- **标签映射**：`creator = labels >= 2`，`editor = labels in {1, 2}`，对应 Human/Human、Human/LLM、LLM/LLM、LLM/Human 四类分解。
- **配置**：新增 `configs/far_race/FAR_RACE_factor_contrast_beta010.json` 和 `configs/far_race/FAR_RACE_factor_contrast_beta005.json`，分别对应 `beta=0.10` 与 `beta=0.05`；其余核心训练设置保持 `batch=16, lr=2.5e-5, seed=3407`。
- **验证**：`conda run -n race python -m py_compile models/flexible_model.py train.py test.py` 通过；小 batch smoke test 前向通过，输出 `logits=(8,4)`、`z_y/z_c/z_e=(8,128)`，CE 与三项 SupCon loss 均为有限值。
- **运行说明更新**：新增 FAR-RACE 两组单 seed 训练命令。

### FAR-RACE Factor Contrast Training
```bash
HF_HOME=/home/dx/.cache/huggingface HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers conda run --no-capture-output -n race python train.py --config configs/far_race/FAR_RACE_factor_contrast_beta010.json
```
- **运行后会发生什么**：使用原始 HART split 和 RACE GNN backbone 训练 FAR-RACE，loss 为四分类 CE 加 final-label/creator/editor 三个 SupCon factor loss。
- **输出什么**：结果写入 `results/far_race/factor_contrast_beta010/<timestamp>/`，包含 `best_model.pth`、`args.json`、`metadata.json`、`logs/`、`test_metrics.json` 和 `test_predictions.json`。

```bash
HF_HOME=/home/dx/.cache/huggingface HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers conda run --no-capture-output -n race python train.py --config configs/far_race/FAR_RACE_factor_contrast_beta005.json
```
- **运行后会发生什么**：与上一条相同，但将 factor contrast 总权重从 `beta=0.10` 降到 `beta=0.05`，用于验证对比约束强度是否过大。
- **输出什么**：结果写入 `results/far_race/factor_contrast_beta005/<timestamp>/`。


### 2026-07-08 — FAR-RACE direct factor SupCon 失败后的诊断与 soft hierarchy 实现
- **背景**：`FAR_RACE_factor_contrast_beta010/beta005` 单 seed 结果显示 direct factor SupCon 明显损害低误报性能，尤其 `Generated TPR@1%FPR` 从 Original 的 72.15 降到 58.55/61.20。
- **结论记录**：直接把 same creator/editor 样本作为完整 SupCon positives 过强；这些样本虽然共享角色，但在四分类 provenance 任务里仍是 hard negatives。
- **代码改动**：`models/supcon_loss.py` 新增 `WeightedSupConLoss`，支持连续 pair weights；`train.py` 新增 `build_soft_hierarchy_pair_weights()`，其中 same-label=1.0，shared creator/editor but different final label=`shared_role_weight`，默认 0.1，no-shared-role=0。
- **训练改动**：`train.py` 新增 `use_soft_hierarchy_contrast`、`soft_hierarchy_beta`、`shared_role_weight`、`soft_hierarchy_warmup_epochs`；同时新增 `loss_factor_scaled` 和 `loss_soft_scaled` 日志，用于检查 `β * contrastive loss` 相对 CE 的量级。
- **诊断配置**：新增 `FAR_diag_final_supcon_only_beta005.json`、`FAR_diag_creator_supcon_only_beta005.json`、`FAR_diag_editor_supcon_only_beta005.json`，分别对应附件要求的 final-only / creator-only / editor-only 三组诊断。
- **Soft hierarchy 配置**：新增 `FAR_soft_hierarchy_beta010_shared010.json` 和 `FAR_soft_hierarchy_beta020_shared010.json`，对应 `soft_hierarchy_beta=0.01/0.02`、`shared_role_weight=0.1`。
- **验证**：`conda run -n race python -m py_compile train.py models/supcon_loss.py models/flexible_model.py` 通过；WeightedSupConLoss 小样例验证通过，权重矩阵符合 same-label=1.0、shared-role=0.1、otherwise=0。

### FAR-RACE Direct SupCon Diagnostics
```bash
HF_HOME=/home/dx/.cache/huggingface HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers conda run --no-capture-output -n race python train.py --config configs/far_race/FAR_diag_final_supcon_only_beta005.json
```
```bash
HF_HOME=/home/dx/.cache/huggingface HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers conda run --no-capture-output -n race python train.py --config configs/far_race/FAR_diag_creator_supcon_only_beta005.json
```
```bash
HF_HOME=/home/dx/.cache/huggingface HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers conda run --no-capture-output -n race python train.py --config configs/far_race/FAR_diag_editor_supcon_only_beta005.json
```
- **运行后会发生什么**：分别只启用 final-label、creator、editor 单项 SupCon，`beta=0.05`，用于定位 direct factor SupCon 中到底是哪一项破坏 Generated/Polished 的低误报排序。
- **重点观察**：`loss_ce`、`loss_factor_y/c/e`、`loss_factor_scaled`、四类 `TPR@1%FPR`，尤其 Generated 与 Polished。

### FAR-RACE Soft Hierarchy
```bash
HF_HOME=/home/dx/.cache/huggingface HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers conda run --no-capture-output -n race python train.py --config configs/far_race/FAR_soft_hierarchy_beta010_shared010.json
```
```bash
HF_HOME=/home/dx/.cache/huggingface HUGGINGFACE_HUB_CACHE=/home/dx/.cache/huggingface/hub TRANSFORMERS_CACHE=/home/dx/.cache/huggingface/transformers conda run --no-capture-output -n race python train.py --config configs/far_race/FAR_soft_hierarchy_beta020_shared010.json
```
- **运行后会发生什么**：启用 weighted SupCon，same-label 作为强 positive，shared-role 作为弱 positive，不再把 creator/editor 二分类样本当完整 positives。
- **重点观察**：`loss_soft_hierarchy`、`loss_soft_scaled`、Generated/Polished/Humanized 的 `TPR@1%FPR` 是否较 direct factor SupCon 恢复。

### 2026-09-12 — 启动 PASTED-RACE 最小实验

- **实验边界**：只使用 `human_written` 与 `human_ai_polished`，以连续 `1-BLEU4` EDU 标签训练 RACE 图编码器的回归头；不处理 AI-generated、人类改写或四分类融合。
- **数据审计**：原 `hart_split` 存在大量跨 split 的同源 `group_id`（train/val=1323、train/test=2350、val/test=695），直接使用会产生 source leakage；完整实验改为合并去重后按 group 重新做 70/10/20 划分。
- **实现文档**：已在 `docs/implementation.md` 追加 PASTED-RACE addendum，并通过实验覆盖、tensor shape 和文件完整性校验。
- **运行策略**：复用 conda `race` 与 RTX 4090 D；自动运行数据/训练 smoke test，成功后运行用户明确要求的完整单 seed 实验；不执行 Git 推送。

### 2026-09-12 — 实现连续 lexical trace 数据生成器

- **完成内容**：新增 `utils/lexical_trace_label_builder.py`，支持合并旧 split、按 `item_id` 去重、筛选完整 human/polished pair、按 domain 分层做 group-safe 70/10/20 划分、句子语义对齐、无平滑 `1-BLEU4` 计算与字符 overlap EDU 投影。
- **防泄漏检查**：生成器在写文件前强制断言三个 split 的 `group_id` 交集为空，并输出 `split_manifest.json`。
- **运行说明影响**：新增标签生成命令将在运行脚本完成后统一列入下方运行说明。

### 2026-09-12 — 修正 lexical trace 生成器语法

- **问题**：首次补丁中 `output` 与 `all_similarities` 两个声明意外落在同一行。
- **修正**：拆分为两个独立变量声明；数据和算法逻辑不变。

### 2026-09-12 — Dataset 接入连续 EDU 标签

- **完成内容**：修改 `utils/flexible_dataset.py`，读取 `edu_lexical_scores`/`edu_lexical_masks`，按 tokenizer 截断后的 `n_valid_edu` 同步截断或补零，并在 collate 中保持逐文档 ragged tensor list。
- **兼容性**：旧 RACE/LE-RACE 数据缺少新字段时自动生成全 0 mask，不改变原训练路径。

### 2026-09-12 — 模型增加 PASTED-style EDU 回归头

- **完成内容**：修改 `models/flexible_model.py`，新增 `use_lexical_trace` 开关、`lexical_trace_head` 和 `_compute_lexical_trace()`；对每个 RGCN EDU 节点拼接 `[h_i; h_root; h_i*h_root]` 后输出一个不经 sigmoid 的连续分数。
- **兼容性**：开关默认关闭；原 RACE、LE-RACE、FCE/FAR 分支保持原返回与训练逻辑。

### 2026-09-12 — 增加 PASTED-RACE 独立训练入口

- **完成内容**：新增 `train_pasted_race.py`，仅优化连续 EDU lexical trace 的严格 masked-mean MSE；验证/测试输出 MSE、Pearson、Spearman、EDU AUROC 与严格 `FPR < 1%` 下的 TPR。
- **复现与产物**：固定 seed 42，按验证 AUROC 保存 best checkpoint，并写出 resolved config、训练历史、最终指标和逐文档 EDU 预测。
- **配置与命令**：新增 `configs/pasted_race/PASTED_RACE_lexical.json`、标签生成/训练前台脚本，并在 README 增加可直接运行的完整实验与 smoke 命令。

### 2026-09-12 — 复用本机完整 RoBERTa 缓存

- **发现**：`roberta-base` 的新 Hub 别名命中了不完整下载并停在权重获取；本机已有完整的 `FacebookAI/roberta-base` snapshot。
- **调整**：配置与标签脚本改用等价的规范仓库名 `FacebookAI/roberta-base`，不改变模型架构或权重，避免重复下载约 500 MB。

### 2026-09-12 — 标签流水线 smoke test 通过

- **结果**：20 个同源 pair 成功重划为 14/2/4 groups，对应 28/4/8 documents；三个 split 均为 human/polished 1:1，缺失 reference 为 0。
- **标签质量快检**：训练集有效 EDU 1517，平均连续 target 0.3505，句子对齐平均余弦相似度 0.9896；生成文件和 split manifest 正常。
- **便捷性**：训练入口新增 `--data_dir` 与 `--output_dir` 覆盖参数，允许 smoke 数据和正式数据共用同一配置而不修改文件。

### 2026-09-12 — 正式 PASTED-RACE seed 42 实验完成

- **数据**：4000 个完整 human/polished pairs 按 group-safe 70/10/20 重划，得到 5600/800/1600 篇；三个 split 的 group overlap 均为 0，生成数据共 229 MiB。
- **训练**：RTX 4090 D、batch 16、RoBERTa 最后一层解冻、RGCN 2 层、最多 10 epochs；按验证 EDU AUROC 选择第 7 轮 checkpoint，checkpoint 约 494 MiB。
- **测试结果（EDU）**：MSE 0.03979、Pearson 0.89567、Spearman 0.84908、AUROC 0.99090、TPR@1%FPR 0.73437。
- **测试结果（文档汇总）**：mean valid EDU score 的文档 AUROC 0.99775、TPR@1%FPR 0.955；分域 TPR 为 arxiv 1.00、essay 1.00、news 0.96、writing 0.77。
- **指标补全**：训练入口的评估函数新增文档级 AUROC/TPR，并支持 `--eval_only` 从 best checkpoint 重写统一指标文件，无需重训。

### 2026-09-12 21:15 — 迭代 #1：确认 PASTED-RACE 联合融合设计

**改动原因**：lexical-only 测试文档 AUROC 0.99775，说明 EDU 连续监督有效，但 EDU 分数尚未参与 RACE 最终分类。
**改动内容**：
- `docs/idea_report.md`：新增聚焦的 Method/Experiment addendum，定义 lexical attention pooling 与联合目标。
- `docs/implementation.md`：补充模型接口、训练逻辑、tensor shape、指标和验证顺序。
- `docs/user_requirements.md`：记录用户确认的二分类联合融合范围与执行策略。
**预期效果**：保留可解释 EDU 连续分数，同时让 RACE root 表示补充局部 lexical 线索，形成端到端文档检测器。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待编码阶段更新

### 2026-09-12 21:18 — 迭代 #1：模型接入 lexical attention fusion

**改动原因**：独立 lexical score 只能作为外部均值检测器，无法与 RACE root 表示端到端互补。
**改动内容**：
- `models/flexible_model.py`：新增 `use_lexical_fusion`、temperature-softmax EDU pooling、`z_lexical`、三路交互融合 MLP 与 warm-up bypass；lexical-only 开关关闭融合时保持原路径。
**预期效果**：分类头同时利用全文结构表示和高 lexical-trace EDU 表示，并保留逐 EDU 可解释输出。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-13 00:08 — 迭代 #2：四分类 group-safe 基线训练完成

**训练结果**：正式基线训练与测试正常结束，最佳模型保存为 `results/pasted_race/fourclass_baseline_seed42/2026-09-12_23-36-32/best_model.pth`。测试 accuracy 0.933125、macro-F1 0.902334、macro-AUROC 0.983673、macro TPR@1%FPR 0.774694。
**分类结果**：Human/Polished/Generated/Humanized 的 F1 分别为 0.980564/0.914701/0.937792/0.776280；对应 TPR@1%FPR 为 0.9875/0.7200/0.659317/0.731959。
**配置更新**：`PASTED_RACE_fourclass_joint.json` 的 `baseline_checkpoint` 已指向上述无跨集合 source leakage 的正式基线，后续联合训练不再使用 legacy checkpoint。
**预期效果**：联合模型从同一 group-safe 数据划分上的强四分类边界初始化，确保与 baseline 的差异主要来自 lexical auxiliary loss 和 residual fusion。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-14 23:05 — 迭代 #10：固定 Creator Retention 后续实验队列

**改动原因**：用户要求按“先验证信号，再接模型”的顺序执行，并跳过已完成的公平划分和强基线实验。

**改动内容**：

- `docs/user_requirements.md`：记录 P2–P6 的严格执行顺序、进入下一阶段的信号有效性条件，以及必做的 `lambda=0` 结构控制。
- 明确 P2 只分析真实编辑 pair，不将 Human/Generated 自保留样本设为 1；必须分方向报告 Creator Retention 与 `1-BLEU4` 的 Pearson/Spearman 及领域分布。
- 当前正式单/双痕迹六组训练继续独占 GPU；本轮只固化后续队列，不启动 Creator 计算或训练。

**预期效果**：先判断 Creator Retention 是否提供独立于 Editor Modification 的信息，再决定是否投入四分类联合训练，避免无法归因的一次性堆叠。

**文档同步**：idea_report.md 已有 Creator/Editor 设计 | implementation.md 已有实现说明 | user_requirements.md 是

### 2026-09-14 22:55 — 迭代 #9 验证：官方设置 trace smoke tests

- `py_compile`、两个 JSON config 解析与 `git diff --check` 通过。
- 官方 `StratifiedBatchSampler` 检查：batch size 16，首批四类各 4 条，每轮 700 batches，与强 baseline 一致。
- 单痕迹 GPU smoke：CE、SupCon、polishing MSE 均为有限值并共同反传；第 1 步后 residual gamma 从 0 变为非零；checkpoint 重载和测试输出通过。
- 双痕迹 GPU smoke：CE、SupCon、polishing/humanization MSE 均为有限值并共同反传；两个 residual gamma 均从 0 变为非零；checkpoint 重载和双分支预测输出通过。
- smoke 子集只有 64/32 条，指标不用于模型结论。

**结论**：六组正式训练可以启动。

### 2026-09-14 22:56 — 迭代 #9 启动修正：固定 Hugging Face cache

首次正式启动在构建第一组 dataset tokenizer 时退出，尚未进入训练。原因是脚本继承了外层已有但不包含 RoBERTa 的 `HF_HOME`，同时 offline mode 禁止回退网络。现已改为与强 baseline 启动脚本相同的绝对 cache 路径 `/home/dx/.cache/huggingface`；失败目录没有 `metrics.json`，重启脚本会安全覆盖其 `resolved_config.json` 并正常执行该组。

### 2026-09-14 — 迭代 #8 验证：Creator/Editor smoke tests

- Python syntax：`creator_retention_label_builder.py`、dataset、model、four-class trainer 全部通过 `py_compile`；`git diff --check` 通过。
- Pair audit：train/val/test 分别解析 3503/518/994 个 edited pair，总计 5015，全部由同 split、同 `group_id` 的 Creator 解析。
- CPU label smoke：每个 split 取 1 group，以缓存 RoBERTa、长度 64 验证 BERTScore Recall 路径，输出分数有限且位于 `[0,1]`，group overlap 为 0。该 smoke 分数不是正式 SciBERT 标签。
- Dataset contract：retention score/mask 正确 collate 为 `[B]`。
- Initialization smoke：强 baseline 精确迁移 216 个参数，两个 Editor checkpoint 各迁移 4 个 head 参数；输出 `logits=(1,4)`、`creator_retention_scores=(1,)`，三个 residual gate 均严格为 0。
- Backward compatibility：旧 dual-trace 配置前向仍输出 `(1,4)`，不启用 Creator Retention，新增 gate 为 `None`。

**结论**：架构与数据链路 smoke test 通过。正式 SciBERT 全量标签生成及完整训练尚未启动，以免与正在运行的 P0 强 baseline 多种子任务争用 GPU。

### 2026-09-14 — 迭代 #7 结果：P0 强 RACE baseline 三种子

- seed 42：accuracy 0.935938，macro-F1 0.906799，macro-AUROC 0.983910，macro TPR@1%FPR 0.779082。
- seed 2026：accuracy 0.942500，macro-F1 0.914046，macro-AUROC 0.983863，macro TPR@1%FPR 0.805350。
- seed 3407：accuracy 0.943438，macro-F1 0.911599，macro-AUROC 0.985618，macro TPR@1%FPR 0.808134。
- 三种子 baseline 均值±样本标准差：accuracy 0.940625±0.004086，macro-F1 0.910815±0.003687，macro-AUROC 0.984464±0.001000，macro TPR@1%FPR 0.797522±0.016030。
- dual trace 对应均值：accuracy 0.940625，macro-F1 0.912077，macro-AUROC 0.982034，macro TPR@1%FPR 0.801117。

**结论**：dual trace 的平均 accuracy 与强 baseline 相同，macro-F1 仅高 0.001263，macro TPR@1%FPR 高 0.003595，但 macro-AUROC 低 0.002430；accuracy/F1 只在 seed 42 上胜出，AUROC 三个种子均更低。因此不能宣称稳定超过真正强的 RACE baseline。dual trace 还复用了固定 seed-42 初始化 checkpoints，并非完整端到端配对三种子。

**产物**：`reports/group_safe_official_baseline_multiseed/{README.md,summary.json}`。

### 2026-09-13 13:24 — 迭代 #4：实现双痕迹四分类 residual fusion

**改动原因**：单 polishing 分支改善 Polished 低误报检测，但不能监督 Humanized；独立 humanization 模型已证明第二方向可学习。
**改动内容**：
- `utils/build_pasted_race_fourclass.py`、`scripts/build_pasted_race_fourclass_dual.sh`：合并两个 target 数据源并生成隔离的 dual 数据；未配对 Generated 的 humanization mask 保持为 0。
- `utils/flexible_dataset.py`：加载、截断并 collate `edu_humanization_scores/masks`。
- `models/flexible_model.py`：增加独立 humanization head、attention pooling、fusion projection 和零初始化 residual gamma；两个 residual 从共同 root 表示计算并相加。
- `train_pasted_race_fourclass.py`：增加第二 checkpoint 映射加载、双 head calibration、双 masked MSE、分离指标与预测输出。
- `configs/pasted_race/PASTED_RACE_fourclass_dual_joint.json`、`scripts/train_pasted_race_fourclass_dual_joint.sh`：固定 seed 42、两个 loss 权重均为 0.2 和正式运行入口。
**数据结果**：dual 数据保持 16,000 篇、4,000 groups、train/val/test 11,200/1,600/3,200，group overlap 为 0；humanization 有效 EDU 为 61,052/10,294/16,404。
**预期效果**：保留 polishing 分支对 Polished 的收益，同时通过方向专属 head 改善 Humanized 概率排序与低 FPR 召回，且初始化时严格等于原四分类 baseline。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-13 13:05 — 迭代 #3 结果：AI-to-Humanized lexical trace

**数据结果**：成功生成 2,030 篇 group-safe 文档（1,015 对）；train/val/test 为 1,406/236/388 篇。三组 group overlap 均为 0，标签/掩码长度、reference 完整性和 Generated 零目标检查均通过。平均句对齐相似度 0.990291，训练集有效 EDU 61,052。
**训练结果**：完整 10 轮训练正常结束，按 validation EDU AUROC 选择第 8 轮。测试 EDU MSE 0.128451、Pearson 0.757476、Spearman 0.748049、AUROC 0.941799、TPR@1%FPR 0.474321；文档 AUROC 0.946806、TPR@1%FPR 0.525773。
**对比诊断**：该方向弱于 Human-to-Polished lexical 模型（文档 AUROC 0.997745、TPR@1%FPR 0.9550），与训练配对仅 703 对、Humanized 改写更异质的预期一致；validation 与 test 接近，暂未观察到明显 split-level 过拟合。
**产物**：`results/pasted_race/humanization_lexical_seed42/best_model.pt`、`metrics.json`、`history.json` 和 val/test predictions。
**结论**：独立 AI-to-Humanized 痕迹可学习，足以作为第二 lexical head 的初始化，但不应替换表现更强的 Human-to-Polished head；下一步应采用双分支而非共享单标量。

### 2026-09-12 23:50 — 迭代 #2：实现四分类分阶段联合训练器

**改动原因**：四分类需要双 checkpoint 安全迁移、partial lexical loss、原 RACE 校准保护及同 split 多类指标，二分类训练器无法直接满足。
**改动内容**：
- `train_pasted_race_fourclass.py`：新增 group-safe 数据加载、分层采样、baseline/shared 与 lexical-head 双迁移、一轮冻结校准、四类 CE+masked MSE、zero-gamma residual、macro/per-class/lexical 指标及预测输出。
**预期效果**：正式模型从无泄漏四分类 baseline 出发，只在有定义的类别学习 lexical target，同时量化 Polished 收益和其他类别退化。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待更新

### 2026-09-12 23:38 — 迭代 #2：smoke 后修正 checkpoint 资格

**改动原因**：2-epoch smoke 中 calibration 与首个 joint epoch 的 macro-F1 相同，严格 `>` 会保留 gamma=0 的 calibration checkpoint，无法保证正式产物实际启用融合。
**改动内容**：
- `train_pasted_race_fourclass.py`：calibration epoch 仅做校准与日志，不参与 best checkpoint/early-stop；至少完成一个 CE+MSE joint epoch 才可保存正式模型；缺类子集的 per-class AUROC 安全返回 NaN。
**预期效果**：正式 best checkpoint 必然来自 residual fusion 已参与训练的阶段，同时 smoke 子集不再产生无意义警告。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-09-14 — 迭代 #8：Creator Retention / Editor Modification 解耦

**改动原因**：现有 dual-trace 的两个分支都监督局部编辑强度，无法区分“原始 Creator 内容保留”与“Editor 修改程度”。用户指定参考 `gyc-nii/CAS-CS-and-dual-head-detector` 的 human-involvement/BERTScore 监督思想迁移到 RACE。

**参考仓库核验**：公开仓库的 Detector 代码仍为 `Coming Soon`；可复用内容是包含 `scibert_bertscore`、`token_labels`、`generated_text/source_text/prompt_text` 的多任务数据契约。因此本实现明确为监督思想迁移，不声称复刻其未公开模型。

**改动内容**：

- `docs/user_requirements.md`：记录 Creator Retention 定义、配对边界、离线标签与无源文本推理约束。
- `docs/idea_report.md`：新增 Creator/Editor 解耦公式、联合目标和消融设计。
- `docs/implementation.md`：新增数据、模型、训练及兼容性契约。
- `utils/creator_retention_label_builder.py`：实现 uniform-weight、unrescaled SciBERT BERTScore Recall；Human/Generated 自保留为 1，Polished/Humanized 使用同组原始 Creator 作为 reference。
- `utils/flexible_dataset.py`：加载并批处理文档级 retention score/mask/reference ID。
- `models/flexible_model.py`：新增 sigmoid Creator Retention 回归头、retention-conditioned residual fusion 和零初始化 gate。
- `train_pasted_race_fourclass.py`：新增 masked scalar MSE、Creator 指标/预测输出、校准阶段与联合损失接线。
- `train_pasted_race_fourclass.py`：stratified sampler 的 generator 改为使用配置 `seed`，避免新实验写 seed 3407 但采样仍固定为 42。
- `configs/pasted_race/PASTED_RACE_fourclass_creator_editor_joint.json`：新增强 baseline 初始化、官方优化设置的完整 Creator/Editor 实验配置。
- `scripts/build_creator_retention_data.sh`、`scripts/train_pasted_race_creator_editor.sh`：新增数据构建与训练入口。
- `README.md`：补充 Creator Retention 数据生成、训练命令和无源文本推理说明。

**预期效果**：Creator 分支学习源内容语义保留，Editor 分支学习 EDU 局部改写强度；三个 residual gate 均从零开始，加载后保持强 RACE 决策边界不变。

**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-14 — 迭代 #7：P0 强 RACE baseline 三种子控制实验

**改动原因**：现有双痕迹三种子结论仅相对 seed-42/lr-2.5e-5 固定基线成立；官方 seed-3407/lr-2.9e-5 单次强基线已显示更高的 Accuracy、AUROC 和 TPR，必须补齐同等三种子控制后才能判断改进是否稳定超过强 RACE。
**改动内容**：
- `docs/user_requirements.md`：记录 P0 强基线多种子范围与公平比较约束。
- `configs/pasted_race/PASTED_RACE_fourclass_baseline_official_seed42.json`：新增 seed 42、lr 2.9e-5 强基线配置。
- `configs/pasted_race/PASTED_RACE_fourclass_baseline_official_seed2026.json`：新增 seed 2026、lr 2.9e-5 强基线配置。
- `scripts/train_pasted_race_fourclass_baseline_official_multiseed.sh`：按顺序运行缺失的 42/2026 两组；已有 3407 结果不重复计算。
**预期效果**：得到相同 split、架构、学习率和评估协议下的三种子 RACE baseline 均值与样本标准差，并据此重新审计双痕迹方法的稳定收益声明。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 是

### 2026-09-13 — 迭代 #6 结果：官方超参数的 group-safe 四分类基线

**训练结果**：使用同一 `group overlap=0` 数据划分、官方 seed 3407、lr 2.9e-5 和 batch 16；第 14 轮取得最佳 validation macro-F1 0.920429，训练在第 19 轮触发 patience-5 早停。最佳模型测试 accuracy 0.943438、macro-F1 0.911599、macro-AUROC 0.985618、macro TPR@1%FPR 0.808134。
**相对 seed-42/lr-2.5e-5 group-safe baseline**：accuracy +0.010313、macro-F1 +0.009265、macro-AUROC +0.001944、macro TPR@1%FPR +0.033440。Polished TPR@1%FPR 从 0.720000 提升到 0.851250；Generated 从 0.659317 降到 0.640114。
**与论文严格 group-aware 结果比较**：macro TPR@1%FPR 仍低 0.010866，但 macro-AUROC 高 0.019718；差距主要集中在 Generated TPR@1%FPR（本地 0.640114，论文 0.752000）。这说明官方超参数显著缩小了低误报指标差距，但不同数据 partition 仍是不可忽略的变量。
**GitHub 轻量产物**：`reports/group_safe_official_seed3407_lr29e-6/` 包含完整 test metrics、resolved args、validation history 和对比说明；603 MB checkpoint 与 42 MB prediction dump 仅保留本地。

### 2026-09-13 — 迭代 #6：修正官方参数训练入口的缓存路径

**改动原因**：首次启动在 tokenizer 加载前被宿主环境已有的只读 `HF_HOME=/media/dx/文档/hf_cache` 阻断，尚未进入训练。
**改动内容**：
- `scripts/train_pasted_race_fourclass_baseline_official.sh`：显式固定 Hugging Face 缓存为已存在且可写的 `/home/dx/.cache/huggingface`。
**预期效果**：复用本机已有 RoBERTa 缓存并正常启动；模型、数据和训练超参数不变。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-09-13 — 迭代 #6：启动官方超参数的 group-safe 四分类基线

**改动原因**：当前 group-safe 基线使用 seed 42、lr 2.5e-5，需要单独检验论文官方 checkpoint 的 seed 3407、lr 2.9e-5 是否能缩小严格划分结果差距。
**改动内容**：
- `configs/pasted_race/PASTED_RACE_fourclass_baseline_official_seed3407.json`：保持同一 group-safe 数据、RACE 架构、batch 16、训练轮数和 macro-F1 checkpoint protocol，只改为官方 seed 3407 与 lr 2.9e-5，并使用独立输出目录。
- `scripts/train_pasted_race_fourclass_baseline_official.sh`：新增前台训练入口。
**预期效果**：获得与 seed-42/lr-2.5e-5 无泄漏基线可直接比较的官方超参数单种子结果；本实验不能消除当前数据 partition 与论文严格 partition 的差异。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 是

### 2026-09-13 14:20 — 迭代 #5 结果：双痕迹三种子稳定性

| Seed | 最佳轮次 | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |
|---:|---:|---:|---:|---:|---:|
| 42 | 5 | 0.942188 | 0.913799 | 0.983616 | 0.798011 |
| 2026 | 4 | 0.940313 | 0.911507 | 0.981996 | 0.792447 |
| 3407 | 3 | 0.939375 | 0.910925 | 0.980490 | 0.812894 |
| 均值 ± 样本标准差 | — | 0.940625 ± 0.001432 | 0.912077 ± 0.001519 | 0.982034 ± 0.001563 | 0.801117 ± 0.010572 |

**相对固定 group-safe baseline**：三种子均值 accuracy +0.007500、macro-F1 +0.009743、macro TPR@1%FPR +0.026423；macro-AUROC -0.001640。四类平均 F1 相对 baseline 的变化分别为 Human +0.000538、Polished +0.007262、Generated +0.008566、Humanized +0.022606。
**结论**：三个种子的 accuracy 和 macro-F1 均超过同一 baseline，说明双痕迹融合的分类提升在联合训练随机性下稳定；极低 FPR 指标的方差较大，尤其 Generated TPR@1%FPR，且 macro-AUROC 有小幅稳定代价。该结论仅覆盖联合融合阶段随机性，因为 baseline 和两个 lexical 初始化 checkpoint 固定为 seed 42。
**产物**：三个独立结果目录及 `results/pasted_race/fourclass_dual_multiseed_summary.md`。

### 2026-09-12 23:57 — 迭代 #2：新增四分类 baseline/joint 配置与脚本

**改动原因**：正式对比必须先得到相同 group-safe split 上的无泄漏原 RACE baseline，再运行 residual PASTED-RACE。
**改动内容**：
- `configs/pasted_race/PASTED_RACE_fourclass_baseline.json`：官方 RACE 结构、CE+SupCon、batch 16、lr 2.5e-5、seed 42、macro-F1 checkpoint。
- `configs/pasted_race/PASTED_RACE_fourclass_joint.json`：双 checkpoint、1 轮 calibration、lambda 0.2、zero-gamma residual、12 epochs。
- `scripts/train_pasted_race_fourclass_{baseline,joint}.sh` 与 `README.md`：新增前台运行入口和顺序。
**预期效果**：baseline 与 joint 使用完全相同的数据划分和指标选择，差异可归因于 lexical integration。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-12 23:43 — 迭代 #2：支持四类 batch 的空 lexical mask

**改动原因**：分层采样虽通常混合四类，但合法 batch 仍可能只含 Generated/Humanized，其 lexical mask 全为 0。
**改动内容**：
- `train_pasted_race.py::masked_mse`：新增 `allow_empty`，在无有效 partial target 时返回与预测图连接的零 loss；二分类默认仍严格报错。
**预期效果**：四分类联合训练不会因某个 batch 没有 Human/Polished 而中断，也不会错误监督无定义类别。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-12 23:39 — 迭代 #2：阻止 legacy checkpoint 级测试泄漏

**改动原因**：旧四分类 checkpoint 在 legacy train 上训练，而新 manifest 会把其中部分旧 train groups 分到新 test；直接迁移会让正式测试包含 checkpoint 已见样本。
**改动内容**：
- `docs/idea_report.md`、`docs/implementation.md`、`docs/user_requirements.md`：正式初始化改为先在新 split 重训官方 RACE 结构，再迁移该 no-leak baseline；旧 checkpoint 仅限 smoke。
**预期效果**：baseline 与 PASTED-RACE 都只从新 train 学习，正式 val/test 比较无 checkpoint-level source leakage。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待更新

### 2026-09-12 21:23 — 迭代 #1：训练器支持联合 CE + lexical MSE

**改动原因**：融合模型需要端到端文档监督、从 lexical-only checkpoint 迁移共享参数，并同时监控分类与回归是否互相伤害。
**改动内容**：
- `train_pasted_race.py`：新增共享权重迁移（排除 classifier/fusion）、一轮 lexical warm-up、`CE + lambda*MSE`、按分类 AUROC 选 checkpoint，以及 classifier AUROC/TPR/accuracy/F1 和预测概率输出。
**预期效果**：新分类头从干净初始化学习 root/local 融合，同时 lexical 辅助目标约束局部解释分支不坍缩。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待更新

### 2026-09-12 21:26 — 迭代 #1：新增联合实验配置与前台脚本

**改动原因**：需要固定可复现的迁移 checkpoint、loss 权重、warm-up 和融合超参数，并保持普通终端可见日志。
**改动内容**：
- `configs/pasted_race/PASTED_RACE_joint.json`：固定 seed 42、batch 16、lr 2e-5、lambda 0.2、1 轮 warm-up 和 lexical seed-42 初始化。
- `scripts/train_pasted_race_joint.sh`：新增联合训练前台命令。
- `README.md`：补充 lexical-only 到 joint integration 的运行路径和产物目录。
**预期效果**：实验可一键复现，且不依赖 tmux 或重新生成标签。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-12 23:03 — 迭代 #1：隔离子集 eval-only 产物

**改动原因**：兼容性检查携带 `--max_eval_samples` 时复用了正式 `metrics.json`/prediction 文件名，会暂时覆盖全量评估产物。
**改动内容**：
- `train_pasted_race.py`：子集 eval-only 自动写入 `metrics_subset.json`、`val_predictions_subset.jsonl` 和 `test_predictions_subset.jsonl`；全量 eval-only 仍写正式文件。
**预期效果**：后续 smoke/兼容性评估不会污染正式结果；本次被覆盖的 lexical 全量产物将立即由全量 eval-only 恢复。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-09-12 23:04 — 迭代 #1：联合路径 smoke test 通过

**验证内容**：旧 lexical-only checkpoint 全量评估已恢复且指标不变；联合 2-epoch smoke 覆盖第 1 轮 MSE-only、第 2 轮 CE+0.2*MSE、softmax pooling、共享权重迁移和 best checkpoint 重载。
**结果**：迁移报告 `missing=8, unexpected=0`，恰好对应被刻意排除的新 classifier/fusion 参数；所有 loss/metric 有限，输出文件完整。
**说明**：4 篇 smoke 验证/测试上的 AUROC/F1 无统计意义，仅用于确认执行链路。

### 2026-09-12 23:16 — 迭代 #1 结果：PASTED-RACE 联合融合

| 指标（test） | lexical-only | joint classifier | 变化 |
|---|---:|---:|---:|
| 文档 AUROC | 0.997745 | 0.997997 | +0.000252 |
| 文档 TPR@1%FPR | 0.9550 | 0.9750 | +0.0200 |
| EDU MSE | 0.039788 | 0.037598 | -0.002190（改善） |
| EDU Pearson | 0.895666 | 0.895075 | -0.000591 |
| EDU AUROC | 0.990902 | 0.988901 | -0.002001 |
| EDU TPR@1%FPR | 0.734372 | 0.708876 | -0.025496 |

**训练结果**：按 validation classifier AUROC 选择 epoch 8；test accuracy 0.9775，F1 0.97786。joint checkpoint 约 497 MiB。

**分域 classifier TPR@1%FPR**：arxiv 1.00、essay 1.00、news 0.97、writing 0.59。全局低误报指标提升，但 writing 的域内低误报 TPR 从 lexical mean baseline 的 0.77 降到 0.59，是下一轮应重点修复的泛化问题。

**结论**：迭代 #1 成功把 PASTED lexical trace 端到端融入 RACE，并提升全测试集文档低误报检测；代价是 EDU 极低误报排序小幅下降和 writing 域内性能退化。下一轮优先考虑增加 lexical loss 权重或对 writing 做 validation-only 的分域诊断，不改变 test 标签或划分。

### 2026-09-12 21:19 — 迭代 #1：修正空 EDU 图的 lexical pooling

**改动原因**：首次融合补丁在空 EDU 分支引用 root 表示早于赋值，虽然当前数据无空图，也会破坏边界输入。
**改动内容**：
- `models/flexible_model.py`：将每个图的 `root_h` 提取提前到空 EDU 判断之前，空图返回同维零 lexical representation。
**预期效果**：保持正常样本逻辑不变，并使空 EDU 边界路径可执行。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-12 23:30 — 迭代 #2：确认四分类 residual PASTED-RACE 设计

**改动原因**：用户确认将 lexical trace 融入 Human/Polished/Generated/Humanized 四分类；二分类 replacement fusion 不能直接复用，否则会破坏原 RACE 四类边界并错误监督 Generated/Humanized。
**改动内容**：
- `docs/idea_report.md`：新增四分类 partial lexical supervision、零初始化 residual fusion、双 checkpoint 初始化与主实验/消融定义。
- `docs/implementation.md`：定义四类 group-safe 数据构建、模型模式、分阶段训练、指标和 mask contract。
- `docs/user_requirements.md`：记录用户确认的四分类执行范围。
**数据审计**：现有 4000-group manifest 覆盖全部 16000 个唯一 item；预期 train/val/test 为 11200/1600/3200，无未分配文档。
**预期效果**：在不牺牲原四分类初始化的前提下，让 Polished 类受益于局部 AI 改写痕迹，并通过 partial MSE 避免给 Generated/Humanized 施加错误标签。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待编码

### 2026-09-12 23:34 — 迭代 #2：实现四类 group-safe partial-label 数据构建

**改动原因**：legacy HART split 有跨集合同源泄漏，且 Generated/Humanized 不应获得伪造的 PASTED lexical target。
**改动内容**：
- `utils/build_pasted_race_fourclass.py`：合并去重 16000 篇，按现有 manifest 重分四类；Human/Polished 复制已生成 target，Generated/Humanized 写全零 mask，并强制检查未分配、缺 target 和 group overlap。
- `scripts/build_pasted_race_fourclass.sh`：新增可复现的数据构建入口。
**预期效果**：四分类 CE 覆盖全部文章，lexical MSE 仅覆盖语义成立的 Human/Polished，且 train/val/test 无 source leakage。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-12 23:36 — 迭代 #2：增加零初始化 residual lexical fusion

**改动原因**：四分类直接替换 `h_root` 会使原 RACE classifier 立刻接收分布不同的新特征，破坏 pretrained decision boundary。
**改动内容**：
- `models/flexible_model.py`：新增 `lexical_fusion_mode=residual` 和可学习标量 `lexical_residual_gamma`；gamma 默认从 0 开始，binary `replace` 模式保持兼容。
**预期效果**：四分类模型初始化时严格退化为原 RACE，训练只在验证信号支持时逐步引入 lexical pooled representation。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-13 00:54 — 迭代 #2 结果：四分类 PASTED-RACE residual fusion

**训练结果**：第 5 轮取得最佳 validation macro-F1 0.918653；此后连续 4 轮未改善，于第 9 轮早停。最佳模型测试 accuracy 0.940000、macro-F1 0.908521、macro-AUROC 0.980931、macro TPR@1%FPR 0.809217。
**相对 group-safe baseline**：accuracy +0.006875、macro-F1 +0.006186、macro TPR@1%FPR +0.034523；macro-AUROC -0.002743。Polished TPR@1%FPR 从 0.7200 提升到 0.8275，是主要收益。
**Lexical 分支**：test MSE 0.045072、Pearson 0.868488、AUROC 0.986498、TPR@1%FPR 0.632388；最佳模型 residual gamma 为 0.004745，说明模型以较小幅度引入 lexical representation。
**产物**：`results/pasted_race/fourclass_joint_seed42/best_model.pt`、`metrics.json`、`history.json` 及 val/test predictions。
**结论**：融合提升了四分类准确率、macro-F1 和低误报召回，尤其改善 Polished；但 macro-AUROC 小幅回落，主要风险仍在 Humanized 类的概率排序。

### 2026-09-13 01:16 — 迭代 #3：新增 AI-to-Humanized 独立 lexical trace 实验

**改动原因**：四分类融合显著改善 Polished，但 Humanized AUROC 回落；现有 Human-to-Polished 监督无法描述 AI 文本被人类化的反向编辑过程。
**改动内容**：
- `utils/humanization_trace_label_builder.py`：新增 Generated/Humanized 唯一同组配对、复用语义句对齐与 `1-BLEU4` EDU 投影，并按现有 manifest 分配数据。
- `utils/flexible_dataset.py`：支持配置级 `label_map`，使独立实验能保留原始类名并映射为二分类 0/1。
- `configs/pasted_race/PASTED_RACE_humanization_lexical.json`：新增 seed-42 masked-MSE 独立训练配置。
- `scripts/generate_humanization_trace_labels.sh`、`scripts/train_humanization_trace.sh`：新增普通前台数据生成和训练入口。
- `README.md`：补充运行命令和隔离产物目录。
**数据审计**：共 1,015 个一对一 group；train/val/test 分别为 703/118/194 对，reference 与 target 均无重复，沿用四分类 group-safe manifest。
**预期效果**：得到专门刻画 AI-to-Humanized 局部改写程度的第二个 lexical checkpoint，为后续双分支四分类融合提供独立初始化，同时避免混淆两个相反改写方向。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-13 13:18 — 迭代 #4：双痕迹 GPU smoke test 通过

**验证内容**：两轮 smoke 覆盖双 checkpoint 参数映射、双 head calibration、第二轮 CE+0.2 polishing MSE+0.2 humanization MSE、双类指标、双 gate 更新、best checkpoint 重载和预测输出。
**结果**：baseline 迁移 216 个参数；两个 lexical checkpoint 各精确迁移 4 个 head 参数；初始两个 gamma 均为 0，联合轮后均获得非零梯度；所有 loss 和输出指标有限。子集指标仅用于执行链路验证，不作模型结论。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-13 13:39 — 迭代 #4 结果：四分类双痕迹联合融合

**训练结果**：第 5 轮取得最佳 validation macro-F1 0.908567；此后连续 4 轮未改善，于第 9 轮早停。最佳模型测试 accuracy 0.942188、macro-F1 0.913799、macro-AUROC 0.983616、macro TPR@1%FPR 0.798011。
**相对 group-safe baseline**：accuracy +0.009063、macro-F1 +0.011465、macro TPR@1%FPR +0.023317；macro-AUROC -0.000058，基本持平。Polished TPR@1%FPR 从 0.7200 提升到 0.8225；Humanized TPR@1%FPR 从 0.7320 提升到 0.7526。
**分类细节**：test F1 分别为 Human 0.983093、Polished 0.923920、Generated 0.947032、Humanized 0.801153。相较 baseline，四类 F1 均提高，其中 Humanized 从 0.776280 提升到 0.801153。
**双痕迹分支**：Human-to-Polished test lexical AUROC 0.987901、TPR@1%FPR 0.656350；AI-to-Humanized test trace AUROC 0.898094、TPR@1%FPR 0.605163。最佳模型 gamma_p=0.012981、gamma_h=0.014510，两个 residual 分支均被模型实际启用。
**相对单 polishing 分支**：accuracy +0.002188、macro-F1 +0.005279、macro-AUROC +0.002685；macro TPR@1%FPR -0.011206。双分支提高总体分类质量和 Humanized F1，但低误报宏平均召回略低于单分支。
**产物**：`results/pasted_race/fourclass_dual_joint_seed42/best_model.pt`、`metrics.json`、`history.json`、初始化报告及 val/test predictions；模型文件约 527 MB。
**结论**：在 seed 42 的同一 group-safe 划分上，双痕迹融合是当前 macro-F1 与 accuracy 最好的方案；macro-AUROC 与 baseline 持平，但尚不能宣称稳定优于单分支，需要补多种子实验确认。

### 2026-09-13 13:45 — 迭代 #5：启动双痕迹多种子稳定性实验

**改动原因**：seed 42 的双分支结果提升了 accuracy 和 macro-F1，但单次运行不足以判断提升是否稳定；用户确认继续运行实验。
**改动内容**：
- `docs/user_requirements.md`：记录 seed 2026/3407 的稳定性实验范围，并固定数据、初始化 checkpoint、架构和其他超参数。
- `train_pasted_race_fourclass.py`：新增命令行 `--seed` 覆盖，允许同一配置安全写入独立种子目录。
**预期效果**：获得三个联合融合训练种子的可比结果；本轮只测联合阶段随机性，不重新训练 seed-42 baseline 与两个独立 lexical checkpoint。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-09-14 22:45 — 迭代 #9：官方设置单/双痕迹三种子控制

**改动原因**：此前单/双痕迹实验使用 `lr=2.0e-5`、12 epochs、patience 4、固定 seed-42 baseline 初始化，且联合训练器没有官方 RACE 的 SupCon，因此不能与新强 baseline 严格比较。

**改动内容**：

- `docs/user_requirements.md`、`docs/idea_report.md`、`docs/implementation.md`：明确官方 outer-loop 对齐与逐 seed baseline 初始化协议。
- `train_pasted_race_fourclass.py`：训练 sampler 改为与官方 `train.py` 相同的 `StratifiedBatchSampler`；可选加入四分类 `SupConLoss(temperature=0.07)`，联合目标为官方 CE+SupCon 再加方法特有 trace MSE。
- `configs/pasted_race/PASTED_RACE_fourclass_single_official.json`：单痕迹官方设置 canonical config。
- `configs/pasted_race/PASTED_RACE_fourclass_dual_official.json`：双痕迹官方设置 canonical config。
- `scripts/train_pasted_race_trace_official_multiseed.sh`：按 single/dual × 42/2026/3407 顺序运行六组，并为每个 seed 选择同 seed 强 baseline checkpoint。
- `scripts/train_pasted_race_trace_official_multiseed.sh`：显式复用现有 Hugging Face cache，避免长实验启动时重复下载 backbone。
- `scripts/train_pasted_race_trace_official_multiseed.sh`：正式六组运行启用 offline mode，固定使用已缓存的同一 RoBERTa revision，避免运行中网络波动或 revision 漂移。

**公平性设置**：`lr=2.9e-5`、train/eval batch 16、20 epochs、linear warmup 0.1、weight decay 0.01、clip 1.0、patience 5、validation macro-F1 选模、SupCon temperature 0.07；calibration epochs=0、fusion warmup=0，所有20轮均为联合训练。方法特有 MSE 权重保持 0.2。

**已知边界**：两个独立 lexical-only checkpoint 只有 seed 42，因此六组实验均复用对应方向的 seed-42 trace head；共享 RACE backbone/RGCN/classifier 则严格逐 seed 配对。

**预期效果**：消除旧实验训练配方和固定 baseline 初始化造成的混杂，直接判断单/双痕迹是否稳定超过强 RACE。

**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-14 23:28 — 迭代 #10 验证与启动：P2 Creator Retention 后续队列

- `utils/creator_retention_label_builder.py`：新增 `--edited_pairs_only`，P2 产物只保留 Polished/Humanized 真实编辑 pair，默认全类行为保持不变。
- `utils/analyze_creator_retention_signal.py`：新增分方向 Pearson/Spearman、数值摘要、四领域统计、行级 CSV 和分布图输出。Editor Modification 文档级标量明确为有效 EDU `1-BLEU4` 的均值。
- `scripts/run_creator_retention_signal_analysis.sh`、`scripts/queue_creator_signal_after_trace_controls.sh`：新增 P2 执行入口和六组控制结果门禁。
- 验证：Python 语法、Bash 语法、相关性函数 smoke test 与 `git diff --check` 全部通过。
- 执行：持久队列会话 `18473` 已启动；目前检测到 1/6 组完成，其余结果出现前不会加载 SciBERT。P2 完成后停止，不自动越过信号有效性判断进入 P3。

**文档同步**：idea_report.md 是 | implementation.md 是 | user_requirements.md 是

### 2026-09-15 00:16 — 迭代 #11：扩展为 Creator Retention 完整实验矩阵

**改动原因**：用户明确要求在当前官方设置单/双痕迹控制补全后，继续完成 Creator Retention 全部实验，不因 P2 相关性过高或 P3 可学习性偏弱而提前停止。

**实验矩阵**：

- P2：真实编辑 pair 的 SciBERT Creator Retention 分布、分域统计及与 Editor Modification 的 Pearson/Spearman，无随机种子。
- P3：`h_i`、`[h_i;h_root]`、`[h_i;h_root;h_i*h_root]` 三种输入 × seeds `42/2026/3407`，共 9 组 Creator-only regression。
- P5：最佳 P3 输入的 Creator+Editor 联合模型 × 3 seeds，共 3 组。
- P6：新训练 `RACE+Creator`、Creator loss/no fusion、全结构 `lambda=0` 各 3 seeds，共 9 组；RACE、Editor-only 和完整 Creator+Editor 分别复用 P1、P4、P5。

**公平性**：所有四分类条件沿用强 RACE 的 group-safe split、官方 CE+SupCon outer-loop 设置和逐 seed baseline 初始化；训练串行使用单 GPU。P3 是任务特定的 MSE-only 回归，按 validation MSE 选模并用 Spearman 破平。

**当前执行边界**：已运行的会话 `18473` 会自动完成 P2；P3–P6 的独立 runner 将在不改动当前 trace trainer 的前提下实现和 smoke，然后接续执行。

**文档同步**：idea_report.md 是 | implementation.md 是 | user_requirements.md 是

### 2026-09-15 00:55 — 迭代 #11 结果：官方设置 trace 控制与 P2 信号分析

**官方设置三种子均值 ± 样本标准差**：

| 方法 | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |
|---|---:|---:|---:|---:|
| Strong RACE | 0.940625 ± 0.004086 | 0.910815 ± 0.003687 | 0.984464 ± 0.001000 | 0.797522 ± 0.016030 |
| Single trace | 0.944271 ± 0.007758 | 0.916606 ± 0.007611 | 0.984127 ± 0.000693 | 0.824575 ± 0.010700 |
| Dual trace | 0.943125 ± 0.007973 | 0.911680 ± 0.013869 | 0.984775 ± 0.001281 | 0.818952 ± 0.005386 |

**配对差值诊断**：Single trace 平均相对强基线的 Accuracy/Macro-F1/Macro-TPR 分别为 +0.003646/+0.005792/+0.027053，但 seed 3407 的 Accuracy/F1 为负；Dual trace 对应为 +0.002500/+0.000865/+0.021430，seed 3407 Macro-F1 下降 0.015789。因此低 FPR 收益更一致，但不能声称分类 Macro-F1 稳定提升，双痕迹尤其受 seed 3407 影响。

**P2 数据审计**：共 5,015 个真实编辑 pair，H→P 4,000，G→Hu 1,015；train/val/test 为 3,503/518/994，group overlap 全为 0，未加入自保留为 1 的样本。

**P2 主结果**：

- H→P：Creator Retention `0.781899 ± 0.099840`，Editor Modification `0.729806 ± 0.245705`，Pearson `-0.823426`，Spearman `-0.872743`。
- G→Hu：Creator Retention `0.779110 ± 0.075592`，Editor Modification `0.813736 ± 0.142824`，Pearson `-0.731668`，Spearman `-0.796033`。
- 分域相关范围：H→P Pearson `[-0.846190,-0.743298]`，G→Hu Pearson `[-0.828989,-0.696860]`。

**结论**：两个信号具有较强负相关，尤其 H→P，说明 Creator Retention 不是完全独立轴；但相关性仍未达到接近 `-1`，且 G→Hu 只为中高强度，仍存在可验证的非冗余信息。根据用户要求，继续 P3–P6，不以此为停止条件。

**产物**：`results/pasted_race/creator_retention_signal/signal_analysis.json`、`edited_pair_signals.csv`、`signal_distributions.png`、`domain_distributions.png`。

### 2026-09-15 10:00 — 迭代 #12：逐种子端到端 trace 补训

**改动原因**：首轮官方 outer-loop 对照在联合阶段使用 42/2026/3407，但所有种子的独立 polishing/humanization trace head 均复用 seed 42，不能表示完整 pipeline 的多种子稳定性。

**改动内容**：

- `docs/user_requirements.md`、`docs/idea_report.md`、`docs/implementation.md`：固定四个 trace-only 补训和四个 paired joint 重跑的公平性、初始化与隔离产物契约。
- `train_pasted_race.py`：新增 `--seed` 覆盖，并将 DataLoader shuffle generator 从硬编码 42 改为 resolved run seed。
- `train_pasted_race_fourclass.py`：新增 `--lexical_checkpoint` 覆盖，与已有 baseline/humanization 覆盖组成可审计的逐 seed 初始化入口。
- `scripts/train_pasted_race_trace_end_to_end_multiseed.sh`：串行执行 polishing × 2026/3407、humanization × 2026/3407，再执行 single/dual × 2026/3407 paired joint；完成产物可安全跳过。

**设置边界**：trace-only 没有 RACE 官方配方，因此严格保留现有 seed-42 canonical trace-only 配方（`lr=2.5e-5`、batch 16、10 epochs、patience 3），只改 seed；四分类 joint 仍使用官方 RACE CE+SupCon outer-loop。

**Smoke 验证**：两个独立 trace 和单/双 joint 共四条 seed-2026 子集路径全部通过；每条均产生 finite loss、`metrics.json`和 `best_model.pt`。Dual 初始化审计显示 baseline 从 seed-2026 迁移 216 个参数，polishing/humanization 分别从对应 seed-2026 smoke checkpoint 迁移 4 个 head 参数，两个 gate 从 0 开始并在联合步更新。

**预期效果**：得到 `RACE seed=s + trace seed=s + joint seed=s` 的完整三种子比较，并将 trace 预训随机性与首轮固定 trace 初始化结论分开报告。

**文档同步**：idea_report.md 是 | implementation.md 是 | user_requirements.md 是 | configs 复用 canonical

### 2026-09-15 10:03 — 迭代 #12 修正与启动：trace-only 学习率统一为官方 RACE

**用户修正**：完整 pipeline 要求学习率与强 RACE 保持一致。因此不再复用 `lr=2.5e-5` 的旧 seed-42 trace checkpoint，而是对 polishing/humanization 两个方向全部重训 seeds `42/2026/3407`。

**统一设置**：`lr=2.9e-5`、train/eval batch 16、20 epochs、warmup 0.1、weight decay 0.01、clip 1.0、patience 5。Trace-only 任务仍为 masked MSE，因没有四分类标签，CE/SupCon/Macro-F1 不适用。

**改动与产物**：新增 `PASTED_RACE_lexical_official.json`、`PASTED_RACE_humanization_lexical_official.json`；串行入口改为 6 个 official-optimized trace-only + 6 个 same-seed joint，输出到 `*_official_seed*` 和 `fourclass_*_official_e2e_seed*`，不覆盖旧表。

**验证**：新 `2.9e-5` polishing trace seed-42 GPU smoke 通过，loss/metrics/checkpoint 均有限；JSON、Bash、Python 语法和 `git diff --check` 通过。

**执行**：正式 12 组串行任务已于 10:03 启动，会话 ID `66613`，当前第 1/12 组为 polishing trace seed 42。Creator Retention P3–P6 顺延到本轮完成后。

**文档同步**：idea_report.md 是 | implementation.md 是 | user_requirements.md 是 | configs 是

### 2026-09-15 12:35 — 迭代 #12 结果：逐种子端到端 trace 补训完成

**完成性审计**：12/12 个正式任务均正常结束并生成 `metrics.json` 与 `best_model.pt`：polishing/humanization trace-only 各 3 seeds，same-seed single/dual 四分类联合模型各 3 seeds。串行会话 `66613` 以退出码 0 结束。

**端到端配对三种子均值 ± 样本标准差**：

| 方法 | Accuracy | Macro-F1 | Macro-AUROC | Macro TPR@1%FPR |
|---|---:|---:|---:|---:|
| Strong RACE | 0.940625 ± 0.004086 | 0.910815 ± 0.003687 | 0.984464 ± 0.001000 | 0.797522 ± 0.016030 |
| Single trace, same-seed E2E | 0.945729 ± 0.006046 | 0.917933 ± 0.005956 | 0.984412 ± 0.000490 | 0.825848 ± 0.011592 |
| Dual trace, same-seed E2E | 0.943229 ± 0.008819 | 0.914408 ± 0.009454 | 0.984484 ± 0.000601 | 0.824065 ± 0.005164 |

**相对 Strong RACE 的均值变化**：Single 为 Accuracy `+0.005104`、Macro-F1 `+0.007118`、Macro-AUROC `-0.000052`、Macro TPR `+0.028326`；Dual 为 `+0.002604`、`+0.003594`、`+0.000021`、`+0.026543`。

**稳定性边界**：Single 的 Macro-F1 在 seed 42/2026/3407 分别为 `0.922064/0.920629/0.911106`，三者均高于各自 Strong RACE；Accuracy 在 seed 3407 下降。Dual 的 Macro-F1 为 `0.921729/0.917761/0.903735`，seed 3407 下降。因此端到端配对后，Single 的 Macro-F1 和低 FPR 收益最有说服力；Dual 仍不能称为所有种子稳定提升。

**Trace-only 测试 AUROC**：polishing seeds 42/2026/3407 为 `0.991074/0.990336/0.990865`；humanization 为 `0.956739/0.953156/0.952588`。Humanization 痕迹显著更难，但三个种子结果一致。

**下一步**：按既定 P3–P6 队列进入 Creator Retention 回归、输入消融、Creator/Editor 联合与 lambda/fusion 消融。

### 2026-09-15 14:26 — 迭代 #13：P3 Creator Retention 实现与启动

**实现**：新增 `train_creator_retention.py`、P3 canonical config 和三输入 × 三种子串行 launcher。模型 Creator head 现支持 `edu`、`edu_root`、`edu_root_interaction`；每个 EDU 产生 logit，文档 logit 取有效 EDU 算术均值后 sigmoid。P3 只在真实编辑最终文本上训练 Creator MSE，按 validation MSE 选模、Spearman 破平，并从同 seed Strong RACE 初始化结构编码器。

**缓存修正**：主机旧 `HF_HUB_CACHE/TRANSFORMERS_CACHE` 指向已不存在的外置路径；P3 launcher 显式固定到 `/home/dx/.cache/huggingface/hub`，全程离线使用现有 RoBERTa snapshot。

**验证**：三个输入模式分别完成 GPU smoke；均成功加载 54 类关系和同 seed baseline，产生 finite MSE/相关系数、`best_model.pt`、`metrics.json`。Python/JSON/Bash 语法与 `git diff --check` 通过。

**执行**：开始串行运行 P3 九组正式实验；完成后按三种子 validation MSE 选择输入模式，再进入 P5/P6。

### 2026-09-15 15:57 — 迭代 #13 结果与 P5 自动接续

**P3 完成性**：9/9 正式任务完成。按三种子平均 validation MSE 自动选择 `edu` (`h_i`)：`edu=0.004445`、`edu_root=0.004761`、`edu_root_interaction=0.004606`。对应三种子测试均值为：`edu` MSE/Pearson/Spearman `0.004049/0.733611/0.723041`，优于其余两种输入。

**P5 实现**：四分类 trainer 新增 P3 Creator head 的形状校验加载和 CLI 覆盖；新增 Creator 标签无重算合并器、official P5 config 和 P3→P5 自动选择/接续队列。合并数据 train/val/test 为 11,200/1,600/3,200，group overlap 为 0，Creator target 全覆盖。

**验证与空间处理**：8 条样本 smoke 因官方分层 sampler 无完整 batch，仅验证前向，不计作训练验证；随后 200 条有效 smoke 产生非零训练 loss，三个 residual gate 均从 0 更新。首次保存因磁盘满失败；删除本轮临时 smoke 目录，并在保留 metrics/history/predictions 的前提下删除 P3 两个落选模式的 6 个可重训 checkpoint，空间从 0 恢复到 5.3 GB。

**执行**：P5 seed 42 已启动；完成后自动运行 seeds 2026/3407。选择报告为 `results/pasted_race/creator_retention_p3_selection.json`。
