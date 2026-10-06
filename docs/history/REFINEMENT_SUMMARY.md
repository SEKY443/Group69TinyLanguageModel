# CITS4012 Group 69 — completed audit and refinement handover

30 September 2026. Read alongside `docs/PROJECT_AUDIT.md` (pre-change findings),
`docs/DEVELOPMENT_LOG.md` sections 22–23 (work actually performed), and `report_support/` (report evidence).
The implementation, evidence and analysis have been strengthened substantially. This does not guarantee
a mark. The final six-page ACL report and full fresh-Colab verification remain outstanding.

## 1. What I Found

The existing submission candidate is `CITS4012_69.ipynb`: an executed A100 study of a scratch-built
Difference-Aware Contrastive Transformer (DACT), five baselines and seven ablations. Its eight embedded
modules matched the original source. The course data, training-only tokenizer, 32 scratch training logs,
saved result summaries and attention figures were present. Historical checkpoints and item predictions
were missing. No final report existed. The inventory and requirement-by-requirement initial audit were
written before changing source. The model already had meaningful PIQA-specific interactions, so it was
preserved rather than replaced with another architecture.

## 2. Requirement Compliance

| Requirement | Before | After | Evidence |
|---|---|---|---|
| Scratch neural QA model with attention | Met | Preserved and dynamically checked | Original/random initialisation; 6.11M-parameter DACT; gradient tests |
| Task-specific design | Met, some overclaims | Exact interaction flow and attribution documented | `architecture.md`, `methodology_equations.md`, figure SVG |
| Actual dataset audit | Partial/truncated statistics | Raw statistics, schemas, duplicates, vocab and hashes | `dataset_audit.json`, `dataset_lengths.csv`, split manifest |
| Separation/no external dataset | Training-only fitting, overlap undisclosed | Same split; overlap and past test exposure disclosed | Dataset audit; new inference deferred; no new dataset |
| Training correctness | Saved execution, few asserted checks | Masks, gradients, invariance, overfit, checkpoint tests pass | `sanity_checks.json` |
| Informative own baselines | Five executed historical baselines | Preserved; B0/B1 independently rerun | Historical notebook; local predictions and fitted LR |
| Controlled ablations | Seven three-seed variants | Controls and confounds accurately explained | `experiments.md`, `ablation_table.csv` |
| Quantitative analysis | Complete results, misleading CI/p claims | Correct labels, exploratory Holm, real local paired analysis | Result/comparison tables and verification notebook |
| Attention and error analysis | Selected heatmaps, weak interpretation | Numeric maps, compact plots, successes/failures and sampled errors | `qualitative_examples.md`, `replication.md` |
| Reproducibility | Output overwrite risk; missing artifacts | Safe builder, provenance, all new checkpoints/predictions | Smoke/verification notebooks; manifests; requirements |
| Complete saved notebook | Original executed study | Original code/outputs retained plus real audit appendix | 27 original pairs preserved; 2 new executed audit cells |
| Clean Colab | Historical direct-upload A100 execution | Current code passes local fresh-kernel smoke; full route pending | No claim that Drive/B3/B4 revised path has been executed |
| Six-page ACL report | Missing | Evidence and narrative ready; final PDF still missing | Report story, equations, figures, BibTeX |

## 3. Problems Found

The most material issues were one exact training/test overlap, directional difference-tag alignment,
statistics measured after truncation, two test pairs made indistinguishable by truncation, missing
historical checkpoint/prediction artifacts, early historical B1/B3 test inference, overinterpretation
of attention and statistical uncertainty, incomplete run metadata, and notebook regeneration that could
erase saved outputs. The audit did not find pretrained weights in the main model, incorrect label
indexing, disconnected attention, or test-based early stopping. Absence of those findings is based on
the inspected code/evidence, not proof that every past manual decision can be reconstructed.

## 4. Changes Made

`report_support/changes.md` is the detailed table requested: **file → original behaviour → new behaviour
→ reason → assignment requirement addressed**. It covers every implementation change and artifact family;
`report_support/files_changed.md` enumerates individual files, including new report/evidence files.

The main changes are strict data validation/provenance, distinct run/split seed metadata, validation
loss, safe checkpoint sidecars, saved predictions, overwrite protection, honoured AMP configuration,
finite-loss checks and efficient selected-position MLM. Future full baseline runs retain B3 and log
pretrained revisions. Optional canonical alignment remains disabled for scored historical-protocol runs.
The original architecture, labels, loss, split, caps and model selection rule were not replaced.

The historical notebook received corrected prose and actual audit outputs; its original 27 code/output
pairs remain intact. A separate current-source notebook and two genuinely executed smoke notebooks
avoid passing old outputs off as results of modified code. New analysis notebooks explicitly verify
saved training artifacts rather than pretending to have executed training in their cells.

## 5. Final Main Model

For each candidate, encode `[CLS] goal[:40] [SEP] solution[:96] [SEP]`, at most 139 tokens.
Random trainable token/position/segment/difference-tag embeddings produce `[B,2,L,256]`. A shared
four-layer pre-LN Transformer uses four 64-dimensional heads and 1,024-wide feed-forward blocks.
Joint self-attention connects the goal and each solution. Cross-solution attention lets each option's
states query the rival's solution states plus a CLS sink. The fusion is
`[q; c; q-c; q*c]` (1,024 features, **signed subtraction**) → 256 → 256 with residual/LN.
Difference-biased additive attention pools solution tokens. Concatenating its 256-wide summary with
the updated CLS gives 512 → 256 → 1 shared scoring. Two scores form `[B,2]` logits for CE/argmax.

Total allocated trainable parameters: **6,114,628**, including the tied-embedding MLM head used only
during warm-up. Ten epochs of MLM use training text and scratch weights; this is not external pretraining.

## 6. Why This Architecture Fits PIQA

Shared weights score candidate suitability without a separate answer-position classifier. Joint
goal/solution encoding supplies context, while explicit rival comparison and label-free difference tags
highlight small changes between often similar alternatives. Pooling creates a compact solution summary.
These are defensible design hypotheses, not established gains for every component. The original study
does not show that the full model beats all simpler ablations. Attribute Transformer, BPE, MLM, additive
attention and matching features to their published origins; claim the specific PIQA adaptation/implementation.

## 7. Attention Mechanism

Encoder Q/K/V come from each joint goal-option sequence; padding keys are masked. Cross-attention Q
comes from one encoded option, K/V from the other solution plus CLS, excluding goal/padding keys.
Pooling normalises learned additive scores over solution tokens, with an initial +1 bias for difference
tokens. Pooled values and updated CLS both feed the shared decision score.

Executed checks established nonzero gradients through encoder/cross/pooling parameters, zero padded-key
mass, normalisation, padding invariance and agreement between explicit and fused attention (max error
1.68e-8). A pooling intervention changed logits by .02761. These establish functional involvement, not
causal interpretability or comparative generalisation benefit. Encoded pairs swap equivariantly, while
the default raw alignment can change on an option swap.

## 8. Dataset Protocol

| Split | Items | Label 0 / label 1 |
|---|---:|---:|
| Supplied training file | 16,113 | 8,053 / 8,060 |
| Training subset | 14,501 | 7,247 / 7,254 |
| Validation subset | 1,612 | 806 / 806 |
| Supplied test | 1,838 | 910 / 928 |

Fields are `goal`, `sol1`, `sol2`; labels are in companion `.lst` files (0=sol1, 1=sol2).
The unchanged split is stratified 90/10, seed 42. BPE vocabulary 8,000/min-frequency 2 is fitted to
training only, with NFKC, lowercasing, punctuation-aware whitespace and individual digits. Unknown-token
rate was zero on the inspected splits. No empty/malformed records or contradictory duplicate labels
were found. Whitespace anomalies and exact/casefold duplicates are explicitly inventoried.

Raw training goal lengths: mean 8.30, median 8, p95 16, max 37 BPE tokens. Combined candidates:
mean 22.28, median 14, p95 64, p99 108, max 492. Caps remain 40/96; the solution cap covers most
training text but sacrifices a tail. Report this tradeoff rather than justifying a cap with capped data.

One test example (index 1545) occurs twice in training. No exact train/validation or validation/test
overlaps were found. This prevents an unqualified disjointness claim. No new data, silent resplitting
or test-based tuning was introduced. The known overlap, historical test inspection and pretrained
baselines' unknown pretraining exposure remain limitations.

## 9. Training Configuration

| Setting | Selected value |
|---|---|
| Model seeds / split seed | 42, 43, 44 / 42 |
| Vocabulary; goal/solution caps | 8,000; 40/96 |
| Width / layers / heads / FFN | 256 / 4 / 4 / 1,024 |
| Dropout / attention dropout | .2 / .1 |
| QA objective | Two-logit CE, label smoothing .1 |
| Optimiser | AdamW, betas (.9,.98) |
| QA LR / weight decay | 3e-4 / .05; no decay on embeddings, bias, LN |
| QA batch / epoch cap / patience | 128 / 20 / 6 |
| Schedule / gradient clipping | 6% linear warm-up then cosine / norm 1 |
| MLM epochs / LR / batch | 10 / 5e-4 / 256 |
| MLM masking | 15% eligible tokens; 80/10/10 mask/random/unchanged |
| Checkpoint rule | Maximum validation accuracy, earliest tie |
| Hyperparameter selection | Five historical seed-42 configurations, validation only |
| Historical / local precision | A100 bf16 / RTX 2070 Max-Q fp16 |
| Determinism | Historical and scored local runs nondeterministic; opt-in utility added |

Full settings, actual epoch counts, parameters and baseline controls are in `experiments.md` and JSON logs.

## 10. Baselines

All five historical baselines have saved executed results. B0 is training-majority label 1; B1 uses
TF-IDF 1–2 grams/min_df=2/sublinear TF and candidate-difference LR, C=3 selected from six values.
B2 is a scratch two-layer BiLSTM (256 per direction), embedding 256, additive pooling, dropout .3,
LR .001, three seeds, no MLM. B3 fine-tunes FacebookAI/roberta-base for four epochs, LR 2e-5, batch 32,
length 128, seed 42. B4 uses open Qwen/Qwen2.5-1.5B zero-shot mean continuation log-likelihood, batch 32,
128-token cap, prompt `Goal: ...\nSolution:`. No proprietary API or additional dataset was used.

B0 measures answer-position bias, B1 lexical preference, B2 recurrent scratch modelling, and B3/B4
pretrained reference points. These are informative comparisons with differing capacity/budgets, not
all matched causal controls. Historical Hub revisions were not saved. Local B0/B1 were rerun with
saved predictors/predictions; current B3/B4 modifications remain unexecuted.

## 11. Ablation Studies

Historical test percentages below are means ± sample SD over three seeds. Common controls are split,
tokenizer/caps, QA optimiser/schedule, batch, epoch cap, patience, metric and seeds. Actual stopping
epochs differ. Changing modules affects parameter counts and RNG consumption. See `experiments.md`
for hypotheses and expected interpretations; no all-attention-off model is claimed.

| Ablation | Research question | Controlled difference | Test result (%) |
|---|---|---|---:|
| No tag embedding | Do input tags help beyond the pool prior? | Remove tag embedding; pool bias remains | 60.52 ± .49 |
| No cross module | Does rival interaction help beyond tags? | Remove cross-attention/fusion/LN, reducing parameters | 61.73 ± .74 |
| No pool tag bias | Does the explicit prior help content pooling? | Retain additive attention, remove tag bias | 61.50 ± 1.27 |
| Mean pooling | Does learned weighting plus prior help? | Uniform solution pooling; encoder/cross remain | 61.46 ± 1.06 |
| Pointwise BCE | Does forced-choice competition help? | CE → independent BCE, same final argmax | 60.92 ± .87 |
| Vanilla Transformer | Does the combined task adaptation help? | Joint removal of tags/cross/attentive pooling | 58.71 ± .95 |
| No MLM | Does training-only warm-up help? | Remove ten MLM epochs, less compute | 61.37 ± 1.25 |

Several single ablations outperform full DACT on test despite validation selection favouring full.
Report this directly. Vanilla is a bundled comparison and mean pooling removes both weighting and bias.
No pooling bias raw p=.01696 becomes .15267 under exploratory Holm correction over 12 historical tests.

## 12. Final Results

**Historical A100 study — keep together:**

| Model | Validation (%) | Test (%) |
|---|---:|---:|
| DACT | 60.09 ± .66 | 60.37 ± 1.17 |
| Vanilla Transformer | 57.88 ± .91 | 58.71 ± .95 |
| B0 majority | 50.00 | 50.49 |
| B1 TF-IDF + LR | 59.24 | 61.21 |
| B2 BiLSTM + attention | 59.57 ± 1.15 | 59.54 ± .96 |
| B3 RoBERTa fine-tuned | 70.04 | 68.50 |
| B4 Qwen zero-shot | 78.10 | 75.35 |

DACT's historical representative is seed 44 (chosen on validation), test 59.03%, CI [56.75,61.21]%;
this is not the CI of its three-seed mean. Historical prediction-level CIs/p-values are preserved with
provenance rather than falsely described as independently recomputed.

**Separate completed local replication:**

| Model | Validation (%) | Test (%) |
|---|---:|---:|
| DACT, seeds 42/43/44 | 59.64 ± .51 | 62.57 ± .27 |
| B0 majority | 50.00 | 50.49 |
| B1 TF-IDF + LR | 59.00 | 60.88 |

The three DACT test accuracies are 62.2960, 62.5680 and 62.8400%. Validation selected seed 43,
which gets 1,150/1,838 correct; its 95% item-bootstrap CI is [60.3917,64.6368]%.
Excluding the known duplicate gives 62.5476% on 1,837 items. DACT minus local B1 is +1.6866 points,
paired-bootstrap CI [-.7617,4.0805], exact McNemar p=.18953 (same after Holm across B0/B1).
No robust advantage over B1 is established. Changed hardware/libraries/numerics prevent attributing
the new main-model score to the code refinements. All 17,250 new saved prediction rows were verified.

## 13. Attention Analysis

Historical cases and corrected interpretation are in `qualitative_examples.md`. Four new attention
cases have full measured JSON/PNG evidence. The compact report SVG/PNG shows success #1672 (ice cream
without the jalapeño addition) and failure #551 (rotation direction for a bulb). For #551, attention
concentrates on the differing direction tokens yet the wrong choice gets p=.974. This is strong evidence
against interpreting attention concentration alone as correct reasoning. #130 is another correct food
contrast; #1244 is a label disagreement on an underspecified weight-loss item, not independently verified
medical truth. These cases were chosen by confidence, so they are illustrative rather than representative.

Figures mark gold/predicted options and differenced tokens. Cross/goal panels average heads and omit
some keys, so sliced rows need not sum to one. The +1 initial difference prior must be disclosed.
FP32 figure versus AMP batch probability differences are at most .000117 for these cases; labels agree.

## 14. Error Analysis

Local seed 43 makes 688 errors. Twelve were sampled reproducibly before one-reviewer, post-hoc categorisation:
four material/causal-knowledge contrasts, four object/tool affordances, two spatial/temporal relations,
one ingredient contrast and one underspecified equipment example. Full text, labels, confidence,
category rationale and limitations are stored in `replication_error_sample.json` and `replication.md`.
Do not generalise those counts to all PIQA errors or treat dataset labels as verification of every claim.

Two test pairs become identical after truncation and both are wrong. However, the truncated-item subgroup
gets 24/32 correct versus 1,126/1,806 for other items: these observations do not establish a general
truncation penalty. At confidence ≥.8, 112 of 420 predictions are wrong. New train/validation loss
curves show substantial overfitting after the first few epochs. Test error inspection was used for
analysis only, not another architecture/hyperparameter search.

## 15. Reproducibility

From the repository directory in PowerShell:

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch
.venv/Scripts/python.exe tools/audit_evidence.py --data-dir ../../PIQA
.venv/Scripts/python.exe tools/verify_model.py --data-dir ../../PIQA
.venv/Scripts/python.exe tools/smoke_notebook.py
.venv/Scripts/python.exe -u tools/reproduce_main.py --data-dir ../../PIQA --out-dir experiments/completed/outputs_replication_new
.venv/Scripts/python.exe tools/analyse_replication.py
.venv/Scripts/python.exe tools/final_checks.py
```

Use a fresh output name for any new experiment. `analyse_replication.py` verifies the named completed
`experiments/completed/outputs_replication_efficient`/`experiments/completed/outputs_replication_baselines` evidence; change its input paths explicitly
to verify a different run. `requirements-a100.txt` records historical versions. Local runs record actual
versions/configs and weights; seeds alone cannot make nondeterministic cross-device results identical.

For Colab, open `notebooks/CITS4012_69_reproducible.ipynb`, select GPU, supply only the four course files (or the
specified Drive ZIP), run all, and download both executed notebook and complete timestamped output folder.
The notebook installs missing packages and saves checkpoints to Drive when mounted. That complete
revised route has not been tested here. The actual local fresh-kernel smoke passed with tiny validation-
derived diagnostics and B3/B4 skipped. The separate artifact-verification notebook executed successfully.

## 16. Files Changed

The exhaustive path-by-path inventory is `report_support/files_changed.md`; the original-to-new rationale
is `report_support/changes.md`. Modified tracked files are `.gitignore`, `README.md`,
`docs/DEVELOPMENT_LOG.md`, `CITS4012_69.ipynb`, all eight `src/*.py` files and `tools/build_notebook.py`.
New files include the pre-change audit, this handover, two requirements files, the reproduction notebook,
audit/test/replication/analysis tools, report-support notes/tables/figures/manifests, historical snapshots,
two executed smoke notebooks and the executed replication-verification notebook.

Completed new experiment artifacts are locally retained in `experiments/completed/outputs_replication_efficient/` and
`experiments/completed/outputs_replication_baselines/` (38 hashed files, including weights and fitted LR). These directories
are ignored by Git; they are not protected by a remote backup. Interrupted/development run folders
remain labelled and are not reported as completed experiments. No push or new remote was performed.

## 17. Files I Should Submit

Required names: **`CITS4012_69.ipynb`** and **`CITS4012_69.pdf`**. The current historical notebook is the
complete executed submission candidate, with original study outputs and corrected audit commentary.
The separate reproducible notebook is a current-source candidate for a future fully reviewed Colab run,
not a replacement with invented saved outputs. The PDF does not yet exist.

Build the report from the actual historical study and, if included, clearly separate the local replication.
Use the official ACL LaTeX template in final mode, Group 69 in the author field, ≤6 main-content pages,
BibTeX references and truthful team contributions outside the main-page limit as specified by the brief.
Do not submit virtual environments, scratch smoke folders, or an unexecuted replacement notebook as
the final evidence. Keep raw logs/checkpoints/predictions available for reproducibility.

## 18. Remaining Risks

The report is unfinished; clean Drive/Colab and modified pretrained pathways are unverified. Original
weights/predictions and Hub revisions are missing. Dataset overlap, test familiarity, directional tags,
length truncation, nondeterministic training and pretrained exposure remain disclosed limitations.
The architecture has useful task motivation but its added modules do not consistently improve test
accuracy. Attention is not a causal explanation. Some real-world labels are underspecified. Three seeds
are limited evidence; multiple comparisons are exploratory. Long full heatmaps need zooming, so use
legible compact figures in the report. New ignored weights/evidence need an intentional backup before
changing machines. No highest-mark claim is warranted before the final report and runtime checks.

## 19. Recommended Report Story

Use the six-page allocation in `report_story.md`: .7 pages introduction/relevant work, 1.4 methodology,
.9 experimental setup, 1.2 quantitative results, 1.2 attention/error analysis and .6 limitations/conclusion.
Ask whether shared, difference-aware comparison helps scratch PIQA learning, which components the
ablations support, and where attention/error cases limit reasoning claims. Show meaningful learning
over majority, a competitive lexical baseline, mixed single-component effects, and stronger pretrained
references. The defensible contribution is a clearly motivated implementation and critical investigation,
not a claim of state-of-the-art accuracy or uniformly beneficial complexity. References and actual
contributions follow outside the main-page limit. If space is tight, treat local replication as a short,
separate reproducibility check rather than replacing the coherent historical comparison.

## 20. Final Readiness Checklist

- [PASS] Main model trained from scratch.
- [PASS] Neural sequence model.
- [PASS] Meaningful, connected attention with executed gradient/mask checks.
- [PASS] Task-specific design and appropriate attribution support.
- [PASS] Actual dataset documented.
- [PASS] No additional dataset deliberately introduced.
- [PARTIAL] Train/validation/test separation: training-only fitting; one supplied overlap disclosed.
- [PARTIAL] Reproducible training: new complete artifacts; historical gaps/nondeterminism remain.
- [PASS] Meaningful baselines.
- [PASS] Own baseline experiments, with saved historical evidence and new B0/B1 reruns.
- [PASS] Controlled ablations, with compound changes/capacity/compute confounds disclosed.
- [PASS] Accuracy justified for balanced forced-choice QA.
- [PASS] Quantitative comparison and uncertainty caveats.
- [PASS] Attention visualisations, including numeric exports and compact report figures.
- [PASS] Successful example analysis.
- [PASS] Failed example analysis from actual predictions.
- [PASS] Limitations stated without unsupported causal claims.
- [PASS] Architecture diagram support.
- [PASS] Methodology equations matching implementation.
- [PASS] Logs stored; historical evidence preserved.
- [PASS] Results stored in original and separate verification notebooks with distinct provenance.
- [PARTIAL] Clean Colab compatibility: local smoke passed; full revised route pending.
- [PASS] Code/results consistency within each separately identified experiment.
- [PASS] Six-page report evidence prepared.
- [FAIL] Final six-page ACL report PDF completed — still required.
