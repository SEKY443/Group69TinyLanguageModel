# Final requirements traceability (CITS4012 Group Project, Group 73)

**Source of truth:** *CITS4012 Natural Language Processing — Group Project* (September 2026, 6 pages), supplied
by the group on 2026-10-06 (SHA-256 `a5b32c6fb3155ad9fcbe116e2a9cd7bea7cd2ea6ed9e34e33868cd8d34ed451c`; the PDF
itself is not committed). Every row was checked against the repository at the final-audit commit. A **PASS** needs
direct evidence (a file, a test, a recomputation or a rendered page), not a statement in an earlier document.

Statuses: PASS · PARTIAL · FAIL · NOT VERIFIED (only the team can confirm) · NOT APPLICABLE.
Verification commands are run from the repository root in the locked environment (`requirements-lock.txt`).

## A. Task, data and main model

| ID | Official requirement (abridged) | Repository evidence | Status | Risk | Required action | Verification |
|---|---|---|---|---|---|---|
| A1 | Select ONE of TweetQA / MultiRC / ReClor / PIQA | PIQA only (`src/data.py`, notebook §1.2). The four files are byte-identical to the course files used by the run (`outputs/data_manifest.json`) and to the official PIQA release (train + dev) | PASS | – | – | `tests/test_data_acquisition.py`; SHA-256 comparison in `experiments/final_audit_20261006/` (log §49) |
| A2 | Design, implement and train your own neural model **from scratch** (a pretrained / fine-tuned model or LLM system does not satisfy this) | DACT (`src/model.py`): random N(0, 0.02) initialisation, BPE learned on the training split, MLM warm-up on training-split text only; no `transformers` import or `from_pretrained` call in the main-model path | PASS | Low | – | `tests/test_model_properties.py::test_main_model_code_path_loads_no_pretrained_weights`, `::test_dact_starts_from_random_initialisation`, `::test_tokenizer_is_learned_from_the_training_split_only` |
| A3 | Use RNN / LSTM / GRU / Transformer for sequence modelling | 2-layer pre-LN Transformer encoder shared by both candidates (selected config d=128, 4 heads, FFN 512) | PASS | – | – | `src/model.py` `EncoderLayer`, `outputs/results/dact_full_val.json` config |
| A4 | Must incorporate an attention mechanism; explain its role | Self-attention, cross-solution multi-head attention (keys = rival solution tokens + rival [CLS]), difference-guided additive pooling. Attention weights are proper distributions over allowed keys only, receive QA gradient and change the prediction | PASS | – | – | `test_attention_weights_are_distributions_over_allowed_keys`, `test_qa_loss_sends_gradient_through_every_attention_component`, `test_pooling_attention_changes_the_prediction`, `test_explicit_attention_equals_fused_kernel` |
| A5 | Originality: thoughtful task-specific adaptations, not a stock architecture; justify with respect to the dataset | Difference-tag embedding, cross-solution comparison of the two PIQA candidates (ESIM fusion adapted), tag-biased pooling, jointly trained hashed lexical head; motivated by the 15 % median share of differing tokens (recomputed: 0.150 on train) | PASS | Medium: individual components show no consistent effect (honestly reported) | – | Report §1.1, §2, Table 2; notebook §2.1 design table |
| A6 | Complete technical description: notation, equations, architecture diagram; enough detail to reproduce; distinguish own design from adapted work; cite; state assumptions/limitations | Report §2 (Eq. 1–4), Figure 1 (redrawn), §2 opening paragraph separates adapted vs project-specific parts; §3.3 hyper-parameters; equations now match the code (LayerNorm on cross-attention queries and fusion input) | PASS | Low | – | Code ↔ equation review (log §49); `report/CITS4012_73.tex` |

## B. Experiments

| ID | Official requirement (abridged) | Repository evidence | Status | Risk | Required action | Verification |
|---|---|---|---|---|---|---|
| B1 | Dataset description, preparation, train/validation/test organisation, preprocessing | Report §3.1–3.2; stratified 90/10 split (seed 42) of the training file; split hashes reproduce exactly; duplicates (6), one train/eval overlap (item 1545) and 1.6 % truncated solutions disclosed | PASS | – | – | `experiments/final_audit_20261006` recomputation; ordered split SHA-256 = manifest |
| B2 | Training procedure and hyper-parameters needed to reproduce; how they were selected | Report §3.3 (all optimiser, schedule, epochs, patience, dropout, MLM settings), one-seed validation grid of five configurations, lexical-LR pilot on validation | PASS | Low: one-seed grid within seed noise (disclosed) | – | `outputs/results/hp*_val.json`, notebook Step 1 output |
| B3 | Baselines: relevant, justified, specific models cited, implementation/prompt described; **all baseline results from the group's own experiments**; no closed-source APIs | B0 majority, B1 TF-IDF+LR, B2 BiLSTM+attention, B3 RoBERTa-base fine-tuned (3 seeds), B4 Qwen2.5-1.5B zero-shot (open weights); all run in the notebook; no proprietary API anywhere | PASS | Low: B3/B4 test predictions were not saved per item (aggregates are in notebook outputs); B1 re-fitted here and matches exactly | – | `tools/verify_b1_reproduction.py` → `experiments/final_audit_20261006/b1_reproduction.json` (0 of 1,612 validation predictions differ; test accuracy, McNemar p, overlap identical) |
| B4 | Evaluation metrics defined, justified, computation explained | Accuracy (balanced labels, forced choice), mean ± sample std over 3 seeds, percentile bootstrap CI (2,000 resamples), exact McNemar | PASS | – | – | `src/evaluate.py`; `tests/test_evidence_consistency.py::test_mcnemar_exact_known_values` |
| B5 | Controlled ablations: question/hypothesis, why informative, how controlled | 9 ablations × 3 seeds on the selected configuration (notebook Step 3 table of hypotheses; report Table 2). Parameter change per ablation now shown; vanilla is declared compound | PASS | Medium: parameter-count and compound confounds (disclosed) | – | `outputs/results/ablation_significance.csv`; `tests/test_evidence_consistency.py` |
| B6 | Test set must NOT guide training, design or model selection; reserve for final evaluation | Final protocol: test labels hidden (placeholder) until `final_eval()`; every read logged (`outputs/test_access.log`: 2 lines, both after selection). **Historical exposure:** earlier runs were evaluated on test, and the decision to *explore* a lexical head followed those results (its configuration was chosen on validation) | PARTIAL | Medium (disclosed in report §3.6, notebook readme) | Cannot be repaired without a fresh held-out split; do not iterate further on test | `tools/check_test_access.py`, `tests/test_eval_hygiene.py`; audit reads logged in `experiments/final_audit_20261006/test_access.log` |
| B7 | No additional external datasets for training/evaluation (pretrained checkpoints allowed for baselines) | Only PIQA; MLM warm-up on training split only; RoBERTa/Qwen checkpoints used only as baselines. The data fallback downloads the *same* files from the official release and accepts them only if byte-identical | PASS | – | – | `src/data.py::download_piqa_release`, `tests/test_data_acquisition.py` |
| B8 | Must NOT copy implementation code from open-source projects | Code is project-specific in style and structure; no licence headers, third-party URLs or attributions in `src/`; standard library components only | NOT VERIFIED | Low | Team confirms authorship | `grep` of `src/` (log §49) |

## C. Results and analysis

| ID | Official requirement (abridged) | Repository evidence | Status | Risk | Required action | Verification |
|---|---|---|---|---|---|---|
| C1 | Overall comparison of main model vs baselines in tables/figures, discuss principal findings | Table 1 + §4.1; every value recomputed from saved predictions/results | PASS | – | – | `tools/check_report_numbers.py` (46 table values, 53 prose claims, 0 errors) |
| C2 | Ablation results: discuss whether they support the design | §4.2: combined design supported (all vanilla seeds below all full-model seeds on both splits); single components not consistently supported, said explicitly; Holm-corrected p-values | PASS | – | – | `tools/ablation_significance.py` |
| C3 | Qualitative attention analysis with informative visualisations of **successful and failed** predictions, giving inputs, expected answers and predictions | Report Figure 3: correct (mindfulness, p = 0.99), incorrect (lotion bars, p = 0.96), most uncertain (shipping/shopping, p = 0.50), each with gold and predicted option; notebook §3.3: 2 successes, 2 failures, 1 uncertain case with full attention maps; quantitative controls (Figure 2) | PASS | – | – | Figures rebuilt from `outputs/figures/*.png.json` (`report/make_figures.py`) |
| C4 | Critical discussion of limitations and unexpected findings; evidence-supported explanations | §4.4–4.6: RoBERTa instability, seed sensitivity, errors attend *more* to differing tokens, lexical over-confidence on long spans, tokenizer/GPU nondeterminism, test exposure; causal attention claims avoided | PASS | – | – | Report text review (log §49) |

## D. Report format

| ID | Official requirement | Repository evidence | Status | Risk | Required action | Verification |
|---|---|---|---|---|---|---|
| D1 | LaTeX with the **official ACL template** | `report/acl.sty` and `acl_natbib.bst` from acl-org; the IEEE citation override was removed so citations and the reference list use the template's own author-year style | PASS | Low | – | `report/CITS4012_73.tex` preamble |
| D2 | `\usepackage[final]{acl}` | Present | PASS | – | – | `tools/build_submission.py --check-only` |
| D3 | Only the group number in `\author` (e.g. `\author{Group 12}`), no names | `\author{Group 73}` (group ID confirmed by the group on 2026-10-06) | PASS | – | – | same |
| D4 | ≤ 6 pages of content incl. figures/tables; Team Contributions and References excluded | pdflatex (TeX Live 2023): abstract–Conclusion on pages 1–6, Team Contributions from page 7, references after; 3 lines of measured slack | PASS | Low: a different TeX distribution could shift lines; rebuild with `report/build.sh` and re-run the check | Re-check if rebuilt elsewhere | `tools/build_submission.py --check-only` (page rule), slack test (log §49) |
| D5 | References generated with BibTeX | `report/references.bib` (23 entries, all cited, all primary sources) | PASS | – | – | `bibtex` log: 0 errors, 0 warnings |
| D6 | Team Contributions section describing each member | Present (Bayron, Ali, Salah) | NOT VERIFIED | Medium | **Each member must confirm their paragraph is true** | Team |
| D7 | Report accurately describes the implemented system and experiments | Equations aligned with code; numbers machine-checked; architecture figure matches `DACT.forward` | PASS | – | – | `tools/check_report_numbers.py`; code review |

## E. Implementation notebook and reproducibility

| ID | Official requirement | Repository evidence | Status | Risk | Required action | Verification |
|---|---|---|---|---|---|---|
| E1 | MUST use the provided code template | Built from `templates/Copy_of_CITS4012_YourGroupID.ipynb`; the three section titles unmodified | PASS | – | – | `tools/build_submission.py --check-only` |
| E2 | Notebook contains the complete implementation | All eight modules embedded and identical to `src/` | PASS | – | – | `tools/sync_notebook.py --check` |
| E3 | Setup and execution instructions | Readme cell: Colab GPU, Run all, data acquisition (shared folder → official release, SHA-256 checked → zip), versions, runtime, re-run expectations | PASS | – | – | Notebook cell 1 |
| E4 | Saved training and evaluation logs and the reported results as outputs | Complete T4 run of `b880108` (33 training runs, 0 error outputs); supplementary cell (labelled, executed 2026-10-06 on CPU) shows the numbers the report takes from separate runs | PASS | Low | – | Notebook outputs; `tests/test_evidence_consistency.py` |
| E5 | Links/code to download and load saved models, if required | Not required: Run all retrains every model; pretrained baselines download from the HF Hub by name | NOT APPLICABLE | – | – | Readme cell |
| E6 | Document dependencies, settings, execution steps; notebook runs from a clean Colab runtime | Clean Colab run of `b880108` (Colab CLI, T4, 2026-10-03/04). Later cell changes proven not to alter results (bit-identical weights/predictions, deterministic CPU), data path tested offline and live; every code cell executes in the end-to-end smoke test. **No GPU re-run of the final cells was possible in this audit** | PARTIAL | Low | Optional: one Colab Run all of the final notebook (≈1 h 50 min on a T4); its outputs would replace the current ones | `tools/verify_post_run_equivalence.py`; `tests/test_notebook_smoke.py` |

## F. Submission

| ID | Official requirement | Repository evidence | Status | Risk | Required action | Verification |
|---|---|---|---|---|---|---|
| F1 | Submit exactly two files: `CITS4012_YourGroupID.pdf`, `CITS4012_YourGroupID.ipynb` | `submission/CITS4012_73.pdf`, `submission/CITS4012_73.ipynb`, byte-identical to the repository sources, with `SHA256SUMS` and `MANIFEST.json` | PASS | – | Upload these two files | `tests/test_submission.py` |
| F2 | One member submits via LMS before 18 Oct 2026 23:59 AWST | – | NOT APPLICABLE (pending) | – | Submit | – |
| F3 | Group of at most 3 students | Three names in Team Contributions | NOT VERIFIED | – | Team | – |

## Notes
- AI-assistance disclosure: the brief does not mention it, but `docs/DEVELOPMENT_LOG.md` (§19, §22, §27) and
  `docs/history/NOTE_FOR_SALAH.md` record extensive AI-assisted work (Claude Code) and that an AI-use paragraph was removed
  from the report at a member's request. Whether a declaration is required is a unit-policy question for the group;
  nothing in that record was removed by this audit.
