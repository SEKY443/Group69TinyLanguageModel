# Final submission checklist (Group 73)

**Final version, 2026-10-07.** After the checks below (2026-10-06), the report text was revised for readability (no number changed) and rebuilt with pdflatex, the notebook's explanatory text was polished (code and saved outputs unchanged), and `submission/` was re-frozen: `CITS4012_73.pdf` `7fa7414fa61d`, `CITS4012_73.ipynb` `4161ff0ff92f` (source commit `294eb64`). `python tools/build_submission.py --check-only`, `python tools/check_report_numbers.py` and `python -m pytest` (79 tests) pass on this version. Unless the group raises further objections, this is the final result; only the Team Contributions row below still needs each member's confirmation.

Checked on 2026-10-06 against the official brief. PASS needs direct evidence; the command or file is given.
Re-run everything with: `python tools/build_submission.py` (checks + package) and `python -m pytest`.

## Required files

| Item | Status | Evidence |
|---|---|---|
| `CITS4012_73.pdf` | PASS | `report/CITS4012_73.pdf`; frozen copy `submission/CITS4012_73.pdf` (SHA-256 in `submission/SHA256SUMS`) |
| `CITS4012_73.ipynb` | PASS | `CITS4012_73.ipynb`; frozen copy `submission/CITS4012_73.ipynb` (SHA-256 in `submission/SHA256SUMS`) |

## Main model

| Item | Status | Evidence |
|---|---|---|
| Implemented by the team | PARTIAL | Project-specific code, no copied third-party code found; authorship can only be confirmed by the team |
| Trained from scratch | PASS | `test_main_model_code_path_loads_no_pretrained_weights`, `test_dact_starts_from_random_initialisation` |
| Approved sequence architecture (Transformer) | PASS | `src/model.py` `EncoderLayer` |
| Meaningful attention | PASS | attention tests in `tests/test_model_properties.py`; report §4.3 controls |
| Task-specific design | PASS | difference tags, cross-solution comparison, tag-biased pooling, lexical head (report §2) |
| Architecture described accurately | PASS | equations aligned with `DACT.forward`; Figure 1 redrawn from the code path |

## Data

| Item | Status | Evidence |
|---|---|---|
| Correct PIQA dataset | PASS | SHA-256 = run manifest = official PIQA release |
| No prohibited external datasets | PASS | only PIQA; pretrained checkpoints only for B3/B4 |
| Split handling documented | PASS | report §3.1; ordered split hashes reproduce |
| No unintended leakage | PASS | labels hidden until `final_eval`; tokenizer/MLM on training split only; label-independent inputs tested; overlap (item 1545) disclosed |
| Preprocessing reproducible | PARTIAL | deterministic except BPE training in `tokenizers` (disclosed); run tokenizer saved |

## Experiments

| Item | Status | Evidence |
|---|---|---|
| Own baseline experiments | PASS | B0–B4 run in the notebook; B1 reproduced exactly |
| Controlled ablations | PASS | 9 × 3 seeds; parameter changes and compound vanilla disclosed |
| Appropriate metrics | PASS | accuracy, seed mean ± std, bootstrap CI, McNemar + Holm |
| Test set used only within a defensible boundary | PARTIAL | final run compliant and logged; historical exposure disclosed |
| Results traceable to artifacts | PASS | `tools/check_report_numbers.py` (46 + 53, 0 errors); `tests/test_evidence_consistency.py` |

## Attention

| Item | Status | Evidence |
|---|---|---|
| Quantitative / technical validation | PASS | distributions, masking, gradient, effect tests; learned vs built-in controls |
| Success example | PASS | Figure 2 top (mindfulness, p = 0.99); notebook #993, #1300 |
| Failure example | PASS | Figure 2 bottom (lotion bars, p = 0.96); notebook #747, #841 |
| Expected answer shown | PASS | gold option marked in every case |
| Prediction shown | PASS | predicted option and probability in every case |
| No unsupported causal interpretation | PASS | report §4.3/§4.4 and notebook discussion use descriptive, hedged wording |

## Notebook

| Item | Status | Evidence |
|---|---|---|
| Complete implementation | PASS | eight modules embedded, identical to `src/` |
| Setup instructions | PASS | Readme cell |
| Execution instructions | PASS | Readme cell |
| Saved training logs | PASS | Step 1–4 outputs |
| Saved evaluation logs | PASS | Step 5 outputs, consistency-check cell |
| Reported results | PASS | results table, key-numbers cell, supplementary-evidence cell |
| Checkpoint/model-loading instructions if needed | NOT APPLICABLE | Run all retrains; stated in the Readme cell |
| No broken paths | PASS | data path tested offline and live; no local-machine paths in code cells |
| Clean-runtime verified | PARTIAL | clean Colab T4 run of `b880108`; later changes proven equivalent; no GPU re-run of the final cells |

## Report

| Item | Status | Evidence |
|---|---|---|
| LaTeX | PASS | `report/CITS4012_73.tex` |
| Official ACL template | PASS | `acl.sty`, `acl_natbib.bst` unmodified; numeric citations via natbib options |
| Final mode | PASS | `\usepackage[final]{acl}` |
| `\author{Group 73}` only | PASS | `tools/build_submission.py --check-only` |
| ≤ 6 pages of content | PASS | content pages 1–6, 3 lines of slack |
| Contribution section excluded from limit | PASS | starts page 7 |
| References excluded from limit | PASS | after Team Contributions |
| BibTeX | PASS | `report/references.bib`, 23 entries, no warnings |
| Architecture diagram | PASS | Figure 1 |
| Equations | PASS | Eq. 1–4 |
| Baselines | PASS | §3.4, Table 1 |
| Ablations | PASS | §4.2, Table 2 |
| Attention | PASS | §4.3 |
| Success/failure analysis | PASS | §4.4, Figure 2 |
| Limitations | PASS | §4.6 |
| Conclusion | PASS | §5 |
| Accurate implementation description | PASS | code review + number checker |
| Team contributions truthful | NOT VERIFIED | **each member must confirm** |

## Consistency

| Item | Status | Evidence |
|---|---|---|
| report == code | PASS | equations/figure vs `src/model.py`; hyper-parameters vs `dact_full_val.json` |
| report == notebook | PASS | every reported number appears in notebook outputs (main run or labelled supplementary cell) |
| report == logs | PASS | `tools/check_report_numbers.py` |
| tables == prediction artifacts | PASS | `tests/test_evidence_consistency.py` |
| figures == real experiments | PASS | Figure 2 built from `outputs/` (`report/make_figures.py`) |
| hyper-parameters == executed configuration | PASS | `outputs/results/dact_full_val.json` |
| final commit == executed version | PARTIAL | outputs from `b880108`; later module-cell changes proven not to alter weights/predictions (`post_run_equivalence.json`); data-acquisition change tested |

## Submission hygiene

| Item | Status | Evidence |
|---|---|---|
| No debug clutter in the deliverables | PASS | no TODO/FIXME/debug prints; no error outputs |
| No irrelevant temporary files | PASS | unused figure PNGs removed; caches git-ignored |
| No assistant/chat transcripts in the deliverables | PASS | grep of the report and notebook |
| No Claude/ChatGPT development chatter in the deliverables | PASS | same |
| No placeholder text | PASS | template instruction line in the title cell replaced |
| No fabricated evidence | PASS | the supplementary cell's provenance is stated; no output was edited |
| Acknowledgements/citations preserved | PASS | 23 references; development-log record of AI assistance untouched |
| Academic-integrity requirements respected | NOT VERIFIED | **group decision on an AI-use statement** (see `FINAL_MARKING_AUDIT.md` §10) |
