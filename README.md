# CITS4012 Group 73: Difference-Aware Contrastive Transformer (DACT) for PIQA

CITS4012 Natural Language Processing group project (UWA, 2026). Dataset: **PIQA** (physical commonsense, two
candidate solutions per goal). Main model: **DACT**, a Transformer trained from scratch that marks where the two
near-identical solutions differ (difference-tag embeddings), compares them with cross-solution attention, pools
around the differing tokens and adds a jointly trained hashed lexical head. Pretrained RoBERTa-base and
Qwen2.5-1.5B appear only as baselines. (The repository keeps its original name, `Group69TinyLanguageModel`; the
group's assigned ID is **73**.)

## Submission

| Required file | In this repository | Source |
|---|---|---|
| `CITS4012_73.pdf` | `report/CITS4012_73.pdf` | frozen copy in `submission/` |
| `CITS4012_73.ipynb` | `CITS4012_73.ipynb` | frozen copy in `submission/` |

`python tools/build_submission.py` checks the brief's format rules (course template, `\author{Group 73}`, final
mode, six-page limit, no error outputs, notebook = `src/`) and rebuilds `submission/` with `SHA256SUMS` and a freeze
`MANIFEST.json`. Before submitting, the group must confirm the Team Contributions paragraphs and decide on an AI-use
statement (`docs/FINAL_MARKING_AUDIT.md` §10).

## Results of the reported run (free Colab T4, commit `b880108`, accuracy %, 3 seeds)

| Model | Validation | Evaluation ("test") |
|---|---|---|
| **DACT (main model, 1.93 M parameters)** | **61.3 ± 0.5** | **61.9 ± 0.3** |
| B1 TF-IDF + logistic regression | 59.2 | 61.2 (McNemar p = 0.62 vs DACT) |
| B2 BiLSTM + attention | 59.3 ± 0.7 | 58.7 ± 1.1 |
| B3 RoBERTa-base fine-tuned (2 of 3 seeds failed; validation re-run 69.0 ± 0.6) | 59.2 ± 5.8 | 57.3 ± 6.9 |
| B4 Qwen2.5-1.5B zero-shot | 78.2 | 75.1 |
| Vanilla Transformer (four DACT components removed) | 57.5 ± 1.0 | 57.6 ± 1.0 |

The evaluation split is PIQA's labelled development set (byte-identical to the unit's `test` files). It was
evaluated in earlier development runs, so it is not an untouched estimate (report §3.6). All numbers are in
`outputs/results/` and are recomputed from per-item predictions by `tests/test_evidence_consistency.py`.

## Repository layout

```text
CITS4012_73.ipynb      executed submission notebook (course template; outputs of the reported run)
report/                ACL report: CITS4012_73.tex/.pdf, acl.sty, references.bib, figures/, make_figures.py, build.sh
submission/            the two files to upload (CITS4012_73.*), SHA256SUMS, MANIFEST.json
src/                   model, data, training, baselines, evaluation, experiment and visualisation modules
                       (embedded verbatim in the notebook; tools/sync_notebook.py keeps them identical)
outputs/               reported run: logs/, predictions/ (gzip), results/, figures/, tokenizer, run_config, test_access.log
tests/                 pytest suite (model properties, evidence consistency, evaluation hygiene, resume, notebook smoke run)
tools/                 checking, verification, analysis and packaging scripts
experiments/           archived earlier runs and separate pre-registered experiments (each folder has its own record)
docs/                  development log, final audit, requirements traceability, submission checklist, earlier audits
archive/               superseded material: original A100 run, development notebooks, notes on the original model,
                       audit-track and retired tools (see archive/README.md)
templates/             the official, unmodified course notebook template
```

## Running the notebook (clean Google Colab)

1. Open `CITS4012_73.ipynb` in Colab and select a GPU runtime.
2. `Runtime → Run all`: about 1 h 50 min on a free T4, 30–35 min on an A100.
   - **Data:** the four PIQA files are downloaded from the unit's shared folder. If that fails, the official PIQA
     release is downloaded from GitHub and used only if its SHA-256 equals the unit's files. The last fallback is
     `MyDrive/Group73/PIQA.zip`. Every source is checked against the hashes of the reported run.
   - **Packages:** everything is pre-installed on Colab; the first cell compares versions with
     `requirements-lock.txt`.
   - **Checkpoints:** none are needed; every model is retrained.
3. Results reproduce up to seed-level variation, not digit for digit: GPU kernels and BPE training in the
   `tokenizers` library are not deterministic.

## Local checks (CPU, about 2 minutes)

```bash
pip install -r requirements-lock.txt pytest nbformat nbclient ipykernel pypdf
python -m pytest                              # 79 tests; -m "not slow" skips the end-to-end notebook run
python tools/sync_notebook.py --check         # notebook module cells == src/
python tools/check_test_access.py             # no test-label access before the final-results cell
python tools/check_report_numbers.py          # every table value and result in the prose == its result file
python tools/build_submission.py --check-only # format rules of the brief
```

The same checks run on GitHub Actions (`.github/workflows/ci.yml`).

## Verification tools (final audit)

**Run without the PIQA files:**

| Command | What it shows |
|---|---|
| `python tools/verify_post_run_equivalence.py` | Code changes after the reported run leave trained weights and predictions bit-identical (DACT with MLM warm-up, BiLSTM; deterministic CPU). |
| `python tools/ablation_significance.py` | McNemar tests of all ablations on both splits, Holm-adjusted, plus seed separation and parameter changes, from saved predictions. |

**Need the PIQA files in `data/piqa/`:**

| Command | What it shows |
|---|---|
| `python tools/verify_b1_reproduction.py --data-dir data/piqa --out-dir <dir>` | Re-fits B1 and matches the saved validation predictions item by item, and the reported test numbers. Logs its test-label read in `<dir>/test_access.log`. |
| `python tools/rescore_without_duplicate.py --data-dir data/piqa` | Effect of the one duplicated evaluation item. |

## Building the report

```bash
python report/make_figures.py   # figures from outputs/ (matplotlib)
sh report/build.sh              # pdflatex + BibTeX; prints pages, overfull boxes and warnings
```

## Known limitations (also in the report)

- The evaluation split was exposed during development; the lexical head was explored after earlier test results
  (configured on validation only).
- The hyper-parameter grid used one seed.
- Ablations change parameter counts, and the vanilla ablation removes four components at once.
- One evaluation item also appears in the training data.
- The `difflib` difference tags depend on the option order (102 evaluation items).
- B3/B4 per-item test predictions were not saved.
- The run's checkpoints are not distributed.

## History

`docs/DEVELOPMENT_LOG.md` records every step, including failed experiments and earlier runs (archived in
`experiments/` and `archive/`). The final audit is `docs/FINAL_MARKING_AUDIT.md`, with
`docs/FINAL_REQUIREMENTS_TRACEABILITY.md` and `docs/FINAL_SUBMISSION_CHECKLIST.md`.
