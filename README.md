# Group 69: PIQA / Difference-Aware Contrastive Transformer (DACT)

CITS4012 group project. Dataset: PIQA (Physical Interaction QA). Main model: **DACT**, a randomly initialised,
shared Transformer with difference-tag embeddings, cross-solution comparison, difference-biased pooling and a
jointly trained lexical head. Pretrained RoBERTa/Qwen are baselines only.

This repository combines two tracks of work (merged on 2026-10-01, see `docs/DEVELOPMENT_LOG.md` section 28):
- **`revision-2`:** the final model with the lexical head, its full Colab run, and the ACL report draft (log sections 22–27).
- **`main` (Salah Elshafey):** an evidence audit of the original A100 run, robustness and provenance code, and a local replication of the original model (log sections S1–S3).

## Where to find things

```text
Group69TinyLanguageModel/
├── CITS4012_69.ipynb        # submission candidate: executed run of the final model (A100, 2026-10-01)
├── report/                  # ACL report draft: CITS4012_69.tex, CITS4012_69.pdf, figures, make_figures.py
├── outputs/                 # logs, results and figures of that final run
├── src/                     # model, data, training and evaluation (merged code of both tracks)
├── tools/                   # sync, pilot, build, audit, experiment and checking scripts
├── references.bib           # BibTeX used by the report
├── docs/                    # development log, audit, handover and folder guide
├── evidence/historical/     # original A100 notebook (exact + audited), sources, SHA-256 manifest,
│   └── a100_outputs/        #   and the original A100 logs, results and figures
├── report_support/          # audit notes, tables and figures about the ORIGINAL model
├── notebooks/               # current-source reproduction candidate from the audit track
├── experiments/             # archived runs: T4 run of the final model, A100 runs of 2026-10-01 incl. Salah's version
├── templates/               # original course notebook template
└── requirements*.txt        # local and historical environments
```

| Artifact | Meaning |
|---|---|
| `CITS4012_69.ipynb` | **Final executed notebook** (commit `0d70cdc` code, Colab Pro A100): DACT 61.2 / 61.6 % val / test, all baselines, 8 ablations, attention analysis, consistency check |
| `report/CITS4012_69.pdf` | Report draft built from that run; Team Contributions still to be written by the group |
| `outputs/` | Logs, results, figures and per-item predictions (gzip) of the final run (40 JSON-lines logs) |
| `experiments/a100_runs_20261001/` | Both A100 runs of 2026-10-01 compared (`COMPARISON.md`): current version vs Salah's version |
| `experiments/t4_run_538cf5b/` | Outputs of the earlier free-T4 run of the final model (61.1 / 61.6 %) |
| `evidence/historical/CITS4012_69.ipynb` | Exact original A100 notebook (run 1) |
| `evidence/historical/CITS4012_69_audited_A100.ipynb` | Original A100 notebook with the audit track's corrected narrative and appended audit cells |
| `evidence/historical/a100_outputs/` | Original A100 logs, results, tokenizer and attention figures, unchanged |
| `report_support/`, `notebooks/` | Audit-track material; describes the **original** model (without the lexical head) |

## Findings of the audit track that apply to every run
- One normalised test item also appears twice in the training data.
- The `difflib` alignment is directional, so swapping the two options can change the difference tags. Set `Config.symmetric_diff_tags=True` to use canonical ordering. This is off by default, so recorded results are unchanged.
- The original A100 run's checkpoints and predictions were not kept. New runs save their predictions to `outputs/predictions/`.

## Running the final notebook (Google Colab)
1. Open `CITS4012_69.ipynb` in Colab and select a GPU runtime.
2. Run `Runtime → Run all`. It takes about 30–35 minutes on an A100, or about 1 h 50 min on a free T4.
   - The PIQA files are downloaded automatically from the unit's shared folder.
   - `MyDrive/Group69/PIQA.zip` is used only if that download fails.
   - Every run directory must be new, because the code refuses to overwrite existing logs or checkpoints.
3. At the end, `outputs.zip` (logs, results, figures) is downloaded to your computer.

**Note:** the saved outputs of `CITS4012_69.ipynb` were produced by the `revision-2` code. After the merge, `src/` also contains the audit track's additions:
- saved predictions;
- run provenance;
- validation loss and finite-loss checks;
- a selected-position MLM head;
- optional determinism.

These add checks and records and don't change the model's defaults. Even so, one fresh run of the merged notebook is needed before submission, so that the saved outputs come from exactly the submitted code.

## Keeping the notebook and `src/` in sync
After editing a module in `src/`, run:

```bash
python tools/sync_notebook.py
```

## Tests and checks (CPU, about 2 minutes)

```bash
pip install -r requirements-lock.txt pytest      # or a CPU torch build plus the locked scientific stack
python -m pytest                                  # 25 tests; add -m "not slow" to skip the notebook run
python tools/sync_notebook.py --check             # the notebook's module cells equal src/
python tools/check_test_access.py                 # no test labels before the final-results cell
python tools/check_report_numbers.py              # report tables equal their result files
```

The same checks run on GitHub Actions (`.github/workflows/ci.yml`). The audit track's old tools (`audit_evidence.py`,
`verify_model.py`, `final_checks.py`, …) were written for the pre-merge model and are kept in `tools/retired/` (see
its README); their outputs remain in `report_support/`.

## Submission
The brief requires exactly two files:
- `CITS4012_69.ipynb`
- `CITS4012_69.pdf` (ACL LaTeX in final mode, "Group 69" as the only author, at most six pages of main content, BibTeX references, and the real team contributions)

Label the historical A100 results and the audit track's local replications separately from the final run.
