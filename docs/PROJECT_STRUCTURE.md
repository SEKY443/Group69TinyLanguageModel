# Project folder guide

Run every tool from the repository root. The repository keeps its original name (`Group69TinyLanguageModel`); the
group's assigned ID is 73.

## Submission files

| Location | Purpose |
|---|---|
| `CITS4012_73.ipynb` | Submission notebook: course template, all code, saved outputs of the reported T4 run (`b880108`) and a labelled supplementary-evidence cell |
| `report/CITS4012_73.tex`, `.pdf` | Report (ACL template, final mode, Group 73); `make_figures.py` builds `figures/` from `outputs/`; `build.sh` compiles |
| `submission/` | Frozen snapshot of the two files to upload, with `SHA256SUMS` and `MANIFEST.json`; built by `tools/build_submission.py` |

## Code and checks

| Location | Purpose |
|---|---|
| `src/` | `config`, `data`, `model`, `baselines`, `train`, `evaluate`, `experiments`, `viz`; embedded verbatim in the notebook |
| `tests/` | pytest suite; `conftest.py` writes synthetic PIQA-format data, so no real data is needed |
| `tools/` | Notebook sync, checks (test access, report numbers, submission format), verification, analysis and packaging scripts |
| `.github/workflows/ci.yml` | CI: sync check, test-access check, report numbers, submission format, pytest |
| `requirements-lock.txt`, `requirements.txt` | Exact package versions checked by the notebook; version ranges for development |
| `templates/` | The official course notebook template (unmodified) |

## Results and experiments

| Location | Purpose |
|---|---|
| `outputs/` | Reported run: `logs/` (JSONL per run), `predictions/` (per item, gzip), `results/` (JSON/CSV, including post-run analyses), `figures/` (with attention data `.png.json`), `tokenizer.json`, `run_config.json`, `data_manifest.json`, `test_access.log` |
| `experiments/a100_run_0d70cdc/` | Previous full run of the final model (Colab Pro A100) |
| `experiments/t4_run_538cf5b/`, `t4_run_e5fdb29/` | Earlier free-T4 runs |
| `experiments/a100_runs_20261001/` | The two A100 runs of 2026-10-01 compared (`COMPARISON.md`) |
| `experiments/goal_matching/`, `length_bucketing/`, `roberta_stability/` | Pre-registered experiments (validation only), each with `PREREGISTRATION.md` and results |
| `experiments/lexical_pilot/`, `b1_versions/`, `calibration/`, `checkpoint_selection/`, `t4_timing/` | Small validation-only analyses |
| `experiments/final_audit_20261006/` | Final audit: post-run equivalence, B1 reproduction, its own `test_access.log` |

## Documentation

| File | Purpose |
|---|---|
| `docs/DEVELOPMENT_LOG.md` | Chronological record of all work |
| `docs/FINAL_MARKING_AUDIT.md` | Final assessment against the brief and the marking guide |
| `docs/FINAL_REQUIREMENTS_TRACEABILITY.md` | Every official requirement, its evidence and status |
| `docs/FINAL_SUBMISSION_CHECKLIST.md` | Pre-submission checklist |
| `docs/history/` | Earlier audits, hand-over notes and the 2026-09-30 reorganisation manifests |

## Archive

`archive/` holds superseded material that is kept for the record: the original A100 run, development notebooks, the
audit track's notes on the original model, and audit-track and retired tools. `archive/README.md` lists every
relocation.

`data/` (the PIQA files), `outputs/checkpoints/` and caches are git-ignored. Checkpoints of the reported run are not
distributed; `Run all` retrains every model.
