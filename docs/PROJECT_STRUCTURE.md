# Project folder guide (current as of the final audit, 2026-10-06)

Run every tool from the repository root. This guide replaces the 2026-09-30 version, which described the
pre-merge layout (folders `notebooks/` and `experiments/completed/` no longer exist; see git history and
`path_relocations.json`).

## Deliverables

| Location | Purpose |
|---|---|
| `CITS4012_69.ipynb` | Executed submission notebook: course template, all code, outputs of the reported T4 run (`b880108`) and a labelled supplementary-evidence cell |
| `report/CITS4012_69.tex`, `.pdf` | ACL report (final mode, Group 73); `make_figures.py` builds `figures/` from `outputs/`; `build.sh` compiles |
| `submission/` | `CITS4012_73.pdf`, `CITS4012_73.ipynb` (byte copies), `SHA256SUMS`, `MANIFEST.json`; built by `tools/build_submission.py` |

## Code and checks

| Location | Purpose |
|---|---|
| `src/` | `config`, `data`, `model`, `baselines`, `train`, `evaluate`, `experiments`, `viz`; embedded verbatim in the notebook |
| `tests/` | pytest suite; `conftest.py` writes synthetic PIQA-format data so no real data is needed |
| `tools/` | sync, checking, verification, analysis and packaging scripts; `tools/retired/` holds superseded audit-track tools (README inside) |
| `.github/workflows/ci.yml` | CI: sync check, test-access check, report numbers, submission format, pytest |
| `requirements-lock.txt` | exact package versions checked by the notebook (`requirements.txt`: ranges; `requirements-a100.txt`: alias) |

## Evidence

| Location | Purpose |
|---|---|
| `outputs/` | Reported run: `logs/` (JSONL per run), `predictions/` (per item, gzip), `results/` (JSON/CSV, incl. post-run analyses), `figures/` (+ attention data `.png.json`), `tokenizer.json`, `run_config.json`, `data_manifest.json`, `test_access.log` |
| `experiments/a100_run_0d70cdc/` | Previous full run of the final model (Colab Pro A100) |
| `experiments/t4_run_538cf5b/`, `t4_run_e5fdb29/` | Earlier free-T4 runs |
| `experiments/a100_runs_20261001/` | The two A100 runs of 2026-10-01 compared (`COMPARISON.md`) |
| `experiments/goal_matching/`, `length_bucketing/`, `roberta_stability/` | Pre-registered experiments (validation only), each with `PREREGISTRATION.md` and results |
| `experiments/lexical_pilot/`, `b1_versions/`, `calibration/`, `checkpoint_selection/`, `t4_timing/` | Small validation-only analyses |
| `experiments/final_audit_20261006/` | Final audit: post-run equivalence, B1 reproduction, its own `test_access.log` |
| `evidence/historical/` | Original A100 notebook, sources and outputs with SHA-256 manifest (unchanged) |
| `evidence/` (other) | Development smoke notebooks and the audit track's replication notebook |
| `report_support/` | Audit-track notes, tables and figures about the ORIGINAL model (historical) |
| `templates/` | Official course notebook template (unmodified) |

## Documentation

| File | Purpose |
|---|---|
| `docs/DEVELOPMENT_LOG.md` | Chronological record of all work (§49: final audit) |
| `docs/FINAL_MARKING_AUDIT.md` | Final assessment against the brief and marking guide |
| `docs/FINAL_REQUIREMENTS_TRACEABILITY.md` | Every official requirement → evidence → status |
| `docs/FINAL_SUBMISSION_CHECKLIST.md` | Pre-submission checklist |
| `docs/PROJECT_AUDIT.md`, `REFINEMENT_SUMMARY.md`, `TRAINING_AUDIT.md`, `NOTE_FOR_SALAH.md` | Earlier audits and hand-over notes (historical; later findings take precedence) |
| `docs/organisation_manifest.json`, `path_relocations.json` | Hashes and path map of the 2026-09-30 reorganisation |

`data/` (the PIQA files), `outputs/checkpoints/` and caches are git-ignored. Checkpoints of the reported run are not
distributed; `Run all` retrains every model.
