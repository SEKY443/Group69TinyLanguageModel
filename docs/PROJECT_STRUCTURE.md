# Project folder guide

The repository root contains the executed submission notebook, README, dependency files and the
main working folders. The supplied PIQA data and assignment PDF remain in the parent assignment
directory. No dataset, training result, checkpoint or original source snapshot was deleted.

| Location from repository root | Purpose |
|---|---|
| `CITS4012_69.ipynb` | Executed submission candidate; original code/output evidence preserved |
| `notebooks/CITS4012_69_reproducible.ipynb` | Current-source reproduction notebook; full Colab execution pending |
| `templates/Copy_of_CITS4012_YourGroupID.ipynb` | Unmodified official notebook template |
| `src/` | Eight implementation modules |
| `tools/` | Notebook generation, experiments, audits, verification and figure generation |
| `docs/DEVELOPMENT_LOG.md` | Chronological work history and corrections |
| `docs/PROJECT_AUDIT.md` | Original pre-refinement audit; intentionally historical |
| `docs/REFINEMENT_SUMMARY.md` | Detailed 20-section handover |
| `report_support/` | Report text support, result tables, figures, references and check manifests |
| `outputs/` | Original A100 logs/results/tokenizer/figures; unchanged |
| `experiments/completed/outputs_replication_efficient/` | Completed local three-seed DACT replication, including weights and predictions |
| `experiments/completed/outputs_replication_baselines/` | Completed local B0/B1 replication and fitted lexical model |
| `experiments/development/` | Preserved smoke/functional-check runs and interrupted first attempt |
| `evidence/historical/` | Exact original notebook/source/builder with original hashes |
| `evidence/archive/` | Previous executed verification notebook, retaining pre-organisation paths |
| `evidence/` | Executed smoke notebooks and current saved-artifact verification notebook |
| `.venv/`, `tmp/` | Local environment and temporary tooling; not submission content |

## Commands and paths

Run tool commands from the **repository root**, not from `notebooks/` or `tools/`. The README has current
commands. The notebook builder reads the course template from `templates/` and writes its unexecuted
candidate into `notebooks/`. Newly generated notebooks put smoke runs in `experiments/development/`
and full runs in `experiments/completed/`. The local main-replication command likewise defaults to a
fresh directory under `experiments/completed/`. The original executed submission notebook is unchanged.

`path_relocations.json` maps the previous layout to this one. Saved experiment JSON/logs may record
old checkpoint paths because changing those would change historical evidence. Active analysis tools
use `tools/project_paths.py` to resolve those paths; do the same when loading old metadata. Historical
notebooks, snapshots and chronological log sections describe the layout at the time of execution.
For a current verification run use `tools/analyse_replication.py` or the new verification notebook;
the notebook in `evidence/archive/` documents the earlier execution and is not the current launcher.

`organisation_manifest.json` records SHA-256 hashes before relocation. Every moved file was verified
immediately after the move. Subsequent deliberate path/documentation updates are recorded in the log;
experiment artifacts and archived executed evidence retain their content.

## Submission and backup

The final required names remain `CITS4012_69.ipynb` and `CITS4012_69.pdf`; the report PDF is still pending.
Keep the working report assets here, and place the final PDF beside the submission notebook when ready.
Completed/development run directories and weights remain Git-ignored. Back up the completed runs
explicitly when moving machines; Git alone does not include them. No remote push was performed.
