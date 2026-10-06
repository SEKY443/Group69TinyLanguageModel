# Archive

Superseded material, kept unchanged for the record. Nothing here is needed to run the notebook, build the report or
run the tests, and nothing here was deleted. Paths written inside these files refer to the layout at the time they
were made (see the relocation table below).

| Folder | Contents |
|---|---|
| `original_a100_run/` | The original A100 notebook (exact and audited copies), its sources, builder and outputs, with the SHA-256 manifest `sha256.json` |
| `development_notebooks/` | Development smoke-test notebooks and the audit track's replication-verification notebooks |
| `original_model_notes/` | The audit track's notes, tables and figures about the ORIGINAL model (before the lexical head) |
| `tools/` | Scripts of the audit track (replication of the original model, notebook review, inventories) and `retired/` tools; several overwrite or regenerate files and must not be run on the current repository |

## Relocations (2026-10-06)

| Old path | New path |
|---|---|
| `CITS4012_69.ipynb` | `CITS4012_73.ipynb` (repository root) |
| `report/CITS4012_69.tex`, `.pdf` | `report/CITS4012_73.tex`, `.pdf` |
| `evidence/historical/` | `archive/original_a100_run/` |
| `evidence/*.ipynb`, `evidence/archive/` | `archive/development_notebooks/` |
| `report_support/` | `archive/original_model_notes/` |
| `tools/retired/` | `archive/tools/retired/` |
| `tools/{analyse_replication, build_replication_evidence, project_paths, refinement_notes, replicate_lexical_baselines, replication_report_figures, report_figures, reproduce_main, review_historical_notebook, write_file_inventory}.py` | `archive/tools/` |
| `docs/{PROJECT_AUDIT, REFINEMENT_SUMMARY, TRAINING_AUDIT, NOTE_FOR_SALAH}.md`, `docs/{organisation_manifest, path_relocations}.json` | `docs/history/` |
| `references.bib` (root copy) | removed; identical to `report/references.bib` |
| `requirements-a100.txt` | removed; it only included `requirements-lock.txt` |

Earlier relocations (2026-09-30) are recorded in `docs/history/path_relocations.json`.
