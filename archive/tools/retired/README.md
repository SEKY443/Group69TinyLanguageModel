# Retired tools

Kept for the record (nothing is deleted); not maintained and not run by CI.

- `build_notebook.py`: the audit track's notebook builder. It regenerated a notebook from the template and `Config()`,
  which after the merge is the lexical-head model, so its output no longer matched the submitted notebook.
  `tools/sync_notebook.py` is the **one** notebook builder: it copies `src/*.py` into the module cells of
  `CITS4012_69.ipynb` and leaves every other cell and output untouched (`--check` is run by CI).
- `CITS4012_69_reproducible.ipynb`: the notebook that `build_notebook.py` generated from the pre-merge `src/`
  (formerly `notebooks/`). Out of date since the merge.
- `final_checks.py`, `smoke_notebook.py`: checked / executed the generated notebook above. Replaced by
  `tests/test_notebook_smoke.py` (end-to-end smoke run of the real notebook, plus a "new VM" re-run) and the CI
  workflow `.github/workflows/ci.yml`.
- `audit_evidence.py`, `verify_model.py`: written for the audit track's intermediate `src/` (which no longer exists)
  and read `outputs/` as the original A100 evidence. With today's `src/` they would rebuild the lexical-head model from
  the original configs; with `evidence/historical/src/` they fail to import (`environment_info` is missing there).
  Their outputs in `report_support/` stay as committed evidence. The checks they made are covered by `tests/`
  (masks, symmetry, truncation, lexical head, guard) and `tools/verify_audit_claims.py` (data audit).
- `append_audit_cells.py`: appended audit cells, executed against the original model, to the root `CITS4012_69.ipynb`.
  Since the merge that file is the submission notebook, so running the tool would modify it. Its result is kept in
  `evidence/historical/CITS4012_69_audited_A100.ipynb`.
