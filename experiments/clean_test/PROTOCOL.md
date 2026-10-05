# Protocol: a clean test split (committed before any run under it)

Author: ALI. Date: 2026-10-05. Reason: the PIQA development file (1,838 labelled items), used as the test set in every
earlier run, was evaluated during development, and the lexical component was introduced after those results had been
seen. The specification requires that the test set never guides training, design or model selection. A test split
that was never evaluated on fixes this.

## Split (`src/data.py: split_indices`, `Config.test_source = "train_holdout"`)
- Source: the official PIQA training file (16,113 items).
- Validation: exactly the earlier split (stratified 10 %, seed 42; 1,612 items), verified identical by index.
- Test: 1,838 items drawn, stratified, with seed 2026, from the remaining 14,501, restricted to items whose goal and
  solution pair (in either order) occurs only once in the training file (13 duplicated items excluded from the pool).
- Train: the remaining 12,663 items.
- Checked without reading model predictions: the three splits are disjoint; no test item is duplicated anywhere in the
  training file; no test item equals a PIQA development item.
- Why these items are clean: in earlier runs they were training items. No model was ever evaluated on them, so no
  decision (architecture, hyper-parameters, lexical component, selection) was based on their results.

## What does not change
The model, every hyper-parameter, the grid, the ablations, the baselines (including the declared RoBERTa restart rule)
and the seeds are exactly those of commit `b880108` and later. No design decision is made in this run.

## Run
One full run of `CITS4012_69.ipynb` on a free Colab T4. The test labels are read once, by `final_eval()` in Step 5,
and the read is logged in `outputs/test_access.log`. Every result reported from this run, including any that look
worse than before, is reported. The earlier development-file results remain in `experiments/` as an exposed
development set and are labelled as such.
