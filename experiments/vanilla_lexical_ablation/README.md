# Vanilla + lexical head ablation (validation only)

Question: how much of the 3.7-point validation drop of the vanilla Transformer (Table 2) is the lexical head alone?

- **Arms** (selected configuration, d=128, 2 layers, 10-epoch MLM warm-up, seeds 42-44):
  `vanilla_plus_lexical` = vanilla (no difference tags, no cross-solution attention, mean pooling) with the lexical head on;
  `vanilla_control` = the plain vanilla model, re-run in the same session to check that sessions are comparable
  (57.40 here against 57.55 in the final run).
- **Validation only.** `experiments.run_experiment` never reads test labels; the evaluation split was not scored and no
  `final_eval` was opened. Table 2 therefore shows `--` for the evaluation column of this row.
- **Run:** 2026-10-13, free Colab T4 through the Colab CLI (`colab run --gpu T4`), repository at commit b4906d3, official PIQA
  files (SHA-256 identical to the unit's files), fresh output directory (the repository ships its own `outputs/`, which is
  never overwritten). `run_abl.py` is the script that was executed; `run_cli.log` is its complete output. The traceback at the end
  of the log is the CLI's reply timeout after the script printed `DONE`; both result lines were printed before it.
- `results.csv`: per-seed best validation accuracy (fractions), mean, sample std (n-1, as `summarise_seeds`) and trainable parameters.
- Not independent of the final run: it uses the same split and tokeniser recipe, but BPE training and GPU kernels are not
  deterministic, so numbers match the final run only up to seed-level variation.
