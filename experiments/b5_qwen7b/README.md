# B5: Qwen2.5-7B zero-shot baseline (separate experiment, 2026-10-07)

A larger pretrained **baseline**, run by the group on a free Colab T4. Kept separate from the main notebook
(`CITS4012_73.ipynb`), the report and `submission/`, which are unchanged. The main model (DACT) is trained from
scratch; this LLM is a baseline only, as the brief allows ("open-source LLMs", our own experiment).

| Model | Validation | Test (PIQA dev file) |
|---|---|---|
| **B5 Qwen2.5-7B, zero-shot, 4-bit NF4** | **81.76%** | **79.43%** (95% bootstrap CI 77.58–81.23) |
| B4 Qwen2.5-1.5B, zero-shot (main run) | 78.2% | 75.1% |
| DACT, main model (main run) | 61.3 ± 0.5% | 61.9 ± 0.3% |

McNemar against DACT's best-validation seed (42) on test: B5 alone correct on 540 items, DACT alone on 216,
exact p = 7.5e-33.

## Protocol
- Same split (`load_piqa`, seed 42), prompt (`Goal: {goal}\nSolution:` + each solution) and score (mean
  log-probability of the solution tokens, `score_continuations` from `src/baselines.py`) as B4; no training.
- Code: repository `main` at `d1e334a`; PIQA files SHA-256-verified (identical to the reported run).
- Model: `Qwen/Qwen2.5-7B`, Hugging Face revision `d149729398750b98c0af14eb82c78cfe92750796` (`main`, unchanged since
  2024-09-25), loaded with bitsandbytes NF4 4-bit (double quantisation, fp16 compute) because the fp16 model does not
  fit a 16 GB T4. Environment: torch 2.11.0+cu130, transformers 5.18.0, bitsandbytes 0.50.2, accelerate 1.15.0.
- Validation first (356 s). The test labels were then read once, by `load_piqa_with_test_labels`, which logged the
  access (`"B5 Qwen2.5-7B zero-shot baseline: final evaluation"`, 1,838 items, 2026-10-07 14:00:02 UTC).
  No decision about the main model uses this run.

## Evidence and limitations
- `B5_Qwen2.5-7B_zero_shot_output.ipynb`: the executed notebook with every output (0 error cells), including the
  validation accuracy, the full results JSON and the test-access log line. `B5_results.json` is copied from that
  output.
- The VM ended immediately after the run, before its files were downloaded, so the per-item prediction files and the
  JSONL log on the VM were lost. The printed outputs above are the record; re-running the notebook (about 35 min on a
  free T4) recreates the prediction files.
- The `revision` printed by the run is `None` (this transformers version did not expose the commit hash); the
  revision above is Hugging Face's `main` on the day of the run, unchanged since 2024-09-25.
- The access-log line shows `commit: unknown` because the notebook ran outside the repository directory; the code
  commit is recorded as `repo_commit: d1e334a` in the results.
