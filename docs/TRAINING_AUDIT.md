# Training audit (TRAINING_UPGRADE_PROMPT.md), 2026-10-04, branch `upgrade/training-hardening`

Each recommendation of the generic "QLoRA / SFTTrainer / W&B" checklist, mapped to this project. Evidence paths are in
the repository; all accuracy comparisons use the validation split only.

| # | Recommendation | Applies? | What was done / why not | Evidence |
|---|---|---|---|---|
| 1 | AMP dtype per GPU | yes, already | bf16 only on native-bf16 GPUs, fp16 + GradScaler on a T4 (`amp_dtype`) | `src/config.py`; DEVELOPMENT_LOG 24 |
| 2 | Avoid per-step host-device syncs | partly | `.item()` per step removed in `train_qa`, `mlm_warmup` and B3; the per-step finiteness check (a safety check) still syncs, so the measured gain is within noise (6.31 s vs 6.25 s per epoch, different sessions) | `experiments/length_bucketing/RESULTS.md`; `outputs/logs/dact_full_seed*.jsonl` |
| 3 | Gradient checkpointing | **no** | peak training memory 0.68 GB on a 15 GB T4 (4.5 %); checkpointing would only cost time | RESULTS.md |
| 4 | LoRA / PEFT / QLoRA for the main model | **no** | DACT has 1.9 M parameters and must be trained from scratch; adapters on a pretrained model would break the brief | prompt rule 2 |
| 5 | LoRA variant of the RoBERTa baseline | not done | optional; needs `peft` and a GPU; it would not answer any question in the report | — |
| 6 | RoBERTa stability: one pre-registered change | declared, not run | learning rate 2e-5 → 1e-5, judged by the number of seeds that learn; needs about 65 min of GPU for both arms | `experiments/roberta_stability/PREREGISTRATION.md` |
| 7 | 4-bit NF4 loading for Qwen | **no** | the fp16 1.5 B model ran on the free T4 in the final run; quantisation would only add a source of difference | notebook Step 4 output |
| 8 | OOM-safe batching for Qwen | yes | `score_continuations` halves the batch on `torch.cuda.OutOfMemoryError`, retries and logs the batch sizes; free GPU memory is printed before loading | `src/baselines.py`; tests |
| 9 | Sequence packing | **no** | PIQA items are pairs scored with a pairwise softmax; packing would mix items in attention | — |
| 10 | Length bucketing | yes, opt-in | `Config.length_bucketing` (off): padding 69.8 % → 6.9 %, step time −17 %, val +0.20 (rejected as a default by the pre-registered rule) | `tools/padding_measurement.py`; `experiments/length_bucketing/` |
| 11 | Loss masking (prompt vs response) | yes, verified | DACT's loss is on the choice only; MLM labels only real tokens (ids ≥ 5, never padding); B4 scores only the continuation; B3 sees [goal, solution_k] pairs; padding never receives attention | `tests/test_training_hardening.py` |
| 12 | Gradient norm, LR, GPU memory, step time in the logs | yes | added per epoch to the JSONL logs of DACT, the MLM warm-up and B3 | `src/train.py`, `src/baselines.py` |
| 13 | Weights & Biases | **no** | would send data to an external service; the JSONL logs, `progress.json` and `run_config.json` already cover it; the group has not agreed to an external service | — |
| 14 | Best checkpoint by validation loss | **no** | on existing logs the loss rule picks an earlier epoch in 35 of 38 (T4) and 32 of 35 (A100) runs and loses 1.4 / 1.9 points of validation accuracy on average (label smoothing raises the loss while accuracy still improves) | `experiments/checkpoint_selection/selection_vs_loss.csv` |
| 15 | Early stopping, warm-up + cosine | yes, kept | patience 6 stops training at epochs 8-13 with the best epoch at 2-7; warm-up + cosine is the standard Transformer schedule and was never the cause of a failure | `outputs/logs`, `experiments/length_bucketing/run_20261004` |
