# Results: length bucketing (pre-registered in PREREGISTRATION.md, commit 1ff22a1, 01:35 AWST 2026-10-04)

Run: free Colab T4, session `group69b`, code at `1ff22a1`, 01:37-01:50 AWST, control first, seeds 42/43/44 per arm.
Validation only: `data["test_labels_loaded"]` was asserted False; no test prediction was made.
Raw outputs: `run_20261004/` (logs, validation predictions, `summary.json`, console log).

| Arm | Val acc per seed (%) | Mean ± std | Step time (ms) | Epoch time (s) | Peak GPU memory (MB) |
|---|---|---|---|---|---|
| control (random batches) | 60.73 / 61.04 / 60.98 | **60.92 ± 0.16** | 50.2 / 51.2 / 51.0 | 6.18 / 6.28 / 6.28 | 680.6 / 682.4 / 683.4 |
| bucketing | 61.54 / 60.86 / 60.98 | **61.12 ± 0.36** | 41.6 / 42.8 / 42.5 | 5.24 / 5.36 / 5.37 | 681.5 / 683.6 / 683.3 |

**Decision (primary rule):** gain +0.20 points. The rule needs ≥ 1.0 point AND more than twice the pooled seed std
(2 × 0.28 = 0.56). **Rejected as a default: `length_bucketing` stays False.**

**Secondary (efficiency):** bucketing cuts the training step time by 17 % and the epoch time by 15 % at no measurable
accuracy cost, so it is documented as an opt-in speed option (a new protocol; its numbers are not mixed with the
reported results). Peak memory is unchanged: it is set by the longest batch, which both arms contain.

**Gradient checkpointing:** the peak training memory of the selected DACT is about 0.68 GB on a 15 GB T4 (4.5 %), so
gradient checkpointing would only cost time; it is not needed.
