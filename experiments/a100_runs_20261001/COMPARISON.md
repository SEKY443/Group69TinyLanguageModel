# A100 runs of 2026-10-01: current version vs Salah's version

Both notebooks were executed top to bottom on a Colab Pro A100 (40 GB, bf16) with the Colab CLI, on clean VMs, with **zero error outputs**.

| Run | Notebook / commit | Model | Where |
|---|---|---|---|
| Current version | `CITS4012_69.ipynb` at `0d70cdc` | DACT + lexical head + difference-aware truncation | repository root: `CITS4012_69.ipynb`, `outputs/` |
| Salah's version | `notebooks/CITS4012_69_reproducible.ipynb` at `e08e80c` | DACT as in the original A100 run (no lexical head, prefix truncation) | `salah_version/` |
| First attempt of the current version | same as current version | same | `first_attempt_reclaimed_before_test.log` (VM reclaimed during the test step; validation results only) |

For reference, the original run of 2026-09-29 (A100) is in the repository history (`f79db0a`), and the free-T4 run of the current model (commit `538cf5b`) is archived in `experiments/t4_run_538cf5b/`.

## Results (accuracy %, validation / test)

| Model | Original 29 Sep | Salah's version | Current version (T4) | **Current version (A100)** |
|---|---|---|---|---|
| **DACT (full)** | 60.1 / 60.4 | 59.2 / 61.2 | 61.1 / 61.6 | **61.2 / 61.6** |
| − difference tags | 59.0 / 60.5 | 59.1 / 60.1 | 59.7 / 60.3 | 59.9 / 60.6 |
| − cross-solution attention | 60.0 / 61.7 | 58.7 / 60.5 | 61.0 / 60.9 | 60.5 / 60.7 |
| − diff bias in pooling | 59.4 / 61.5 | 59.0 / 60.4 | 61.4 / 61.6 | 61.7 / 61.8 |
| mean pooling | 59.8 / 61.5 | 59.7 / 60.6 | 60.9 / 60.7 | 61.1 / 61.0 |
| pointwise objective | 60.1 / 60.9 | 59.2 / 60.8 | 60.1 / 60.8 | 60.5 / 61.6 |
| vanilla Transformer | 57.9 / 58.7 | 58.4 / 59.0 | 58.7 / 58.6 | 58.7 / 59.2 |
| − MLM warm-up | 59.1 / 61.4 | – | – | – |
| − lexical head | – | – | 59.2 / 60.7 | 59.4 / 61.2 |
| tag bias initialised at 0 | – | – | 61.4 / 61.6 | 61.6 / 61.7 |
| B2 BiLSTM + attention | 59.6 / 59.5 | 59.6 / 60.4 | 59.8 / 58.7 | 59.5 / 60.1 |
| B0 majority | 50.0 / 50.5 | 50.0 / 50.5 | 50.0 / 50.5 | 50.0 / 50.5 |
| B1 TF-IDF + LR | 59.2 / 61.2 | 59.2 / 61.2 | 59.2 / 61.2 | 59.2 / 61.2 |
| B3 RoBERTa-base (fine-tuned) | 70.0 / 68.5 (1 seed) | 67.3 / 65.9 (1 seed) | 69.0 / 67.9 | 68.0 / 66.7 |
| B4 Qwen2.5-1.5B (zero-shot) | 78.1 / 75.4 | 78.1 / 75.4 | 78.2 / 75.1 | 78.1 / 75.4 |

DACT seed std (val / test): original 0.7 / 1.2, Salah's version 1.0 / 0.1, current T4 0.8 / 0.7, current A100 0.4 / 0.1.

## Conclusions
- **The current version is the best model**: highest DACT accuracy (61.2 / 61.6) and the only version above TF-IDF + LR on both splits. Against Salah's version it is +2.0 (val) / +0.4 (test); without its lexical head it drops to 59.4 / 61.2, the level of Salah's version.
- **It reproduces**: the T4 and A100 runs of the current version give the same DACT test accuracy (61.6) and validation within 0.1.
- **Salah's version reproduces the original model**: 59.2 / 61.2 vs 60.1 / 60.4, within seed noise; DACT and TF-IDF + LR are tied there.
- **Fragile claims**: between the two runs of the current version, the BiLSTM test accuracy moved from 58.7 to 60.1 (DACT vs BiLSTM on test: McNemar p = 0.021 → 0.10), and the pointwise-objective ablation from −0.8 to 0.0 on test. The grid choice also differs between versions (base + MLM in the original run, small in all later runs). Differences below about 1 point are within run-to-run noise.
- **RoBERTa fine-tuning is unstable**: in the first attempt of the current version, seed 42 stayed at chance (loss 0.694 for all 4 epochs); in the completed A100 run all seeds trained (65.8, 69.1, 69.1 % val).
