# Results: RoBERTa stability test (pre-registered in PREREGISTRATION.md, committed in 482f904 before any run)

Free Colab T4, code `main` at `2db3f51`, validation only (the test split's labels are hidden placeholders and the split
is not used; no test prediction was made). Same settings in both arms except the learning rate; the declared restart
rule (val < 52 % after epoch 1) applied in both. Raw evidence: `run_20261004/` (per-seed JSON results, B3 logs,
consoles, driver logs, the scripts).

| Arm | Seeds 42 / 43 / 44, best val acc (%) | Mean ± std | Median | Seeds that learned* | Restarts |
|---|---|---|---|---|---|
| control, lr 2e-5 | 69.60 / 68.49 / 68.80 | **68.96 ± 0.58** | 68.80 | **3 / 3** | none |
| candidate, lr 1e-5 | 65.20 / 67.99 / 68.24 | **67.14 ± 1.69** | 67.99 | **2 / 3** | none |

\* Pre-registered criterion: training loss below 0.68 by the end of epoch 2. Candidate seed 42 was at 0.693 after
epoch 2 and only started learning in epochs 3-4: at half the learning rate, 4 epochs are too short.

**Decision:** the candidate is adopted only if more of its seeds learn than the control's AND its mean is not lower.
It has fewer learning seeds (2 vs 3) and a lower mean (67.1 vs 69.0). **Rejected: the learning rate stays 2e-5.**

**What this shows about the final run's failure.** With unchanged settings on the same GPU type, all three control
seeds trained (69.0 % mean validation, close to the A100 run's 68.0). The final run's 2-of-3 failure (59.2 % mean) was
therefore an unlucky draw of fine-tuning instability, not a systematic T4/fp16 problem; a stricter restart rule (for
example, also restart if the training loss has not left ln 2 after epoch 2) would have caught it. Only validation
numbers were produced here, so the report's test numbers for B3 are unchanged.

**How the run went (three Colab sessions):**
- Session 1 (`group69r`, 02:15-02:28): control seed 42 finished (69.3 %, `attempt1_session1_reclaimed.log`), then the
  VM was reclaimed, most likely because this PC went to sleep overnight. Not used for the decision.
- Session 2 (`group69r2`, 12:30-13:35): re-run with one `colab exec` per seed and an immediate download of each seed's
  result. Control 42/43/44 and candidate 42/43 finished; the VM was reclaimed during candidate seed 44 after epoch 1
  (`candidate_lr1e-5_seed44.reclaimed_attempt1.console`).
- Session 3 (`group69r3`, 14:08-14:20): candidate seed 44 only. **Deviation from the protocol:** this seed did not run in
  the same session as the others. It cannot change the decision, which was already fixed once the control had 3 of 3
  learning seeds.
