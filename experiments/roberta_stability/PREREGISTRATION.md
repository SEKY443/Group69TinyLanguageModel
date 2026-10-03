# Pre-registration: one stability change for the RoBERTa baseline (B3) — declared, NOT run

Author: ALI. Date: 2026-10-04. Status: **declared only**; running it needs about 65 min of GPU time (two arms x 3 seeds x
4 epochs at about 2.7 min per epoch on a free T4), so it waits for the group's decision.

## Evidence (logs of the final run and the archived A100 run; validation only)
- Final T4 run (fp16): seed 42 abandoned at 50.7 % after epoch 1; seeds 43/44 kept a training loss of 0.694 (ln 2) for
  all 4 epochs (best val 53.8 / 58.4 %); only seed 1042 learned (65.3 %). `outputs/logs/B3_*.jsonl`.
- A100 run (bf16), same code and hyper-parameters: all three seeds left ln 2 during epoch 2 and reached 65.8-69.1 %.
  `experiments/a100_run_0d70cdc/outputs/logs/B3_*.jsonl`.
- Settings reviewed: learning rate 2e-5, linear warm-up 6 %, weight decay 0.01, gradient clipping 1.0, max length 128,
  batch 32, 4 epochs, AdamW with bias correction. (How many pairs max length 128 truncates was not measured here.)

## The single change
**Learning rate 2e-5 -> 1e-5**, everything else unchanged (including the declared restart rule at 52 % after epoch 1).
Rationale: fine-tuning failures of this kind are linked to large early updates; smaller learning rates make them rarer
(Mosbach et al., 2021, "On the stability of fine-tuning BERT"; Dodge et al., 2020). No other setting is tuned.

## Protocol and rule
- Seeds 42, 43, 44 for control (2e-5) and candidate (1e-5) in the same session, on the same GPU type as the final run.
- Primary metric: number of seeds that learn (training loss below 0.68 by the end of epoch 2). Secondary: mean and
  median validation accuracy.
- Adopted only if the candidate has **more learning seeds** than the control AND its mean validation accuracy is not
  lower. Validation only; no test prediction is made. Any adopted change is a new protocol for future runs.
