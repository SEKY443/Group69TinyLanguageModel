# Pre-registration: length bucketing for the DACT training loader

Committed **before any training** for this experiment (TRAINING_UPGRADE_PROMPT.md, section 3). Author: ALI. Date: 2026-10-04.

## Change
`Config.length_bucketing = True`: the training loader uses `BucketBatchSampler` (`src/data.py`). Each epoch shuffles all
items, sorts chunks of 50 x 128 items by length, cuts them into batches and shuffles the batch order. Items are never
packed or split (PIQA pairs are scored together). Evaluation loaders are unchanged. No parameter is added (0 parameters).
Measured on the training split without training (`tools/padding_measurement.py`): padding falls from 69.6-69.9 % to
6.8-6.9 % of the batch tensors (seeds 42-44).

## Fixed configuration
The configuration of the final run's DACT (`outputs/results/dact_full_val.json`: small, d = 128, 2 layers, 10-epoch MLM
warm-up, lexical head, every other setting unchanged). Control = that configuration; treatment = the same with
`length_bucketing=True`. Seeds 42, 43, 44 for both arms, trained in **the same free Colab T4 session**, control first.

## Split and outputs
Validation only (1,612 items). The test labels are not loaded (`load_piqa` hides them; `final_eval` is not called);
no test prediction is made. Logged per epoch: validation accuracy and loss, step time, peak GPU memory, gradient norm.

## Decision rule
- **Primary (accuracy):** bucketing becomes the default only if its mean validation accuracy is at least **1.0 point**
  higher than the control's AND the gain is more than **twice the pooled seed std**. Otherwise the default stays off.
- **Secondary (efficiency, reported, not a reason to change the default):** mean training step time and peak memory
  per epoch. If the primary rule fails, bucketing may still be documented as an opt-in speed option, together with its
  validation difference; its numbers are never mixed with the reported results.
- The control's peak memory is also the evidence for the claim that gradient checkpointing is not needed.

## Deadline
Results by 2026-10-05; an unfinished run is reported as unfinished.
