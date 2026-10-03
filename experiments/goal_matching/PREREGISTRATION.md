# Pre-registration: goal-to-solution matching (written and committed before any training)

**Date:** 2026-10-03. **Branch:** `honest-pass`. **Experiment script:** `tools/goal_matching_experiment.py`.

## Motivation
The analysis of the final model shows that its most uncertain test items tie at p = 0.50. In those ties, the correct option repeats the goal's key words (for example, *pry pallet apart*). DACT compares the two solutions with each other, but it has no component that relates a solution to its goal.

This motivation came from inspecting test items in earlier runs. That is why the decision rule below uses **validation only**.

## The exact change
A `GoalMatching` block (`src/model.py`), switched on with `Config.use_goal_matching = True`:
- **Where:** it is applied to the encoder output of each option, before the cross-solution attention.
- **Attention:** every token of an option attends to the goal segment of the same option (`[CLS] goal [SEP]`, segment 0). It uses the same multi-head attention as the rest of the model: 4 heads, attention dropout 0.1.
- **Fusion:** gated, in ESIM style:
  - m = FFN([q; c; q ⊙ c])
  - g = σ(W[q; c])
  - h' = LayerNorm(h + dropout(g ⊙ m))
  - where q is the layer-normalised token state, c the goal context, and the FFN is Linear(3d → d), GELU, dropout, Linear(d → d).
- **Parameters:** **165,504**. The main model has 1,927,492; with the block it has 2,092,996.

## Fixed configuration (no tuning of the new block)
- **Main model:** exactly as selected on validation in the final run. Small configuration: d = 128, 2 layers, 4 heads, d_ff = 512, dropout 0.2. Lexical head with lr 1e-2. Difference tags, cross-solution attention and difference-guided pooling all on. Pairwise objective.
- **Training:** the same budget for both: AdamW lr 3e-4, ≤ 20 epochs, early stopping on validation accuracy with patience 6, batch size 128.
- **Variant:** the same configuration plus `use_goal_matching = True`. Nothing else changes.

## Protocol
- **Seeds:** 42, 43, 44, for both the control (the main model) and the variant.
- **Environment:** both run in the same session, on one Colab GPU.
- **Data:** the same TRAIN/VAL split as everywhere else. The test split is deleted from memory before training and is never evaluated.

## Acceptance rule (validation accuracy)
Accept only if the 3-seed **mean gain ≥ 1.0 point** AND **gain > 2 × pooled seed standard deviation**, where pooled sd = √((sd_control² + sd_variant²) / 2).

## Decision procedure (fixed in advance)
- **Rejected:** report as a negative result with validation numbers only. DACT stays the main model. **No test run.**
- **Accepted:** add the block as a component, with its own ablation.
  - Run the complete notebook once, on a clean runtime. That is the only permitted test evaluation.
  - Report the test result whatever it is, labelled "post-hoc; test previously consulted".
  - Add one goal-attention figure (one success, one failure).

## Time limit
If the three validation seeds of both arms are not finished by **10 October 2026**, the experiment is abandoned and reported as not run.
