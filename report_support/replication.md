# Independent local replication — completed 30 September 2026

These results come from `experiments/completed/outputs_replication_efficient/`, not the original A100 comparison.
The fixed recipe, seeds and evaluation rule were written to `protocol.json` before training.
No new hyperparameter or architecture selection used test data. Hardware, library versions, precision,
loader workers and selected-position MLM projection differ; improved accuracy cannot be attributed
to any one of those changes. Main architecture, mathematical objective and historical preprocessing
remain the same. All ten MLM epochs and three QA runs completed; checkpoints and item predictions exist.

| Model | Validation accuracy (%) | Test accuracy (%) |
|---|---:|---:|
| DACT seed 42 | 59.0571 | 62.2960 |
| DACT seed 43 | 59.9876 | 62.5680 |
| DACT seed 44 | 59.8635 | 62.8400 |
| DACT mean ± sample SD | 59.6361 ± 0.5052 | 62.5680 ± 0.2720 |
| B0 majority | 50.0000 | 50.4897 |
| B1 TF-IDF + LR, C=3 | 58.9950 | 60.8814 |

Seed 43 was selected by validation. Its accuracy is 1,150/1,838 and its item-bootstrap 95% CI is
[60.3917, 64.6368]%. This CI does not incorporate seed selection/training variance. Excluding the
known training/test duplicate gives 62.5476% on 1,837 examples, a sensitivity analysis only.

`tools/analyse_replication.py` checks ten prediction files (17,250 rows), supplied text/labels,
probability/decision consistency, all accuracies, seed statistics, validation maxima, checkpoint
existence and figure metadata. `replication_manifest.json` hashes the complete local evidence.

## Exploratory paired comparisons

Same-item bootstrap resampling uses 10,000 replicates, seed 20260930. McNemar tests compare correctness
on identical items. The post-run family consists of representative DACT versus B0 and B1; Holm corrects
these two tests. These are exploratory, not preregistered confirmatory claims.

| Comparison | Difference (percentage points) | Paired bootstrap 95% CI | DACT-only / baseline-only correct | Exact p | Holm p |
|---|---:|---|---:|---:|---:|
| DACT − B0 | +12.0783 | [8.8683, 15.1795] | 564 / 342 | 1.60e-13 | 3.20e-13 |
| DACT − B1 | +1.6866 | [-0.7617, 4.0805] | 277 / 246 | .18953 | .18953 |

DACT clearly exceeds this majority reference in the local run. Evidence for an advantage over lexical
LR remains uncertain; failure to reject equality is not evidence of equivalence. Do not compare this
new DACT score to old ablations as if only their architectural component differed.

## Training and structural error analysis

`figures/replication_training_curves.png` shows falling training loss but rising validation loss after
the first few epochs. Best validation checkpoints occur at QA epochs 2, 3 and 2 for seeds 42, 43 and 44.
The later high-confidence errors are consistent with overfitting; this does not identify their cause.

| Seed-43 subgroup | Correct / n | Accuracy (%) |
|---|---:|---:|
| All supplied test items | 1,150 / 1,838 | 62.57 |
| Either solution truncated | 24 / 32 | 75.00 |
| Neither solution truncated | 1,126 / 1,806 | 62.35 |
| Identical retained option tokens | 0 / 2 | 0.00 |
| Predicted confidence at least .8 | 308 / 420 | 73.33 |

Truncation demonstrably makes two distinctions unavailable, but the aggregate table does not show
that truncated items are generally harder. Subgroups are small, overlap and were analysed after training.
Of the 420 high-confidence predictions, 112 are wrong: confidence is not a guarantee of correctness.

## Random sample of actual errors

Twelve of the 688 seed-43 errors were sampled without replacement using NumPy RNG seed 20260930,
before assigning the following qualitative categories. Full inputs and probabilities are in
`replication_error_sample.json`. One reviewer assigned one primary category per example after reading
it. Categories overlap conceptually; these counts describe this small exploratory sample only.
Gold labels are dataset annotations, not independent verification of every real-world assertion.

| Test index | Gold → predicted option | Category | Observed contrast / limitation |
|---:|---|---|---|
| 307 | 1 → 2 | Material/causal knowledge | Dechlorinated versus chlorinated aquarium-filter rinse; full 96-token input retained. |
| 588 | 2 → 1 | Spatial/temporal relation | Adhesive over versus under protective painter's tape. |
| 648 | 1 → 2 | Object/tool affordance | Peanut butter versus a cat as mouse-trap bait. |
| 702 | 2 → 1 | Ingredient/lexical compatibility | Blood-orange juice versus pigs' blood in fudge; 117-token option is truncated, but the retained options still differ. |
| 713 | 2 → 1 | Material/causal knowledge | Penny versus button in a flower vase; the dataset's claimed benefit was not independently validated. |
| 735 | 1 → 2 | Material/causal knowledge | Coolest versus hottest gas-purchase timing; the distractor also contradicts its own cooler-temperature explanation. |
| 779 | 2 → 1 | Material/causal knowledge | Biaxial versus woven fiberglass; domain-specific property distinction, not independently validated. |
| 861 | 2 → 1 | Object/tool affordance | Hot-glue gun versus sanding belt for attaching cardboard. |
| 880 | 1 → 2 | Object/tool affordance | Colander versus pot for rinsing and shaking pumpkin seeds dry. |
| 1050 | 1 → 2 | Object/tool affordance | Masking tape versus magnets on either side of an unspecified table. |
| 1360 | 1 → 2 | Underspecified equipment | Oven versus Dutch oven for macaroni and cheese; the wording does not fully establish equipment constraints. |
| 1421 | 1 → 2 | Spatial/temporal relation | Hairspray after versus before inflating a balloon; label-level contrast only. |

Sample category counts: material/causal knowledge 4, object/tool affordance 4, spatial/temporal relation 2,
ingredient/lexical compatibility 1, underspecified equipment 1. They are not an estimate of category
prevalence across PIQA, and no architecture change was selected from these test examples.

## Measured attention: success and failure

Four cases were chosen by the pre-existing deterministic rule: two most-confident correct and two
most-confident wrong predictions. Full head-averaged attention PNGs and numeric JSON sidecars are in
the run's `figures/` directory. All were inspected. For the six-page report, prefer the compact SVG/PNG
in `report_support/figures/replication_attention_compact.*`; long full heatmaps need zooming and should
not be squeezed into an ACL column. The compact plot displays six largest pooling weights plus the
sum of all omitted weights, without renormalisation or changed model inference.

- **Success #1672, p(gold option 1)=.984:** the ice-cream distractor adds jalapeño juice. The added
  subwords and `juice` dominate option 2's pooling; option 1 has no insertion tokens and distributes
  weight across its recipe. This pattern is compatible with detecting a local contrast, but does not
  prove the model knows food preparation or that these weights caused the correct decision.
- **Success #130, p(gold option 2)=.979:** brown-butter basting beats the rice-ball alternative. The
  largest gold-option pooling weight is on `##ty` in “nutty” (.228), followed by “meat” (.124).
  Subword fragmentation makes word-level interpretations approximate.
- **Failure #551, p(wrong option 1)=.974:** the bulb-removal options differ in rotation direction.
  Pooling puts .333 on “clockwise” and .861 in total on the three “counterclockwise” subwords, yet
  selects the dataset's wrong direction. Attending to the distinction is insufficient to resolve it.
- **Failure #1244, p(wrong option 1)=.974:** a weight-loss item contrasts protein substitution with
  carbohydrate removal. The prediction disagrees with the dataset label; this underspecified health
  question is weak evidence of physical-reasoning ability and is not treated as dietary advice or an
  independently established factual error. It remains disclosed because the selection rule chose it.

The last-layer goal-attention panels sometimes concentrate on punctuation/function words, and the
cross-option matrices often have vertical bands rather than interpretable one-to-one matching.
All maps are descriptive. Slices omit other keys/CLS sink and need not sum to one. The +1 initial
difference bias remains a confound in explaining concentration. Figure inference uses fp32 while
saved batch evaluation used AMP; maximum probability discrepancy across these four cases is .000117,
with identical predicted labels. The untouched batch predictions define reported accuracy.
