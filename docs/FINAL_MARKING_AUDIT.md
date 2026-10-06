# Final marking audit (2026-10-06)

Scope: the complete repository on branch `claude/inspiring-mayer-t5ckuv`, judged against the official brief
(*CITS4012 Group Project*, September 2026). Earlier audits (`history/PROJECT_AUDIT.md`, `history/REFINEMENT_SUMMARY.md`,
`history/TRAINING_AUDIT.md`, `DEVELOPMENT_LOG.md`) were used as leads only; every conclusion below was re-checked against code,
artifacts, recomputation or the rendered PDF. The requirement-by-requirement matrix is in
`FINAL_REQUIREMENTS_TRACEABILITY.md`; the submission checklist in `FINAL_SUBMISSION_CHECKLIST.md`; the work log in
`DEVELOPMENT_LOG.md` §49.

## 1. Executive summary

**Verdict: submission ready with disclosed minor limitations**, provided the group (a) confirms the Team
Contributions paragraphs and (b) decides on an AI-use statement under the unit's policy. Neither can be settled by
an audit.

- Every hard rule of the brief that can be checked mechanically passes: ACL template in `[final]` mode,
  `\author{Group 73}`, main content on pages 1–6 (3 lines of slack), BibTeX, the course template with unmodified
  section titles, files named `CITS4012_73.pdf` / `CITS4012_73.ipynb`.
- Every number in the report's tables and prose was traced to a result file and, where saved predictions exist,
  recomputed from per-item predictions: 46 table values and 53 prose claims, 0 errors.
- The saved notebook outputs come from one clean Colab T4 run of commit `b880108`. The later edits to its module
  cells were proven not to change any trained weight or prediction (bit-identical under deterministic CPU execution).
- Remaining limitations are disclosed in the report and are scientific rather than procedural:
  - historical test exposure (the lexical head was explored after earlier test results had been seen);
  - seed sensitivity;
  - confounded ablations (parameter counts; the compound vanilla model);
  - no fresh GPU run of the final cells.

## 2. Status of the hard requirements

| Area | Status | Evidence |
|---|---|---|
| Own model, from scratch, Transformer + attention | PASS | `src/model.py`; `tests/test_model_properties.py` (no pretrained path, random init, attention distributions, gradient flow, attention affects output) |
| PIQA only, no external data | PASS | SHA-256 of the four files = run manifest = official PIQA train/dev release |
| Own baselines, no closed APIs | PASS | B0–B4 in the notebook; B1 reproduced item by item |
| Test set not guiding decisions | PARTIAL (historical, disclosed) | final protocol enforced and logged; lexical-head exploration followed earlier test results |
| ACL `[final]`, Group 73, ≤ 6 pages, BibTeX | PASS | `tools/build_submission.py --check-only` |
| Course template, complete implementation, logs and results as outputs | PASS | sync check, 0 error outputs, supplementary-evidence cell |
| Clean Colab reproducibility | PARTIAL | clean T4 run of `b880108`; later changes proven equivalent / tested on CPU; no GPU re-run here |
| Two correctly named files | PASS | `submission/` with `SHA256SUMS` |

## 3. Model design assessment

**Submitted architecture.** The selected configuration is recorded in `outputs/results/dact_full_val.json`. Every
item below was checked against `DACT.forward`.

1. **Input.** Each option is `[CLS] goal [SEP] solution [SEP]`. Goals are cut to 40 BPE tokens and solutions to 96;
   a solution that is too long keeps a window centred on the differing span. The BPE vocabulary has 8,000 entries
   and is learned on the training split (NFKC, lower-casing, digits split).
2. **Embedding.** Token, position, segment and difference tag (other / shared / different, from a `difflib`
   alignment of the two solutions), followed by LayerNorm and dropout 0.2.
3. **Encoder.** A shared pre-LN Transformer encoder: 2 layers, d = 128, 4 heads, FFN 512 with GELU, attention
   dropout 0.1, final LayerNorm.
4. **Cross-solution attention.** Shared multi-head attention. The queries are every LayerNorm'd token of one option;
   the keys and values are the rival's solution tokens plus the rival's [CLS] as a sink. ESIM fusion
   MLP([q; c; q−c; q⊙c]), then a residual connection and LayerNorm. Both directions use the pre-update states, so the
   block is symmetric.
5. **Pooling.** Additive attention over solution tokens, with a learned bias per tag (initialised [0, 0, 1]). An
   empty solution falls back to [CLS].
6. **Scorer.** MLP([p; h_CLS]) gives the neural score s_k.
7. **Lexical head.** 2^18 hashed solution unigram and bigram weights (bucket 0 reserved for padding), initialised at
   0, trained with its own learning rate (1e-2) under weight decay. Its sum ℓ_k is added to s_k.
8. **Objective.** Softmax over (z_1, z_2) with cross-entropy and label smoothing 0.1.
9. **MLM warm-up.** 10 epochs on training-split text, 15 % masking (80/10/10), with the output layer tied to the
   token embedding.
10. **Optimisation and selection.**
    - AdamW with β = (0.9, 0.98), learning rate 3e-4 and weight decay 0.05 (none on biases, LayerNorm or
      embeddings).
    - 6 % warm-up followed by cosine decay; batch size 128; gradient clipping at norm 1.
    - At most 20 epochs, early stopping on validation accuracy with patience 6; the best checkpoint is restored.
    - fp16 autocast on the T4.
11. **Size.** 1,927,492 trainable parameters, 262,144 of them lexical.

**Technical correctness (tests, not inspection).**

- *Masks:* padding is never attended to, and pooling is restricted to solution tokens.
- *Attention:* every attention row sums to 1 over the allowed keys. The explicit (figure) path equals the fused
  SDPA path. The QA loss reaches the tag embedding, self-attention, cross-attention, pooling, scorer and lexical
  weights, and the MLM head receives no QA gradient. Replacing the learned pooling with uniform pooling changes the
  scores.
- *Inputs:* a padding-invariance test exists, and swapping already-encoded candidates swaps the scores.
- *Training mechanics:* checkpoint round trips are exact, the same seed gives the same initialisation, and the model
  can fit a tiny training set.
- *Raw-input swap is not invariant:* `difflib` is directional, so 102 test items get different tags when the options
  are swapped. This is disclosed, and an opt-in canonical alignment exists.

**Originality.**

- The PIQA-specific parts are the difference tags, the adaptation of ESIM-style comparison to two competing answers,
  the tag-biased pooling and the jointly trained lexical head.
- The report separates these from the adapted work it cites: Transformer, ESIM, additive attention, BPE,
  wide-and-deep and MLM.
- The evidence supports the **combination** (vanilla −3.7 / −4.3 points; every vanilla seed is below every full
  seed on both splits), the **pairwise objective** (pointwise −1.7 / −1.4; also seed-separated on both splits) and
  the **lexical head on validation** (−2.3).
- It does **not** consistently support the tags, cross-attention, tag bias or MLM warm-up individually. The report
  says so and offers a redundancy explanation, framed as a hypothesis.

## 4. Experiment assessment

| Experiment | Question | Control / confound | Seeds | Evidence |
|---|---|---|---|---|
| B0 majority | chance level | — | deterministic | notebook Step 4/5 |
| B1 TF-IDF + LR | how far shallow lexical cues go | C on validation | deterministic | reproduced here (`b1_reproduction.json`) |
| B2 BiLSTM + attention | recurrent from-scratch alternative | same tokenizer; different lr/dropout, no MLM | 3 | `b2_bilstm_*` logs/predictions |
| B3 RoBERTa-base fine-tuned | pretrained encoder reference | declared restart rule; 2 of 3 seeds failed in this session; validation-only re-run trained 3/3 | 3 (+3 re-run) | logs; `experiments/roberta_stability` |
| B4 Qwen2.5-1.5B zero-shot | LLM knowledge without training | length-normalised log-likelihood | deterministic | notebook outputs |
| 9 ablations | one component each (vanilla: four) | same data, seeds, budget; parameter counts differ (−0.4k to −411k) | 3 each | Table 2, `ablation_significance.csv` |
| Grid (5 configs) | size, dropout, MLM | one seed; top three within 0.43 points | 1 | `hp*_val.json` |
| Goal-matching block | pre-registered extension | acceptance rule committed before training | 3 | rejected (−0.31 on validation) |
| Length bucketing, RoBERTa lr | pre-registered engineering tests | validation only | 3 | rejected as defaults |

**Statistics.**

- Accuracy uses mean ± sample std over 3 seeds.
- The bootstrap CI and the exact McNemar test use the best-validation seed, as the report states.
- The report now applies the Holm correction it cites: the vanilla ablation has p = 0.007 on test and 0.009 on
  validation uncorrected, and 0.06 and 0.08 after correction.

## 5. Results-analysis assessment

**Final results** (T4 run of `b880108`; all recomputed):

| Model | Val | Test |
|---|---|---|
| DACT | 61.29 ± 0.49 | 61.86 ± 0.30 (seed-42 CI 59.6–64.0) |
| B1 TF-IDF + LR | 59.24 | 61.21 (McNemar p 0.62 vs DACT) |
| B2 BiLSTM + attention | 59.28 ± 0.68 | 58.72 ± 1.12 (p 0.09) |
| B3 RoBERTa-base fine-tuned | 59.16 ± 5.77 (two failed seeds; validation re-run 69.0 ± 0.6) | 57.29 ± 6.91 |
| B4 Qwen2.5-1.5B zero-shot | 78.16 | 75.08 |
| B0 majority | 50.0 | 50.5 |

**Attention analysis.**

- The pooling puts 78.0 % of its weight on differing tokens, which make up 24.8 % of solution tokens. With the tag
  bias set to 0 the share is 62.0 %; an untrained model reaches 38.9 %.
- Errors attend slightly more to differing tokens than correct predictions (79.0 vs 77.4 %, Mann–Whitney
  p = 0.03).
- The pooling entropy is the same for correct and wrong predictions (p = 0.98).
- A linear probe shows that without tags the encoder barely learns where the candidates differ.
- The report now visualises one success, one failure and one tie, each with its inputs, gold answer, prediction and
  probability.
- Attention is described as aggregation weight, not as a causal explanation.
- The negative and unexpected findings are reported:
  - no significant test gain over TF-IDF;
  - RoBERTa instability;
  - individual components without a consistent effect;
  - errors attending more to the differing tokens;
  - lexical over-confidence on long differing spans.

## 6. Report assessment

**Format.**

- ACL template in final mode; the template files are unmodified, and citations are numeric through natbib's `numbers` option (the group's choice).
- `\author{Group 73}`.
- 8 pages in total: content on pages 1–6, Team Contributions on page 7, references alone on page 8.
- No overfull boxes, LaTeX warnings or BibTeX warnings.

**Rendered pages.** All eight pages were inspected visually:

- Figure 1 was redrawn: no clipped labels, the alignment → tags and lexical → softmax flows are shown, and the
  legend is legible.
- Figure 2 (attention cases) has larger fonts; a redundant bar chart of four numbers already given in the text was removed to keep the main content within six pages.
- The orphan page that held only "model." was removed.

**Content.**

- Introduction, Methodology (four equations matching the code), Experimental Setup, Results, Limitations and
  Conclusion follow the suggested structure.
- The equations of the cross-solution block were corrected to the implemented LayerNorm placement.
- One unverifiable sentence (a "prepared but not yet run" test split, which is not in this repository) was replaced
  by a statement of what would be needed.

## 7. Research engagement

- There are 23 BibTeX entries, all cited and all primary sources: PIQA, Transformer, pre-LN, ESIM, decomposable
  attention, OCN, DCMN+, additive attention, BPE, BERT, RoBERTa, Qwen2.5, wide-and-deep, annotation artefacts (×2),
  attention-as-explanation (×2), bootstrap, McNemar, Dror et al., Dodge et al., UnifiedQA and AdamW.
- Literature motivates the design (OCN/DCMN+/ESIM → option comparison; annotation artefacts → lexical baseline and
  head) and the interpretation (Jain & Wallace; Wiegreffe & Pinter; Dodge et al.; Dror et al.).
- Published scores (Bisk et al.) appear for context only.

## 8. Reproducibility assessment

**Executed in this audit** (CPU container: Python 3.13.16, torch 2.11.0, numpy 2.1.3, scipy 1.16.3,
scikit-learn 1.6.1, pandas 2.2.3, matplotlib 3.10.0, tokenizers 0.23.1, transformers 5.16.1):

- the full test suite (79 tests);
- the end-to-end notebook smoke run (every code cell, on synthetic data, B3/B4 stubbed);
- the post-run equivalence proof;
- the B1 reproduction;
- the report-number, test-access and sync checkers;
- the live data download;
- the LaTeX build.

**Not executed:** a GPU run of the final notebook. Google Drive and the Hugging Face Hub were unreachable from the
container, and no GPU was available.

**Known non-determinism:** GPU kernels, and BPE training in `tokenizers`. Two trainings on the same split differ,
including single-threaded. A re-run therefore reproduces results only up to seed-level variation; this is now
disclosed in the report, the notebook and the README.

## 9. Remaining limitations (all disclosed)

1. The test split was used in earlier development runs; the lexical head was explored after those results.
2. A one-seed hyper-parameter grid; three seeds per experiment; seed std up to 1.1 points.
3. Ablations change parameter counts; the vanilla model removes four components.
4. One test item duplicates training data; the re-scoring effect is 0.02 points.
5. Difference tags depend on the option order (102 test items).
6. B3/B4 per-item test predictions were not saved; their numbers exist as notebook outputs only.
7. Run checkpoints are not distributed; Run all retrains everything (≈1 h 50 min on a T4).

## 10. Mark risks and unresolved blockers

| Risk | Likelihood | Mitigation in place / action |
|---|---|---|
| Team Contributions not confirmed by every member | — | **Blocker for the group**: each member checks their paragraph |
| AI-use declaration required by the unit but absent from the report | unknown | **Group decision**; the record in the development log is intact |
| Marker re-runs on Colab and gets slightly different numbers | medium | readme explains the expected variation; versions are checked; data is SHA-256 verified |
| Marker reads the test-exposure disclosure as a protocol weakness | medium | disclosed honestly; validation used for every decision in the final run |
| Report rebuilt with another TeX distribution and spills past page 6 | low | 3 lines of slack; `report/build.sh` + `tools/build_submission.py --check-only` |
| Individual components unsupported by ablations | certain, already reported | framed as findings, not hidden |

## 11. Marker simulation

Indicative bands, not a prediction of the mark.

| Criterion | Judgement | Where marks could still be lost |
|---|---|---|
| Model design and implementation | Strong: coherent, tested, clearly task-motivated, attention integrated at three levels | Individual components lack consistent empirical support; the design is an ensemble of known ideas rather than a new mechanism |
| Experimental investigation | Strong: five baselines spanning chance to LLM, nine ablations × 3 seeds, pre-registered follow-ups, significance with multiplicity control | Historical test exposure; one-seed grid; confounded ablations |
| Results and critical analysis | Strong: honest negative results, controls separating learned and built-in attention, probe, error overlap, success and failure cases, discussion of the most uncertain item | Small effect sizes limit the conclusions |
| Writing and presentation | Good to strong: dense but coherent, all numbers traceable, readable figures | Six pages are tightly packed |
| Engagement with research | Good: literature informs both design and interpretation | Related work is brief (space) |
| Reproducibility (mandatory) | Good: complete logs, predictions, hashes, tests, equivalence proof | No fresh GPU run of the final cells; non-deterministic tokenizer |

## 12. Final recommendation

Submit `submission/CITS4012_73.pdf` and `submission/CITS4012_73.ipynb` (hashes in `submission/SHA256SUMS`) once the
group has confirmed the Team Contributions and decided on the AI-use statement. Do not run further experiments
against the test split.
