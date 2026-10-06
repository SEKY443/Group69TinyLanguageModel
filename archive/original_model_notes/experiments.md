# Experimental investigation

## Primary historical study

Keep the A100 comparison table together. Do not replace one row with a later laptop replication or compare different preprocessing protocols as if they were controlled ablations. Historical data: 14,501 train / 1,612 validation / 1,838 test; exact duplicate limitation is documented in dataset_statistics.md. Accuracy = correctly selected solutions / items. Three-seed means and sample SD are descriptive estimates of training variability, not uncertainty over the entire study design.

## Exact selected configuration

| Setting | Value |
|---|---|
| Split seed / validation fraction | 42 / 0.10 stratified |
| Training seeds | 42, 43, 44 |
| Learned tokenizer | BPE 8,000 vocabulary entries, minimum frequency 2; NFKC/lowercase; punctuation split; individual digits |
| Goal / solution caps | 40 / 96 BPE tokens |
| Positional capacity | 160 (actual maximum input 139) |
| Embedding / encoder width | 256 |
| Encoder layers / heads / head width | 4 / 4 / 64 |
| Feed-forward width | 1,024 |
| Dropout / attention dropout | 0.2 / 0.1 |
| Objective | Two-logit CE, label smoothing 0.1 |
| Optimiser | AdamW, betas (0.9,0.98) |
| QA learning rate / weight decay | 0.0003 / 0.05; no decay on embeddings, biases or LN |
| QA batch / maximum epochs / patience | 128 / 20 / 6 |
| Schedule / clipping | 6% linear warm-up, cosine decay / gradient norm 1 |
| MLM epochs / learning rate / batch | 10 / 0.0005 / 256 |
| MLM masking | 15% eligible tokens; 80/10/10 mask/random/unchanged |
| Selection | Highest validation accuracy; earliest epoch wins ties |
| Precision / device (historical) | bf16 autocast, A100, TF32 enabled, nondeterministic CUDA |
| Total trainable parameters | 6,114,628 (includes MLM-only head) |

The five-grid validation accuracies were 59.24 (base), 59.93 (small), 59.93 (dropout .3),
60.24 (base + 10 MLM epochs), 58.93 (small + MLM). The chosen warm-up was not the best test variant;
selection was made on validation. Running the selected configuration again with seed 42 did not reproduce
the grid score exactly, which illustrates the nondeterministic CUDA setting. Historical log configurations
use `seed=42` for the split even in run 43/44; the run filename and results JSON identify the model seed.
The refined code records both separately.

## Ablation questions, hypotheses and controls

All historical variants share the fixed split, BPE, truncation, QA optimiser/schedule/batch/epoch cap,
patience, metric and seeds unless stated. Three seeds are separate initialisations; early stopping leads
to different actual epoch counts. Removing modules changes random-number consumption and parameter
count, so equal seeds do not constitute identical initial weights. Parameter counts are in experiment_summary.csv.
Some allocated parameters are inactive (MLM head in QA, pooling projections in mean mode); total
parameter count is not an exact count of active QA capacity.

| Ablation | Research question / hypothesis | Controlled change and interpretation | Validation / test mean (%) |
|---|---|---|---|
| No tag embedding | Do encoder input tags add value beyond pooling's tag bias? Expect a gain if early difference marking helps. | Removes embedding only; tag-biased pooling remains. Also changes MLM encoding. | 58.97 / 60.52 |
| No cross-solution module | Does explicit rival attention add value beyond tag information? Expect a gain if contextual rival matching matters. | Removes cross attention, fusion, residual LN; fewer parameters. Tag-based comparison remains. | 60.05 / 61.73 |
| No pooling tag bias | Does the additive tag prior improve a learned solution summary? | `diff_attn` -> `attn`; learned content scores remain. Tests the prior, not all pooling attention. | 59.43 / 61.50 |
| Mean pooling | Does learned, tag-biased weighting improve over uniform solution averaging? | Removes learned weighting and tag prior together; encoder/cross attention remain. | 59.80 / 61.46 |
| Pointwise objective | Does direct competition between candidates improve over independent BCE? | CE -> BCE with matched smoothing; shared architecture/argmax decision. | 60.07 / 60.92 |
| Vanilla Transformer | Does the combined task-specific package help over shared joint encoding? | Joint removal of tag embedding, cross module and pooling attention; not single-factor evidence. | 57.88 / 58.71 |
| No MLM | Does training-only masked-token warm-up improve downstream generalisation? | Removes ten extra warm-up epochs; inherently less training compute. | 59.14 / 61.37 |

## Baseline controls and limits

| Baseline | Implemented configuration | What it tests |
|---|---|---|
| B0 | Most common training label, 1 | Position/majority reference; no fitted neural weights |
| B1 | TF-IDF unigrams/bigrams, min_df=2, sublinear TF; LR C in {.03,.1,.3,1,3,10}, max_iter=2000; both option orders | Lexical solution preference without explicit goal-conditioned scoring |
| B2 | 2-layer BiLSTM, hidden 256 each direction, embedding 256; additive pooling; lr .001, dropout .3; 3 seeds | Recurrent scratch baseline; no MLM and not a matched budget |
| B3 | FacebookAI/roberta-base multiple-choice; 4 epochs, lr .00002, batch 32, max_len 128, seed 42; weight decay .01, 6% linear warm-up, clipping 1 | Adapted pretrained encoder; best validation epoch 4 |
| B4 | Qwen/Qwen2.5-1.5B base checkpoint, no training; prompt `Goal: {goal}\nSolution:`; mean continuation token log-probability, 128-token cap, batch 32 | Open-model zero-shot reference; no proprietary API |

Baseline checkpoint identities are known; their historical Hub commit revisions were not saved. The
refined code logs resolved revisions and retains B3 locally. A future exact replication must use recorded
revisions, not assume the current Hub main branch is identical. Pretraining exposure to PIQA is unknown.
No external dataset was deliberately added by this project.

Historical B1 selected C=3.0 (saved notebook output, baseline cell).

## Interpretation and uncertainty

The primary story is useful but modest scratch learning, a strong lexical baseline, and no robustly isolated
benefit for every added component. Single-ablation reversals between validation and test must be reported.
Do not label this proof of component redundancy or dismiss them as noise. Historical raw p=.01696 for
no-pooling-bias versus DACT becomes .15267 after Holm adjustment over 12 comparisons. This is exploratory,
post-hoc multiplicity reporting, not a preregistered confirmatory analysis. The corrected table separates
representative-seed CIs from the across-seed mean. The historical paired p-values are copied with provenance
because their prediction arrays are missing, not represented as independently recomputed.

## New audit work

Functional sanity checks and tiny overfit are separate development tests, not additional baseline results.
The laptop reproduction uses the previously selected full DACT recipe, fixed saved tokenizer and historical
split, without a hyperparameter search. Its source changes are evidence retention and numerically checked
selected-position MLM projection; Windows uses zero loader workers and fp16. Keep its results, checkpoints
and prediction arrays separate from the A100 comparison. The positive +1 pooling prior, raw-input alignment
limitation and dataset overlap remain properties of the historical protocol.
