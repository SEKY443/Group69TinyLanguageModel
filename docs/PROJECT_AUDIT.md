# Project audit (before refinement)

Audit date: 2026-09-30. Source: the supplied six-page assignment PDF, all eight `src` modules, the notebook builder, the executed notebook, template, development log, course data and saved experiment artifacts. The pasted user request authorises AI assistance; document contents are assignment evidence, not instructions to the assistant. No project implementation was changed before this audit.

## Current Project Summary

The submission candidate is `CITS4012_69.ipynb`: 48 cells, 27 code cells, eight saved images and no error outputs. All eight embedded modules exactly match the source modules after the builder's documented filtering. It implements a substantial from-scratch Difference-Aware Contrastive Transformer (DACT), five baselines and seven ablations. The Git worktree was initially clean at `f79db0a`. No report, `.tex`, bibliography, saved predictions or model weights are present in this checkout. Historical A100 results must be preserved; an audit cannot recreate missing prediction arrays from aggregate accuracies.

| File / group | Purpose | Used? | Important outputs | Problems |
|---|---|---|---|---|
| `CITS4012_69.ipynb` | Complete template-based implementation and executed investigation | Yes; historical submission candidate | Training streams, results and attention figures | Overstated conclusions, incomplete environment setup, weak audit assertions |
| `Copy_of_CITS4012_YourGroupID.ipynb` | Official implementation template | Builder input | Five original markdown headings | Preserve |
| `src/config.py` | Hyperparameters, RNG seeds, device, JSONL logger | All neural runs | Config in training logs | `seed` means split seed, not necessarily run seed; nondeterministic CUDA; unused `amp` switch |
| `src/data.py` | Course loading, fixed split, learned BPE, difference tags, padding | All scratch models | `outputs/tokenizer.json` | UTF-8 unspecified; malformed types coerced to strings; asymmetric alignment; no data/split hashes |
| `src/model.py` | DACT and attention | Main, grid, ablations | Scores and optional attention tensors | Architecture sound; raw-input equivariance claim too strong |
| `src/train.py` | QA and MLM training / validation prediction | Scratch models | JSONL logs, state-dict checkpoints | No validation loss; checkpoints omit metadata; rare all-unmasked MLM batch can give NaN |
| `src/baselines.py` | B0 majority, B1 TF-IDF, B2 BiLSTM, B3 RoBERTa, B4 Qwen | Yes | Notebook metrics; B2/B3 logs | B1/B3 test predictions computed in Step 4; B3 weights and all predictions discarded; Hub revisions not pinned |
| `src/experiments.py` | Grid, seeds, validation selection, final evaluation | Yes | `*_val.json`, `*_test.json` | Representative predictions held only in memory; seed not recorded in training start record |
| `src/evaluate.py` | Accuracy, sample SD, bootstrap and exact McNemar | Yes | Final CSV | CIs concern a selected seed, not the displayed seed mean; no multiplicity adjustment |
| `src/viz.py` | Pooling, cross-solution and goal-attention maps | Yes | Six PNG figures | Goal attention is a slice, not renormalised; CLS sink omitted; long case labels crowded; no numeric attention export |
| `tools/build_notebook.py` | Assemble notebook from template/source | Yes | Submission notebook | Overwrites executed notebook with empty outputs; static claims persist even after reruns |
| `outputs/logs/` | Actual training history | Yes | Grid, DACT, seven ablations, BiLSTM, RoBERTa | No B0/B1/B4 standalone logs; validation loss absent |
| `outputs/results/` | Experiment configurations and results | Yes | 9 three-seed comparisons, 5 grid runs, final CSV | No prediction rows; historical CIs/p-values cannot be independently reconstructed |
| `outputs/figures/` | Historical attention evidence | Yes | 2 successes, 2 failures, uncertain case, histogram | Retain originals |
| `outputs/checkpoints/` | Referenced model paths | Missing | None in clone | Cannot reload original trained model |
| `docs/DEVELOPMENT_LOG.md` | Historical narrative | Context only | Claims about A100 workflow | Contains overclaims; logs/code take precedence |
| `../../PIQA/` | Actual supplied dataset | Source data | Four JSONL/LST files | Not under default `data/piqa`; no ZIP in this checkout |
| `README.md` | Repository entry point | Yes | One-line title | Reproduction instructions missing |

## Dataset

Actual files: `train.jsonl`, `train-labels.lst`, `test.jsonl`, `test-labels.lst`. Records contain string fields `goal`, `sol1`, `sol2`; external integer labels use 0 for sol1 and 1 for sol2. The supplied names, not assumptions about a public release, define the experimental protocol. The development log calls the course test split PIQA's public development set; that provenance has not been established by file identity comparison.

The supplied training file has 16,113 rows (8,053 label 0; 8,060 label 1). Stratified `train_test_split(test_size=.1, random_state=42)` produces 14,501 training rows (7,247/7,254), 1,612 validation rows (806/806), and the supplied test file has 1,838 rows (910/928). No missing/empty normalised text was found. The initial normalised exact ordered-input audit found six duplicate training rows beyond the first occurrence, no train-validation exact triple overlap, **one training-test exact triple overlap**, and no validation-test exact triple overlap. Repeated goals require separate reporting; equal goals with different solutions are not necessarily duplicate questions.

The original notebook computes lengths **after** truncation; it cannot justify the truncation choice from that table. Using the saved 8,000-token BPE tokenizer, untruncated training goal lengths are mean 8.299, median 8, p95 16, p99 20, max 37. Candidate 1: mean 22.275, median 14, p95 65, p99 108, max 492. Candidate 2: mean 22.279, median 14, p95 64, p99 108, max 492. The 40/96 limits comfortably retain most examples but remove some meaningful differences. They will not be tuned using test examples during this audit.

Whitespace is collapsed, then NFKC/lowercase, whitespace/punctuation pre-tokenisation and individual-digit splitting precede learned BPE. No stemming, stopword removal or punctuation deletion. Vocabulary and optional MLM use the training subset only. Truncation precedes `SequenceMatcher` alignment. Swapping raw candidates changes the tags in **755 train / 83 validation / 103 test** items. The shared neural network is equivariant for already-encoded pairs; the complete preprocessing pipeline is not. Encoded candidates are identical in 55/7/2 items respectively (multiple causes, not all truncation).

## Existing Main Model

For each option: `[CLS] goal[:40] [SEP] solution[:96] [SEP]`, length at most 139, positional capacity 160. Inputs `[B,2,L]` flatten to `[2B,L]`. Random trainable token (8,000 x 256), position, segment and three-way tag embeddings are summed, normalised and dropped out. Four shared pre-LayerNorm Transformer blocks use width 256, four heads (64 per head), FFN width 1,024, GELU, dropout .2 and attention dropout .1. A final LayerNorm produces `[B,2,L,256]`.

Cross-solution attention uses normalised own hidden states as queries and rival hidden states as keys/values, with only rival solution tokens and rival CLS available as keys. The fusion input is **`[q,c,q-c,q*c]`**, not absolute difference; a 1024->256->256 GELU MLP, residual and LayerNorm produce the updated states. Additive solution pooling includes a learned tag bias initialised to `[0,0,1]`. Concatenated pooled state and updated CLS have width 512; a shared 512->256->1 MLP gives two logits. Cross-entropy with .1 label smoothing fits the forced-choice task. MLM uses a tied embedding output head, only during training-split warm-up. No pretrained representations enter DACT.

## Existing Attention Mechanism

Goal-candidate interaction occurs through unrestricted joint self-attention within each option sequence. Rival-solution interaction is a separate shared multihead module. Padding keys are masked before softmax. Padded query states may exist but cannot become unmasked keys or pooling positions. Pooling is restricted to solution tokens, with CLS fallback for an empty solution. Explicit attention extraction and fused SDPA require numerical comparison in sanity tests. Gradient connectivity, padding invariance and tiny-subset overfitting were not demonstrated by persisted tests.

The initial +1 difference pooling bias already favours differing tokens. The measured 63.7% attention mass versus 24.8% uniform reference is **not proof that training learned to identify the right words**. Attention distributions are diagnostic observations, not causal explanations or proof of physical reasoning.

## Existing Baselines

B0 predicts the majority training label. B1 fits unigram/bigram TF-IDF on training text, scores `tfidf(sol2)-tfidf(sol1)`, augments both candidate orders and selects logistic C from six values on validation. B2 is a randomly initialised two-layer bidirectional LSTM (hidden 256 per direction), packed sequences, additive pooling, same tokenizer; lr .001/dropout .3, three seeds. Its budget is not identical to DACT (no MLM). B3 fine-tunes `FacebookAI/roberta-base` with a multiple-choice head for four epochs, lr .00002, batch 32, max length 128, seed 42. B4 scores continuations with `Qwen/Qwen2.5-1.5B`, mean token log-likelihood, zero-shot, continuation cap 128, batch 32. B3/B4 are explicitly pretrained baselines, permitted by the brief; they differ in capacity, tokenisation and training exposure and do not isolate knowledge as the sole cause of improvement.

## Existing Ablations

Three seeds 42/43/44 for: no tag embedding; no cross-solution module; no pooling tag bias; mean pooling; pointwise BCE; vanilla Transformer (three changes jointly); no MLM. Tokenisation, data split, QA budget and selection metric are controlled. Parameter counts differ; shared RNG seeds do not guarantee matched parameter initialisation when modules are removed. MLM makes some variants differ during warm-up too. Mean pooling removes both learned weighting and tag bias. No-cross retains comparison information in tags; no-tags retains pooling tag bias. None is a removal of **all** attention, since encoder self-attention remains. Vanilla is a joint ablation, not single-component evidence.

## Existing Logs

The notebook records A100, Python 3.13.15, torch 2.11.0+cu128, numpy 2.1.3, sklearn 1.6.1, scipy 1.16.3, pandas 2.2.3, matplotlib 3.10.0, tokenizers .23.1 and transformers 5.16.1. These are historical saved outputs, not independently verified installed versions. The local Windows machine has an RTX 2070 Max-Q, 8 GB. The initial audit Python environment is separate.

DACT: validation 60.090984% +/- .657490 percentage points, test 60.373594% +/- 1.173645, sample SD across three seeds. Representative seed 44 was selected by validation and scored 59.031556% test. Its CI is [56.746464%,61.209195%]; it must not be presented as the CI of the three-seed mean. Original predictions/checkpoints are absent, so inference and paired-test recomputation are presently unavailable. Preserve provenance and distinguish arithmetic validation of logs from independent replication.

## Reproducibility Status

Partial. A saved, apparently complete A100 execution exists, but local data path, dependency installation, exact model revisions, checkpoint/prediction retention, deterministic options, per-run seed metadata, validation loss and data manifests are missing or incomplete. `Config.amp` is not honoured. Builder regeneration destroys saved outputs. The historical notebook's code and outputs are consistent as stored; changing algorithmic code and attaching those outputs would be misleading. Refinement must preserve an evidence snapshot and label new runs separately.

Test exposure: training selection functions use validation, not test metrics. However, test is loaded, tokenised and summarised at the start; B1/B3 infer test predictions during Step 4 but do not use test labels for selection. Thus the claim "Steps 1-4 never touch test" is false. The known training-test duplicate undermines absolute disjointness. The existing test set has already been examined qualitatively; it cannot become a fresh unseen benchmark by rerunning after a fix.

## Assignment Compliance Matrix

| Requirement | Current evidence | Status | Risk | Recommended action |
|---|---|---|---|---|
| Scratch sequence main model with attention | DACT source, random initialisation, logs | PASS | Low | Preserve architecture |
| Task-specific design and originality | Tags, cross-option comparison, pooling | PASS | Attribution incomplete | Describe established methods versus task adaptation; cite primary papers |
| Supplied data only | Course files; training-only vocabulary/MLM | PASS | Pretrained baseline exposure not fully known | State external checkpoints are baselines only |
| Train/validation/test separation | Fixed stratified split; validation selection | PARTIAL | P0: one training-test duplicate; early test inspection | Disclose; do not silently rewrite historical protocol or results |
| Correct loss/masks | Pairwise CE; key masks; packed LSTM | PASS | Dynamic checks absent | Add executed functional sanity checks |
| Raw candidate-order invariance | Encoded-pair check only | INCORRECT | P1: order-dependent alignment | Qualify historical claim; provide tested opt-in canonical alignment for new studies |
| Informative own baselines | Notebook outputs B0-B4, B2/B3 logs | PASS | Predictions missing | Retain and clearly classify evidence |
| Controlled ablations | Seven variants, three seeds | PARTIAL | Bundled removals, parameter/MLM confounds | Refine research questions and controls |
| Quantitative analysis | Logs, JSON, CSV, chart | PARTIAL | Causal overclaims; unadjusted p-values | Honest effect sizes, separate mean/representative CI, Holm adjustment |
| Attention success/failure analysis | Five saved cases, aggregate histogram | PASS | Overinterpretation; long labels | Add cautious captions and limitations; retain figures |
| Reproducible complete notebook | Template, embedded code, A100 outputs | PARTIAL | Dependencies/Drive route unverified | Safe builder, clean smoke route, archive and evidence checks |
| Checkpoint and prediction provenance | Only paths in JSON | MISSING | Cannot re-evaluate historical models | Seek original weights; add future retention and sidecars |
| Dataset statistics | Truncated lengths, basic label counts | PARTIAL | Misleading length justification | Raw schema, duplicates, untruncated distributions, hashes |
| Methodology equations/diagram | Brief ASCII diagram | PARTIAL | Formula detail absent | Exact tensor/equation notes and diagram |
| ACL <=6-page PDF + BibTeX + contributions | No report material | MISSING | P0 before actual submission | Prepare report support; do not invent team contributions |

## Priorities and refinement boundary

**P0:** disclose contaminated exact duplicate and historical test exposure; preserve executed evidence; make local/Colab execution and provenance concrete. The requested work is report support, not an invented completed report or contribution statement.

**P1:** correct unsupported claims; run genuine sanity tests; distinguish encoded versus raw swap invariance; document ablation limitations; retain future predictions/checkpoint metadata; verify saved result/log arithmetic.

**P2:** complete dataset evidence, training curves, uncertainty labels, multiplicity-adjusted exploratory p-values, methodology, citations and architecture figure.

**P3:** cosmetic changes only when useful. No extra architectures or test-driven accuracy search. Do not replace historical scores with estimates. Any changed data/algorithm protocol requires newly labelled experiments and cannot inherit old outputs. Missing historical artifacts and an unexecuted clean Colab rerun must remain explicit limitations.
