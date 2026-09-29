# Development Log — CITS4012 Group 69 (PIQA, DACT)

A step-by-step record of how this project was planned, built, tested, run and analysed, with the reasoning behind each decision.
All times are local (CST, UTC+8) on 2026-09-29. Times of the Colab run come from the JSONL logs in `outputs/logs/`, which were written in UTC by the Colab VM and are converted here.

---

## 0. Task requirements (from the assignment brief)

| Requirement | How it is met |
|---|---|
| QA task on one of TweetQA / MultiRC / ReClor / **PIQA** | PIQA (physical commonsense, 2-option forced choice) |
| No external data; test set only for final evaluation | Validation split carved from the training file; test used only in "Step 5" of Section 3 |
| Main model trained from scratch, RNN/LSTM/GRU/Transformer, with attention | DACT: Transformer encoder + cross-attention + attention pooling, random initialisation |
| Original, dataset-motivated architecture changes | Difference tags, contrastive cross-solution attention, difference-guided pooling (Section 2 of notebook) |
| Baselines run by the group (not copied from papers) | B0–B4, all produced by code in the notebook |
| Suitable metric with justification | Accuracy (balanced labels, official PIQA metric) + seeds, bootstrap CI, McNemar |
| Ablation studies | 7 ablations × 3 seeds |
| Attention visualisation, success and failure cases | 5 case figures + aggregate attention statistic |
| File `CITS4012_YourGroupID.ipynb`, official template, reproducible on clean Colab, saved logs | `CITS4012_69.ipynb`, generated from `Copy_of_CITS4012_YourGroupID.ipynb`; logs in `outputs/` |

---

## 1. Discovery (≈18:15)

**What was done**
- Listed the repository: only `README.md`, the official template `Copy_of_CITS4012_YourGroupID.ipynb` and a data zip.
- Read the template: a title cell, a Readme cell and three fixed section titles (`1.Dataset Processing`, `2. Model Implementation`, `3.Testing and Evaluation`) that must not be modified.
- Inspected the zip: `train.jsonl` (16,113 items), `train-labels.lst`, `test.jsonl` (1,838 items), `test-labels.lst`. Each item is `{goal, sol1, sol2}` with label 0/1.
- Basic statistics: goals average 7 words and solutions 19 words; labels are balanced (train 8,053 / 8,060; test 910 / 928).

**Key observations and why they matter**
1. **There is no validation split.** The provided "test" (1,838) is the official PIQA dev set. To respect "never use the test set for design decisions", a validation split had to be carved out of the training file.
2. **The two solutions are usually near-duplicates** (e.g. *"…bedding made of ripped paper strips"* vs *"…ripped jeans material"*). The answer is determined by a few words. This became the central design idea of the model.

## 2. Planning (≈18:20)

A plan was presented before any code was written: model design, baselines, metrics, ablations, attention analysis, code layout. Three decisions were confirmed with the user:

| Question | Decision |
|---|---|
| Notebook file name | `CITS4012_69.ipynb` |
| Where the data comes from on Colab | Google Drive, file at `MyDrive/Group69/PIQA.zip` |
| Local environment for testing | Create `.venv` for smoke tests |

The user then added: *Colab Pro is available, use the GPU to maximise efficiency*. The code was therefore written GPU-first (bf16 autocast, TF32, pre-tokenised tensors, pinned-memory loaders).

## 3. Environment setup (≈18:22)

- `python3.12 -m venv .venv`; installed `torch 2.14`, `numpy`, `scikit-learn`, `matplotlib`, `tokenizers 0.23`, `pandas`; later `transformers 5.17`, `scipy`, `nbclient`, `ipykernel`.
- **Security check:** `pip-audit` on the environment reported **no known vulnerabilities**. All packages are mainstream and pre-installed on Colab, so the notebook itself needs no `pip install`.
- Created `.gitignore` (venv, data, zips, checkpoints, scratch outputs, local changelog).

## 4. Writing the Python modules (`src/`, ≈18:25–18:50)

The code was written as normal Python modules first so it could be tested outside a notebook, then assembled into the notebook (Step 8).
Cross-module imports are tagged `# nb-skip` so the notebook builder can drop them; in the notebook everything lives in one namespace.

### 4.1 `config.py`
- One `Config` dataclass holding all hyper-parameters **and the ablation switches** (`use_diff_tags`, `use_cross_solution`, `pool_mode`, `objective`, `mlm_epochs`).
  *Why:* every experiment, including each ablation, is then a documented, logged variation (`cfg.but(...)`) of one config.
- `get_device()` (CUDA → MPS → CPU; enables TF32 on CUDA), `amp_dtype()` (bf16 if supported, else fp16), `set_seed()`, and a `JsonlLogger` that appends one JSON record per event so **every run leaves a log file**.

### 4.2 `data.py`
- **Data location:** on Colab, mount Drive and read `Config.drive_zip`; locally, find a zip that contains the four expected files.
  *Security:* only the four expected file names are extracted, so a malicious zip cannot write arbitrary paths (no path traversal).
- **Split:** stratified 90/10 split of the training file with a fixed seed (42) → **14,501 train / 1,612 validation / 1,838 test**.
- **Tokenizer:** BPE with an 8k vocabulary, trained with `tokenizers` **on the training split only** (NFKC + lower-case, whitespace + digit pre-tokenisation).
  *Why:* PIQA contains many rare domain words (tools, materials, recipes). Sub-words avoid a high `[UNK]` rate, and learning the tokenizer only from training text keeps everything "from scratch".
- **Difference tags:** `difflib.SequenceMatcher` aligns the BPE sequences of `sol1` and `sol2`; each solution token is tagged *shared* (1) or *different* (2), goal/special tokens get 0.
  Measured: **median share of differing solution tokens = 14 %** (mean 25 %), which confirmed the design idea.
- **Lengths:** goal p99 = 20 BPE tokens; solution p95 = 65, p99 = 108, max = 492. Limits were set to 40 (goal) and 96 (solution) tokens.
- **Encoding:** each option is `[CLS] goal [SEP] solution [SEP]` with segment ids, tags and a solution mask; a batch has shape `[B, 2, L]`. Everything is tokenised once up front (`PIQADataset`) so the GPU is not starved by Python work.

### 4.3 `model.py` — DACT (Difference-Aware Contrastive Transformer)
| Component | Detail | Reason |
|---|---|---|
| `MultiHeadAttention` | Own implementation; uses fused `scaled_dot_product_attention` in training, explicit softmax when weights are requested | Speed on GPU, plus exact attention maps for visualisation |
| Embedding | token + position + segment + **difference-tag** embedding | Tells the encoder from layer 1 where the options disagree |
| `EncoderLayer` × 4 | Pre-LayerNorm, d = 256, 4 heads, FFN 1024, GELU, dropout 0.2 | Pre-LN trains more stably from scratch |
| `ContrastiveCrossSolution` | Tokens of option *i* attend to the solution tokens of option *j* (+ `[CLS]` as a "no counterpart" sink); ESIM-style fusion `[h; c; h−c; h⊙c]` | Plausibility is relative; the model should encode *what replaced what* |
| `AttentionPooling` | Additive attention with a learned bias per tag (`diff_attn`), or plain `attn`, or `mean` | Focus the decision on the few decisive tokens; weights are interpretable |
| Scorer | MLP on `[pooled; CLS]` → one logit per option; softmax over the two | Forced-choice objective |
| MLM head | Tied to the token embedding; used only for the optional warm-up | In-domain pre-training on training-split text only |

Both options go through the same weights, so the model is **permutation-equivariant**: swapping the options swaps the scores. This was later verified in the notebook (`swap option order -> scores swap: True`).
Size: **6.11 M parameters** (base configuration).

### 4.4 `train.py`
- AdamW (β₂ = 0.98), linear warm-up + cosine decay, gradient clipping at 1.0, bf16 autocast on CUDA, GradScaler only for fp16.
- **Bug found and fixed while writing:** the first version excluded embeddings from weight decay by checking the *parameter name* for `"Embedding"`, which never matches (`tok.weight`). It was rewritten to identify embedding weights by **module type**.
- `train_qa`: early stopping on **validation accuracy**, best checkpoint saved and restored, one JSONL record per epoch.
- `mlm_warmup`: 15 % masking (80/10/10), training-split sequences only.
- `predict`: returns probabilities, predictions and labels in dataset order.

### 4.5 `evaluate.py`
- `accuracy`, percentile **bootstrap 95 % CI** (2,000 resamples), **exact McNemar test** (binomial on discordant pairs, implemented with `math.comb`), `summarise_seeds` (mean, sample std).

### 4.6 `baselines.py`
| ID | Baseline | Notes |
|---|---|---|
| B0 | Majority class | Chance reference |
| B1 | TF-IDF (1–2-grams) of `sol2 − sol1` + logistic regression | Trained on both option orders; C tuned on validation |
| B2 | 2-layer BiLSTM + additive attention | Same BPE tokenizer, trained from scratch, packed sequences |
| B3 | `FacebookAI/roberta-base` + multiple-choice head | Fine-tuned 4 epochs, lr 2e-5, best epoch chosen on validation |
| B4 | `Qwen/Qwen2.5-1.5B` zero-shot | Length-normalised log-likelihood of each solution given `"Goal: …\nSolution:"` |

### 4.7 `viz.py`
- `plot_attention_case`: four panels per item — pooling weights of each option, sol1→sol2 cross-attention, last-layer solution→goal attention; differing tokens shown in bold red.
- `diff_attention_mass` / `diff_token_share`: share of pooling attention on differing tokens vs the uniform-attention reference.
- `select_cases`: most confident correct, most confident wrong, most uncertain predictions.

### 4.8 `experiments.py`
- `run_experiment`: trains an architecture for each seed, touching **only train and validation**, and saves checkpoints and a `*_val.json` summary.
- `test_experiment`: the **only** function that reads the test split; it reloads saved checkpoints and predicts once. The "representative run" for CI and McNemar is the seed with the best **validation** accuracy.
- `prepare_everything` (later moved into `data.py`): load splits, train the tokenizer, pre-tokenise.

## 5. Local smoke tests (≈18:32–18:45)

1. **Data check:** split sizes, vocab size 8,000 and diff-tag output looked correct, e.g. `how do you scissor something?` → `cut`(diff) `it`(shared) `open`(diff) `.`(shared).
2. **End-to-end smoke** (256 items per split, 1 epoch, tiny model): DACT, every ablation switch, MLM warm-up, BiLSTM and the separate test pass all ran without errors.
3. **Baseline smoke on CPU:** B1 ran; B3 (RoBERTa, 64 training items, 1 epoch) ran; B4 (Qwen2.5-0.5B, 40 validation items) scored 85 %, a plausible sanity check for the scoring code.

## 6. Full-data probe on the Mac GPU (≈18:31–19:00)

**Why:** to check that the architecture actually learns before spending Colab compute.
- DACT base config, full training data, Apple MPS: about 75 s per epoch.
- Result: validation accuracy **55.7 % → 58.6 % → … best 58.9 % at epoch 6**, while training accuracy kept rising (82 % at epoch 7, 87.5 % at epoch 12).
- **Conclusion:** the model learns (well above 50 %), but **overfits after 2–6 epochs**. Actions taken:
  - `epochs` 40 → 20 and `patience` 8 → 6 (a long cosine schedule never decayed before early stopping);
  - a validation grid over **model size, dropout and the MLM warm-up** was added to the notebook.
- The second probe (vanilla variant) was stopped early to free the Mac GPU; the Colab run repeats that comparison properly.

## 7. Notebook builder (`tools/build_notebook.py`, ≈18:50)

- Reads the official template, **keeps its title, Readme text and the three section titles verbatim**, and fills each section:
  - Section 1: environment check, `Config`, data code, loading, dataset statistics table and histograms, design implications.
  - Section 2: architecture diagram and component table, model code, baselines, training utilities, evaluation code, sanity checks (parameter count, shapes, permutation equivariance).
  - Section 3: protocol, experiment code, Step 1 grid → Step 2 main model → Step 3 ablations → Step 4 baselines → **Step 5 final test evaluation**, results table and bar chart, discussions, attention analysis, copy of outputs to Drive.
- Module code is inserted from `src/` with `# nb-skip` lines removed, so the notebook and the modules can never drift apart.
- A `PIQA_SMOKE=1` environment variable shrinks everything (256 items, 1 epoch, small model, CPU) so the whole notebook can be tested locally in minutes. It is always off on Colab.
- Model classes stay in Section 2, although the template suggests putting OOP code at the bottom, because the notebook must run top to bottom. This is explained to the marker in the Readme.

## 8. Notebook smoke runs (≈19:05–19:30)

1. **Run 1 (Mac GPU / MPS):** every cell passed up to the final test cell (grid, main model, ablations, B0–B4 on validation). It then **stalled inside the Qwen test pass on MPS** (kernel near 12 % CPU, no progress for over 7 minutes). The same function had just worked on validation, so this was diagnosed as an MPS backend issue, not a code bug. The run was interrupted and the partial outputs inspected.
2. **Fix:** smoke mode now forces the CPU device (`DEVICE = cpu if SMOKE else get_device()`).
3. **Run 2 (CPU):** **all 27 code cells passed**, including the results table, bar chart, 5 attention figures and the attention statistics. One figure was inspected visually to confirm the layout and the red highlighting of differing tokens.

## 9. Full run on Colab Pro via the Colab CLI (19:31–20:17)

**Setup**
- Found the installed `colab` CLI (`~/.local/bin/colab`, already authenticated) and read its bundled guide. The existing `pinball` CPU session belongs to other work and was left untouched.
- `colab new -s group69 --gpu A100` → **NVIDIA A100-SXM4-40GB**, torch 2.11 + CUDA 12.8.
- **Problem:** `colab drivemount` is interactive and cannot be run from an automated agent.
  **Solution:** uploaded the four data files directly to `/content/data/piqa/` (byte sizes verified against the local copies). `prepare_data()` finds the files and skips the Drive step. On a normal Colab run the notebook still reads `MyDrive/Group69/PIQA.zip`.
- `colab exec -s group69 -f CITS4012_69.ipynb --timeout 10800` executed every cell and saved the executed notebook locally.

**Timeline and results** (validation accuracy; about 4 s per DACT epoch on the A100 vs about 75 s on the Mac)

| Time | Step | Result |
|---|---|---|
| 19:34–19:38 | Step 1 grid (1 seed) | base 59.24 · small 59.93 · dropout 0.3 59.93 · **base + MLM 60.24** · small + MLM 58.93 → **selected: base + MLM warm-up** |
| 19:38–19:43 | Step 2 DACT, 3 seeds | **60.09 ± 0.66** |
| 19:43–20:10 | Step 3 ablations, 7 × 3 seeds | − diff tags 58.97 · − cross-attn 60.05 · − diff bias 59.43 · mean pool 59.80 · pointwise 60.07 · vanilla 57.88 · − MLM 59.14 |
| 20:10–20:14 | B2 BiLSTM, 3 seeds | 59.57 ± 1.15 |
| 20:15–20:16 | B3 RoBERTa-base, 4 epochs | 61.9 → 69.0 → 68.9 → **70.0** |
| ≈20:16 | B0 / B1 / B4 | 50.0 / 59.2 / **78.1** |
| ≈20:17 | Step 5 test, figures, statistics | all cells finished, exit code 0 |

A warning about `HF_TOKEN` appeared during B3/B4. It is harmless: both models are public, and Colab only tries to read the secret, which is unavailable outside its web UI.

**Wrap-up**
- Packed `outputs/{logs,results,figures,tokenizer.json}` on the VM (≈1 MB, checkpoints excluded), downloaded it, and **stopped the A100 session** so it stopped consuming compute units.

## 10. Final results

| Model | From scratch | Val acc | Test acc | Test 95 % CI* | McNemar p vs DACT* |
|---|---|---|---|---|---|
| **DACT (full)** | ✓ | **60.09 ± 0.66** | **60.37 ± 1.17** | [56.7, 61.2] | — |
| − difference tags | ✓ | 58.97 ± 0.16 | 60.52 ± 0.49 | [58.7, 63.0] | 0.18 |
| − cross-solution attention | ✓ | 60.05 ± 0.27 | 61.73 ± 0.74 | [58.7, 63.1] | 0.12 |
| − diff bias in pooling | ✓ | 59.43 ± 1.02 | 61.50 ± 1.27 | [59.6, 63.9] | 0.017 |
| mean pooling | ✓ | 59.80 ± 0.75 | 61.46 ± 1.06 | [58.5, 62.9] | 0.13 |
| pointwise objective | ✓ | 60.07 ± 0.46 | 60.92 ± 0.87 | [58.3, 62.7] | 0.23 |
| vanilla Transformer | ✓ | 57.88 ± 0.91 | 58.71 ± 0.95 | [57.2, 61.5] | 0.83 |
| − MLM warm-up | ✓ | 59.14 ± 0.37 | 61.37 ± 1.25 | [58.9, 63.3] | 0.09 |
| B2 BiLSTM + attention | ✓ | 59.57 ± 1.15 | 59.54 ± 0.96 | [57.9, 62.5] | 0.42 |
| B0 majority | – | 50.00 | 50.49 | [48.2, 52.7] | <0.001 |
| B1 TF-IDF + LR | ✓ | 59.24 | 61.21 | [59.0, 63.4] | 0.12 |
| B3 RoBERTa-base (fine-tuned) | ✗ | 70.04 | 68.50 | [66.4, 70.6] | <0.001 |
| B4 Qwen2.5-1.5B (zero-shot) | ✗ | 78.10 | 75.35 | [73.4, 77.3] | <0.001 |

\* CI and McNemar use each model's best-**validation** seed. For DACT this is seed 44, which is its weakest seed on test (59.0 % vs 60.9 % and 61.2 %). It was kept because choosing a seed by test accuracy would break the protocol.

## 11. Analysis and fact-checking (20:18–20:30)

**Quantitative conclusions (written into the notebook)**
- DACT is the best from-scratch *neural* model: +2.2 (val) / +1.7 (test) over the vanilla Transformer and +0.5 / +0.8 over the BiLSTM. The gain is modest and not significant against B1/B2.
- TF-IDF + LR is as strong as the neural from-scratch models. From 14.5k items, much of the learnable signal is word association, not physical reasoning.
- Pretrained models lead by 8–15 points: world knowledge is the bottleneck.
- Ablations: on validation every single removal ties or hurts (diff tags and MLM ≈ −1 point each; cross-attention and the pairwise objective ≈ 0). Removing everything hurts on both splits. On test, single ablations score 0.5–1.4 points higher than the full model, which is within noise and opposite to validation. This was reported as observed and **no design change was made based on test results**. Interpretation: the components are partly redundant, since they all encode where the options differ.

**Qualitative conclusions**
- DACT puts **64 % of its pooling attention on differing tokens, which are only 25 % of tokens**. The attention mass is almost the same for correct (63.0 %) and wrong (64.7 %) predictions (Mann-Whitney p = 0.08). Errors come from judging the right tokens wrongly, not from looking in the wrong place.

**Claims corrected during fact-checking** (every statement was checked against the figures and the training data before it was accepted)
1. **Cobbler case (test item 1433):** the first draft said the model attended to the differing words. Checking the token positions showed the only difference (*ground coffee* vs *granulated sugar*) is at BPE positions **97–98, just past the 96-token truncation**. Both inputs are identical after truncation, hence p = 0.50 exactly. The text was rewritten, and "truncate around the differing span" was added as a concrete improvement.
2. **Lexical priors quantified:** in training items, *baby wipe(s)* appears only in the correct option 26× vs only in the wrong option 5×; *tissue paper* 3× vs 24×; *cooler* 2× vs 7×, which misleads the model on the *store food without a refrigerator* failure; *crinkle* appears in only 2 items.
3. **Cross-attention in failure 1:** the draft said *cooler* aligned with *chest*. The figure shows the attention spreading over shared words instead, so the text was corrected.
4. **Two overstatements removed:** "holds for every seed on both splits" (true only on validation) and "+0.9 over BiLSTM on test" (actually +0.8).

## 12. Assembling the final notebook (≈20:20)

- Rebuilt the notebook from the builder so it contains the final discussion text, then **copied the saved outputs cell-by-cell** from the executed A100 notebook. The script asserted that every code cell was identical to the executed one, so the outputs belong to exactly that code.
- Result: 48 cells (27 code), 8 figures, **0 error outputs**.
- Checks: all template titles present unchanged; no tokens, e-mail addresses or local paths in the notebook.
- Removed scratch artifacts (`outputs_smoke/`, `outputs_nbsmoke/`, `outputs_probe/`, the intermediate executed copy). Updated `.gitignore` so `outputs/` logs are committed but checkpoints are not.

## 13. Version control (20:23)

Four atomic commits, pushed to `origin/main` (`c65ddb6..9db1a8e`):
1. `d356988` — `.gitignore`
2. `02ca061` — `src/` modules and the notebook builder
3. `cefe3c7` — executed `CITS4012_69.ipynb` and the official template
4. `9db1a8e` — logs, results and figures of the A100 run

A local `CHANGELOG.local.md` (git-ignored) records the release notes.

---

## How to reproduce

**On Google Colab (intended path)**
1. Put the data zip at `MyDrive/Group69/PIQA.zip`.
2. Open `CITS4012_69.ipynb`, choose `Runtime → Change runtime type → GPU`, then `Runtime → Run all` (about 45 min on an A100).

**Locally (smoke test only)**
```bash
python3.12 -m venv .venv
.venv/bin/pip install torch numpy scikit-learn scipy pandas matplotlib tokenizers transformers nbconvert ipykernel
# put the zip in the repo root, then:
PYTHONPATH=src .venv/bin/python src/experiments.py      # module-level smoke test
.venv/bin/python tools/build_notebook.py                 # regenerate the notebook from src/
PIQA_SMOKE=1 jupyter nbconvert --to notebook --execute CITS4012_69.ipynb --output /tmp/smoke.ipynb
```
Note: regenerating the notebook produces a version **without** outputs. The submitted notebook contains the outputs of the A100 run.

**With the Colab CLI (what we did)**
```bash
colab new -s group69 --gpu A100
for f in train.jsonl train-labels.lst test.jsonl test-labels.lst; do colab upload -s group69 data/piqa/$f /content/data/piqa/$f; done
colab exec -s group69 -f CITS4012_69.ipynb --timeout 10800   # writes CITS4012_69_output.ipynb
colab stop -s group69
```

## Known limitations / possible improvements
- Prefix truncation at 96 BPE tokens hides the difference in about 1–2 % of long items; truncating around the differing span would fix this.
- The Google Drive loading path was not exercised in the CLI run (the files were uploaded directly).
- Results come from one Colab run with 3 seeds; with about ±1 point seed std and a ±2.3-point test CI, differences under about 2 points are not conclusive.
- A pretrained model was used only as a baseline, as required. The gap to B3/B4 shows that combining this architecture with pretrained representations would be the natural next step outside the assignment constraints.
