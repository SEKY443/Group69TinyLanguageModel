# Development Log — CITS4012 Group 69 (PIQA, DACT)

The complete step-by-step record of this project, from the creation of the repository to the final push, with the reasoning behind each decision. It covers the dead ends, the smoke tests, the data cleaning and the clean-up of the repository.

All times are local (CST, UTC+8) on **2026-09-29**. Times of the Colab run come from the JSONL logs in `outputs/logs/`, which the Colab VM wrote in UTC and which are converted here.

**Contents**
0. Requirements
1. Repository created
2. Discovery
3. Plan and decisions
4. Environment setup
5. Data preparation and cleaning
6. Model and training code
7. Smoke test 1: data pipeline
8. Smoke test 2: end-to-end modules
9. Probe run on full data (Mac GPU)
10. Smoke test 3: pretrained baselines
11. Notebook builder
12. Smoke test 4: notebook on Mac GPU (stalled)
13. Smoke test 5: notebook on CPU (passed)
14. Full run on Colab Pro A100
15. Results
16. Analysis and fact-checking
17. Final notebook assembly and checks
18. Clean-up
19. Version control
20. Reproduction
21. Limitations

---

## 0. Requirements (from the assignment brief)

| Requirement | How it is met |
|---|---|
| QA task on one of TweetQA / MultiRC / ReClor / **PIQA** | PIQA (physical commonsense, 2-option forced choice) |
| No external data; test set only for the final evaluation | Validation split carved from the training file; test used only in "Step 5" of Section 3 |
| Main model trained from scratch; RNN/LSTM/GRU/Transformer; attention required | DACT: Transformer encoder + cross-attention + attention pooling, random initialisation |
| Original, convincing, dataset-specific design (not just a standard architecture) | Difference tags, contrastive cross-solution attention, difference-guided pooling |
| Baselines produced by the group (no numbers copied from papers) | B0–B4, all produced by code in the notebook |
| Suitable metric with justification | Accuracy + seed statistics, bootstrap CI, McNemar test |
| Ablation studies | 7 ablations × 3 seeds |
| Attention visualisation; success and failure cases | 5 case figures + an aggregate attention statistic |
| `CITS4012_YourGroupID.ipynb`, official template, reproducible on a clean Colab, saved logs | `CITS4012_69.ipynb` generated from the template; logs in `outputs/` |
| No copying of open-source projects | Whole pipeline written for this project; only standard libraries used |

---

## 1. Repository created (17:59)

- GitHub repository `SEKY443/Group69TinyLanguageModel` created with a one-line `README.md` (commit `c65ddb6 first commit`).
- The official template `Copy_of_CITS4012_YourGroupID.ipynb` (18:13) and the data zip `drive-download-20260929T100449Z-1-001.zip` (18:04) were added to the folder. A copy named `PIQA.zip` was also placed there later and uploaded to Google Drive at `MyDrive/Group69/PIQA.zip`.

## 2. Discovery (≈18:15)

**Actions**
- Listed the files and read `README.md`; there was no project `CLAUDE.md` or contributing guide.
- Read the template: a title cell, a Readme cell and three section cells whose titles **must not be modified**: `# 1.Dataset Processing`, `# 2. Model Implementation`, `# 3.Testing and Evaluation`. Each has one empty code cell.
- Listed the zip without extracting it into the repository: `train.jsonl` (4.38 MB), `train-labels.lst`, `test.jsonl`, `test-labels.lst`.
- Extracted a copy into the session's scratch directory and computed first statistics:

| | train | test |
|---|---|---|
| items | 16,113 | 1,838 |
| label 0 / 1 | 8,053 / 8,060 | 910 / 928 |
| mean goal length (words) | 7.06 | 7.08 |
| mean solution length (words) | 18.8 | 18.5 |

- Checked for `torch` in the system Python: not installed, so a virtual environment was needed.

**Findings that shaped the project**
1. **There is no validation split.** The provided "test" file (1,838 items) is the official PIQA dev set. A validation set therefore had to be carved from the training file, so that the test set is used for nothing except the final evaluation.
2. **The labels are balanced**, so accuracy is an unbiased metric and chance is 50 %.
3. **The two solutions are near-duplicates.** Example from `train.jsonl`: *"…bedding made of **ripped paper strips**…"* vs *"…bedding made of **ripped jeans material**…"*. The answer depends on a few words, which became the core design idea.

## 3. Plan and decisions (≈18:20)

A written plan (model, baselines, metrics, ablations, attention analysis, code layout) was presented before any code. Three questions were asked:

| Question | Answer |
|---|---|
| Notebook name | `CITS4012_69.ipynb` |
| How Colab gets the data | Google Drive; the user uploads the provided zip (later: `MyDrive/Group69/PIQA.zip`) |
| Local `.venv` for smoke tests? | Yes |

While the environment was being set up the user added: *"I have Colab Pro and you can use it"* and *"use GPU to maximise the efficiency"*. The code was therefore designed GPU-first (bf16 autocast, TF32 matmuls, pre-tokenised tensors, pinned-memory loaders), with the Mac used only for correctness tests.

## 4. Environment setup (≈18:22)

- Checked the available interpreters (3.11, 3.12, 3.13) and chose **Python 3.12** for broad wheel support.
- `python3.12 -m venv .venv`, then installed `torch 2.14.0`, `numpy 2.5.3`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`, `tokenizers 0.23.2`, `pandas`. `torch.backends.mps.is_available()` returned `True`, so the Mac GPU (MPS) could be used.
- Later additions: `transformers 5.17.0` and `scipy 1.18.1` (baselines), `nbclient`, `nbformat`, `ipykernel` (to execute the notebook headlessly), `pip-audit`.
- **Dependency security check:** `pip-audit` on the whole environment → **"No known vulnerabilities found"**. All packages are widely used and pre-installed on Colab, so the notebook needs no `pip install` cell.
- Created `src/`, `tools/` and a first `.gitignore` (`.venv/`, `__pycache__/`, `data/`, `outputs/`, `*.zip`, `CHANGELOG.local.md`, `.DS_Store`).

## 5. Data preparation and cleaning (`src/data.py`, ≈18:25)

Everything below is implemented in `src/data.py` and appears in Section 1 of the notebook.

### 5.1 Safe extraction
- On Colab the code mounts Google Drive and reads `Config.drive_zip` (`/content/drive/MyDrive/Group69/PIQA.zip`). Locally it searches `*.zip` for an archive that contains all four expected files.
- **Only the four expected files are extracted, by base name**, into `data/piqa/`. Any other entry in the zip is ignored.
  *Why:* a zip can contain paths like `../../x`. Writing only known file names removes the path-traversal risk and ignores folders or junk inside the archive.
- If the four files already exist in `data/piqa/`, extraction is skipped. This is what made the Colab CLI run possible (Section 14).

### 5.2 Validation of the raw data
`load_split()` checks the data before anything uses it:
- the number of JSON rows equals the number of labels (`assert`);
- every label is 0 or 1 (`assert`);
- blank lines are skipped.

Data-quality audit of the raw files:

| Check | train (16,113) | test (1,838) |
|---|---|---|
| Items with irregular whitespace (double spaces, leading/trailing spaces, newlines) | 2,335 | 281 |
| – fields with double spaces | 2,759 | – |
| – fields with leading/trailing spaces | 1,942 | – |
| – fields with newlines/tabs | 3 | – |
| Items with an empty goal or solution | 0 | 0 |
| Items whose two solutions are identical | 0 | 0 |
| Label mismatches | 0 | 0 |

### 5.3 Cleaning
- **Whitespace normalisation:** every `goal`, `sol1`, `sol2` is rewritten as `" ".join(text.split())`, which collapses runs of spaces, tabs and newlines and strips the ends. This affected 2,335 training and 281 test items.
  *Why:* stray whitespace carries no meaning, but it would change token alignment and the difference tags.
- **Unicode and case normalisation** in the tokenizer: NFKC (unifies look-alike characters, e.g. full-width forms) followed by lower-casing.
  *Why:* PIQA comes from informal how-to text (instructables.com). Case carries almost no information here, and merging case variants gives rare words more training examples.
- **Nothing else was removed or rewritten.** No items were dropped, and spelling mistakes and informal phrasing were kept, because they are part of the task.

### 5.4 Split
- Stratified 90/10 split of the training file with seed 42 (`sklearn.model_selection.train_test_split`) → **14,501 train / 1,612 validation / 1,838 test**, with label balance kept (50.0 % / 50.0 % / 50.5 % label-1).
- The split is fixed across all runs, so every model is selected on the same validation items.

### 5.5 Tokenizer
- Byte-pair encoding (vocabulary 8,000, minimum pair frequency 2, `##` continuation prefix), trained with the `tokenizers` library **on training-split text only** (goal + both solutions of 14,501 items). Pre-tokenisation splits on whitespace/punctuation, and digits are split individually so numbers do not flood the vocabulary.
- Special tokens: `[PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4`.
- Measured **unknown-token rate on the test split: 0.0 %**; every word can be spelled from learned sub-words.

### 5.6 Difference tags
- The BPE sequences of `sol1` and `sol2` are aligned with `difflib.SequenceMatcher(autojunk=False)`. Tokens inside "equal" blocks are tagged *shared* (1), all others *different* (2); goal and special tokens get 0.
- Example: `how do you scissor something?` / `cut it open.` vs `close it.` → `cut`(2) `it`(1) `open`(2) `.`(1).
- Median share of differing solution tokens: **14.9 % train, 14.3 % val, 15.4 % test** (mean ≈ 25 %). This confirmed that the decisive signal is a small part of each input.

### 5.7 Length limits
- BPE lengths on the training split: goal p95 = 16, p99 = 20, max = 37; solution p95 = 65, p99 = 108, max = 492.
- First limits were 40 (goal) and 80 (solution). The solution limit was **raised to 96** after this measurement, so that about 98–99 % of solutions are kept whole. On the test split **58 of 3,676 solutions (1.6 %)** are still truncated. Section 16 shows one case where this matters.
- `max_len` (position embeddings) was raised from 128 to 160 to fit 40 + 96 + 3 special tokens.

### 5.8 Encoding and batching
- Each option becomes `[CLS] goal [SEP] solution [SEP]` with segment ids (0 = goal part, 1 = solution part), tags and a solution mask. A batch has shape `[B, 2, L]` and is padded to the longest sequence in the batch.
- `PIQADataset` tokenises everything **once** at start-up so the GPU is not held up by Python work.
- Safety fallbacks: if an option had no solution tokens, attention pooling falls back to `[CLS]`, and cross-attention always has `[CLS]` as a key, which avoids NaN from fully masked softmax rows. Neither case occurs in PIQA (0 empty fields), but the code does not assume that.

## 6. Model and training code (≈18:30–18:50)

The code was written as normal Python modules so it could be tested outside a notebook, then assembled into the notebook (Section 11). Imports between modules end in `# nb-skip` so the notebook builder can drop them.

### 6.1 `config.py`
- One `Config` dataclass with all hyper-parameters **and all ablation switches** (`use_diff_tags`, `use_cross_solution`, `pool_mode`, `objective`, `mlm_epochs`); `cfg.but(**changes)` makes a variant.
  *Why:* every experiment is a logged, reproducible variation of one config.
- `get_device()` (CUDA → MPS → CPU; enables TF32 and cuDNN benchmark on CUDA), `amp_dtype()` (bf16 if supported, else fp16, none off-GPU), `set_seed()`, `JsonlLogger`.

### 6.2 `model.py` — DACT
| Component | Detail | Reason |
|---|---|---|
| `MultiHeadAttention` | Own implementation; fused `scaled_dot_product_attention` in training, explicit softmax when weights are requested | GPU speed plus exact attention maps for visualisation |
| Input embedding | token + position + segment + **difference tag** | Marks where the options disagree from the first layer on |
| `EncoderLayer` × 4 | Pre-LN, d = 256, 4 heads, FFN 1024, GELU, dropout 0.2, attention dropout 0.1 | Pre-LN trains stably from scratch |
| `ContrastiveCrossSolution` | Option *i*'s tokens attend to option *j*'s solution tokens (+ `[CLS]` sink); fusion MLP over `[h; c; h−c; h⊙c]`, residual + LayerNorm | Plausibility is relative; encodes *what replaced what* |
| `AttentionPooling` | `v·tanh(Wh) + b[tag]`, with `b` initialised to [0, 0, 1] (other, shared, diff); modes `diff_attn` / `attn` / `mean` | Steers the summary towards the decisive tokens; weights can be interpreted |
| Scorer | MLP on `[pooled; CLS]` → one logit per option | Pairwise softmax over the two logits |
| MLM head | Tied to the token embedding | Optional warm-up on training-split text only |

- Initialisation: N(0, 0.02) for Linear/Embedding, zero bias, zero padding row.
- The two options share all weights, so the model is **permutation-equivariant** (swapping the options swaps the scores). This was verified numerically in every notebook run.
- Size: 6.11 M parameters (base); 0.65 M in the tiny smoke configuration.

### 6.3 `train.py`
- AdamW (β = 0.9 / 0.98), linear warm-up (6 %) + cosine decay, gradient clipping at 1.0, bf16 autocast on CUDA, GradScaler only for fp16.
- `train_qa`: one JSONL record per epoch, early stopping on **validation** accuracy, best checkpoint saved and restored.
- `mlm_warmup`: 15 % of non-special tokens selected; 80 % → `[MASK]`, 10 % → random token, 10 % unchanged.
- `qa_loss`: pairwise cross-entropy with label smoothing 0.1, or pointwise BCE for the ablation.
- **Bug found and fixed while writing:** the first `make_optimizer` excluded embeddings from weight decay by checking the *parameter name* for `"Embedding"`, which never matches names like `tok.weight`, so embeddings would have been decayed. It was rewritten to identify embedding weights by **module type**.

### 6.4 `evaluate.py`
- `accuracy`, percentile **bootstrap 95 % CI** (2,000 resamples, seeded), **exact two-sided McNemar test** (binomial on discordant pairs via `math.comb`, so no extra dependency), `summarise_seeds` (mean, sample std).
- A `majority_vote` seed-ensemble helper was drafted and then **removed** before use. It would have compared an ensemble instead of the models as trained, so the best-validation-seed rule was used instead.

### 6.5 `baselines.py`
| ID | Baseline | Details |
|---|---|---|
| B0 | Majority class | Most frequent training label |
| B1 | TF-IDF (1–2-grams, `min_df=2`, sublinear tf) of `sol2 − sol1` + logistic regression | Trained on both option orders (x, y) and (−x, 1−y); C ∈ {0.03 … 10} chosen on validation |
| B2 | 2-layer BiLSTM (256 per direction) + additive attention | Same BPE tokenizer, from scratch, packed sequences, lr 1e-3, dropout 0.3 |
| B3 | `FacebookAI/roberta-base` + multiple-choice head | 4 epochs, lr 2e-5, batch 32, linear warm-up; best epoch chosen on validation |
| B4 | `Qwen/Qwen2.5-1.5B`, zero-shot | Mean log-probability of the solution tokens after the prompt `"Goal: …\nSolution:"`; the higher-scoring option wins |

An unused import (`TAG_DIFF`) was removed from `baselines.py` during review.

### 6.6 `viz.py` and `experiments.py`
- `viz.py`: the 4-panel attention figure (pooling weights per option, sol1→sol2 cross-attention, last-layer solution→goal attention, differing tokens in bold red), `diff_attention_mass`, `diff_token_share`, `select_cases`.
- `experiments.py`: `run_experiment` uses **only train and validation** and saves checkpoints and `*_val.json`. `test_experiment` is the **only** code path that reads the test split: it reloads saved checkpoints and predicts once. The representative run for CI and McNemar is the best-**validation** seed. `prepare_everything` was later moved into `data.py` because it is data code. An unused `mcnemar_exact` import was removed.

## 7. Smoke test 1 — data pipeline (≈18:28)

**Goal:** check extraction, cleaning, split, tokenizer, tags and lengths on the real files.
- Extraction picked up `PIQA.zip` from the project folder and wrote the four files to `data/piqa/`.
- Output: `14501 1612 1838`, vocabulary `8000`, the length percentiles in 5.7, and the tag example in 5.6.
- Mean share of differing tokens over 3,000 items = 25.0 %, median 14.3 %.
- **Action taken:** raised `max_sol_len` from 80 to 96 and `max_len` from 128 to 160. The first `sed` edit of `max_sol_len` silently did not apply because its pattern included a trailing space. This was noticed when checking the file and fixed with a second edit.

## 8. Smoke test 2 — end-to-end modules (≈18:35)

**Goal:** prove that every component runs together before using real compute.
**Setup:** `src/experiments.py` main block: 256 items per split, 1 epoch, 1 MLM epoch, tiny model (d = 64, 2 layers), MPS device.

- **First attempt failed:** `FileNotFoundError: No zip containing the PIQA files was found`. The script had been started from inside `src/`, so relative paths did not resolve. It was re-run from the repository root with `PYTHONPATH=src`. This was an operator error, not a code bug.
- **Second attempt passed.** Every path ran: MLM warm-up, DACT training, the separate test pass (bootstrap CI computed), all five ablation switches, and the BiLSTM.

| Run (1 epoch, 256 items) | Accuracy |
|---|---|
| DACT, separate test pass | 55.1 % (bootstrap CI [49.2, 60.9]) |
| − diff tags / − cross-attention / mean pooling / attn pooling / pointwise (val) | 48.0 / 56.3 / 50.0 / 50.0 / 49.6 % |
| BiLSTM, val and test pass (27 s/epoch on MPS) | 53.9 % |

These accuracies are **meaningless as results** (tiny data, one epoch). The test only proved that the code runs. The identical `mean` and `attn` pooling numbers were checked: attention pooling is almost uniform at initialisation, so after one tiny epoch the two modes match.

## 9. Probe run on full data — Mac GPU (18:31–19:05)

**Goal:** check that DACT actually learns before spending Colab compute.
**Setup:** base config, full 14,501 training items, 12 epochs, MPS, running in the background.

- Output was not visible at first because Python buffers stdout under `nohup`. Progress was read from the JSONL log instead, which is one reason every run writes its own log.
- About 75 s per epoch on MPS (one epoch took 156 s while other jobs competed for the GPU).

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| train acc | 52.4 | 62.8 | 69.5 | 74.0 | 77.0 | 79.6 | 82.2 | 84.1 | 85.9 | 87.5 |
| val acc | 55.7 | 58.6 | 58.6 | 58.8 | 58.1 | **58.9** | 58.6 | 58.1 | 57.8 | 57.8 |

**Conclusions and actions**
- The model learns: validation accuracy is well above 50 %, in the range expected for PIQA from scratch.
- It **overfits after about 2–6 epochs** (training accuracy keeps climbing, validation does not). So:
  - `epochs` 40 → 20 and `patience` 8 → 6, because with a 40-epoch cosine schedule the learning rate never decays before early stopping;
  - the validation grid in the notebook was given regularisation options: a smaller model, dropout 0.3, and the MLM warm-up.
- A second probe (the "vanilla" variant) was started, then **stopped** at 19:05 to free the Mac GPU for smoke test 4. The Colab run repeats that comparison properly.

## 10. Smoke test 3 — pretrained baselines (≈18:50)

**Goal:** make sure the baseline code works before the long run. Run on CPU so it would not disturb the probe on MPS.
- **B0** majority and **B1** TF-IDF + LR on the full splits: both ran, and C was chosen on validation.
- **B3** RoBERTa-base on 64 training / 32 validation / 32 test items, 1 epoch: ran end-to-end (109 s on CPU). The expected warnings about the new classifier head appeared.
- **B4** Qwen2.5-**0.5B** on 40 validation items: **85 %**, a plausible sanity check that the log-likelihood scoring is correct.

## 11. Notebook builder — `tools/build_notebook.py` (≈18:55)

- Reads the official template, **keeps its title, Readme text and the three section titles verbatim**, and fills the sections:
  - Section 1: environment/version check, `Config`, data code, loading, dataset statistics table and histograms, design implications.
  - Section 2: architecture diagram and component table, DACT code, baselines table and code, training utilities, evaluation code, sanity checks (parameter count, output shapes, permutation equivariance).
  - Section 3: protocol and metric justification, experiment and visualisation code, Step 1 grid → Step 2 main model → Step 3 ablations → Step 4 baselines → **Step 5 final test evaluation**, results table and bar chart, discussions, attention analysis, copy of outputs to Drive.
- Module code is copied from `src/` with `# nb-skip` lines removed, so the notebook and the modules cannot drift apart.
- `PIQA_SMOKE=1` shrinks everything (256 items per split, 1 epoch, tiny model) so the whole notebook can be tested in minutes. It is always off on Colab.
- Two leftovers were cleaned up in the builder: a clumsy string-replace hack in the sanity-check cell, and a no-op title replacement.
- The template suggests putting OOP code at the bottom. The model classes stay in Section 2 because the notebook must run top to bottom, and the Readme explains this to the marker.

## 12. Smoke test 4 — notebook on the Mac GPU (≈19:00–19:18) ❌ stalled

**Setup:** the generated notebook executed headlessly with `nbclient`, `PIQA_SMOKE=1`, on MPS.

What passed (all 21 cells up to the final test cell):
- environment check, data loading (256/256/256), statistics table and figure;
- sanity checks: `DACT parameters: 0.65M | BiLSTM baseline: 3.14M`, output shapes correct, `swap option order -> scores swap: True`;
- Step 1 grid (smoke values 54.3 / 53.1 / 54.7 / 56.3 / 53.5 %, "selected on validation: base + MLM warm-up"), Step 2 main model, Step 3 ablations;
- Step 4 baselines on validation: B0 48.8 %, B1 55.1 % (C = 0.03), B2 54.3 % (120 s per epoch on MPS), B3 47.7 %, B4 71.5 %.

What failed:
- **The run stalled in Step 5, inside the Qwen test pass on MPS.** The test JSON files were written at 19:10, but no figure appeared. At 19:17 the kernel was at about 12 % CPU with no progress for more than 7 minutes, and the first watcher timed out after 30 minutes without events.
- **Diagnosis:** the same function had just finished on the validation split, and on CPU in smoke test 3. That points to an Apple MPS backend issue, not a bug in the code.
- **Action:** interrupted the run with SIGINT so `nbclient` saved the partial notebook, inspected every cell's output, and changed smoke mode to run on CPU (`DEVICE = torch.device("cpu") if SMOKE else get_device()`). Colab is not affected, because `SMOKE` is false there.

## 13. Smoke test 5 — notebook on CPU (19:18–19:31) ✅ passed

- **All 27 code cells executed without an error.**
- The results table (13 rows, with the CI and McNemar columns), the bar chart, all five attention figures and the attention statistics were produced.
- Selected cases: correct #127 / #43, wrong #248 / #197, uncertain #62 / #135 (on the smoke subset).
- Attention statistic after one tiny epoch: 38.6 % of pooling mass on differing tokens (uniform reference 24.6 %); learned tag bias ≈ [0, 0, 1.0].
- One figure (`confident_correct_127.png`) was opened and inspected: correct layout, labels readable, differing tokens in bold red.
- **Decision:** the notebook is ready for the real run.

## 14. Full run on Colab Pro via the Colab CLI (19:31–20:19)

### 14.1 Finding the tool
The user asked to "use colab cli". `colab` was not on `PATH`, but `~/.config/colab-cli/` existed, and the binary was found at `~/.local/bin/colab` (google-colab-cli 0.7.1), already authenticated. Its bundled guide (`colab skill`) was read before use. It contained the key facts: sessions are billable until stopped, `exec -f notebook.ipynb` runs every cell and saves `<name>_output.ipynb`, and `drivemount` is interactive. An existing CPU session called `pinball` belongs to other work and was **left untouched**. The OAuth token file was deliberately not opened.

### 14.2 Session and data
- `colab new -s group69 --gpu A100` → **NVIDIA A100-SXM4-40GB**, torch 2.11.0 + CUDA 12.8, Python 3.13.
- **Problem:** `colab drivemount` needs a human at the terminal, so the Drive path could not be used from the CLI.
- **Solution:** `colab upload` of the four files directly to `/content/data/piqa/`. The byte sizes were checked against the local copies (4,382,162 / 32,226 / 495,929 / 3,676). `prepare_data()` finds the files and skips Drive (Section 5.1). On a normal Colab run the notebook still reads `MyDrive/Group69/PIQA.zip`.
- `colab exec -s group69 -f CITS4012_69.ipynb --timeout 10800` (the default timeout is 30 s, far too short for training).

### 14.3 Timeline (validation accuracy; about 4 s per DACT epoch on the A100 vs about 75 s on the Mac)

| Time | Step | Result |
|---|---|---|
| 19:31 | Cells 1–11: environment, data, statistics, sanity checks | 14,501 / 1,612 / 1,838; 6.11 M parameters; permutation check `True` |
| 19:34–19:38 | Step 1 grid, 1 seed | base 59.24 · small 59.93 · dropout 0.3 59.93 · **base + MLM 60.24** · small + MLM 58.93 → **selected: base + MLM warm-up (10 epochs)** |
| 19:38–19:43 | Step 2 DACT, seeds 42/43/44 | 59.99 / 59.49 / 60.79 → **60.09 ± 0.66** |
| 19:43–20:10 | Step 3 ablations, 7 × 3 seeds | − diff tags 58.97 · − cross-attn 60.05 · − diff bias 59.43 · mean pool 59.80 · pointwise 60.07 · vanilla 57.88 · − MLM 59.14 |
| 20:10–20:14 | B0, B1, B2 BiLSTM (3 seeds) | 50.00 · 59.24 · 59.57 ± 1.15 |
| 20:15–20:16 | B3 RoBERTa-base | epochs: 61.85 → 68.98 → 68.86 → **70.04** (26 s/epoch) |
| ≈20:16 | B4 Qwen2.5-1.5B zero-shot | **78.10** |
| ≈20:17 | Step 5 test evaluation, bar chart, figures, statistics | done; `EXIT 0` |

- A warning `Error while fetching HF_TOKEN secret` appeared during B3/B4. It is harmless: both models are public, and Colab only tries to read the secret, which is unavailable outside its web UI.
- The progress watcher expired once (30-minute limit) and was re-armed; the run itself was unaffected.

### 14.4 Wrap-up on the VM
- Packed `outputs/{logs,results,figures,tokenizer.json}` into a 995 KB archive (checkpoints of about 600 MB excluded), downloaded it and extracted it into the repository.
- **`colab stop -s group69`** at about 20:19, so the A100 stopped consuming compute units. Afterwards `colab sessions` showed only the user's own `pinball` session.

## 15. Results

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
| B0 majority | – | 50.00 | 50.49 | [48.2, 52.7] | < 0.001 |
| B1 TF-IDF + LR | ✓ | 59.24 | 61.21 | [59.0, 63.4] | 0.12 |
| B3 RoBERTa-base (fine-tuned) | ✗ | 70.04 | 68.50 | [66.4, 70.6] | < 0.001 |
| B4 Qwen2.5-1.5B (zero-shot) | ✗ | 78.10 | 75.35 | [73.4, 77.3] | < 0.001 |

\* The CI and McNemar test use each model's best-**validation** seed. For DACT this is seed 44, which is its weakest seed on test (59.0 % vs 60.9 % and 61.2 %). It was kept because choosing a seed by test accuracy would break the protocol.

## 16. Analysis and fact-checking (20:18–20:30)

### 16.1 Quantitative conclusions (in the notebook)
- DACT is the best from-scratch *neural* model: +2.2 (val) / +1.7 (test) over the vanilla Transformer and +0.5 / +0.8 over the BiLSTM. The gain is modest and not significant against B1/B2.
- TF-IDF + LR is as strong as the from-scratch neural models. From 14.5k items, much of the learnable signal is word association.
- The pretrained models lead by 8–15 points, so world knowledge is the bottleneck.
- Ablations: on validation every single removal ties or hurts (difference tags and MLM ≈ −1 point each; cross-attention and the pairwise objective ≈ 0). Removing all components hurts on both splits. On test, single ablations score 0.5–1.4 points higher, which is within noise and opposite to validation. This is reported as observed, and **no design change was made based on test results**. Interpretation: the components are partly redundant, since all of them encode where the options differ.

### 16.2 Qualitative conclusions (in the notebook)
- DACT puts **64 % of its pooling attention on differing tokens, which make up only 25 % of solution tokens**. The mass is almost the same for correct (63.0 %) and wrong (64.7 %) predictions (Mann-Whitney p = 0.08). The model looks at the right tokens; errors come from judging them wrongly.
- Six figures were inspected one by one: two confident successes (*chalk boards* and *cup coaster*, both *baby wipes*), two confident failures (*food without a refrigerator*, *aluminium-foil ball*), one uncertain case (*blackberry cobbler*) and the attention-mass histogram.

### 16.3 Corrections made during fact-checking
Every claim in the draft text was checked against the figures, the logs and the training data. Six changes followed:
1. **Cobbler case (test item 1433):** the draft said the model attended strongly to the differing words. Checking the token positions showed the only difference (*ground coffee* vs *granulated sugar*) lies at BPE positions **97–98, just beyond the 96-token limit**, so both inputs are identical after truncation (hence p = 0.50 exactly). The text was rewritten, and "truncate around the differing span" was added as a concrete improvement.
2. **Lexical priors measured instead of assumed:** in training items, *baby wipe(s)* appears only in the correct option 26× vs only in the wrong option 5×; *tissue paper* 3× vs 24×; *cooler* 2× vs 7× (this misleads the refrigerator failure); *crinkle* appears in only 2 items. The count for *ice* was discarded because substring matching (*rice*, *slice*, *nice*) made it meaningless.
3. **Cross-attention in the refrigerator failure:** the draft said *cooler* aligned with *chest*. The figure shows the attention spreading over the shared words (*put the*) and *small*, so the text was corrected.
4. **Aluminium-foil failure:** added the verified observation that cross-attention aligns *fold … in* with *cr ##ink ##le*, i.e. the substitution is found correctly.
5. **"Holds for every seed on both splits"** → corrected. It is true only on validation; on test one vanilla seed (59.4 %) beats DACT's weakest seed (59.0 %).
6. **"+0.9 over BiLSTM on test"** → corrected to +0.8.

## 17. Final notebook assembly and checks (≈20:20)

- Rebuilt the notebook with the builder so it contains the final discussion text, then **copied the saved outputs cell by cell** from the executed A100 notebook. The merge script asserted that all 27 code cells were **identical** to the executed ones, so every output belongs to exactly that code.
- Result: 48 cells (27 code, 21 markdown), 8 images, **0 error outputs**.
- Template check: all five template headings present and unchanged.
- Secret/privacy scan of the notebook: no JWT/session tokens, no Hugging Face tokens, no e-mail address, no local file paths (0 matches).
- The Readme cell was updated to describe exactly how the saved outputs were produced (A100, Colab CLI, data uploaded directly) and the real runtime (about 45 min).

## 18. Clean-up

**Processes and cloud resources**
- Stopped the background probe run on the Mac (19:05).
- Interrupted the stalled MPS smoke run (19:18).
- Stopped the A100 session (≈20:19).
- Stopped the last leftover log watcher (`tail -f`) after the run.
- Final check: no project processes running locally; only the user's own `pinball` Colab session remains.

**Files removed**
| Removed | What it was |
|---|---|
| `outputs_smoke/` | output of smoke tests 2 and 3 |
| `outputs_nbsmoke/` | output of notebook smoke tests 4 and 5 |
| `outputs_probe/` | output of the full-data probe run |
| `CITS4012_69_output.ipynb` | intermediate executed notebook written by `colab exec`, merged into `CITS4012_69.ipynb` |
| `src/__pycache__/` | Python bytecode cache |

**Kept, but not committed**
- `.venv/` (local environment), `data/piqa/` (extracted data; the dataset is not redistributed), the zips, and `CHANGELOG.local.md` (local release notes).

**`.gitignore` final version**
`outputs/` was originally ignored as a whole. It was changed so the **logs, results and figures are committed** (they are required deliverables) while `outputs/checkpoints/` (about 600 MB of weights) and all scratch folders `outputs_*/` stay ignored.

## 19. Version control

| Time | Commit | Content |
|---|---|---|
| 17:59 | `c65ddb6` | first commit (README) |
| 20:23 | `d356988` | `.gitignore` |
| 20:23 | `02ca061` | `src/` modules and notebook builder |
| 20:23 | `cefe3c7` | executed `CITS4012_69.ipynb` and the official template |
| 20:23 | `9db1a8e` | logs, results and figures of the A100 run |
| 20:26 | `2f734ef` | first version of this development log |

The commits were pushed to `origin/main` after the user asked. At the user's request, no AI co-author lines were added. Local release notes are kept in `CHANGELOG.local.md` (git-ignored).

---

## 20. How to reproduce

**On Google Colab (intended path)**
1. Put the data zip at `MyDrive/Group69/PIQA.zip`.
2. Open `CITS4012_69.ipynb`, choose `Runtime → Change runtime type → GPU`, then `Runtime → Run all` (about 45 min on an A100). Allow Drive access when asked.

**Locally (smoke test only)**
```bash
python3.12 -m venv .venv
.venv/bin/pip install torch numpy scikit-learn scipy pandas matplotlib tokenizers transformers nbconvert ipykernel
# put the zip in the repository root, then:
PYTHONPATH=src .venv/bin/python src/experiments.py      # module-level smoke test
.venv/bin/python tools/build_notebook.py                 # regenerate the notebook from src/
PIQA_SMOKE=1 .venv/bin/jupyter nbconvert --to notebook --execute CITS4012_69.ipynb --output /tmp/smoke.ipynb
```
Regenerating the notebook produces a version **without** outputs. The submitted notebook contains the outputs of the A100 run.

**With the Colab CLI (what we did)**
```bash
colab new -s group69 --gpu A100
for f in train.jsonl train-labels.lst test.jsonl test-labels.lst; do
  colab upload -s group69 data/piqa/$f /content/data/piqa/$f
done
colab exec -s group69 -f CITS4012_69.ipynb --timeout 10800   # writes CITS4012_69_output.ipynb
colab stop -s group69                                        # always stop: sessions are billed
```

## 21. Known limitations / possible improvements
- **Prefix truncation** at 96 BPE tokens affects 1.6 % of test solutions and can hide the difference (test item 1433). Truncating around the differing span would fix this.
- The **Google Drive loading path** was not exercised in the CLI run (the files were uploaded directly). It is standard `google.colab.drive` code, but it was not tested end-to-end.
- Results come from **one Colab run with 3 seeds**. With about ±1 point seed std and a ±2.3-point test CI, differences under about 2 points are not conclusive.
- Pretrained models were used only as baselines, as required. The gap to B3/B4 suggests that pairing this architecture with pretrained representations would be the natural next step outside the assignment constraints.

## 22. Evidence audit and refinement — 30 September 2026

This section records a new audit and local replication. Sections 1–21 describe the earlier A100 work;
the corrections here supersede their interpretations where noted. No original A100 result file was
overwritten. No remote commit or push was made in this audit. The user explicitly requested continued
refinement and maintenance of this log. The official six-page project brief and actual supplied files
were reviewed before changing implementation. `PROJECT_AUDIT.md` records the pre-change inventory,
requirements comparison, findings and priorities. `evidence/historical/` preserves the original notebook,
eight source modules and builder; its SHA-256 manifest records the original evidence.

### 22.1 Findings that affect report claims

- The supplied training file has 16,113 items. The unchanged stratified split (seed 42) gives 14,501
  training and 1,612 validation items; the supplied test has 1,838 items. Labels index sol1/sol2 as 0/1.
  No missing/malformed fields were found. The 8,000-entry BPE was learned from training text only.
- One normalised exact test item (index 1545, vodka and soda) occurs twice in training. This is a
  supplied-data overlap, so an unqualified claim of strict example disjointness is incorrect. The
  historical split and scores remain intact; duplicate-excluded evaluation is a sensitivity check.
- Historical raw candidate swapping can change SequenceMatcher's directional alignment tags:
  755 training, 83 validation and 103 test items. Encoded-pair network equivariance is valid, but it
  does not establish raw-pipeline invariance. An optional canonical alignment was implemented and
  tested; it stays OFF in historical-protocol replication and has no benchmark score claimed.
- The original length table measured already-truncated inputs. New raw length tables document the
  40/96-token caps and 58 of 3,676 test solutions exceeding 96 tokens. Truncation erases the tokenised
  distinction for two test pairs. A difference-centred truncation strategy is only future work; it
  is not established that it would solve reasoning errors, and it was not test-tuned here.
- Pooling starts with a positive difference-token bias of +1. High mass on differing words therefore
  is not evidence that training independently discovered them, nor proof of causal reasoning.
- The no-pooling-bias historical comparison has raw p=.01696 and post-hoc Holm-adjusted p=.15267
  across 12 comparisons. Overlapping CIs do not establish that every observed change is noise.
  Historical CIs refer to the best-validation single seed, not the three-seed mean.
- B1/B3 historically generated test predictions before the final reporting stage. Inspection found
  no test-based checkpoint/hyperparameter selection, but the earlier claim of no test inference
  before that stage was too strong. The new notebook defers this inference.
- Original weights and per-item predictions are absent locally. Original metrics remain supported
  by saved notebook outputs, result files and logs; they cannot all be independently recomputed.
  New checkpoints/predictions do not retroactively recover those original arrays.

### 22.2 Implemented improvements

Preserved the DACT architecture, scratch initialisation, objective, historical split, length caps,
default alignment and selected hyperparameters. Added strict UTF-8/schema validation, separate run
and split seed records, environment/data/split provenance, validation loss, parameter counts, finite
loss checks, safe checkpoint metadata, per-item prediction export and overwrite protection. AMP now
honours the configuration flag. B3 retention and pretrained revision logging were added for future
full runs. The modified pretrained paths have not yet been executed locally.

MLM now projects only supervised positions through the vocabulary head, keeping the same mathematical
objective while reducing memory. Attention figures use adaptive sizing, 300 dpi, comparable pooling
scales, explicit predicted/gold labels and numeric sidecars. The architecture diagram is available
as SVG and PNG. `report_support/changes.md` documents each file, old/new behaviour and marking relevance.

The notebook builder defaults to `CITS4012_69_reproducible.ipynb` and refuses to overwrite executed
notebooks even with `--force`. The submission candidate retains all 27 original code/output pairs,
corrected prose and two genuinely executed audit cells. The separate current-source notebook avoids
presenting historical results as outputs of modified code. Report-support tables, equations, verified
primary references/BibTeX, architecture, training curves, experiment controls and limitations were created.

### 22.3 Checks actually executed

- Historical audit reconciled 32 scratch training runs with logs/results and checked three-seed
  arithmetic and model parameter counts (`report_support/consistency_checks.json`).
- Full-model checks passed: padding keys masked, normalisation, padding invariance, encoded swaps,
  gradient flow through encoder/cross/pooling attention and checkpoint reload/overwrite protection.
  Explicit versus fused attention maximum error was 1.68e-8. Altering pooling changed logits by .02761.
- Full versus selected-position MLM loss error was 0; maximum gradient discrepancy was 6.71e-8.
- A reduced diagnostic DACT reached 100% accuracy on 16 training items by step 10; final step-20
  loss was 5.36e-6. This is an overfit check, not a generalisation result.
- On 128 training items, original alignment failed raw-swap tag consistency for 7; canonical alignment
  failed for 0. Optional canonical model raw-swap checks passed.
- Fresh-kernel development notebook execution completed without errors. Latest evidence:
  `evidence/CITS4012_69_development_smoke_20260930_132631.ipynb`. Its current code matches the generated
  notebook and all eight source modules. Smoke skips pretrained downloads and uses validation-derived
  diagnostic data in place of official test data. It does not establish full clean-Colab compatibility.

### 22.4 Completed local fixed-recipe replication

The first attempt in `outputs_replication/` was interrupted for excessive GPU memory before a complete
model was produced; its logs are retained and it contributes no result. After the tested MLM projection
optimisation, the fresh `outputs_replication_efficient/` run completed at approximately 14:01 Perth time
on 30 September. Completion was verified from all logs, three saved checkpoints, six prediction files,
result JSON and four attention figures; the old interactive process handle was no longer available.

Protocol was saved before training. Seeds 42/43/44 used the historical validation-selected recipe and
saved training-only tokenizer, with no new search. All seeds trained before final test inference.
Local hardware: RTX 2070 Max-Q 8 GB, fp16, zero data-loader workers; Python 3.13.12, torch 2.11.0+cu128,
NumPy 2.5.3, sklearn 1.9.1, tokenizers 0.23.2. These differ from the nondeterministic historical A100
environment, so the run is an independent replication, not a controlled ablation or exact replay.

| Seed | QA epochs completed | Best validation accuracy (%) | Final test accuracy (%) |
|---|---:|---:|---:|
| 42 | 8 | 59.0571 | 62.2960 |
| 43 | 9 | 59.9876 | 62.5680 |
| 44 | 8 | 59.8635 | 62.8400 |
| Mean ± sample SD | — | 59.6361 ± 0.5052 | 62.5680 ± 0.2720 |

All three runs completed ten MLM epochs. The representative model is seed 43, selected by validation
accuracy before test evaluation. Its item-bootstrap 95% CI is [60.3917, 64.6368]%, a single-model CI.
Excluding the pre-identified duplicate yields 62.5476% on 1,837 items; this is sensitivity only.
Separate local lexical runs in `outputs_replication_baselines/` produced B0 test 50.4897% and B1 test
60.8814% (validation 58.9950%, C=3.0). Keep all these local results separate from the A100 table.

The historical main result remains validation 60.0910 ± 0.6575%, test 60.3736 ± 1.1736%. A higher local
replication score is not evidence that logging or MLM memory optimisation improves the model.

### 22.5 Submission status

The implementation and report evidence are substantially stronger. The complete historical study,
local replication and development tests have distinct provenance. Full revised B3/B4 execution and
the Drive/clean-Colab route remain unverified. No final ACL report PDF or invented team contribution
statement has been created. The required final filenames remain `CITS4012_69.ipynb` and
`CITS4012_69.pdf`; report notes are supporting evidence, not the finished six-page report.

## 23. Completed replication analysis and final handover — 30 September 2026

`tools/analyse_replication.py` independently verified all ten local prediction files, **17,250 rows**,
against the supplied validation/test text and labels. Stored correctness, probabilities, argmax labels,
per-run accuracy, seed aggregates, validation maxima and representative-seed CI all reconciled.
All three checkpoint/metadata pairs exist. The completed main/B0/B1 evidence has **38 file hashes**
in `report_support/replication_manifest.json`, including model weights and fitted lexical predictor.

Exploratory paired analysis used 10,000 same-item bootstrap resamples (seed 20260930) and exact McNemar
tests, with a declared two-comparison family (validation-selected local DACT seed 43 versus B0/B1).
DACT minus B1 = +1.6866 percentage points, paired CI [-.7617,4.0805], p=.18953 before/after Holm.
DACT minus B0 = +12.0783 points, CI [8.8683,15.1795], Holm p=3.20e-13. Thus the local lexical advantage
is uncertain. These tests are post-run exploratory evidence, not a preregistered confirmatory study.

The 688 local seed-43 failures now have structural analysis and a deterministic random sample of 12
with explicitly post-hoc, single-reviewer categories. Two identical retained-token pairs both fail;
24/32 items with some truncation are correct, so the analysis does not claim all truncated examples
perform worse. Among 420 predictions with confidence at least .8, 112 are wrong. Validation loss rises
while training loss falls. `report_support/replication.md` records exact examples, counts and caveats.

All four new full attention figures were visually inspected and their metadata reconciled with saved
predictions. Long heatmaps require zooming; compact report-ready SVG/PNG pooling plots were generated
from numeric sidecars, showing the six largest weights plus the omitted-weight sum without renormalising.
Success #1672 and failure #551 illustrate that focus on differing words can accompany either outcome.
The confidence-selected weight-loss case #1244 is treated as dataset-label disagreement, not validated
medical truth. Figure fp32 versus saved AMP inference differs by at most .000117 probability, with the
same labels. The compact attention figure and new training curves were visually checked.

`evidence/CITS4012_69_replication_verification.ipynb` was executed in a fresh kernel, reading actual
training logs, rerunning metric/error checks and displaying actual figures. It clearly states that it
verifies an already-completed training run, rather than claiming training executed in those cells.
`REFINEMENT_SUMMARY.md` supplies all 20 requested sections, full configurations/results, before/after
compliance and a candid readiness checklist. `report_support/files_changed.md` lists individual files.

Final consistency checks passed: **10 archived originals verified; 88 original output files unchanged;
27 original notebook code/output pairs preserved; five current/evidence notebooks schema-valid with
no error outputs; eight generated modules match source; latest fresh-kernel smoke code matches current
generated code; 38 replication artifact hashes verified; Python compilation and Git whitespace checks
passed.** A first version of the final source comparison mistakenly included module-only command-line
smoke sections omitted by the notebook builder; the checker was corrected to use the builder's exact
embedding transformation and then passed. This was a verification-script issue, not a source/notebook
mismatch. Windows line-ending notices and local kernel transport/event-loop notices were informational.

Remaining work is explicit: complete the final six-page ACL PDF with truthful contributions, verify the
full revised clean-Colab/Drive/B3/B4 workflow before promoting the new notebook, and preserve/backup
the Git-ignored new experiment artifacts. Historical missing weights/revisions cannot be retroactively
recovered by the new run. No external dataset, test-driven retuning, invented result, remote push or
unrequested final report was introduced.

## 24. Project folder organisation — 30 September 2026

At the user's request, reorganised project files for easier navigation. Kept the executed submission
notebook, README and dependency files at the repository root. Moved the audit and refinement handover
to `docs/`, the official template to `templates/`, and the current-source reproduction notebook to
`notebooks/`. Completed local runs now live under `experiments/completed/`; smoke/functional-test and
interrupted runs live under `experiments/development/`. The original `outputs/` A100 evidence and
`evidence/historical/` snapshots retain their existing locations. The supplied dataset and brief remain
outside the repository in the assignment directory. Added an assignment-level README pointing to the
project, a root folder map, and `docs/PROJECT_STRUCTURE.md` explaining artifact roles.

Before relocation, checked all source/destination paths were inside the intended project and confirmed
no relevant Python training process was running. Recorded the path mapping in `docs/path_relocations.json`
and pre-move SHA-256 hashes in `docs/organisation_manifest.json`. **Eleven entries containing 258 files
were relocated and verified byte-for-byte immediately after moving.** Nothing was deleted.

Updated active build, smoke, experiment, figure, analysis and inventory scripts to use the new locations.
`tools/project_paths.py` resolves old relative checkpoint paths from saved metadata without rewriting
the original experimental JSON/logs. Future generated notebook runs now go to completed/development
folders according to execution mode. Updated current README/report-support/handover path references;
chronological history and original snapshots deliberately retain their historical path descriptions.

Archived the previous executed replication-verification notebook unchanged in `evidence/archive/`,
then genuinely executed a new verification notebook with the new paths at its normal evidence location.
Regenerated the separate unexecuted reproduction notebook and ran a fresh smoke notebook:
`evidence/CITS4012_69_development_smoke_20260930_234421.ipynb`, **zero errors**. This checks the layout
change without rerunning full training or pretending that smoke metrics are benchmark results.

Final consistency checks passed after reorganisation: all 88 original output hashes and 27 original
code/output pairs preserved; all 38 completed local artifact hashes verified; generated source matches
the latest executed smoke; seven current/archive evidence notebooks validate with no error outputs;
Python compilation and Git whitespace checks pass. The new verification again reconciled 17,250
prediction rows. The folder changes do not alter model settings or benchmark scores. Git-ignored run
artifacts remain locally retained, not remotely backed up. No commit or push was performed.
