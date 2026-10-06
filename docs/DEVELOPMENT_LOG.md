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
22. Revision 2: review-driven improvements
23. Revision 2b: council review and follow-up changes
24. Re-run attempts with the Colab CLI, and changes for a browser run
25. Full re-run on a free T4 and updated results
26. Lexical head: closing the gap to TF-IDF
27. Report draft
28. Merge of the two tracks (`merge-review`)
29. Report updated with the audit findings
30. Review note for Salah; replication script fixed
31. Submission files prepared from `merge-review`
S1–S3. Parallel track on `main`: evidence audit and replication (Salah Elshafey)
32–39. Report polish, honest final pass and upgrades (robustness, reproducibility, tests, evaluation hygiene)
40. Final run of `main` on a free T4 (the reported run, `b880108`)
41–43. Training hardening; RoBERTa stability test; RoBERTa footnote
44–48. Report passes, group number 73, clean-test run stopped
49. Final audit and submission preparation (2026-10-06)

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

---

## 22. Revision 2: review-driven improvements (2026-09-29, ≈21:00–22:00)

The project was reviewed against the marking guide in the brief. The review scored it at about 70/100 and found the weaknesses below. Revision 2 addresses the ones that code can fix. The changes were made by Ali with **Claude Code (an AI coding assistant)**, which Ali directed and reviewed. Declare this AI assistance in the way the unit requires.

### 22.1 Review findings that motivated the changes
| # | Finding | Evidence | Action |
|---|---|---|---|
| 1 | The attention claim is partly built in: the diff-token pooling bias starts at 1.0 and only reached ≈1.02 | Section 3.3 of the notebook | Attention controls + new ablation (22.2 b, c) |
| 2 | Prefix truncation can delete the only difference between the two options | Cobbler item 1433, listed in section 21 | Difference-aware truncation (22.2 a) |
| 3 | RoBERTa (B3) ran with 1 seed while the from-scratch models ran with 3 | `final_results.csv` | B3 with 3 seeds (22.2 d) |
| 4 | Accuracy alone can't show whether DACT and TF-IDF + LR solve the same items | DACT 60.4 % vs B1 61.2 % on test | Error-overlap analysis (22.2 e) |
| 5 | Almost no references (only ESIM) | Notebook text | `references.bib` + References cell (22.2 f) |
| 6 | Empty template copy left in the repository | `Copy_of_CITS4012_YourGroupID.ipynb` | Removed (22.2 g) |
| – | No ACL report PDF yet | Repository | **Not done here**: the group still has to write the report |

### 22.2 Changes
**a) Difference-aware truncation** (`src/data.py`, `Config.diff_aware_truncation = True`)
- New function `truncate_around_diff`. When a solution is longer than `max_sol_len` (96), it keeps the 96-token window centred on the span where the two options differ, instead of the first 96 tokens.
- The difference tags are now computed on the **full** solutions before truncation. Before, they were computed on the truncated ones.
- Only over-long solutions are affected: about 1.6 % of test solutions, per section 5. Every other item encodes exactly as before.
- Setting the flag to `False` restores the original behaviour exactly.

**b) New ablation "diff-bias prior 0"** (`Config.tag_bias_init`, `src/model.py`)
- `AttentionPooling` now takes the initial value of the differing-token bias from the config. The default stays 1.0, so the main model is unchanged.
- The ablation starts the bias at 0. It tests whether DACT *learns* to focus on differing tokens or whether the focus comes only from the initial bias.
- It is appended after the existing ablations, so earlier run names stay stable. It runs as `abl7` when the selected config uses the MLM warm-up.

**c) Attention controls** (`diff_attention_controls` in `src/viz.py`, notebook Section 3.3)
- Measures the mean pooling mass on differing tokens under four conditions:
  - the trained model;
  - the trained model with the tag bias set to 0 (focus from learned content only);
  - an untrained, freshly initialised model (focus from the initial bias only);
  - uniform attention (reference).
- The sentence "So the model does base its decision on the tokens where the options disagree, as intended" was replaced. The new text explains the confound and points to the controls.

**d) RoBERTa baseline with 3 seeds** (notebook Step 4 and Step 5)
- B3 is fine-tuned with the same `SEEDS` as the from-scratch models.
- The result table reports its mean ± std on validation and test.
- The McNemar test and the CI use the best-**validation** seed, as for DACT.

**e) Error-overlap analysis** (`error_overlap` in `src/evaluate.py`; new notebook cell before the quantitative discussion)
- Compares DACT against B1, B2, the vanilla Transformer and B3.
- For each pair it reports the share of test items that both models, only one model, or neither model get right, plus their agreement and the oracle accuracy (correct if either model is correct).
- Saved to `outputs/results/error_overlap.csv`.

**f) References**
- New `references.bib` with 15 entries: PIQA, Transformer, pre-LN, ESIM, additive attention, BERT, RoBERTa, Qwen2.5, BPE, annotation artefacts, the two attention-as-explanation papers, McNemar, the bootstrap, and significance testing in NLP.
- A References cell was added at the end of the notebook.

**g) Housekeeping**
- `tools/sync_notebook.py` (new) copies `src/*.py` into the notebook's module cells. It drops `# nb-skip` lines and collapses the blank lines that removed imports leave behind. `--check` reports cells that are out of date. The original `tools/build_notebook.py` from section 11 was never committed.
- `README.md` now describes the repository layout and how to run it.
- `Copy_of_CITS4012_YourGroupID.ipynb` was removed. The official template is still available from the unit page.
- Notebook Readme updated with a *Revision 2* bullet, the new runtime estimate (≈60 min on an A100) and the sync-script note. The protocol cell and ablation table mention the new items.

### 22.3 Verification (local, CPU only; no GPU was available)
- Environment: Python 3.12 virtual environment in a temporary folder, PyTorch 2.14.0+cpu, current releases of numpy, scikit-learn, scipy, pandas, matplotlib and tokenizers.
- Data: a **synthetic** PIQA-format dataset (600 train, 120 test). Every 20th item had a long solution whose only difference was at BPE position > 96. The real data was not available locally. **None of these numbers are results.**
- `src/experiments.py` module smoke test: all paths ran, including the new `tag_bias_init=0` ablation, the BiLSTM and the separate test pass.
- The whole notebook ran top to bottom in `PIQA_SMOKE=1` mode. RoBERTa and Qwen were **stubbed** with random predictions, because `transformers` and the model downloads weren't available. Everything else was real code. The final table showed B3 with mean ± std, and the error-overlap and attention-control tables printed.
- Truncation check on the synthetic long items:
  - prefix truncation: 6 of 6 items had identical inputs for both options, and 0 differing tokens were kept;
  - difference-aware truncation: 0 of 6 identical, and 12 differing tokens kept.
  - Unit checks of the window (differing span in the middle, at the end, no difference, short input) passed.
- `python tools/sync_notebook.py --check` reports no out-of-date cells.

### 22.4 What the group still has to do
1. **Re-run the notebook on Colab** (`Runtime → Run all`, ≈60 min on an A100). The outputs of every changed code cell were cleared because they no longer matched the code. The saved outputs must come from one clean run of the current code.
2. **Update the numbers in the two discussion cells.** Each one is preceded by a ⚠️ note. Delete the notes afterwards. Things to check:
   - the new truncation fixes the cobbler item, so the "uncertain" example in Section 3.3 will change;
   - the ablation table gains a *diff-bias prior 0* row;
   - B3 gets a ± std;
   - interpret the controls: if *trained, tag bias = 0* is close to the uniform reference, the focus is built in rather than learned, and the text must say so.
3. **Commit and push** once the re-run has passed. Nothing from revision 2 has been committed or pushed yet.
4. **Write the ACL-format report** (≤ 6 pages) using `references.bib`, and make sure every number in it matches the re-run notebook.
5. **Declare AI assistance** as the unit's policy requires. Section 19 says AI co-author lines were deliberately left out of the commits. The brief requires the pipeline to be implemented by the team, so check with the unit coordinator how AI-assisted work must be acknowledged.

---

## 23. Revision 2b: council review and follow-up changes (2026-09-29, ≈22:30–23:15)

After revision 2 was pushed, Ali asked Claude Code to "check the work then improve it" with an **LLM council**:
- Five independent AI advisors (Contrarian, First Principles, Expansionist, Outsider, Executor) each assessed the repository.
- Five anonymous peer reviews ranked their answers.
- A chairman agent combined everything into a verdict.

All of these were Claude sub-agents, not people. This revision was also written with Claude Code.

### 23.1 Council verdict (summary)
- **Agreed:**
  - The GPU re-run is required.
  - Don't chase accuracy. The honest result (DACT below TF-IDF, focus built in by the bias initialisation) is the story.
  - The report must open by saying what is claimed and what is not.
  - The AI-authorship record is the biggest risk: 4 of 5 advisors and all 5 reviewers said so.
- **Resolved clashes:**
  - The integrity question comes first, ahead of technical work.
  - Add the cheap diagnostics, but no new model variants and no Qwen few-shot.
  - Write predicted outcomes down *before* the re-run, so the discussion can't be reshaped around the numbers.
- **Blind spots raised in peer review:**
  - Disclosing AI use may not satisfy "implemented by your team".
  - The group must be able to explain the code if questioned.
  - Revision 2 was made with the same AI tool.
  - There was no fallback plan if the GPU run fails.
  - The gap to TF-IDF needs a testable explanation, not "small dataset".
- **The chairman's first step:** the group reads the unit's AI-use policy and emails the coordinator before the re-run.

### 23.2 Changes made (the five items the council assigned to Claude Code)
**a) Crash-safe resume** (`src/experiments.py`)
- `run_experiment(..., resume=True)` loads an experiment that already finished in the same runtime, with the same config, the same seeds and its checkpoints on disk, instead of retraining it.
- If the VM kernel crashes, `Run all` continues where it stopped.
- A changed config always retrains.

**b) Seed counts in the results** (notebook Step 5)
- `final_results.csv` and the table gain an `n seeds` column and a `note` column ("single run: no seed std").
- Blank std/CI cells no longer look like missing data.

**c) What DACT adds beyond word statistics** (after the error-overlap table)
- Counts the test items that DACT solves and TF-IDF + LR does not, and prints three of them for the qualitative analysis.

**d) Printed key numbers**
- A new cell before the quantitative discussion prints every number that discussion quotes (%, seed counts, Δ vs DACT) straight from `RESULTS`, so nothing has to be retyped by hand.

**e) Diagnostic for the gap to bag-of-words** (Section 3.3, validation split only). It tests two explanations:
- **H1, the pooling does not select:** the normalised entropy of the pooling weights (1 = uniform), overall and for correct vs wrong predictions. Functions: `pooling_entropy` in `src/viz.py`.
- **H2, the difference signal fades or is never learned:** a logistic-regression probe for "differing vs shared" solution tokens at each layer, for DACT and the vanilla Transformer.
  - Saved to `outputs/results/diff_probe_by_layer.csv`.
  - Functions: `diff_probe_by_layer` and `_layer_states` in `src/viz.py`.
  - The text says to read each layer against the embedding row, because word identity alone already predicts some differing tokens.

**f) Consistency check** (new cell before the Drive copy)
- Asserts that `final_results.csv` matches the table.
- Asserts that every `*_val.json` matches its experiment.
- Asserts that every training run has a JSONL log.
- Runs `tools/sync_notebook.py --check` when the notebook runs from a repository checkout.

### 23.3 Verification (CPU, synthetic data as in 22.3; not results)
- The whole notebook ran top to bottom in smoke mode (59 cells), with RoBERTa and Qwen stubbed.
- The new outputs printed:
  - the table with `n seeds` and `note`;
  - the items only DACT solves;
  - the key-numbers table;
  - H1 entropy and the H2 probe table;
  - `OK: 13 result rows, 9 training runs with logs`.
- Resume test: the first call trained in 14.4 s, the identical second call loaded in 0.01 s, and a changed config retrained.
- On the synthetic data the attention controls behaved as designed: *trained* 0.339 ≈ *untrained* 0.337, and *bias = 0* 0.163 ≈ *uniform* 0.165. So a one-epoch model's focus came entirely from the initial bias, which is exactly the confound the controls exist to reveal.
- `tools/sync_notebook.py --check`: in sync.

### 23.4 Still open: only the group can do these (in the order the council recommends)
1. **Read the unit's AI-use policy and email the coordinator in writing.** Disclose the full extent of AI use, including revisions 2 and 2b, and ask whether the work is still eligible. Don't delete this log or rewrite git history: that would turn a disclosure problem into misconduct.
2. **Each member works through the code** until they can explain DACT, the difference tags, the pooling bias and the truncation without notes. Rewrite parts yourselves where you can.
3. **Commit the predicted outcomes before the re-run.** For example: does *diff-bias prior 0* match the full model? Is *trained, tag bias = 0* near uniform? Does the vanilla probe rise above its embedding row?
4. **Re-run on Colab (A100).** Then fill the two discussion cells from the printed key numbers and delete the ⚠️ notes.
5. **Write the 6-page ACL report.** Open with what is claimed and what isn't, and put the AI-use statement where a marker will see it.

---

## 24. Re-run attempts with the Colab CLI, and changes for a browser run (2026-09-29 22:10 – 2026-09-30 11:30, AWST)

### 24.1 Setting up the Colab CLI on Windows (Ali, using Claude Code)
- Google's `google-colab-cli` 0.7.4 (the `googlecolab` GitHub org) was installed with `uv tool install`.
- It **doesn't run on native Windows**: it imports `termios`, which exists only on Unix. It was therefore installed inside **WSL Ubuntu**, together with the Google Cloud SDK 587.0.0 for sign-in.
- Sign-in uses Application Default Credentials. Ali ran `~/colab-login.sh` in WSL and signed in with their own account in the browser. Claude Code never saw or handled the credentials.
- The account has **0 compute units**:
  - `colab new --gpu A100` was rejected ("no quota or entitlement"), so nothing was charged.
  - A free **Tesla T4** session was created instead.
- Data on the VM:
  - the `revision-2` branch was cloned;
  - PIQA was fetched with `gdown` from the unit's shared Drive folder, with byte sizes identical to section 14.2;
  - `/content/data` was symlinked to the checkout, because each `colab exec` starts in `/content`.

### 24.2 Attempt 1 (session `group69`, commit `de3e3da`): VM lost after ≈35 min
- The run printed `autocast dtype: torch.bfloat16` on the T4. The log stopped at the start of `hp0`, and the session then vanished ("Session 'group69' not found"). No outputs were saved.
- **Diagnosis:** timing test on a throwaway T4 (`colab run`, released automatically), measuring DACT training steps:

| autocast | ms/step | s/epoch (114 steps) |
|---|---|---|
| fp16 | 108 | ≈12 |
| fp32 | 254 | ≈29 |
| bf16 | 377 | ≈43 |

- `torch.cuda.is_bf16_supported()` returns True on the T4 (compute capability 7.5) only through **emulation**. Native support is False.
- **Fix (commit `29fce76`):** `amp_dtype` now uses bf16 only with *native* support (`including_emulation=False`), and fp16 otherwise. A100/L4 behaviour is unchanged.

### 24.3 Attempt 2 (commit `29fce76`): healthy, but the VM was lost again after ≈40 min
- fp16 worked: 13.8–14.3 s per epoch. `hp0`–`hp3` ran with validation accuracy in line with section 14 (e.g. `hp3` fine-tuning peaked at 59.7 %).
- During `hp3` the session disappeared again, and `colab sessions` showed nothing.
- **Conclusion:** free-tier sessions created from the CLI (no browser tab attached) are reclaimed after roughly 35–40 minutes. The full run needs about 2.5–3 h on a T4, so it can't finish on that path. No results from these attempts are used anywhere.

### 24.4 Changes for a browser ("Run all") run (commit after `29fce76`)
The group member chose to run the notebook in the Colab web UI on a free T4. To make that a one-click run:
- **Automatic data download** (`download_piqa` in `src/data.py`, `Config.data_folder_url`):
  - On Colab, if the four files aren't present, they are downloaded from the unit's shared Drive folder with `gdown` (installed only if missing), then moved into `data/piqa/`, and the temporary folder is removed.
  - Any failure (package, network, Drive quota) falls back to the original `MyDrive/Group69/PIQA.zip` path.
  - Tested locally with the real folder: the four files have the exact byte sizes above, the split is 14,501 / 1,612 / 1,838, and the temporary folder is cleaned up.
- **Results downloaded to the browser:** the last cell also packs `outputs/` (logs, results, figures; no checkpoints) into `outputs.zip` and calls `google.colab.files.download`. It still copies to Drive when Drive is mounted.
- The notebook Readme's "How to run" was rewritten to match: GPU choice (bf16/fp16), automatic data, the zip fallback, and `outputs.zip`.
- The notebook smoke test (section 23.3 setup) still passes end to end.

### 24.5 Next step (group)
1. Open the notebook from the `revision-2` branch in Colab, choose `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`. Keep the tab open; it takes about 2.5–3 h.
2. When it finishes, save the executed notebook (`File → Download → .ipynb`) and keep the downloaded `outputs.zip`.
3. Claude Code can then merge both into the repository and update the discussion cells from the printed key numbers.

---

## 25. Full re-run on a free T4 and updated results (2026-09-30, ≈17:00–19:30 AWST)

### 25.1 How the run was made (Ali, using Claude Code with the authorisation "to do all actions to finish the task")
- The files the group member reported downloading from their own browser run were not on this computer. Neither `CITS4012_69*.ipynb` nor `outputs*.zip` was found on drives C:, D: or E:, so that run's outputs could not be used.
- A new free **T4** session (`group69`) was created with the CLI.
- The VM was set up with `setup_vm.py`: clone of `revision-2` at **`5ef7cda`**, PIQA fetched with gdown, and the `/content/data` symlink.
- The notebook was executed with `colab exec -f CITS4012_69.ipynb` from the committed local copy, which was byte-identical to `5ef7cda`.
- **Keeping the VM alive:** the session's `colab url` was opened in the group member's **Brave** browser, where they were already signed in. With a browser tab attached, the VM was *not* reclaimed at 35–40 minutes. That confirms the diagnosis in 24.3.
- **Local record lost:** after about 60 minutes a periodic `colab ls` check coincided with an access-token refresh. The CLI pruned its *local* record of the session (`sessions.json` became `{}`), although the VM kept running: `colab sessions` still listed it as `[?]`, and output kept streaming.
  - The record was rebuilt with a 15-line script (`adopt.py`) that calls the CLI's own `State`, `list_assignments()` and `_apply_proxy_info`. Tokens were handled inside the library and never printed.
  - After that, `colab ls` and `download` worked again.
  - Later monitoring watched only the local output stream.
- **Result:** the run finished with `EXIT 0`. The executed notebook has **0 error cells**, and the consistency check printed `OK: 13 result rows, 27 training runs with logs, 37 log files`.
- Training took 09:30–10:48 UTC (17:30–18:48 AWST). Including data preparation, RoBERTa and Qwen, the whole notebook took about 1 h 40 min.
- `outputs.zip` (0.9 MB: 37 logs, result JSON/CSV files, 6 figures, tokenizer) was downloaded. **`colab stop -s group69`** was run afterwards, and `colab sessions` then showed no active session. No compute units were used (balance 0.00 throughout).

### 25.2 Files replaced
- `CITS4012_69.ipynb` is now the executed notebook from this run: all code cells are identical to `5ef7cda` and have outputs.
- `outputs/` was replaced completely by this run's outputs. The old A100 logs, results and figures (e.g. `confident_wrong_660/708.png`, `uncertain_1433.png`) were removed and remain in git history.
- A partial notebook from attempt 2 (`CITS4012_69_output.ipynb`, stopped at `hp3`, exit 1) had been written into the repository. It was moved to a scratch folder and is not used.

### 25.3 Results (all numbers printed by the notebook)
- **Grid (Step 1).** *small* (d = 128, 2 layers, 1.67 M parameters) and *base + MLM* tied at 60.30 % validation. The first maximum, *small* without warm-up, was selected. The A100 run had selected *base + MLM*: grid differences of about 1 point are within seed noise. With no warm-up in the selected config, the *− MLM* ablation does not apply. **Correction:** during the run, an interim status message called `abl6` the "no MLM warm-up" ablation. In this run `abl6` is *diff-bias prior 0*, and `abl5` is the vanilla Transformer.
- **Main table (val / test, 3 seeds):**

| Model | Val | Test |
|---|---|---|
| DACT | 59.4 ± 0.9 | 60.2 ± 0.8 |
| Vanilla | 58.9 ± 0.2 | 58.7 ± 0.9 |
| BiLSTM | 59.6 ± 0.8 | 59.0 ± 0.5 |
| TF-IDF + LR | 59.2 | 61.2 |
| RoBERTa | 61.5 ± 6.8 | 59.9 ± 8.0 |
| Qwen zero-shot | 78.2 | 75.1 |

- **RoBERTa seeds.** Validation accuracy was 64.4 / **53.7** / 66.3 %. Seed **43** was a degenerate fine-tuning run: its loss stayed at ≈ 0.694 for all 4 epochs. An interim status message wrongly named seed 42 as the degenerate one; the printed per-seed dictionary shows it was 43. The representative run (seed 44) has a test CI of 62.3–66.5 %.
- **Ablations:** all single-component changes are within ±0.5 on validation. On test they range from 0 to +1.2. *diff-bias prior 0* is the same as *− diff bias*. Only the vanilla model is lower on both splits (−0.5 / −1.5, McNemar p = 0.054).
- **Error overlap DACT vs TF-IDF:** 73.5 % agreement; 13.2 % of items only DACT gets right, 13.3 % only TF-IDF; oracle 74.4 %. 243 test items are solved only by DACT.
- **Attention controls:**
  - pooling mass on differing tokens: trained **80.9 %**, trained with the bias set to 0 **67.2 %**, untrained 38.9 %, uniform 24.8 %;
  - the bias moved 1.0 → 1.025, and the shared-token bias 0 → −0.025;
  - so **the focus is mostly learned in this run**. The worry from the review, that the focus was built in, is not supported.
  - Wrong predictions put *more* attention on differing tokens: 81.9 % vs 80.2 % (p = 0.007).
- **H1** is rejected: pooling entropy is 0.55. **H2:** the vanilla probe scores 58.4 % at the embedding layer and at most 61.1 % after the encoder, while DACT scores 100 % at every layer.
- **Qualitative cases:**
  - successes: *baby wipes* (804, 280);
  - failures: *clockwise/counterclockwise* (551; BPE splits the decisive word) and *peppermint spray vs shade tree* (154; no shared words, so the tags carry no information);
  - uncertain: *outside vs inside glass* (1458, p = 0.50; no goal–solution overlap term);
  - the cobbler item is not among the two most uncertain items.

### 25.4 Text changes
- The two ⚠️ "update after re-run" notes were removed. The quantitative and qualitative discussion cells were rewritten from the printed numbers and the saved figures.
- The Readme cell now describes this run and its runtime.
- `README.md` has the new run steps.
- `references.bib` and the References cell now include Dodge et al. (2020), cited for the RoBERTa seed instability.

### 25.5 Still for the group
- The AI-use disclosure and the coordinator email (section 23.4), and each member understanding the code.
- The 6-page ACL report. Every number must come from this run (sections 25.3 and the notebook), not from the first A100 run.
- Merge `revision-2` into `main` when the group agrees.

---

## 26. Lexical head: closing the gap to TF-IDF (2026-09-30, ≈19:00–22:00 AWST)

### 26.1 Motivation and rule
- After the run in section 25, DACT was 1 point below TF-IDF + LR on test (60.2 vs 61.2 %). The group member asked to fix this.
- **Rule followed throughout:** the brief forbids using the test set for design decisions. So the fix was chosen on **validation only**, and the final test result would be reported whatever it turned out to be.
- **Idea:** the validation evidence showed that word-level cues carry most of the learnable signal (TF-IDF 59.2 vs DACT 59.4 % val), and that the Transformer uses those cues only indirectly. So a **lexical ("wide") head** was added, as in wide & deep models (Cheng et al., 2016): one learned weight per hashed solution BPE unigram and bigram (2^18 buckets, starting at 0, with weight decay), added to each option's score and trained jointly from scratch. Under the pairwise softmax only the difference of the two lexical scores matters.

### 26.2 Implementation (`src/model.py`, `src/train.py`, `src/config.py`)
- `DACT.lexical_score` hashes the unigrams and bigrams inside the solution span (bucket 0 = padding, always 0). It uses int64-safe multipliers.
- `Config.use_lexical`, `lex_buckets = 2**18` and `lex_lr = 1e-2`. `make_optimizer` puts the head in its own AdamW group with learning rate `lex_lr`.
- Local checks passed: sparse weights learn, the padding weight stays at 0, swapping the options still swaps the scores, and the head is weight-decayed.

### 26.3 Validation pilots (`tools/pilot_lexical.py`; free T4 via `colab run`, VMs released automatically; the test split was deleted from memory before training)
Setup: the small configuration (d = 128, 2 layers, no warm-up), 3 seeds, validation accuracy.

| Variant | Val (3 seeds) |
|---|---|
| no head | 59.37 ± 0.81 |
| head, lr 3e-4 (shared) | 59.47 ± 0.90 |
| head, lr 3e-4, weight decay 0.01 | 59.45 ± 0.88 |
| head, lr 3e-3 | 60.48 ± 0.72 |
| **head, lr 1e-2** | **60.94 ± 0.56** |
| head, lr 3e-2 | 60.63 ± 0.34 |

- With the shared learning rate the head barely moved from zero before early stopping. That is why a separate `lex_lr` was added after pilot 1.
- **Decision:** `use_lexical=True`, `lex_lr=1e-2`.
- A new ablation, *− lexical head*, measures the head's contribution. The *vanilla* ablation now also removes the head.
- Both pilots ended with a CLI `TimeoutError` while tearing the VM down, after all results had printed. `colab sessions` confirmed no VM was left running.

### 26.4 Full run (commit `538cf5b`, free T4 attached in Brave, 11:50–13:15 UTC training; ≈1 h 50 min total)
- `EXIT 0`, 0 error cells. The consistency check printed `OK: 14 result rows, 30 training runs with logs, 40 log files`.
- The session was stopped afterwards, and no compute units were used.
- **Grid:** *small* was selected (61.4 %), ahead of *base* 61.2 %, *small + MLM* 60.9 %, *base + dropout 0.3* 60.8 % and *base + MLM* 60.8 %.
- **Main results (val / test):**

| Model | Val | Test |
|---|---|---|
| **DACT** | **61.1 ± 0.8** | **61.6 ± 0.7** |
| − lexical head | 59.2 ± 0.9 | 60.7 ± 0.4 |
| vanilla | 58.7 ± 0.3 | 58.6 ± 0.8 |
| BiLSTM | 59.8 ± 0.3 | 58.7 ± 0.9 |
| TF-IDF + LR | 59.2 | 61.2 |
| RoBERTa | 69.0 ± 0.5 | 67.9 ± 0.8 (all 3 seeds trained) |
| Qwen | 78.2 | 75.1 |

- **Outcome:** DACT is now above TF-IDF on both splits (+1.9 val, +0.4 test). The test gap is **not significant** (McNemar p = 0.88), so the report must say "at least as good as, and above on both splits", not "clearly better".
- **Ablations (Δ val / Δ test):**

| Ablation | Δ val | Δ test |
|---|---|---|
| − tags | −1.4 | −1.3 |
| − lexical head | −1.9 | −0.9 |
| pointwise | −1.0 | −0.8 |
| mean pooling | −0.2 | −0.9 |
| − cross-attention | −0.1 | −0.7 |
| − diff bias | +0.3 | 0.0 |
| diff-bias prior 0 | +0.3 | 0.0 |
| vanilla | −2.4 | −3.0 |

- **Attention:**
  - pooling mass on differing tokens: trained 84.9 %, trained with bias 0 **73.2 %**, untrained 38.9 %, uniform 24.8 %; the bias moved 1.0 → 1.022;
  - wrong predictions put more attention on differing tokens than correct ones: 86.7 vs 83.8 % (p < 0.001);
  - H1: entropy 0.51;
  - H2: the vanilla probe scores 59.6 → 60.4 %.
- **Error overlap with TF-IDF:** 78.8 % agreement, oracle 71.9 %, 197 items solved only by DACT.
- **Qualitative:** all four most-confident cases (p ≥ 0.99) are option pairs that differ in most of their words. This is reported as an over-confidence side effect of the summed lexical score:
  - successes: camping lantern (993), frying oil (1123);
  - failures: lotion bars (747), exotic trip (841);
  - uncertain: *pry pallet apart* (1258, p = 0.50; goal-word overlap not exploited).

### 26.5 Files
- `CITS4012_69.ipynb` is the executed notebook from this run (its code is identical to `538cf5b`). The discussion cells and the Readme were rewritten from its printed numbers and figures.
- `outputs/` was replaced by this run's outputs.
- Added `tools/pilot_lexical.py`, and Cheng et al. (2016) in `references.bib` and the References cell.

### 26.6 Still for the group
- Unchanged from 23.4 and 25.5: the AI-use disclosure and the coordinator email, understanding the code, and the ACL report. The report must use **this** run's numbers.

---

## 27. Report draft (2026-10-01, Ali, using Claude Code)

Ali asked for the project report "with the proof of works", including what failed as well as what worked. Claude Code wrote a complete **draft**, which the group must review, correct and complete.

### 27.1 Files (`report/`)
- `CITS4012_69.tex`: the report, using the official ACL style with `\usepackage[final]{acl}` and `\author{Group 69}`.
- `CITS4012_69.pdf`: the compiled report, 6 pages including the references.
- `acl.sty` and `acl_natbib.bst`: downloaded from github.com/acl-org/acl-style-files.
- `references.bib`: a copy of the repository's bibliography.
- `make_figures.py` builds `figures/` from `outputs/`:
  - the architecture diagram;
  - the attention-control bar chart;
  - pooling strips cropped from the saved figures for test items 747 and 1258.
- Compiled with Tectonic 0.17.0, installed in WSL. There are no LaTeX errors and no undefined citations or references.

### 27.2 Content
- **Sections:** Introduction; Methodology (equations for the tag embedding, cross-solution attention with ESIM fusion, difference-guided pooling, the lexical head and the objective; own contributions separated from adapted components); Experimental Setup (data, training and hyper-parameters, baselines B0–B4, metrics); Results and Analysis (main table, ablation table, error overlap, attention controls, H1/H2, qualitative cases); **What Worked and What Did Not**; Conclusion with Limitations; Team Contributions; Use of AI tools; References.
- **Numbers:** all taken from the final run (commit `538cf5b`, `outputs/results/*`, printed notebook outputs). Earlier runs appear only in the development-history section.
- **"What Worked and What Did Not" covers:**
  - run 1: DACT below TF-IDF, noisy ablations, the attention confound, the truncation defect;
  - the fixes that worked: difference-aware truncation, attention controls, 3 RoBERTa seeds (and one degenerate seed in run 2);
  - the engineering failures: bf16 emulation on the T4, free-tier VMs reclaimed;
  - the lexical-head pilots, including the failed shared-learning-rate version;
  - a **protocol note:** the decision to explore a lexical component was made after test results had been seen, so the small test gap to TF-IDF is treated as indicative.
- **"Proof of work":** the evidence paragraph (executed notebook, 30 training runs, 40 JSON-lines logs, the consistency-check cell, this log) and the three-run history.
- **Fact-check before compiling:**
  - "four full runs" was corrected to "three complete runs (and two aborted ones)";
  - the *baby wipes* success was labelled as coming from an earlier run, and the final run's own success (camping lantern) was added;
  - the truncation claim now says it was checked on synthetic data.
- **Notebook:** the protocol wording in the quantitative discussion was changed to match the report (the head was "selected on validation only", with the protocol note). This is a markdown-only change, so no outputs changed.
- **References:** Loshchilov & Hutter (2019) (AdamW) was added to `references.bib`.

### 27.3 Placeholders the group must fill in
- **Team Contributions:** each member's actual contribution.
- **Use of AI tools:** the draft statement says Claude Code was used extensively, including for drafting the report. The group must confirm and complete it according to the unit's AI-use policy.

### 27.4 Before submission
- Read the whole report, and rewrite it in your own words where the unit requires.
- Check every number against the notebook.
- Check the AI-use disclosure with the unit coordinator.
- Submit `CITS4012_69.pdf` and `CITS4012_69.ipynb`, as the brief requires.

### 27.5 Change on 2026-10-01
- At the group member's request, the "Use of AI tools" paragraph was removed from the report (`report/CITS4012_69.tex`, recompiled PDF). The record of AI assistance in this log is unchanged.

---

# Parallel track on `main`: evidence audit and replication (Salah Elshafey, 30 September 2026)

The sections below were written on `main` in parallel with sections 22–27 above (branch `revision-2`). They were merged unchanged on 2026-10-01; only their section numbers were changed from 22–24 to S1–S3 to avoid clashing with sections 22–24 above. They describe the **original** A100 model and its local replication (see section 28 for how the two tracks fit together).

## S1. Evidence audit and refinement — 30 September 2026

This section records a new audit and local replication. Sections 1–21 describe the earlier A100 work;
the corrections here supersede their interpretations where noted. No original A100 result file was
overwritten. No remote commit or push was made in this audit. The user explicitly requested continued
refinement and maintenance of this log. The official six-page project brief and actual supplied files
were reviewed before changing implementation. `PROJECT_AUDIT.md` records the pre-change inventory,
requirements comparison, findings and priorities. `evidence/historical/` preserves the original notebook,
eight source modules and builder; its SHA-256 manifest records the original evidence.

### S1.1 Findings that affect report claims

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

### S1.2 Implemented improvements

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

### S1.3 Checks actually executed

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

### S1.4 Completed local fixed-recipe replication

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

### S1.5 Submission status

The implementation and report evidence are substantially stronger. The complete historical study,
local replication and development tests have distinct provenance. Full revised B3/B4 execution and
the Drive/clean-Colab route remain unverified. No final ACL report PDF or invented team contribution
statement has been created. The required final filenames remain `CITS4012_69.ipynb` and
`CITS4012_69.pdf`; report notes are supporting evidence, not the finished six-page report.

## S2. Completed replication analysis and final handover — 30 September 2026

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

## S3. Project folder organisation — 30 September 2026

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

---

## 28. Merge of the two tracks into `merge-review` (2026-10-01, Ali, using Claude Code)

The group member asked to merge `revision-2` into `main` "after checking and confirming that it's better". The check found that `main` had moved on: Salah Elshafey's commit `e08e80c` ("updates", 30 Sep 23:53, 90 files) added an evidence audit of the original A100 run, robustness and provenance code, a local replication of the original model, and report notes. The two branches had diverged in direction, not only in content.

### 28.1 Comparison
| | `revision-2` | `main` (`e08e80c`) |
|---|---|---|
| DACT val / test | **61.1 / 61.6 %** (lexical head, free T4) | 60.1 / 60.4 % (original A100 run) |
| vs TF-IDF (61.2 % test) | above on both splits (not significant on test) | below on test |
| Unique work | truncation fix, attention controls, diagnostics, 3 RoBERTa seeds, lexical head, report PDF | audit (duplicate test item, directional alignment), determinism and validation code, prediction export, local replication, `report_support/` |

**Conclusion:** `revision-2` has the better model and results, and `main` has real work that `revision-2` lacks. Pushing either branch over the other would lose work. So the group member chose a **combined merge on a review branch**, with `main` left untouched until the group (including Salah) agrees.

### 28.2 Conflict resolution (5 conflicts)
- `src/model.py`: kept `lexical_score` (revision-2) and the `mlm_logits(..., selected=None)` signature and body (main).
- `src/data.py`: kept difference-aware truncation (revision-2) and passed `cfg.symmetric_diff_tags` (main) to both `diff_tags` calls.
- Auto-merged code was reviewed. One real interaction was fixed: `run_experiment` on `main` now stores `run_seed` per seed in the saved config, which would have stopped the crash-resume check from ever matching. `_load_finished` now ignores `run_seed`. Main's evidence guards (never overwrite logs, checkpoints, tokenizer or predictions) were kept.
- `CITS4012_69.ipynb`: kept the executed final notebook (revision-2), and synced its module cells to the merged `src/` (definition cells only, so no outputs changed). Main's version, the original A100 notebook with the audit narrative and audit cells, is preserved as `evidence/historical/CITS4012_69_audited_A100.ipynb`.
- `outputs/`: keeps the final run. Main's README described `outputs/` as the unchanged original A100 evidence, so those files were copied from `e08e80c` to `evidence/historical/a100_outputs/`, renamed from `outputs_a100` because `outputs_*/` is git-ignored on `main`.
- Course template: kept main's move to `templates/`.
- `README.md`: rewritten to describe both tracks. Main's statements that stopped being true (`outputs/` and the top-level notebook as the historical A100 evidence) now point to `evidence/historical/`, and the audit findings are listed.
- `docs/DEVELOPMENT_LOG.md`: both tracks kept in full. Main's sections 22–24 are appended unchanged under "Parallel track on `main`", renumbered S1–S3 to avoid clashing with sections 22–24 of revision-2.

### 28.3 Checks
- `tools/sync_notebook.py --check`: in sync.
- Smoke run of the merged notebook (CPU, synthetic data, stubbed B3/B4, as in 22.3): every cell ran. The consistency check printed `OK: 15 result rows, 11 training runs with logs, 27 log files`, and predictions were written to `outputs_nbsmoke/predictions/`.
- **Not re-run:** main's tools (`audit_evidence.py`, `verify_model.py`, `final_checks.py`, the reproducible notebook in `notebooks/`). They were written for the original model and need the data at `../../PIQA`.

### 28.4 Open items before `merge-review` goes into `main`
1. The group, including Salah, agrees on the submitted model and story (the final model with the lexical head).
2. **One fresh Colab run of the merged notebook.** Its saved outputs come from the revision-2 code, and the merged code adds records and checks. The defaults don't change the model, but the submitted outputs should come from exactly the submitted code.
3. Add the audit's duplicate-test-item finding and the directional-alignment note to the report's limitations.
4. Re-run or update main's audit tools for the new default model, if the group keeps them.

---

## 29. Report updated with the audit findings (2026-10-01, on `merge-review`)

At the group member's request, the report (`report/CITS4012_69.tex`, recompiled PDF) was updated with the findings of the audit track (section S1, `docs/PROJECT_AUDIT.md`).

**Report changes:**
- **Figure 1 caption:** equivariance is now qualified. The network is equivariant for already-encoded inputs, but the full pipeline is not, because the `difflib` alignment depends on option order.
- **Data paragraph:**
  - Test labels are used only in the final step. However, the test file is loaded and tokenised at the start, and B1/B3 compute their test predictions early without inspecting them. This replaces "used only in the final evaluation".
  - Disclosed: one test item also appears in the training file, and there are six duplicate training rows. They were kept so all runs use the same split.
- **New paragraph, "Independent evidence audit",** in "What Worked and What Did Not":
  - the overlap and duplicates;
  - the order-dependent alignment (103 test items change tags when the raw options are swapped, in the original pipeline);
  - the early test loading;
  - the lost checkpoints and predictions of the original run;
  - the code the audit added (opt-in canonical alignment, prediction export, provenance, validation loss, finite-loss checks);
  - its replication of the original model, kept separate from the report's tables.
- **Evidence paragraph:** now mentions `evidence/historical/`.
- **Limitations:** now also list the overlap, the order-dependent tags, and the changed parameter counts in module-removal ablations.

**Notebook:** in the protocol cell (3.1), the false claim "Steps 1-4 never touch the test split" was replaced with the accurate description and the duplicate disclosure. This is a markdown change only.

**Checks:**
- The PDF is 6 pages with no errors or undefined references. The main text ends on page 5; Team Contributions and the references start on page 6.
- `sync_notebook --check`: in sync.
- All numbers are unchanged, taken from the final run.

---

## 30. Review note for Salah; replication script fixed (2026-10-01, on `merge-review`)

**The note:** `docs/NOTE_FOR_SALAH.md`, written at the group member's request so that Salah can review `merge-review` and decide whether to merge. It explains:
- what `revision-2` changed;
- what happened to his work;
- which of his tools the merge may affect;
- what remains before submission;
- options A/B/C;
- that AI assistance was used and that the report currently has no AI-use statement.

**Problem found while writing it:** `tools/reproduce_main.py` (audit track) read the "historical" config and tokenizer from `outputs/`. After the merge, `outputs/` holds the final lexical-head run. The saved A100 config also has no `use_lexical` or `diff_aware_truncation` field, so the new defaults (both on) would have applied. Together, these would have made the "replication of the original model" silently train the new model.

**Fix:** the script now reads `evidence/historical/a100_outputs/` (config, tokenizer and its checksum) and pins `use_lexical=False, diff_aware_truncation=False` before applying the saved config.
- It compiles. The replication was not re-run, because the data is not at `../../PIQA` here.

**Left for Salah to decide, and listed in the note:** `tools/audit_evidence.py`, `tools/verify_model.py` and `tools/append_audit_cells.py` still read `outputs/` as the A100 evidence. `notebooks/CITS4012_69_reproducible.ipynb` is also out of date after the merge.

---

## 31. Submission files prepared from `merge-review` (2026-10-01)

The group member chose the combined `merge-review` version for submission, with **files only**: `main` was not changed, pending Salah's agreement. The upload to the LMS is done by the group, and only one member submits.

**Changes:**
- **Report, Team Contributions:** Ali and Salah are filled in from the repository record (this log; Salah's commit `e08e80c`). Bayron's entry is a marked placeholder, `[to be completed by Bayron]`, because his contribution cannot be determined from the repository.
- **Notebook Readme cell:** a new bullet, "Code version of the saved outputs". It states that:
  - the saved outputs come from commit `538cf5b`;
  - the code cells include the audit track's additions, which are off or record-only by default and do not change the model;
  - the merged re-run was stopped by the free Colab GPU limit, so the outputs were not regenerated.

**Files:** the folder `CITS4012_69_submission` next to the repository (in `D:/Alarms/semester 2/NLP/`) contains exactly the two files the brief requires:
- `CITS4012_69.ipynb`: 57 cells, 23 of 32 code cells with outputs, 0 error cells.
- `CITS4012_69.pdf`: 6 pages; the main text ends on page 5.

**Before uploading:**
1. Bayron's contribution must be written in.
2. The group's AI-use decision.
3. Salah's agreement to the combined version.

### 31.1 Change on 2026-10-01
- At the group member's request, Team Contributions was replaced with a group statement: the work was divided among Ali, Salah and Bayron; each member implemented and tested his own part of the code and wrote the report sections describing his own work. The PDF was recompiled and copied to the submission folder.

---

## 32. Report polish for marks (2026-10-03, branch `report-polish` from `main` `1816b58`)

The group member asked to fix the three weaknesses from the 82/100 evaluation. All three changes are in the report only; the notebook and outputs are unchanged.

1. **Team Contributions are now specific.** The group statement is kept (work divided; each member implemented and tested his own part and wrote the report sections about it), followed by one line per member taken from the git history. The group member confirmed that the account SEKY443 is Bayron.
   - **Bayron:** original pipeline and baselines, original A100 run, final A100 runs and comparison, integration.
   - **Ali:** model revision, lexical head and pilot, attention controls and diagnostics, three RoBERTa seeds, T4 runs, first report draft.
   - **Salah:** evidence audit, robustness and provenance code, replication.
2. **New "Related work" paragraph** in the Introduction. It covers:
   - PIQA's pretrained and human reference points from Bisk et al. (RoBERTa-large 77.1 %, humans 94.9 %);
   - UnifiedQA;
   - option-comparison models for multiple choice (Option Comparison Network, DCMN+) and text alignment (ESIM, decomposable attention), and how DACT adapts them;
   - annotation artefacts and hypothesis-only baselines, linked to the TF-IDF baseline and the lexical head.

   Five references were added to both `references.bib` files: `khashabi2020unifiedqa`, `ran2019option`, `zhang2020dcmn`, `parikh2016decomposable`, `poliak2018hypothesis`.
3. **Figure 3 redrawn as highlighted text** (`report/make_figures.py` → `report/figures/attention_cases.pdf`). It is built from the attention data the notebook saved (`outputs/figures/confident_wrong_747.png.json`, `uncertain_1724.png.json`):
   - each solution token is shaded by its pooling weight, and differing tokens are bold;
   - text is white on dark shading;
   - this replaces the cropped strips with rotated, very small labels.

**Checks:**
- The PDF compiles with no errors or undefined references. It is 7 pages in total.
- The main text ends on page 6, within the limit; Team Contributions and the references follow.
- The new PDF was copied to `CITS4012_69_submission/`.
- `main` was not changed. The branch `report-polish` is pushed for the group to review and merge.

---

## 33. Honest final pass, steps 1–2: verification and documentation (2026-10-03, branch `honest-pass`)

This section follows the "honest final improvement pass" prompt (`IMPROVEMENT_PROMPT.md`, produced by an LLM-council review).

**Base.** Branch `honest-pass` starts from `report-polish` (`f6f6975`), which is `main` / `v1.0.0-rc2` (`1816b58`) plus the report fixes of section 32. The prompt says to start from `main`; `report-polish` was used instead so those fixes are not lost. `v1.0.0-rc2` remains the fallback.

**Rule followed in this section:** no training was done, and no test number was used for any decision.

### 33.1 Step 1: audit claims reproduced (`tools/verify_audit_claims.py` → `outputs/results/audit_checks.csv`)

- **Duplicates: confirmed.**
  - There are 6 duplicate rows within the training file.
  - One test item, **index 1545** ("How to make a vodka and soda?"), is identical to training-file rows **9975 and 11667**, with the same label. Both copies are in TRAIN, none in VAL.
  - No test item matches a training item with the options swapped.
- **Order-dependent difference tags: confirmed.**
  - **102** test items change tags when the two options are swapped with the current pipeline. The audit's 103 was for the original pipeline (prefix truncation).
  - With `symmetric_diff_tags=True`: 0.
  - No test item has identical encoded inputs.
- **Overfitting: confirmed** (final A100 run, `outputs/logs/dact_full_seed*.jsonl`).

  | Seed | Best epoch | Train acc at best epoch | Val acc at best epoch | Train acc, last epoch |
  |---|---|---|---|---|
  | 42 | 5 | 0.917 | 0.615 | 0.978 |
  | 43 | 4 | 0.890 | 0.614 | 0.976 |
  | 44 | 4 | 0.891 | 0.608 | 0.975 |

- **Test access** (`tools/check_test_access.py`):
  - `src/`: only `test_experiment` scores the test set, and the representative seed is chosen by validation.
  - Notebook, before the final-results section: cells 9, 33 and 34 load test data or compute held-out baseline predictions **without labels**.
  - **Cell 10 read test labels**, for the label share in the dataset-statistics table. This was descriptive and fed no decision, but it broke the rule.
  - **Guard added:**
    - cell 6 sets `FINAL_EVAL = False`;
    - cell 10 shows the test label share only if `FINAL_EVAL` is set;
    - cell 37 (Step 5) sets `FINAL_EVAL = True` and prints the test label share there;
    - `test_experiment` raises an error while `FINAL_EVAL` is False.
  - **Checks:**
    - the checker passes on the new notebook (exit code 0) and fails on the pre-guard notebook (exit code 1);
    - a unit check confirmed that `test_experiment` is blocked when `FINAL_EVAL` is False;
    - a smoke run of the whole notebook passed (CPU, synthetic data, stubbed B3/B4).
  - The saved outputs of cells 6, 10 and 37 still come from the code before the guard (see 33.3).

### 33.2 Re-scoring without the duplicate (`tools/rescore_without_duplicate.py` → `outputs/results/rescore_without_duplicate.csv`)

- This is evaluation only: the saved test predictions (`outputs/predictions/*_test.jsonl.gz`) were scored with item 1545 excluded. Nothing was retrained.
- **Effect:** about 0.02 points.
  - DACT: 61.57 → 61.55 %.
  - TF-IDF + LR: 61.21 → 61.19 %.
  - Every ablation moved by 0.004–0.021 points.
- **B1 and B3/B4:** B1's predictions were not saved, so it was recomputed. B3 and B4 predictions were not saved; one item changes their accuracy by at most 0.054 points.
- **Reproducibility finding:** B1 reproduces 61.21 % only with the Colab library versions (scikit-learn 1.6.1, scipy 1.16.3, numpy 2.1.3). With scipy 1.18 / numpy 2.5 the identical code gives **60.88 %**, because the optimiser behaves differently. Library versions must therefore be pinned (step 4).

### 33.3 Step 2: documentation

**Report:**
- **Abstract and findings:** DACT is the best from-scratch model on validation; on test it is statistically tied with TF-IDF (p = 0.88).
- **New paragraph "Validity of the test numbers":** validation is the only split behind design decisions. The test set was evaluated in several development runs, and the lexical head was introduced after earlier test results had been seen.
- **Overall comparison:** opens with the tie.
- **Data paragraph:** names the guard and duplicate item 1545, with the re-scored numbers.
- **Audit paragraph:** 103 order-dependent items (original pipeline) / 102 (current).
- **Limitations:** the order dependence (102, and 0 with the canonical alignment) and overfitting (89–92 % train vs about 61 % val at the selected epoch; 97–98 % by the last epoch).
- **Checks:** the PDF compiles with no errors or undefined references. It is 7 pages; the main text ends on page 6.

**Notebook:**
- A validity and audit note at the top of the quantitative discussion (markdown only).

**Not done:**
- **AI-use statement:** not added. The group member had earlier asked for it to be removed from the report (section 27.5). The prompt says not to write it on the group's behalf, so this decision stays with the group, and it is listed in the summary.
- **No re-run yet:** the code cells changed by the guard (6, 10, 37 and the `experiments` module cell) keep the outputs of the A100 run of commit `0d70cdc`. Their saved outputs differ from what the new code would print only in the dataset-statistics table (the test label share) and in one extra printed line in Step 5.

---

## 34. Honest final pass, steps 3–5: pre-registered experiment, reproducibility, proof (2026-10-03, branch `honest-pass`)

### 34.1 Step 3: goal-to-solution matching, pre-registered and rejected

**Pre-registration:** `experiments/goal_matching/PREREGISTRATION.md`, committed in **`2931eae` at 2026-10-03T07:03:57Z**, before any training.
- **The change:** a `GoalMatching` block with 165,504 parameters. Every token attends to the goal segment of its own option, followed by a gated fusion.
- **Fixed configuration:** the main model's configuration with the block added; no tuning of the block.
- **Rule:** a 3-seed validation gain of at least 1.0 point AND more than twice the pooled seed std.
- **Deadline:** 10 October.
- **Default:** the block is off (`Config.use_goal_matching = False`), so the main model is unchanged. Unit checks (forward passes only, no training) confirmed 1,927,492 parameters without the block, and swap symmetry with and without it.

**Run:**
- One free Colab T4 session (`goalexp`) cloned `honest-pass` at `2931eae` and ran `tools/goal_matching_experiment.py`.
- The first training log started at **07:05:54Z**, after the pre-registration commit.
- The test split was deleted from memory before training. **No test predictions exist:** the run folder holds only validation prediction files.
- The session was stopped afterwards.

**Result (validation accuracy, seeds 42/43/44; `experiments/goal_matching/results.csv`):**

| Arm | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
|---|---|---|---|---|
| Control (main model) | 61.23 | 62.28 | 60.24 | **61.25 ± 1.02** |
| Goal matching | 61.17 | 60.61 | 61.04 | **60.94 ± 0.29** |

- The gain was **−0.31 points**, against a pooled std of 0.75.
- **REJECTED.** As the protocol requires: no test run, DACT unchanged, and the negative result is reported with validation numbers only.
- The run's logs, validation predictions and decision file are in `experiments/goal_matching/run_20261003_070526/`. The checkpoints (47 MB) are git-ignored.

**Report:**
- A new paragraph, "A pre-registered change that did not help", in "What Worked and What Did Not".
- Limitations: "an explicit goal-matching block did not help in a pre-registered validation test" replaces the earlier suggestion of it as future work.

### 34.2 Step 4: reproducibility

- **Pins:** `requirements-a100.txt` already pins the A100 environment (torch 2.11.0, numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1, …). Section 33.2 shows these pins matter: B1 changes from 61.21 to 60.88 % with newer scipy/numpy.
- **Seeds:** the split seed is 42; training seeds are 42/43/44 for every trained model and for the experiment.
- **Code version of the saved outputs:** the notebook Readme previously said "produced by exactly the code shown (commit `0d70cdc`)". It now says the outputs were produced by `0d70cdc`, and lists the changes made since: the `FINAL_EVAL` guard, and the goal-matching block switched off. It states that neither changes the model or any result. The only saved output the current code would print differently is the test label share in the statistics table.
- **No re-run:** none was needed or made. The experiment was rejected, so the protocol allows no further test evaluation.

### 34.3 Step 5: proof tools

- **`tools/check_test_access.py`** passes (exit 0). It fails (exit 1) on the notebook as it was before the guard.
- **`tools/check_report_numbers.py`** (exit 0):
  - **Strict check:** every cell of Tables 1–2 and the headline p-value, matched row by row against `final_results.csv`. 35 values, **0 errors**.
  - **Traceability check:** all 126 distinct numbers in the main text, **0 mismatches**. 90 trace to logged result files and 36 are documented non-result numbers, each with a source.
  - **The traceability check is weak.** A random one-decimal number matches some logged value about half the time. So only the strict check is presented as verification.
  - **While building it:**
    - Three ablation deltas in Table 2 are differences of **rounded** means (e.g. 60.6 − 61.6 = −1.0, where the exact value is −0.94). This matches the notebook's key-numbers cell, so the numbers were kept, and the caption now states the convention.
    - The analysis numbers that the notebook prints but did not save are exported from the executed notebook's outputs to `outputs/results/analysis_numbers.csv` (`tools/export_analysis_numbers.py`). Nothing was recomputed.
    - The T4 timing results are recorded in `experiments/t4_timing/timing.csv`.
- **Notebook check:**
  - 57 cells, 32 of them code cells (23 with outputs); **0 error cells**.
  - The saved consistency check reads `OK: 14 result rows, 30 training runs with logs, 40 log files`.
  - A smoke run of the whole current notebook passed (CPU, synthetic data, stubbed B3/B4).
- **PDF:** 8 pages; **the main text ends on page 6** (Team Contributions starts on page 6).

### 34.4 Still for the group
- The AI-use statement (see 33.3).
- Merge `honest-pass` into `main` if the group agrees. `v1.0.0-rc2` stays the fallback.

---

## 35. Presentation pass to address "clearer claim" and "density" (2026-10-03, branch `polish-90`)

**Step 1.** At the group member's request, `main` was first fast-forwarded to `honest-pass` (`1816b58` → `8cc0ad1`) and pushed. `polish-90` starts from that commit.

The evaluation in this conversation (≈ 87/100) named two presentation weaknesses: a modest main result that the report did not frame as a clear answer, and very dense pages. No result changed: there was no new run, and the test set was not used.

### 35.1 Changes
**Abstract, rewritten around a research question:** can a from-scratch model learn *where* two solutions differ, and is that enough?
- **Yes to the first.** 86 % of pooling attention goes to the differing words, against 25 % uniform, and ablations support the difference tags.
- **No to the second.** DACT is best on validation, tied with TF-IDF on test (p = 0.88), and 5–14 points behind pretrained models. Wrong predictions attend more to the differing words: the bottleneck is knowledge, not focus.

**Introduction:** the findings use the same three-part answer.

**"What Worked and What Did Not":** about 700 words of prose replaced by:
- **Table 3, the development history.** One row each for run 1, run 2, the pilot, run 3, the final run and the pre-registered experiment, with DACT val / test accuracy and the outcome.
- **Three short paragraphs:** what worked, what did not, and protocol / audit / evidence.

The main text now ends at the top of the right column of page 6, about half a column shorter than before.

**Conclusion:** "the best from-scratch model" is now qualified with "on validation, although on test it only ties a lexical baseline".

**Layout:** `\usepackage{array}` added for the left-aligned table columns. There are no overfull boxes.

### 35.2 Traceability
- **Run 2's results** existed only in git history. They were restored to `experiments/t4_run_e5fdb29/outputs/results/final_results.csv` (`git show e5fdb29:…`).
- **The lexical-head pilot results** were recorded from section 26.3 to `experiments/lexical_pilot/pilot_results.csv`, per seed.
- **`tools/check_report_numbers.py`** now also checks **Table 3 row by row** against each run's own results file, plus the TF-IDF caption.
- **Result:** the strict check covers **49 values with 0 errors**, and the traceability check finds 0 mismatches (121 distinct numbers).
- The PDF compiles with no errors, undefined references or overfull boxes. It is 7 pages; the main text ends on page 6.

## 36. Upgrade 1: robustness to interruption (2026-10-03, branch `upgrade/robustness` from `main` `e52e087`)
Source: section 1 of the upgrade prompt and `FUTURE_FIXES_NOTES.md` (A1, A2). Free Colab VMs were reclaimed in 4 of 7 full attempts, and `resume` only worked inside one VM because `outputs/` is local. No result number changed; no test split was evaluated.

**Changes:**
- **`Config.persist_dir`** (`src/config.py`): a persistent run folder. The notebook sets it to `/content/drive/MyDrive/Group69/runs/<RUN_ID>` when Drive is *already* mounted (`drive.mount()` waits for a login and hangs CLI runs), otherwise `./Group69/runs/<RUN_ID>`. `RUN_ID` comes from `GROUP69_RUN_ID` (default `main`, or `smoke` in smoke mode).
- **Per-seed persistence** (`src/experiments.py`): after each seed, a completion marker `results/<run>_done.json` (run summary + config) is written and the seed's log, MLM log, validation predictions, checkpoint and sidecar are copied to `persist_dir`. After the last seed the experiment summary is copied too. Copies never overwrite an existing file.
- **Resume across VMs:** `run_experiment(resume=True)` first restores the experiment's files from `persist_dir`, then loads a finished experiment, or finished seeds of a half-finished one. Checkpoint paths are resolved against the current `out_dir`. Experiment identity ignores `run_seed`, `out_dir` and `persist_dir`.
- **Evidence is never overwritten:**
  - the files of an interrupted seed are renamed to `*.partial-<time>.*` (for example `logs/dact_seed43.partial-20261003_201500.jsonl`) before that seed restarts;
  - a finished seed or summary with a *different* config raises `FileExistsError` ("use a new out_dir / RUN_ID");
  - a summary with the same config that cannot be resumed (missing checkpoint) is kept as `*_val.stale-<time>.json`.
- **Heartbeat** (`write_progress` in `src/config.py`): `progress.json` in `out_dir` and `persist_dir` is replaced atomically at the start of every seed and after every epoch, with time, git commit (or `GROUP69_COMMIT`), step, experiment/run, seed, epoch, val_acc and best val_acc. Monitoring failures never stop training.
- **Notebook:** module cells re-synced with `tools/sync_notebook.py`; the config cell sets `persist_dir`. (On this branch only; the submitted notebook on `main` is unchanged.)

**Acceptance test** (`tests/test_resume.py`, synthetic PIQA-format data, CPU, deterministic, about 40 s):
1. an uninterrupted 2-seed run gives the reference table;
2. the same run is "reclaimed" after epoch 1 of seed 2: seed 1 is already in the persistent folder, seed 2 is not, and `progress.json` shows seed 2;
3. a restart on the same VM trains only seed 2, keeps its partial log (1 epoch) under a `.partial-` name, and gives a table identical to the reference (seeds, val acc, val loss, epochs, mean/std);
4. a new VM (empty `out_dir`) trains nothing and gives the identical table;
5. extra tests: a restart on a new VM after the interruption trains only the missing seed; persistence never overwrites; a different config is refused; `persist_dir` is not part of the identity.

**Notebook evidence** (`CITS4012_69.ipynb` of this branch, smoke mode, synthetic data, CPU, RoBERTa/Qwen stubbed by the external harness):
- run 1 (fresh): all code cells ran, exit 0, 293 s, 27 training epochs logged; the persistent folder `./Group69/runs/smoke` received every experiment;
- run 2 ("new VM": `outputs_nbsmoke/` moved away, persistent folder kept): exit 0, 44 s, 16 "already finished, loaded" messages, 0 training epochs;
- `final_results.csv`, `error_overlap.csv` and `diff_probe_by_layer.csv` of the two runs are byte-identical (`cmp`).

**Not covered:** a real Colab VM loss with Google Drive. The code path is the same (`persist_dir` is just a folder), but Drive sync latency was not tested.

## 37. Upgrade 2: reproducibility (2026-10-03, branch `upgrade/reproducibility` from `upgrade/robustness`)
Source: section 2 of the upgrade prompt and `FUTURE_FIXES_NOTES.md` (B7, B10, B11, E24). Every decision here used the validation split only; no test accuracy was computed.

**One lock file.** `requirements-lock.txt` holds the exact versions of the final A100 run. `requirements-a100.txt` now only includes it (`-r requirements-lock.txt`), so old references still work. The notebook's environment cell compares the running versions with the same list: a mismatch prints a loud warning, or raises when `GROUP69_STRICT_VERSIONS=1`. A test checks that the notebook list equals the lock file.

**B1 version sensitivity, validation only** (`tools/b1_version_check.py`, rows appended to `experiments/b1_versions/b1_versions.csv`, real PIQA, CPU):

| Environment | scikit-learn | scipy | numpy | C | B1 val acc |
|---|---|---|---|---|---|
| locked | 1.6.1 | 1.16.3 | 2.1.3 | 3.0 | **59.2432 %** (equals the A100 value, 59.24) |
| newer | 1.6.1 | 1.18.1 | 2.5.3 | 3.0 | **58.9950 %** |

- **The prompt asked for test numbers here** (61.21 vs 60.88 %); that conflicts with its own rule that test accuracy is never computed. The test values come from the earlier run in section 33.2 and were **not** recomputed; only validation was measured.
- **The "locked" environment is not the full A100 environment:** Python 3.12.10, CPU torch 2.14.0, tokenizers 0.23.2. B1 uses only scikit-learn, scipy and numpy, which match the lock.

**`run_config.json`** (`write_run_config` in `src/config.py`, called after data preparation): run id, time, git commit, full config, environment, the data manifest (file and split checksums, tokenizer hash) and any version mismatch. It is written to `out_dir` and the persistent run folder; a second run gets a time-stamped name.

**Validation predictions for every baseline.** `save_baseline_val` writes `predictions/B1_tfidf_lr_val.jsonl`, `B3_roberta_seed<seed>_val.jsonl` and `B4_qwen_zero_shot_val.jsonl` in DACT's format, with probabilities:
- B1: `predict_proba`;
- B3: softmax of the multiple-choice logits;
- B4: softmax of the two mean log-probabilities, documented as a score, not a calibrated probability.

Test predictions are unchanged: B1 and B3 still make held-out test predictions without scoring them (this avoids keeping 3 × 500 MB checkpoints), and B4 predicts test only in Step 5.

**RoBERTa restart rule** (declared before any run; NEW, not used for the reported A100 numbers):
- if validation accuracy after epoch 1 is below 0.52, the seed is abandoned and re-run with seed + 1000, at most twice;
- the abandoned log is kept and ends with an `"abandoned"` event;
- the notebook prints, per seed actually used, the mean, std and **median**, plus the list of restarts.
- *Why 0.52:* chance is 0.50 and one standard error on 1,612 items is 1.25 points.

**Evidence:**
- `tests/test_reproducibility.py` (6 tests): notebook list = lock file, run config, never-overwrite, baseline validation predictions, restart rule (restart, then give up after 2).
- Notebook smoke run of this branch (synthetic data, CPU, RoBERTa/Qwen stubbed): exit 0. The version warning fired as expected (CPU torch 2.14, pandas 3.0.6, …), `run_config.json` was written to both folders, and `B1_tfidf_lr_val.jsonl`, `B3_roberta_seed42_val.jsonl` and `B4_qwen_zero_shot_val.jsonl` were saved.

## 38. Upgrade 3: tests, CI, tools; three wrong numbers found in the report (2026-10-03, branch `upgrade/tests` from `upgrade/reproducibility`)
Source: section 3 of the upgrade prompt and `FUTURE_FIXES_NOTES.md` (C15, E21–E23).

**Test suite** (`pytest`, CPU, synthetic PIQA-format data, `pytest.ini`): **25 tests**.
- `test_core.py` (13 tests):
  - padding masks; padding does not change scores;
  - swap equivariance with symmetric alignment; order-free symmetric tags;
  - `truncate_around_diff`, including the cobbler case where the only difference lies past token 96 and prefix truncation would make both inputs identical;
  - the lexical head: bucket 0 gets no gradient and stays 0, its own learning rate, initialisation at 0;
  - the `FINAL_EVAL` guard;
  - `save_predictions` (format, probabilities, no overwrite, length check);
  - the duplicate re-scoring (`rescore()` was factored out of `tools/rescore_without_duplicate.py`; its output is unchanged).
- `test_resume.py` (5 tests) and `test_reproducibility.py` (6 tests): sections 36–37.
- `test_notebook_smoke.py` (1 test, marked `slow`):
  - executes every code cell of `CITS4012_69.ipynb` in smoke mode, with RoBERTa/Qwen stubbed;
  - checks the run config, the baseline predictions and `FINAL_EVAL`;
  - re-runs the notebook as on a new VM: nothing may be retrained, and `final_results.csv` must be byte-identical.
- *Run time:* 1 min 54 s on an idle machine; 3 min 21 s while other programs were using the CPU (the same single test took 7.5 s vs 14 s). `-m "not slow"` skips the notebook run.
- *A wrong assumption corrected by a test:* the first lexical-head test assumed bucket 0 is never read in the forward pass. In fact `padding_idx` only blocks the gradient. The property that actually holds, and is now tested, is that bucket 0 receives zero gradient and stays at its initial 0.

**CI** (`.github/workflows/ci.yml`, GitHub Actions, CPU): `sync_notebook.py --check`, `check_test_access.py`, `check_report_numbers.py`, then `pytest`, using the locked scientific stack.

**Tools:**
- **One notebook builder:** `tools/sync_notebook.py`.
- **Retired** (moved to `tools/retired/` with a README; nothing deleted):
  - `build_notebook.py` and its generated `notebooks/CITS4012_69_reproducible.ipynb`;
  - `final_checks.py` and `smoke_notebook.py`, which depended on that notebook;
  - `audit_evidence.py` and `verify_model.py`: written for the audit track's intermediate `src/`. With today's `src/` they would rebuild the lexical-head model from the original configs; with `evidence/historical/src/` they fail to import. Run in an isolated copy, both failed as expected.
  - `append_audit_cells.py`: it writes into the root `CITS4012_69.ipynb`, which is now the submission notebook.
- Their outputs remain in `report_support/` and `evidence/historical/`.

**Stronger number check.** `tools/check_report_numbers.py` has a second STRICT check: **42 prose claims**. Each claim is the exact LaTeX text a result must appear in, built from one named result-file cell (final results, re-scoring, analysis numbers, error overlap, probe, grid, parameter counts, goal-matching results, and the run-3 vs final spread). A coverage pass also reports any decimal number in the prose that is neither inside a claim nor a documented setting. One value has no archived result file: run 2's RoBERTa logs were never archived, so the 53.7 % at-chance seed is checked against `DEVELOPMENT_LOG.md` section 24.

**It found three wrong numbers in the submitted report and notebook.**
- **Where:** the Step 1 grid in the report's Training paragraph, and the "Selected configuration" bullet of the notebook's quantitative discussion, read base 61.0 / small 61.3 / dropout 0.3 60.9 %.
- **The logged values** (`outputs/results/hp*_val.json` and the notebook's own Step 1 output) are **61.2 / 61.5 / 60.7 %**. The two MLM values (60.9 / 61.0) were right.
- **Origin:** the numbers came from commit `15ae038`, whose result files already held the correct values, so this was a transcription error. The loose traceability check could not catch it, because each wrong value happens to equal some other logged number.
- **Fixed on this branch only** (report `.tex` and notebook markdown):
  - The selection is unchanged (small wins).
  - The corrected spread (0.8 points, not 0.4) contradicts the notebook's claim that it is "below one seed std". The sentence now says the spread is about twice DACT's seed std, so a one-seed grid cannot separate the candidates (cf. FUTURE_FIXES B8).
- **Main is unchanged.** `main` (`e52e087`) and its PDF still contain the old numbers; resubmitting is the group's decision.

**Check outputs on this branch:**
- `sync_notebook.py --check`: out of date: none;
- `check_test_access.py`: OK;
- `check_report_numbers.py`: STRICT tables 49 values / 0 errors, STRICT prose 42 claims / 0 errors, loose 0 mismatches.

## 39. Upgrade 4: evaluation hygiene (2026-10-03, branch `upgrade/eval-hygiene` from `upgrade/tests`)
Source: section 4 of the upgrade prompt and `FUTURE_FIXES_NOTES.md` (C12, C15, D17). No test accuracy was computed; the calibration study uses validation predictions only.

**4.1 Test labels only through `final_eval`.**
- `load_piqa` now returns the test split with a placeholder label (`HIDDEN_LABEL = 0`): the real labels are not in memory before Step 5.
- `experiments.final_eval(DATA, CFG, reason)` is the only notebook function that reads `test-labels.lst`. It loads the labels into the test rows and dataset, appends `{time, commit, reason, n_items}` to `test_access.log` in `out_dir` and the persistent run folder, and opens the guard.
- `test_experiment` refuses to run unless `final_eval` has been called, even when `FINAL_EVAL` is forced to True.
- **Notebook:** Step 5 starts with `yte = final_eval(...)`, and the statistics cell shows the test label share only after that call.
- **Tools that score test predictions** (`rescore_without_duplicate.py`, `analyse_replication.py`, `replicate_lexical_baselines.py`, `reproduce_main.py`) use `data.load_piqa_with_test_labels(cfg, reason, log_root)`, which logs to `outputs/test_access.log`. None of them was run in this upgrade. `verify_audit_claims.py` does not need labels and no longer loads them.
- `tools/check_test_access.py` also flags `final_eval(`, `load_labels(` and `test-labels` before the final cell.
- **Side effect:** the test split's `ordered_sha256` in `data_manifest.json` now covers the inputs with placeholder labels, so it differs from the hash recorded by the A100 run. The file checksums are unchanged.
- **`test_access.log`:** none exists in the repository, so there are **zero test-label reads** in this upgrade. A test asserts that `outputs/test_access.log` is absent.

**4.2 Unrounded ablation deltas everywhere.** The notebook's key-numbers cell, the report's Table 2 and prose, the notebook discussion and `check_report_numbers.py` now use differences of the unrounded means, rounded once.
- **Three test deltas change:** − difference tags −1.0 → **−0.9**; − cross-solution attention −0.9 → **−0.8**; tag bias initialised at 0 +0.1 → **+0.2**.
- **The rest is unchanged,** including all validation deltas.
- The Table 2 caption now says "differences of the unrounded 3-seed mean accuracies".
- The key-numbers cell's saved output was cleared, because it came from the old code; it reappears when the notebook is run.

**4.3 Calibration on validation** (`tools/calibration.py` → `experiments/calibration/val_calibration.csv`). Uses 15-bin top-label ECE, plus Brier score and NLL. Sources: the saved validation predictions of the A100 run (DACT, 8 ablations, B2; 3 seeds each except B2 with 1), and B1 recomputed on validation in the locked environment. B3 and B4 are not included: their validation probabilities were never saved, and the notebook now saves them for the next run.

| Model (validation) | Acc | ECE | Mean confidence | Brier | NLL |
|---|---|---|---|---|---|
| DACT (full) | 61.21 | 9.85 | 69.86 | 0.2474 | 0.7039 |
| − lexical head | 59.45 | 8.03 | 65.38 | 0.2498 | 0.7045 |
| − difference tags | 59.88 | 9.00 | 68.42 | 0.2514 | 0.7162 |
| − cross-solution attention | 60.50 | 7.68 | 67.00 | 0.2429 | 0.6901 |
| pointwise objective | 60.50 | 19.56 | 79.30 | 0.2832 | 0.8788 |
| vanilla Transformer | 58.75 | 6.08 | 62.68 | 0.2498 | 0.7050 |
| B2 BiLSTM + attention | 59.49 | 8.54 | 63.84 | 0.2505 | 0.7207 |
| B1 TF-IDF + LR | 59.24 | 5.25 | 63.75 | 0.2418 | 0.6815 |

- **Sanity check:** every accuracy equals the logged validation mean.
- **Reading:**
  - DACT is over-confident: 69.9 % mean confidence for 61.2 % accuracy.
  - Removing the lexical head lowers the confidence by 4.5 points and the ECE by 1.8 points. This supports the over-confidence note (FUTURE_FIXES D17), but the head is not the only source.
  - Pointwise scoring is by far the worst calibrated (ECE 19.6).
  - B1 is the best calibrated.
- **Not yet in the report:** this is a validation-only diagnostic. Candidate 1 of section 5 (a length-normalised or temperature-scaled lexical head) would use it as its pre-declared secondary metric.

**Report PDF (this branch only):** rebuilt with Tectonic after the corrections of sections 38 and 39 (three grid numbers, three ablation deltas, caption). Still 7 pages; the main text ends on page 6; only underfull-box warnings. `main`'s PDF is unchanged.

**Checks on this branch:**
- `sync_notebook.py --check`: none out of date;
- `check_test_access.py`: OK;
- `check_report_numbers.py`: tables 49 / 0 errors, prose 42 / 0 errors;
- `pytest`: **32 tests pass** (31 fast in 67 s, plus the notebook end-to-end test in 2 min 27 s, which now opens Step 5 through `final_eval` on synthetic data).

## 40. Final run of `main` on a free T4; report and notebook updated to it (2026-10-03/04, branch `t4-rerun-b880108`)
Ali asked for a fresh Colab run of the merged `main` (`b880108`), because seven saved outputs had been cleared when the upgrades changed their cells, and then for the report to be updated to this run.

**Run:**
- `colab new --gpu A100` was rejected (no compute units), so nothing was charged; the run used a free **T4** session (`group69`), kept alive in Brave.
- `setup_vm.py` cloned `main` at `b880108` and downloaded PIQA. The notebook was executed with `colab exec -f CITS4012_69.ipynb --env GROUP69_COMMIT=b880108`, 23:14–01:06 AWST.
- The CLI's output stream stalled after the first minute (as in section 24). The run was followed through `outputs/progress.json`, the heartbeat added in section 36, which reported every experiment and seed. RoBERTa does not write the heartbeat, so its log files were watched instead.
- **Result:**
  - the executed notebook has 0 error cells, and its code cells are identical to `b880108`;
  - the consistency check printed `OK: 15 result rows, 33 training runs with logs, 71 log files`;
  - the version check flagged tokenizers 0.23.2 and transformers 5.17.0 against the lock (0.23.1 and 5.16.1); scipy, numpy and scikit-learn match.
- `outputs.tgz` (no checkpoints) was downloaded and the session stopped; `colab sessions` showed none left.

**Test-label access:** `outputs/test_access.log` has exactly two lines, both from commit `b880108`: the notebook's single `final_eval` (Step 5), and `tools/rescore_without_duplicate.py` re-scoring the saved test predictions for the duplicate disclosure. `tests/test_eval_hygiene.py` asserts this.

**Files:**
- The previous `outputs/` (A100 run of `0d70cdc`) was moved unchanged to `experiments/a100_run_0d70cdc/outputs/` (with a README).
- `outputs/` now holds this run; predictions are gzip-compressed, as before.
- `CITS4012_69.ipynb` is the executed notebook.
- `tools/export_analysis_numbers.py`, `tools/verify_audit_claims.py` and `tools/rescore_without_duplicate.py` were re-run on it. `verify_audit_claims.py` had a bug that this run exposed: its glob also matched the new `*_mlm.jsonl` warm-up logs. Fixed.

**What changed compared with the A100 run** (all numbers from `outputs/results/`):

| | A100 (`0d70cdc`) | T4 (`b880108`) |
|---|---|---|
| Grid choice (1 seed) | small, 61.5 | **small + MLM warm-up, 61.85** (base 61.79, small 61.41) |
| DACT val / test | 61.2 ± 0.4 / 61.6 ± 0.1 | 61.3 ± 0.5 / 61.9 ± 0.3 |
| TF-IDF val / test, McNemar p | 59.2 / 61.2, p = 0.88 | 59.2 / 61.2, p = 0.62 |
| RoBERTa val / test | 68.0 ± 1.9 / 66.7 ± 1.1 | **59.2 ± 5.8 / 57.3 ± 6.9** |
| Qwen zero-shot | 78.1 / 75.4 | 78.2 / 75.1 |
| Ablation Δ tags / cross / lexical / pointwise / vanilla | −1.3/−0.9, −0.7/−0.8, −1.8/−0.4, −0.7/0.0, −2.5/−2.4 | −0.2/−1.3, +0.6/−0.6, −2.3/−0.3, −1.7/−1.4, −3.7/−4.3 |
| Attention on differing tokens: trained / bias 0 / untrained | 85.9 / 75.1 / 38.9 % | 78.0 / 62.0 / 38.9 % |
| Wrong vs correct predictions | 87.7 vs 84.7 %, p < 0.001 | 79.0 vs 77.4 %, p = 0.03 |

- **RoBERTa:** the declared restart rule fired: seed 42 was at 50.7 % after epoch 1 and was replaced by seed 1042, which learned (65.3 % val). Seeds 43 and 44 passed the 52 % threshold after epoch 1 (53.8, 52.2 %) and then stalled with their training loss at ln 2 (best 53.8 and 58.4 % val). This is reported as a failed baseline, with the A100 values as reference. Re-running only RoBERTa would be selective and was not done; a stricter rule or a lower learning rate must be declared before any new run.
- **Ablations:** what survives both runs is the combined effect (vanilla), the lexical head on validation, and the forced-choice objective. Cross-solution attention and the tags no longer help consistently on both splits. The report and notebook now say so instead of the A100 run's stronger claims.

**Report** (`report/CITS4012_69.tex`, rebuilt PDF: 8 pages, main text ends on page 6, no overfull boxes):
- **Updated to this run:** abstract, findings, Training paragraph (grid, selected configuration, fp16 on T4), Tables 1–3 (with a RoBERTa footnote, a new *− MLM warm-up* row, and A100 run 4 and the final T4 run in the history), ablation paragraph (nine comparisons), complementary errors, attention controls, H1/H2, case studies, history paragraphs, conclusion and limitations.
- **Case studies:**
  - lotion bars, a confident error in both runs;
  - the new uncertain item, *shipping* vs *shopping* container;
  - the camping lantern;
  - the new most confident error, an exotic trip, which replaces the tea kettle.
- **Figures:** `report/make_figures.py` rebuilt with the new attention-control values and cases.
- **Number checks:**
  - `check_report_numbers.py` was updated to the new text: Table 2 has the MLM row; Table 3 has run 4 (archived) and the final run; there are 50 prose claims.
  - It caught two rounding errors in my first draft: the selected grid value is 61.848, so 61.8 not 61.9, a margin of 0.06 over base; and the entropy p is 0.985, so 0.98.
  - **Result:** tables 53 values / 0 errors, prose 50 claims / 0 errors, loose 0 mismatches.

**Notebook text:** the Readme (code version, logs, runtime), the quantitative discussion (cell 43) and the qualitative discussion (cell 52) were rewritten from this run's printed outputs, including the RoBERTa failure, the grid instability and the new cases.


## 41. Training hardening (2026-10-04, branch `upgrade/training-hardening` from `main` `e2a4b2a`)
Source: `TRAINING_UPGRADE_PROMPT.md` (the generic QLoRA/SFT/W&B prompt rewritten for this project). The full audit table is in `docs/TRAINING_AUDIT.md`. No test number was computed; the repository's `outputs/test_access.log` is unchanged (2 lines, both from the final run).

**Code** (commit `1ff22a1`; the notebook was synced):
- **Diagnostics:** `train_qa`, `mlm_warmup` and B3's `finetune_pretrained` now log, per epoch, the gradient norm before clipping (mean, and max for DACT), step time, peak GPU memory and learning rate. The per-step `.item()` calls were replaced by running sums on the device. The per-step finiteness check is kept as a safety check, and it still syncs.
- **Length bucketing:** `Config.length_bucketing` (default False) and `bucket_chunk` with `BucketBatchSampler`. Used only for the training loader; evaluation keeps the dataset order. *Side effect:* the two new config fields mean that runs persisted by older code are refused by the resume identity check, as intended for a config change.
- **B4:** `lm_batch` and `score_continuations` were factored out, with an out-of-memory fallback that halves the batch, retries and logs it. Free GPU memory is printed before loading.
- **B3:** `encode_multiple_choice` was factored out for testing; behaviour is unchanged.

**Measurements and decisions (validation only):**
- **Padding** (`tools/padding_measurement.py`, real training split, no training): 69.6–69.9 % of the batch tensors are padding with random batches, 6.8–6.9 % with bucketing.
- **Pre-registered bucketing test** (`experiments/length_bucketing/`):
  - the pre-registration was committed at 01:35 and training ran 01:37–01:50, on a free T4, same session, seeds 42–44;
  - accuracy: control 60.92 ± 0.16, bucketing 61.12 ± 0.36; the gain of +0.20 is below 1.0 and below 2 × pooled std (0.56), so it is **rejected as a default**;
  - speed: step time −17 % (50.8 → 42.3 ms), epoch time −15 %; peak memory unchanged (0.68 GB). Documented as an opt-in speed option.
  - The control's 60.92 is 0.4 points below the final run's DACT (61.29), with the same configuration, GPU type and seeds. That is the session-to-session noise of non-deterministic GPU training.
- **Gradient checkpointing:** not needed, at 0.68 GB peak on a 15 GB T4.
- **Removing per-step syncs:** no measurable gain (6.31 s per epoch in the final run with the old code, against 6.25 s here, from different sessions), because the finiteness check still syncs. It is reported as such and not claimed as a speed-up.
- **Checkpoint selection** (`tools/selection_analysis.py`, existing logs): selecting by validation loss would pick an earlier epoch in 35/38 (T4) and 32/35 (A100) runs and lose 1.4 / 1.9 points of validation accuracy on average. The accuracy rule is kept. The comparison favours the accuracy rule by design, because it is scored on the split it selects on; the size and consistency of the gap still make the loss rule the worse choice here.
- **RoBERTa:** one change (learning rate 2e-5 → 1e-5) is pre-registered in `experiments/roberta_stability/`, with the number of seeds that learn as the primary metric. It is **not run**: it needs about 65 min of GPU time and the group's go-ahead. One unmeasured claim (how many pairs max length 128 truncates) was removed from that document before committing.
- **Not done, with reasons in the audit:** LoRA for B3, 4-bit Qwen, sequence packing, W&B.

**Tests:** `tests/test_training_hardening.py`, 11 tests:
- bucketing off by default, covers every item once, reshuffles, reproducible, less padding, only for the training loader;
- padding never receives attention; MLM labels only real tokens; B4 scores only the continuation; B3 builds [goal, solution_k] pairs;
- OOM fallback gives the same scores and raises at batch 1;
- new log fields present.

The full suite has **43 tests**, passing in 1 min 51 s on CPU.


## 42. RoBERTa stability test: lower learning rate rejected (2026-10-04, branch `upgrade/roberta-stability`)
Ali asked to run the pre-registered RoBERTa test (`experiments/roberta_stability/PREREGISTRATION.md`, committed in `482f904` before any run). It ran on a free T4 with code `2db3f51`, validation only.

| Arm | Val acc (%) seeds 42/43/44 | Mean ± std | Learned (loss < 0.68 by epoch 2) |
|---|---|---|---|
| control, lr 2e-5 | 69.60 / 68.49 / 68.80 | 68.96 ± 0.58 | 3/3 |
| candidate, lr 1e-5 | 65.20 / 67.99 / 68.24 | 67.14 ± 1.69 | 2/3 |

- **Decision: rejected.** The candidate has fewer learning seeds and a lower mean. The learning rate stays 2e-5.
- **Finding:** the unchanged settings trained all three seeds on a T4. The final run's failure (2 of 3 seeds at ln 2, mean 59.2 %) was bad luck, not a systematic fp16 problem. A stricter restart rule would have caught it (e.g. also restart when the loss has not left ln 2 after epoch 2). That rule should be declared before any future run.
- **Report:** its test numbers for B3 are unchanged, because no test prediction was made here.
- **Runs:** three sessions were needed.
  - **Session 1:** reclaimed after the first seed, most likely because the PC slept overnight; that seed is not used.
  - **Session 2:** used one `colab exec` per seed, with each result downloaded immediately, and the PC kept awake. It was reclaimed during the last seed.
  - **Session 3:** ran that seed alone. That is a documented deviation from "same session"; it cannot change the outcome, because the decision was fixed once the control had 3 of 3 learning seeds.
- **Evidence:** `experiments/roberta_stability/RESULTS.md` and `run_20261004/`.

## 43. Report: RoBERTa footnote and a figure reference (2026-10-04, branch `report/roberta-footnote`)
- **Table 1 footnote:** added "A separate validation-only re-run on a T4 trained all three seeds (69.0% validation), so the failure was chance, not the hardware." The 69.0 is the mean of `experiments/roberta_stability/run_20261004/control_lr2e-5_seed*.json` (section 42) and is checked by a new strict claim in `check_report_numbers.py` (51 claims, 0 errors). No test number changed.
- **Figure 1 legend:** it said the orange components are "ablated in Table 3"; the ablations are in Table 2. Fixed in `report/make_figures.py`, and the figure was rebuilt.
- **PDF:** rebuilt; 8 pages, main text ends on page 6, no overfull boxes.

## 44. Report: framing and precision pass (2026-10-04, branch `report/rhetoric-pass`)
Ali asked for drop-in replacement text, then for it to be applied. No result changed, and nothing was invented. Each new number is a strict claim in `tools/check_report_numbers.py`, built from a named file.

**Replaced:**
- the abstract's result sentence;
- the grid part of Training (seed sensitivity, the top-3 spread of 0.43 points against the seed std of 0.5);
- the B3 baseline sentence (125 M parameters; the validation-only re-run);
- Table 1: a new row, "validation re-run 69.0 ± 0.6 / --", and a rewritten caption;
- Overall comparison: the TF-IDF lead on validation in both runs, the test tie in both runs (p = 0.62 / 0.88), the parameter ratios (65× RoBERTa, about 800× Qwen) and the pretraining gaps (about 8 points on validation, 13.2 on test);
- "Complementary errors" became "Where the design succeeds": the vanilla gap in both runs (3.7 / 4.3 and 2.5 / 2.4), the error overlap and the minimal pairs;
- one clause in H1/H2;
- the seed-sensitivity sentence in "What did not";
- the first paragraph of the Conclusion;
- the start of Limitations.

**Not written,** because the data does not support it: that DACT beats TF-IDF on test; that DACT is more parameter-efficient than TF-IDF (TF-IDF's feature count was never measured); a test number for the RoBERTa re-run.

**The checker caught one error in the drafted text:** the top-3 grid spread is 61.848 − 61.414 = 0.43, not 0.44 (it had been computed from rounded values). Corrected before committing. "About 800" uses Qwen2.5-1.5B's nominal size, because its exact parameter count was not logged.

**Layout:**
- two redundant sentences were cut, and the new Table 1 row label was shortened (it was 5 pt overfull);
- PDF: 8 pages, no overfull boxes;
- the main text through the Conclusion ends on page 6; the Limitations section now starts on page 7, which the ACL rules exclude from the page limit.

**Checks:** strict tables 53 / 0 errors, strict prose 57 claims / 0 errors, loose 0 mismatches; 42 fast tests pass.

## 45. Report: main content fitted into 6 pages (2026-10-04, branch `report/limitations-fit`)
Ali asked to cut the Limitations section so that it fits in 6 pages.

- **Limitations:** now keeps only the points stated nowhere else:
  - the over-confident lexical head;
  - fast overfitting (85–92 % training against about 61 % validation accuracy at the selected epoch);
  - ablations changing parameter counts;
  - next steps.

  Seed sensitivity, the test-protocol caveat and the option-order dependence are referred to as "above". The over-one-point noise, one-seed grid, RoBERTa instability and the training/test overlap were removed, because Sections 3–5 already state them.
- **Protocol paragraph:** the sentence on the lexical head's test exposure was removed, because the "Validity of the test numbers" paragraph in Section 3 says the same thing.
- **Qualitative analysis:** the "baby wipes" example was removed. It came from an earlier run; the paragraph's other cases are from the final run.
- **Result:** the abstract through Limitations ends on page 6; Team Contributions and the references start page 7. No overfull boxes. Strict checks: 53 table values and 57 prose claims, 0 errors.

## 46. Report rewritten as a technical project report with IEEE numbered references (2026-10-05, branch `report/ieee-report-style`)
Ali asked for the report to be revised from research-paper style into a university technical report, keeping the ACL template, the "Group 69" author line and the 6-page content limit, with IEEE numbered citations and the references on a standalone final page. The official specification is `CITS4012_A2_2026.pdf`, which requires the ACL template in `[final]` mode, only the group number as author, at most six pages of content (Team Contributions and references excluded) and BibTeX.

**Structure:**
- Introduction: Background and Motivation, Project Objectives, Related Work.
- Methodology: Task Formulation, Input Representation and Difference Tags, Shared Transformer Encoder, Cross-Solution Attention, Difference-Guided Attention Pooling, Lexical Scoring Component, Prediction and Training Objective.
- Experimental Setup: Dataset and Data Split, Data Preprocessing, Training Configuration, Baseline Models, Evaluation Metrics, Experimental Protocol.
- Results and Analysis: Overall Performance, Ablation Study, Attention Analysis, Qualitative Analysis, Experimental Stability, Limitations.
- Conclusion; Team Contributions; References.

The rhetorical title, abstract phrasing and question-style headings were replaced. Table 3 (development history) was folded into "Experimental Stability". The training configuration is a paragraph, so Table 1 = results and Table 2 = ablations. The 1,838 labelled development items are called the "evaluation split" throughout, because PIQA's official test labels are not public.

**Facts checked against the code before restating them:**
- whitespace collapsing (`load_split`), NFKC and lower-casing (tokenizer);
- AdamW betas (0.9, 0.98), MLM masking (15 %, 80/10/10);
- the final configuration (`outputs/results/dact_full_val.json`: d=128, 2 layers, 4 heads, FFN 512, MLM warm-up 10 epochs at 5e-4).

No result number changed.

**Citations:**
- natbib is switched to `numbers,square,sort&compress` via `\PassOptionsToPackage` before `\usepackage[final]{acl}`. `acl.sty`'s own `\bibliographystyle{acl_natbib}` call is suspended while the package loads, and `\bibliographystyle{IEEEtran}` is used. `acl.sty`, the page geometry, fonts and columns are unchanged.
- References are numbered by first appearance.
- The one `\citet` became "Bisk et al.~\cite{...}", because IEEE numeric entries carry no author-year data and natbib printed "(author?)".
- All 23 entries are cited, with no undefined citations and no BibTeX warnings.
- Venue names use IEEE abbreviations ("Proc.", "Conf.", "Assoc. Comput. Linguistics", …) in both identical copies of `references.bib`, so that the list fits on one page.

**Layout:**
- the content (title through Conclusion) is on pages 1–6;
- Team Contributions is on pages 6–7;
- `\clearpage`, then References alone on page 8;
- no overfull boxes.

**Checker:** `tools/check_report_numbers.py` was updated: Table 1's label is now "(main model)"; the Table 3 row checks were removed; the claims were re-pointed to the new wording. Result: tables 37 values / 0 errors, prose 51 claims / 0 errors, loose 0 mismatches. 42 fast tests pass.

## 47. Group number set to 73 (2026-10-05)
Ali confirmed that the group's assigned ID is 73. Changed:
- the report's `uthor{Group 73}` (PDF rebuilt; layout unchanged);
- the notebook title line "Group 73 · Dataset: PIQA …";
- the submission files, now named `CITS4012_73.pdf` and `CITS4012_73.ipynb` as the specification requires.

The Google Drive folder names in the code (`MyDrive/Group69/...`) and the repository's file names were left unchanged: they are storage paths, not the group identity, and changing them would make the code differ from the saved run.

## 48. Clean-test run stopped; report cuts applied (2026-10-05)
- **Clean-test protocol** (branch `protocol/clean-test`, `f11d5fe`, `experiments/clean_test/PROTOCOL.md`): the code and protocol are committed, but the Colab run was stopped on Ali's request during the first grid candidate. `final_eval` was never reached, so the new test split remains unevaluated and can still be used cleanly. The branch is not merged.
- **Report cuts**, applied as agreed and shortening only repeated or anecdotal text:
  - shorter abstract; Related Work sentence merged;
  - shorter Table 1 footnotes; probe sentence tightened;
  - exotic-trip example cut to one clause;
  - Experimental Stability without the truncation and bf16 anecdotes;
  - Conclusion without the repeated parameter ratios.
- **One sentence added to Limitations:** a replacement test split, held out of the training file and never evaluated on, has been prepared but not yet run. This is true at this commit; no clean-test result is claimed.
- **Checks:** strict tables 37 / 0 errors, prose 50 claims / 0 errors. The main text ends mid-column on page 6; references alone on page 8.

## 49. Final audit and submission preparation (2026-10-06, branch `claude/inspiring-mayer-t5ckuv` from `main` `c701eba`)
Requested by the group (Salah): a final technical, scientific, reproducibility and format audit against the official
brief, with every safe fix applied. The work was carried out with Claude Code (an AI coding assistant). Its commits
carry a `Co-Authored-By` trailer. The full assessment is in `docs/FINAL_MARKING_AUDIT.md`, the requirement matrix in
`docs/FINAL_REQUIREMENTS_TRACEABILITY.md` and the checklist in `docs/FINAL_SUBMISSION_CHECKLIST.md`.

**Inputs and environment.**
- The official brief (*CITS4012 Group Project*, September 2026, 6 pages) was supplied by the group and read in full.
  Its SHA-256 is recorded in the traceability matrix; the PDF itself is not committed.
- The group confirmed that its assigned ID is **73**.
- Container: Linux on CPU, no GPU; Google Drive and the Hugging Face Hub were not reachable.
- Locked environment installed from PyPI: Python 3.13.16 with exactly the versions in `requirements-lock.txt`.
- LaTeX: TeX Live 2023 (pdflatex + BibTeX) from Ubuntu packages. Tectonic, which built the earlier PDFs, could not
  fetch its bundle here.

### 49.1 Verification without changes
- **Data identity.**
  - The official PIQA files (`train.jsonl`, `train-labels.lst`, `valid.jsonl`, `valid-labels.lst` from the PIQA
    GitHub release) have exactly the SHA-256 that the run recorded for the unit's four files
    (`outputs/data_manifest.json`).
  - The unit's `test` split is therefore PIQA's development set, byte for byte. The earlier audit could not establish
    this.
  - The files were placed in the git-ignored `data/piqa/`.
- **Splits.** The ordered SHA-256 of train / validation / test (14,501 / 1,612 / 1,838) reproduce the manifest.
- **Dataset facts in the report, all recomputed:**
  - 6 duplicate rows in the training file;
  - test item 1545 appears twice in the training split (no train–validation or validation–test overlap);
  - 1.58 % of evaluation solutions exceed 96 BPE tokens;
  - median differing-token share 0.150;
  - raw option swap changes the tags of 759 / 83 / 102 train / validation / test items;
  - 1,927,492 parameters, 262,144 of them lexical; BiLSTM 5,072,385.
- **New finding: the BPE tokenizer is not reproducible by retraining.**
  - Two trainings of `tokenizers` BPE on the same split differ (merge tie-breaking), even with `RAYON_NUM_THREADS=1`.
  - The A100 and T4 runs' tokenizers also differ.
  - The run's tokenizer is saved (`outputs/tokenizer.json`), but a re-run reproduces results only up to seed-level
    variation. Now disclosed in the report (§4.6), the notebook readme and the README.
- **Results.**
  - Every per-seed validation/test accuracy, mean, std, bootstrap CI and McNemar p of DACT, the nine ablations and
    the BiLSTM in `final_results.csv` was recomputed from `outputs/predictions/*.jsonl.gz`. All match exactly.
  - Error overlaps also match.
- **Post-run code changes.** The notebook's module cells differ from the run commit `b880108` (training hardening,
  section 41).
  - Every change was classified from the diff: logging and diagnostics, an off-by-default sampler, a B4 OOM
    fallback, a B3 encoding helper.
  - `tools/verify_post_run_equivalence.py` (new) trains DACT (MLM warm-up + QA) and the BiLSTM with the `b880108`
    sources and with the current `src/`. It uses the same synthetic data, tokenizer and seed, deterministically on
    CPU.
  - Parameters and validation probabilities are **bit-identical**. The only difference is the logged training loss,
    which differs by ≤ 3.4e-8 because it is now accumulated in float32 on the device.
  - Evidence: `experiments/final_audit_20261006/post_run_equivalence.json`.
- **B1.** `tools/verify_b1_reproduction.py` (new) re-fitted TF-IDF + LR in the locked environment.
  - 0 of 1,612 validation predictions differ from the saved file.
  - Test accuracy (61.21 %), McNemar p against DACT (0.6176) and every DACT/B1 overlap share equal the logged values.
- **Test-label reads in this audit.** Two reads, both evaluation-only:
  - a direct read of `test-labels.lst`, to confirm the gold labels stored in the saved prediction files;
  - one logged read by the B1 tool.
  - Both are recorded in `experiments/final_audit_20261006/test_access.log`; the first was entered manually, with an
    approximate time. No decision depended on them, and `outputs/test_access.log` was not modified.

### 49.2 Problems found and fixed

**Report** (`report/CITS4012_69.tex`, rebuilt PDF).
1. **Citation style (format compliance; P0 risk).** The ACL template's own `\bibliographystyle{acl_natbib}` had been
   suppressed in favour of IEEE numbered citations (section 46). Because "the official ACL template" is mandatory,
   the override was removed. Citations and the reference list are back in the template's author-year style, and the
   bibliography's full venue names were restored from before section 46 (keys and every other field unchanged).
   The group can revert this in one commit if it prefers the IEEE style.
2. **Significance overclaim.** "The only significant one on test (p = 0.007)" ignored the nine-comparison family
   that the same paragraph cites.
   - The Holm-adjusted p is 0.061 on test (0.084 on validation, where p = 0.009).
   - The sentence now gives both values and the stronger evidence: every vanilla seed lies below every full-model
     seed on both splits.
   - New tool: `tools/ablation_significance.py` → `outputs/results/ablation_significance.csv`.
3. **No visualised successful prediction.** The brief asks for examples of successful *and* failed predictions;
   Figure 3 showed a failure and a tie. Figure 3 now shows a correct prediction (test item 1300, p = 0.99), the
   failure (747) and the tie (1516), each with gold and predicted option, all from the saved attention data. The
   text now describes the shown success.
4. **Ambiguous footnote.** "The working seed reaches 65.3 %" read as a test score; 65.3 % is its validation accuracy.
5. **Equation accuracy.** The cross-solution equations now show the LayerNorm on queries, keys/values and fusion
   input, as in `ContrastiveCrossSolution`. The embedding now mentions LayerNorm and dropout.
6. **Unverifiable sentence.** "A replacement test split … has been prepared but not yet run" refers to an unmerged
   branch that is not in this repository. It was replaced by what an untouched estimate would need.
7. **Presentation.**
   - Figure 1 redrawn: its labels overflowed their boxes, the alignment → tags and lexical → softmax flows were
     missing, and the scorer input is now shown.
   - Figures 2 and 3 have larger fonts.
   - The forced page break that left page 7 holding only "model." was removed.
   - Table 2 gained a "Δ Par." column (parameter change of each ablation), because the parameter confound was only
     mentioned in words.
   - Interpretive wording made descriptive ("locates … but lacks the knowledge" → what the attention shows).
   - One sentence noting that the unit's test file is byte-identical to PIQA's development set.
   - A few redundant sentences cut to stay within six pages.
8. **Page limit.** Main content on pages 1–6, Team Contributions from page 7, references after it; 8 pages in total.
   - Measured slack: 3 more lines fit before the Conclusion would spill.
   - No overfull boxes, LaTeX warnings or BibTeX warnings (two redundant `number` fields removed).
   - `tools/check_report_numbers.py` updated to the new wording and to the parameter column: 46 table values,
     53 prose claims, 0 errors, 0 untraceable numbers.

**Notebook** (`CITS4012_69.ipynb`; no saved output of the run was edited).
- The title cell names Group 73 and the submitted file names (it still held the template's instruction line).
- **Readme cell:**
  - both sets of post-run changes and their equivalence proof;
  - the data fallback and SHA-256 check;
  - re-run expectations (tokenizer/GPU nondeterminism);
  - why no saved-model download is needed.
- **Discussion cells:** the Holm correction, and hedged causal wording in the qualitative discussion.
- **New supplementary-evidence cell** (markdown + code). It prints the report's numbers that come from separate
  runs or post-run analyses: ablation McNemar/Holm, the RoBERTa validation re-run, the A100 comparison, goal
  matching, the data audit and the duplicate re-scoring.
  - It reads saved files only and skips absent ones.
  - Its saved output was produced on 2026-10-06 in a separate CPU kernel from the repository root, and its markdown
    says so.
- **Colab default GPU** set to T4 in the notebook metadata (it requested an A100, unavailable on the free tier; the reported run used a T4).
- **Schema.** `nbformat_minor` set to 5: the cells carry `id` fields, which made the file fail strict nbformat
  validation since before this audit. No content change.

**Data acquisition** (`src/data.py`; synced into the notebook).
- `prepare_data` now:
  1. verifies the four files against the run's SHA-256 (`verify_piqa_files`; a loud warning on mismatch);
  2. if the unit's shared folder fails, downloads the official PIQA release, accepting it only if byte-identical
     (`download_piqa_release`), before falling back to the Drive zip, which needs an interactive mount.
- Tested offline (mocked downloads) and once live.
- Nothing that is trained or computed changes.

**Repository.**
- New tests (36), all passing:
  - `tests/test_model_properties.py`: no pretrained weights in the main path; random init and seed determinism;
    attention rows are distributions over allowed keys only; the explicit path equals SDPA; QA gradient reaches every
    attention component; pooling attention changes the output; inputs are label-independent; the tokenizer sees only
    the training split; exact checkpoint round trip; tiny-set overfit; no tags on padding.
  - `tests/test_evidence_consistency.py`: tables recomputed from predictions; Holm; McNemar known values; the three
    repository checkers.
  - `tests/test_data_acquisition.py`.
  - `tests/test_submission.py`.
- New tools: `verify_post_run_equivalence.py`, `verify_b1_reproduction.py`, `ablation_significance.py`,
  `build_submission.py`, `report/build.sh`.
- `requirements-lock.txt`: its comment wrongly called it the final run's versions; the T4 run used tokenizers
  0.23.2 and transformers 5.17.0. Comment corrected; pins unchanged.
- CI also runs the submission-format check.
- Two unreferenced PNGs of an earlier report version were removed from `report/figures/` (they remain in git
  history).
- `README.md` and `docs/PROJECT_STRUCTURE.md` were rewritten: they referred to folders that no longer exist, a draft
  report, 25 tests and `CITS4012_69.*` as submission names.
- **Submission package.** `tools/build_submission.py` checks the template section titles, error outputs, the
  `src/` sync, the group ID in the title cell, `[final]{acl}`, `\author{Group 73}` and the six-page rule. It then
  writes `submission/CITS4012_73.pdf`, `submission/CITS4012_73.ipynb`, `SHA256SUMS` and `MANIFEST.json`. A negative
  test confirmed that the page rule rejects a PDF whose Conclusion starts on page 7.

### 49.3 Not changed, and why
- **No retraining and no new experiments.** No result or design decision changed, so none was needed, and the test
  split must not guide further work.
- **No GPU re-run of the final notebook.** No GPU was available. The saved outputs remain those of `b880108`,
  with the proofs above.
- **Team Contributions text.** Only the members can confirm it.
- **AI-use statement.** The brief is silent and the unit's policy decides; the records in §19, §22, §27 and
  `NOTE_FOR_SALAH.md` are untouched.
- **Historical evidence.** Untouched: `outputs/` (except the new derived `ablation_significance.csv`), `evidence/`,
  `experiments/` (except the new `final_audit_20261006/`), logs and earlier audits.

### 49.4 Checks at the end
- `pytest`: 79 passed (43 earlier + 36 new), including the end-to-end notebook smoke run.
- `tools/sync_notebook.py --check`: none out of date.
- `tools/check_test_access.py`: OK.
- `tools/check_report_numbers.py`: 46 / 53, 0 errors.
- `tools/build_submission.py --check-only`: OK.
- Retraining required: **no**. Report changes: **yes** (listed above).
