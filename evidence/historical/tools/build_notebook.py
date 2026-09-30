"""Builds CITS4012_69.ipynb from the official template and the modules in src/.

Usage: python tools/build_notebook.py
The template's section titles are kept verbatim; code comes from src/*.py (lines tagged `# nb-skip`, which only
exist so the modules can import each other outside the notebook, are dropped).
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "Copy_of_CITS4012_YourGroupID.ipynb")
OUT = os.path.join(ROOT, "CITS4012_69.ipynb")


def module(name):
    src = open(os.path.join(ROOT, "src", f"{name}.py")).read()
    src = src.split('\nif __name__ == "__main__":')[0]
    lines = [ln for ln in src.splitlines() if not ln.rstrip().endswith("# nb-skip")]
    return re.sub(r"\n{3,}", "\n\n\n", "\n".join(lines)).strip() + "\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def sources(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


README = r"""
# Readme
*If there is something to be noted for the marker, please mention here.*

*If you are planning to implement a program with Object Oriented Programming style, please put those the bottom of this ipynb file*

**Group 69 · Dataset: PIQA (Physical Interaction QA) · Main model: DACT (Difference-Aware Contrastive Transformer), trained from scratch.**

### How to run (clean Google Colab)
1. `Runtime → Change runtime type → GPU` (any GPU works; we used an A100/L4 on Colab Pro, bf16 mixed precision is enabled automatically).
2. Put the provided data zip at **`MyDrive/Group69/PIQA.zip`** (contains `train.jsonl`, `train-labels.lst`, `test.jsonl`, `test-labels.lst`). The first data cell mounts Google Drive and extracts only these four files. To use another location change `Config.drive_zip`.
3. `Runtime → Run all`. No extra installation is needed: every package used (`torch`, `tokenizers`, `transformers`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`) is pre-installed on Colab; the first code cell prints their versions. The pretrained baselines (B3/B4) download `FacebookAI/roberta-base` and `Qwen/Qwen2.5-1.5B` from the Hugging Face Hub.

### Notes for the marker
* **Data usage.** No external data. The official PIQA training file is split 90/10 (stratified, seed 42) into *train* / *validation*. **All design decisions, hyper-parameter selection, early stopping and ablation comparisons use the validation split only.** The *test* split is touched only in the clearly marked *Final test evaluation* part of Section 3.
* **From scratch.** DACT and the BiLSTM baseline start from random initialisation; their BPE tokenizer is learned from the training split only. The optional masked-LM warm-up also uses only training-split text. Pretrained models appear only as baselines (B3, B4), and every baseline number is produced by the code in this notebook.
* **Logs.** Every run writes a JSON-lines log (`outputs/logs/*.jsonl`) with the full config and per-epoch metrics, and result summaries to `outputs/results/*.json|csv`. When Google Drive is mounted, the last cell copies `outputs/` (without checkpoints) to `MyDrive/Group69/outputs/`. The outputs saved in this notebook come from one clean top-to-bottom run on a Colab Pro A100 (2026-09-29), executed with the Colab CLI. In that run the four data files were uploaded directly to `/content/data/piqa/`, so the Drive step was skipped. The saved logs, result tables and figures of that run are submitted with this notebook in `outputs/`.
* **Code organisation.** Model classes (`nn.Module`s) are in Section 2 (*Model Implementation*) because the notebook must run top-to-bottom; all remaining code is plain functions. The same code also exists as modules in our repository (`src/`), from which this notebook is generated.
* **Runtime.** About 45 min on an A100 for everything (DACT grid, 3 seeds of the main model, 7 ablations x 3 seeds, all baselines). One DACT epoch takes about 4 s.
"""

S1_INTRO = r"""
## 1.1 Environment check and configuration
All hyper-parameters live in one `Config` dataclass so that every experiment (including ablations) is a documented, logged variation of it.
"""

S1_ENV = r"""
import os, sys, platform, importlib
SMOKE = os.environ.get("PIQA_SMOKE") == "1"   # local smoke test only; always False on Colab
for pkg in ["torch", "numpy", "sklearn", "scipy", "pandas", "matplotlib", "tokenizers", "transformers"]:
    print(f"{pkg:13s}", importlib.import_module(pkg).__version__)
import torch
print("python       ", platform.python_version())
print("GPU          ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (switch the runtime to GPU)")
"""

S1_CFG_RUN = r"""
CFG = Config()
if SMOKE:
    CFG = CFG.but(epochs=1, d_model=64, d_ff=128, n_layers=2, batch_size=32, out_dir="outputs_nbsmoke")
DEVICE = torch.device("cpu") if SMOKE else get_device()
os.makedirs(CFG.out_dir, exist_ok=True)
print("device:", DEVICE, "| autocast dtype:", amp_dtype(DEVICE))
"""

S1_DATA_MD = r"""
## 1.2 Loading, splitting and tokenising PIQA
* Each item: a **goal** and two candidate **solutions**; the label says which solution is physically sensible.
* The provided `test` split (1,838 items) is only used at the very end; a stratified 10 % of the training file is held out as the **validation** split.
* **Tokenizer:** byte-pair encoding (8k merges vocabulary) *learned on the training split only*. PIQA has many rare, domain-specific words (tools, materials, recipes), so sub-words avoid a large `[UNK]` rate that a word-level vocabulary would have.
* **Difference tags:** the two solutions of an item are usually near-duplicates. We align their BPE sequences with `difflib.SequenceMatcher` and tag every solution token as *shared* or *different*. These tags are consumed by the model (Section 2).
"""

S1_DATA_RUN = r"""
DATA = prepare_everything(CFG)
if SMOKE:
    for split in ("train", "val", "test"):
        ds = DATA[f"{split}_ds"]
        ds.items, ds.labels, ds.rows = ds.items[:256], ds.labels[:256], ds.rows[:256]
        DATA[split] = DATA[split][:256]
print({s: len(DATA[s]) for s in ("train", "val", "test")}, "| vocab size:", DATA["vocab_size"])
print("example:", DATA["train"][0])
enc = DATA["train_ds"].items[0]
print([(t, tag) for t, tag in zip(token_strings(DATA["tok"], enc[0][0]), enc[0][2])])
"""

S1_STATS = r"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def split_stats(ds):
    goal_len = [sum(1 for s in it[0][1] if s == 0) - 2 for it in ds.items]
    sol_len = [sum(it[k][3]) for it in ds.items for k in (0, 1)]
    diff_share = [np.mean([t == TAG_DIFF for k in (0, 1) for t, m in zip(it[k][2], it[k][3]) if m] or [0]) for it in ds.items]
    return goal_len, sol_len, diff_share

rows = []
for split in ("train", "val", "test"):
    g, s, d = split_stats(DATA[f"{split}_ds"])
    rows.append({"split": split, "items": len(DATA[f"{split}_ds"]),
                 "label=1 share": np.mean(DATA[f"{split}_ds"].labels),
                 "goal BPE len (mean)": np.mean(g), "solution BPE len (mean)": np.mean(s),
                 "solution BPE len (p99)": np.percentile(s, 99),
                 "differing-token share (median)": np.median(d)})
display(pd.DataFrame(rows).round(3))

g, s, d = split_stats(DATA["train_ds"])
fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
ax[0].hist(g, bins=30); ax[0].set_title("goal length (BPE tokens)")
ax[1].hist(np.clip(s, 0, 150), bins=50); ax[1].set_title("solution length (BPE tokens, clipped at 150)")
ax[2].hist(d, bins=40); ax[2].set_title("share of solution tokens that differ between the 2 options")
plt.tight_layout(); plt.show()
"""

S1_STATS_MD = r"""
**What the statistics imply for the design.**
* Labels are balanced, so **accuracy** is an unbiased metric and chance level is 50 %.
* In the median item only a small fraction of solution tokens differ between the two options (right histogram): the answer depends on a few words (*paper strips* vs *jeans material*, *weld* vs *nail*) placed inside otherwise identical text. A model that encodes each option independently must discover this small signal on its own from only 14.5k training items. DACT therefore (i) marks the differing tokens explicitly, (ii) compares the two options with cross-attention and (iii) biases its pooling towards the differing tokens.
* Solutions are short (p99 of about 100 BPE tokens), so we truncate solutions to 96 and goals to 40 tokens.
"""

S2_MD = r"""
## 2.1 DACT: Difference-Aware Contrastive Transformer

```
 goal + sol1 ─┐                                     ┌─ diff-guided attention pooling ─┐
              ├─ shared Transformer encoder ─ h1 ─┐ │                                 ├─ scorer → s1 ┐
 tags(sol1)  ─┘   (token+pos+segment+DIFF-TAG)    ├─┤ contrastive cross-solution attn │              ├─ softmax → P(option)
 goal + sol2 ─┐                                   │ │   h_i attends to h_j, fuse      │              │
              ├─ shared Transformer encoder ─ h2 ─┘ │   [h; c; h−c; h⊙c]              ├─ scorer → s2 ┘
 tags(sol2)  ─┘                                     └─────────────────────────────────┘
```

| Component | Standard choice | Our PIQA-specific change | Why |
|---|---|---|---|
| Input embedding | token + position (+ segment) | **+ difference-tag embedding** (goal/special, shared, different) | Tells the encoder from layer 1 where the two options disagree, so it does not have to learn this from 14.5k items |
| Option interaction | options scored independently | **Contrastive cross-solution attention**: each token of option *i* attends to the tokens of option *j* (plus a `[CLS]` sink for "no counterpart"), fused ESIM-style with `[h; c; h−c; h⊙c]` | Physical plausibility is relative ("jar" vs "plate"); the comparison features encode what replaced what |
| Summary vector | `[CLS]` or mean pooling | **Difference-guided additive attention pooling** with a learned bias per tag | Puts the decision on the few decisive tokens while the pooling weights remain interpretable |
| Objective | independent binary classification | **Pairwise softmax** over the two option scores (label smoothing 0.1) | Matches the forced-choice task |
| Encoder | post-LN Transformer | Pre-LN, shared across options, GELU, dropout 0.2 | Stable training from scratch; weight sharing makes the model **permutation-equivariant** (swapping options swaps scores, verified below), so it cannot learn a position bias |
| (optional) warm-up | none | Masked-LM warm-up on training-split text only | Compensates for the small labelled set without external data; kept only if it helps on validation |

Every component can be switched off in `Config`, which is how the ablations in Section 3 are run.
"""

S2_BASE_MD = r"""
## 2.2 Baselines
| ID | Baseline | Trained from scratch? | Role |
|---|---|---|---|
| B0 | Majority class | n/a | chance level |
| B1 | TF-IDF (1-2-grams) of *sol2 − sol1* + logistic regression (C tuned on validation) | yes | How far do shallow lexical cues get? |
| B2 | 2-layer BiLSTM + additive attention, same BPE tokenizer | yes | Same budget, recurrent instead of our Transformer design |
| B3 | `roberta-base` fine-tuned with a multiple-choice head | no (pretrained) | Reference for pretrained encoders |
| B4 | `Qwen2.5-1.5B` zero-shot, length-normalised log-likelihood | no (open LLM) | Reference for LLM knowledge without any training |
"""

S2_TRAIN_MD = r"""
## 2.3 Training utilities
AdamW (β₂ = 0.98, no decay on biases/LayerNorm/embeddings), linear warm-up + cosine decay, gradient clipping, bf16 autocast on GPU, early stopping on **validation** accuracy, and a JSON-lines log for every run.
"""

S2_SANITY = r"""
# Sanity checks: parameter count, output shapes, permutation equivariance of DACT
set_seed(0)
_m = DACT(CFG, DATA["vocab_size"]).to(DEVICE).eval()
print(f"DACT parameters: {count_parameters(_m) / 1e6:.2f}M | BiLSTM baseline: {count_parameters(BiLSTMAttention(CFG, DATA['vocab_size'])) / 1e6:.2f}M")
_items = [DATA["train_ds"][i] for i in range(8)]
_b = to_device(collate(_items), DEVICE)
_swapped = to_device(collate([((it[1], it[0]), 1 - y, i) for it, y, i in _items]), DEVICE)
with torch.no_grad():
    _o, _os = _m(_b, return_attn=True), _m(_swapped)
print("logits:", tuple(_o["logits"].shape), "| pooling:", tuple(_o["pool"].shape),
      "| self-attn per layer:", tuple(_o["self_attn"][0].shape))
print("swap option order -> scores swap:", torch.allclose(_o["logits"], _os["logits"].flip(-1), atol=1e-4))
del _m
"""

S3_PROTOCOL = r"""
## 3.1 Experimental protocol
* **Metric: accuracy** (official PIQA metric; the labels are balanced and each item has exactly one correct option, so accuracy is also the expected 0/1 forced-choice score). For every from-scratch model we report **mean ± std over 3 seeds**. For the test split we also report a **95 % bootstrap CI** and an **exact McNemar test** against DACT, both computed on the seed with the best *validation* accuracy.
* **Selection:** hyper-parameters are chosen on validation (Step 1). The ablations (Step 3) change one component of the selected configuration at a time, with 3 seeds each.
* **Test:** Steps 1-4 never touch the test split. Step 5 loads the saved checkpoints and evaluates every model on the test split once.
"""

S3_SETUP = r"""
import pandas as pd
SEEDS = [42] if SMOKE else [42, 43, 44]
EXP = {}        # name -> result dict of from-scratch experiments
"""

S3_HP_MD = r"""
### Step 1: hyper-parameter selection (validation only, 1 seed)
A small grid chosen after an initial run showed fast over-fitting (training accuracy far above validation accuracy after a few epochs): model size, dropout, and the in-domain MLM warm-up.
"""

S3_HP = r"""
GRID = {
    "base (d=256, L=4, drop=0.2)": CFG,
    "small (d=128, L=2)": CFG.but(d_model=128, d_ff=512, n_layers=2),
    "base, dropout 0.3": CFG.but(dropout=0.3, attn_dropout=0.15),
    "base + MLM warm-up (10 ep)": CFG.but(mlm_epochs=1 if SMOKE else 10),
    "small + MLM warm-up (10 ep)": CFG.but(d_model=128, d_ff=512, n_layers=2, mlm_epochs=1 if SMOKE else 10),
}
grid_res = {k: run_experiment(f"hp{i}", c, "dact", DATA, DEVICE, seeds=[42]) for i, (k, c) in enumerate(GRID.items())}
display(pd.DataFrame([{"config": k, "val acc": r["val"]["mean"], "epochs run": r["runs"][0]["epochs"]}
                      for k, r in grid_res.items()]).round(4))
BEST_HP = max(grid_res, key=lambda k: grid_res[k]["val"]["mean"])
BEST_CFG = GRID[BEST_HP]
print("selected on validation:", BEST_HP)
"""

S3_MAIN_MD = r"""
### Step 2: main model (selected configuration, 3 seeds)
"""

S3_MAIN = r"""
EXP["DACT (full)"] = run_experiment("dact_full", BEST_CFG, "dact", DATA, DEVICE, SEEDS)
"""

S3_ABL_MD = r"""
### Step 3: ablation studies (3 seeds each, one change at a time)
| Ablation | What is removed / replaced | Hypothesis tested |
|---|---|---|
| − difference tags | tag embedding removed | explicit difference marking helps |
| − cross-solution attention | options are no longer compared | comparing the options matters |
| − diff bias in pooling | additive attention pooling without tag bias | steering the pooling towards differing tokens helps |
| mean pooling | uniform average instead of attention pooling | an attention-based summary is better than a uniform one |
| pointwise objective | independent BCE per option instead of pairwise softmax | the forced-choice objective helps |
| vanilla Transformer | no tags, no cross-attention, mean pooling | the combined contribution of our changes |
| − MLM warm-up | (only if the selected config uses it) | in-domain pre-training on training-split text helps |
"""

S3_ABL = r"""
ABLATIONS = {
    "− difference tags": dict(use_diff_tags=False),
    "− cross-solution attention": dict(use_cross_solution=False),
    "− diff bias in pooling": dict(pool_mode="attn"),
    "mean pooling": dict(pool_mode="mean"),
    "pointwise objective": dict(objective="pointwise"),
    "vanilla Transformer": dict(use_diff_tags=False, use_cross_solution=False, pool_mode="mean"),
}
if BEST_CFG.mlm_epochs > 0:
    ABLATIONS["− MLM warm-up"] = dict(mlm_epochs=0)
for i, (name, kw) in enumerate(ABLATIONS.items()):
    EXP[name] = run_experiment(f"abl{i}", BEST_CFG.but(**kw), "dact", DATA, DEVICE, SEEDS)
"""

S3_BASE_MD = r"""
### Step 4: baselines (validation)
"""

S3_BASE = r"""
yva = np.array([r["label"] for r in DATA["val"]])
BASE = {}   # name -> {"val_acc", "val_pred", and later "test_pred"}
BASE["B0 majority"] = {"val_pred": majority_baseline(DATA["train"], DATA["val"])}
b1 = tfidf_lr_baseline(DATA["train"], DATA["val"], DATA["test"])
BASE["B1 TF-IDF + LR"] = {"val_pred": b1["val_pred"], "held_out_test_pred": b1["test_pred"]}
print("B1 selected C on validation:", b1["C"])
EXP["B2 BiLSTM + attention"] = run_experiment("b2_bilstm", CFG.but(lr=1e-3, dropout=0.3), "bilstm", DATA, DEVICE, SEEDS)
"""

S3_B3 = r"""
# B3: pretrained encoder fine-tuning. Model selection uses validation only; the test predictions are stored
# unseen for Step 5 (they are produced here only to avoid keeping a 500 MB checkpoint around).
B3_NAME = "FacebookAI/roberta-base"
_tr, _va, _te = (DATA["train"][:64], DATA["val"], DATA["test"]) if SMOKE else (DATA["train"], DATA["val"], DATA["test"])
b3 = finetune_pretrained(B3_NAME, _tr, _va, _te, DEVICE, CFG.out_dir, epochs=1 if SMOKE else 4,
                         batch_size=8 if SMOKE else 32, lr=2e-5, seed=42)
BASE["B3 RoBERTa-base (fine-tuned)"] = {"val_pred": b3["val_pred"], "held_out_test_pred": b3["test_pred"]}
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
"""

S3_B4 = r"""
# B4: zero-shot open LLM (no training at all)
B4_NAME = "Qwen/Qwen2.5-0.5B" if SMOKE else "Qwen/Qwen2.5-1.5B"
BASE["B4 Qwen2.5-1.5B (zero-shot)"] = {"val_pred": llm_zero_shot(B4_NAME, DATA["val"], DEVICE, verbose=False)}
for k, v in BASE.items():
    v["val_acc"] = accuracy(v["val_pred"], yva)
    print(f"{k:32s} val acc {v['val_acc']:.4f}")
"""

S3_TEST_MD = r"""
## 3.2 Step 5: final test evaluation (the only use of the test split)
"""

S3_TEST = r"""
yte = np.array([r["label"] for r in DATA["test"]])
for name, res in EXP.items():
    test_experiment(res, DATA, DEVICE)
BASE["B0 majority"]["test_pred"] = majority_baseline(DATA["train"], DATA["test"])
BASE["B1 TF-IDF + LR"]["test_pred"] = BASE["B1 TF-IDF + LR"]["held_out_test_pred"]
BASE["B3 RoBERTa-base (fine-tuned)"]["test_pred"] = BASE["B3 RoBERTa-base (fine-tuned)"]["held_out_test_pred"]
BASE["B4 Qwen2.5-1.5B (zero-shot)"]["test_pred"] = llm_zero_shot(B4_NAME, DATA["test"], DEVICE, verbose=False)

ref = EXP["DACT (full)"]["test_preds"]["pred"]
def row(name, val_mean, val_std, test_mean, test_std, pred, scratch):
    lo, hi = bootstrap_ci(pred, yte)
    b, c, p = mcnemar_exact(ref, pred, yte)
    return {"model": name, "from scratch": scratch, "val acc": val_mean, "val std": val_std,
            "test acc": test_mean, "test std": test_std, "test 95% CI": f"[{lo:.3f}, {hi:.3f}]",
            "McNemar p vs DACT": p if name != "DACT (full)" else np.nan}

table = []
for name, res in EXP.items():
    table.append(row(name, res["val"]["mean"], res["val"]["std"], res["test"]["mean"], res["test"]["std"],
                     res["test_preds"]["pred"], True))
for name, v in BASE.items():
    table.append(row(name, v["val_acc"], np.nan, accuracy(v["test_pred"], yte), np.nan, v["test_pred"],
                     name.startswith(("B0", "B1"))))
RESULTS = pd.DataFrame(table)
RESULTS.to_csv(os.path.join(CFG.out_dir, "results", "final_results.csv"), index=False)
pd.set_option("display.width", 200)
display(RESULTS.round(4))
"""

S3_PLOT = r"""
order = ["DACT (full)"] + list(ABLATIONS) + ["B2 BiLSTM + attention"] + list(BASE)
R = RESULTS.set_index("model").loc[order]
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#2b6cb0"] + ["#90cdf4"] * len(ABLATIONS) + ["#a0aec0"] * (1 + len(BASE))
ax.bar(range(len(R)), R["test acc"], yerr=R["test std"].fillna(0), color=colors, capsize=3)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
ax.set_xticks(range(len(R))); ax.set_xticklabels(R.index, rotation=35, ha="right")
ax.set_ylabel("test accuracy"); ax.set_ylim(0.45, 1.0); ax.legend()
ax.set_title("PIQA test accuracy (mean ± std over 3 seeds for from-scratch models)")
plt.tight_layout(); plt.savefig(os.path.join(CFG.out_dir, "results", "test_accuracy.png"), dpi=120); plt.show()
"""

S3_QUANT_MD = r"""
### Quantitative discussion

**Main comparison (test accuracy, mean ± std over 3 seeds for trained-from-scratch models).**

| Model | Val | Test |
|---|---|---|
| **DACT (ours)** | **60.1 ± 0.7** | **60.4 ± 1.2** |
| Vanilla Transformer (same budget, no DACT components) | 57.9 ± 0.9 | 58.7 ± 0.9 |
| B2 BiLSTM + attention | 59.6 ± 1.2 | 59.5 ± 1.0 |
| B1 TF-IDF + logistic regression | 59.2 | 61.2 |
| B3 RoBERTa-base, fine-tuned | 70.0 | 68.5 |
| B4 Qwen2.5-1.5B, zero-shot | 78.1 | 75.4 |

1. **DACT is the best from-scratch neural model.** It beats the vanilla Transformer by +2.2 points on validation and +1.7 on test, and the BiLSTM by +0.5 / +0.8. The gain over the vanilla Transformer is the combined effect of our PIQA-specific changes. It holds on both splits in the mean, and on validation all three vanilla seeds are below all three DACT seeds. It is still modest.
2. **It is not a large gain, and we do not over-claim it.** With 1,838 test items the 95 % CI of a single model spans about ±2.3 points, and seed std is about 1 point. McNemar tests between DACT's representative run and B1/B2 are not significant (p = 0.12 and 0.42). One caveat: the representative DACT run is the seed with the best *validation* accuracy (seed 44), which happens to be DACT's weakest seed on test (59.0 % vs 60.9 % and 61.2 %). We keep it because choosing a seed by test accuracy would break the test-set protocol.
3. **Shallow lexical cues are a strong baseline on PIQA.** TF-IDF + LR on the *difference* of the two solutions reaches 59.2 % (val) and 61.2 % (test). With only 14.5k training items and no external text, a large share of the learnable signal is which words tend to appear in plausible solutions (*baby wipes* vs *index cards*). Real physical reasoning needs world knowledge that cannot be learned from this data alone.
4. **Pretrained knowledge is the main bottleneck.** Fine-tuned RoBERTa gains about 8 points and a zero-shot 1.5B-parameter LLM about 15 points over every from-scratch model, without any architectural specialisation. This agrees with PIQA being designed to test physical commonsense rather than reading skill.

**Ablations** (Δ vs full DACT; the validation split is the one we used for decisions):

| Ablation | Val Δ | Test Δ |
|---|---|---|
| − difference tags | −1.1 | +0.1 |
| − MLM warm-up | −1.0 | +1.0 |
| − diff bias in pooling | −0.7 | +1.1 |
| mean pooling | −0.3 | +1.1 |
| − cross-solution attention | −0.0 | +1.4 |
| pointwise objective | −0.0 | +0.5 |
| **all DACT components removed (vanilla)** | **−2.2** | **−1.7** |

* On **validation**, every single-component ablation ties with or is worse than the full model. The difference tags and the MLM warm-up matter most (about −1 point each). Cross-solution attention and the pairwise objective show **no measurable individual contribution**.
* On **test**, most single-component ablations score 0.5-1.4 points *above* the full model. These gaps are about one seed-std and inside the CIs. They go the other way from validation, which points to noise from a validation split of 1.6k items rather than a real effect. We report them as observed and did not change the model afterwards, because the test split must not drive design decisions.
* Only removing *all* components hurts consistently on both splits. Our reading is that the components are **partly redundant**: the difference tags, the diff-biased pooling and the cross-solution attention all give the model the same information, namely where the options differ. Any one of them recovers most of the benefit, so removing a single one costs little. The honest conclusion is that *making the difference between the options explicit* helps, and none of the individual mechanisms is indispensable.
"""


S3_ATT_MD = r"""
## 3.3 Qualitative analysis of attention
We analyse the best-validation-seed checkpoint of the full DACT. For each selected test item the figure shows:
1. **top row**: the difference-guided pooling weights over each option's tokens (the vector the scorer sees);
2. **bottom-left**: contrastive cross-solution attention from option-1 tokens (rows) to option-2 tokens (columns), averaged over heads;
3. **bottom-right**: last-encoder-layer attention from each option's tokens to the goal tokens.

Tokens that differ between the two options are shown in **bold red**. Cases: the two most confident correct predictions, the two most confident errors, and the most uncertain item.
"""

S3_ATT = r"""
best_run = max(EXP["DACT (full)"]["runs"], key=lambda r: r["val_acc"])
DACT_BEST = DACT(BEST_CFG, DATA["vocab_size"]).to(DEVICE)
DACT_BEST.load_state_dict(torch.load(best_run["ckpt"], map_location=DEVICE))
DACT_BEST.eval()
tp = EXP["DACT (full)"]["test_preds"]
CASES = select_cases(tp["pred"], tp["prob"], tp["label"], k=2)
print(CASES)
os.makedirs(os.path.join(CFG.out_dir, "figures"), exist_ok=True)
"""

S3_ATT_CASE = r"""
for kind in ("confident_correct", "confident_wrong"):
    for i in CASES[kind]:
        r = DATA["test"][i]
        print(f"--- {kind} | test item {i} | gold = option {r['label'] + 1}\n  goal: {r['goal']}\n  sol1: {r['sol1']}\n  sol2: {r['sol2']}")
        plot_attention_case(DACT_BEST, r, DATA["tok"], BEST_CFG, DEVICE, title=f"{kind} #{i}",
                            save_path=os.path.join(CFG.out_dir, "figures", f"{kind}_{i}.png"))
"""

S3_ATT_UNC = r"""
i = CASES["uncertain"][0]
r = DATA["test"][i]
print(f"--- uncertain | test item {i} | gold = option {r['label'] + 1}\n  goal: {r['goal']}\n  sol1: {r['sol1']}\n  sol2: {r['sol2']}")
_ = plot_attention_case(DACT_BEST, r, DATA["tok"], BEST_CFG, DEVICE, title=f"uncertain #{i}",
                        save_path=os.path.join(CFG.out_dir, "figures", f"uncertain_{i}.png"))
"""

S3_ATT_STATS_MD = r"""
### Does attention focus on the differing tokens, and does that relate to correctness?
For every test item we measure the share of pooling attention that falls on difference-tagged tokens and compare it with the share those tokens would get under uniform attention, separately for correct and wrong predictions.
"""

S3_ATT_STATS = r"""
from scipy.stats import mannwhitneyu
test_loader = make_loader(DATA["test_ds"], BEST_CFG, False, DEVICE)
mass, correct = diff_attention_mass(DACT_BEST, test_loader, DEVICE)
uniform = diff_token_share(test_loader)
print(f"learned pooling tag bias [other, shared, diff]: {DACT_BEST.pool.tag_bias.detach().cpu().numpy().round(3)}")
print(f"mean attention mass on differing tokens: {mass.mean():.3f} (uniform-attention reference {uniform.mean():.3f})")
print(f"  correct predictions: {mass[correct].mean():.3f} | wrong predictions: {mass[~correct].mean():.3f}")
print("  Mann-Whitney U (correct vs wrong): p = %.4g" % mannwhitneyu(mass[correct], mass[~correct]).pvalue)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(mass[correct], bins=30, alpha=0.6, density=True, label="correct")
ax.hist(mass[~correct], bins=30, alpha=0.6, density=True, label="wrong")
ax.axvline(uniform.mean(), c="k", ls="--", lw=1, label="uniform reference")
ax.set_xlabel("pooling attention mass on differing tokens"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(CFG.out_dir, "figures", "diff_attention_mass.png"), dpi=120); plt.show()
"""

S3_QUAL_MD = r"""
### Qualitative discussion

**What the attention does in general.** The learned pooling bias for difference-tagged tokens went from its initial value 1.0 to about 1.02 in training, while the bias for shared tokens stayed near 0. On average DACT puts **64 % of its pooling attention on the differing tokens, although they make up only 25 % of solution tokens** (uniform reference, dashed line). So the model does base its decision on the tokens where the options disagree, as intended. The attention mass on differing tokens is almost the same for correct (63.0 %) and wrong (64.7 %) predictions (Mann-Whitney p = 0.08). **Looking at the right tokens is therefore necessary but not sufficient.** Errors come from judging those tokens wrongly, not from looking elsewhere.

**Success: "Remove dust residue from chalk boards", *baby wipes* ✓ vs *index cards* (p = 0.91).** Pooling attention sits almost entirely on *baby wipes* / *index cards*. In the last encoder layer, *baby* and *wipes* attend strongly to the goal verb *remove*, so the differing tokens are read in the context of the action. The cross-solution map shows every option-1 token attending to *index cards*: the comparison centres on the substituted object. The second success (*Clean dirty cup coaster*: *baby wipes* ✓ vs *tissue paper*, p = 0.91) has the same pattern: pooling on the substituted nouns, *baby*/*wipes* attending to *clean*, and cross-attention from option 1 concentrated on *tissue paper*. Both successes reuse a **lexical prior from the training split**. In training items, *baby wipe(s)* appears only in the correct option 26 times and only in the wrong option 5 times, while *tissue paper* appears only in the correct option 3 times and only in the wrong option 24 times. The model is confident because it has seen these phrases before, not because it reasons about absorbency. This is the association a from-scratch model can learn from 14.5k items, and it also explains why TF-IDF + LR does well.

**Failure 1: "To store food without a refrigerator", *cooler filled with ice* (gold) vs *small chest of water* (p = 0.85 for the wrong option).** Pooling again concentrates on the differing span (*cooler filled* / *small chest*). However, *ice*, the word that decides the item, gets relatively little pooling weight. The cross-solution attention from *cooler … ice* does not line up with *chest … water*; it spreads over the shared words *put the* and over *small*. The lexical prior also works against the model here: in training, *cooler* appears only in the wrong option 7 times and only in the correct option 2 times. The goal token *refrigerator*, which ties the goal to temperature, gets almost no attention from either option. The model finds the differing phrase but lacks the physical knowledge that ice keeps food cold. It falls back on a misleading word statistic.

**Failure 2: "To make a ball of aluminum foil", *fold it in half* vs *crinkle it into a ball* (gold) (p = 0.83 for the wrong option).** The pooling focuses on *fold … half* and on *cr ##ink ##le*, and the cross-solution attention aligns option 1's *in* / *fold* with *cr ##ink ##le*. The model therefore correctly identifies *fold in half ↔ crinkle into a ball* as the substitution. But *crinkle* occurs in only 2 training items and is split into three sub-word pieces, so its representation is weak. The gold option also repeats the goal words *a ball*, a strong cue a human would use, but the solution-to-goal attention does not link *ball* to the goal's *ball*. The model apparently prefers the common, "safe" instruction *fold it in half*. This shows two limits of training from scratch: a weak representation of rare words, and no reliable goal-solution matching learned from 14.5k items.

**Uncertain case: blackberry cobbler, *ground coffee* vs *granulated sugar* (gold) (p = 0.50).** The two 162-token options differ only at BPE positions 97-98, **just past our 96-token truncation limit**. After truncation both inputs are identical (no red tokens; both end at *sprinkle 1*), so every attention map and both scores are identical. The model does not fail here through reasoning: preprocessing removes the evidence. This long-tail case (about 1-2 % of solutions exceed 96 tokens) points to an easy improvement: **truncate around the difference span** (keep a window centred on the differing tokens) instead of keeping the prefix.

**Summary.** The attention maps confirm that DACT's components work as designed. The decision is concentrated on the differing words, and the cross-solution attention aligns substituted phrases. The failures show that DACT's limit is mostly **knowledge, not focus**. It finds the decisive tokens but judges them with word statistics from the training split (*baby wipes* good, *cooler* bad) instead of physical knowledge, which only pretraining provides (B3/B4). The one clear focus failure (the cobbler item) is caused by prefix truncation, and difference-centred truncation would fix it.
"""


S3_SAVE = r"""
# Persist logs, results and figures to Google Drive (checkpoints excluded to save space)
import shutil
if not SMOKE and os.path.isdir("/content/drive/MyDrive"):
    dst = "/content/drive/MyDrive/Group69/outputs"
    shutil.copytree(CFG.out_dir, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoints"))
    print("copied to", dst)
print(sorted(os.listdir(os.path.join(CFG.out_dir, "logs")))[:10], "...")
"""


def main():
    nb = json.load(open(TEMPLATE))
    t = nb["cells"]
    title, readme, sec1, sec2, sec3 = (c for c in t if c["cell_type"] == "markdown")
    cells = [title, md(README),
             sec1, md(S1_INTRO), code(S1_ENV), code(module("config")), code(S1_CFG_RUN),
             md(S1_DATA_MD), code(module("data")), code(S1_DATA_RUN), code(S1_STATS), md(S1_STATS_MD),
             sec2, md(S2_MD), code(module("model")), md(S2_BASE_MD), code(module("baselines")),
             md(S2_TRAIN_MD), code(module("train")), code(module("evaluate")), code(S2_SANITY),
             sec3, md(S3_PROTOCOL), code(module("experiments")), code(module("viz")), code(S3_SETUP),
             md(S3_HP_MD), code(S3_HP), md(S3_MAIN_MD), code(S3_MAIN), md(S3_ABL_MD), code(S3_ABL),
             md(S3_BASE_MD), code(S3_BASE), code(S3_B3), code(S3_B4),
             md(S3_TEST_MD), code(S3_TEST), code(S3_PLOT), md(S3_QUANT_MD),
             md(S3_ATT_MD), code(S3_ATT), code(S3_ATT_CASE), code(S3_ATT_UNC), md(S3_ATT_STATS_MD),
             code(S3_ATT_STATS), md(S3_QUAL_MD), code(S3_SAVE)]
    for c in cells:
        c["source"] = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    nb["cells"] = cells
    nb["metadata"]["accelerator"] = "GPU"
    nb["metadata"]["colab"]["gpuType"] = "A100"
    json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT, "with", len(cells), "cells")


if __name__ == "__main__":
    main()
