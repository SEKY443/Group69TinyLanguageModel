"""Evidence-correct narrative shared by historical-notebook review and safe builder."""

QUANTITATIVE = r"""
### Quantitative discussion (historical A100 experiment)

The table above reports the original run, not a rerun of the audit instrumentation. DACT achieved
60.09 +/- 0.66% validation and 60.37 +/- 1.17% test (sample SD over seeds 42/43/44).
It exceeds the vanilla Transformer's means (57.88/58.71) and BiLSTM's means (59.57/59.54),
but **several single-component ablations and TF-IDF score higher on test**. DACT is not the
best from-scratch variant on that split. The full-model-minus-vanilla difference is +2.21 validation
and +1.67 test percentage points; the three mechanisms are removed together, so this does not
identify the causal benefit of any single component.

The single-component ablation means tie or fall below the full model on validation but mostly rise on test.
This pattern is compatible with selection variability, component redundancy, capacity differences and
limited sample size; these explanations are hypotheses, not established facts. In particular, removing
cross-solution attention does not remove encoder attention or difference-tag information. Mean pooling
removes learned weighting and the tag bias together. BiLSTM uses different learning rate/dropout and no MLM,
so it is an informative baseline rather than an exactly matched training-budget ablation.

**Uncertainty:** displayed means/SD describe three training runs. Each bootstrap interval and McNemar test
instead describes the single seed chosen by validation. DACT's representative seed is 44, test 59.03%,
CI [56.75,61.21]%. This is not a CI for 60.37%. The original p-values are unadjusted exploratory comparisons.
Removing pooling tag bias gives p=0.01696 against the representative DACT run; it must not be dismissed
solely because intervals overlap. Holm correction across all 12 reported comparisons makes this
comparison nonsignificant (see the executed audit's results_table.csv). Non-significance does not prove
equivalence. Original prediction arrays and checkpoints are absent from this checkout, so the saved
paired-test results cannot currently be independently recomputed.

TF-IDF reaches 61.21% test without goal-dependent neural interaction, demonstrating a useful lexical baseline.
RoBERTa reaches 68.50% and Qwen 75.35%; this is consistent with benefits from pretraining, but model size,
tokenisation, objective, training budget and possible benchmark exposure also differ. The experiments do
not isolate world knowledge as the sole bottleneck.

**Protocol qualification:** one normalised exact test item occurs twice in training. Test inputs/statistics
were read before final evaluation; B1/B3 generated predictions early without using test accuracy for
selection. No evidence of test-based checkpoint selection was found in the code, but complete disjointness
and literal non-access before Step 5 must not be claimed. Historical numbers and splits remain unchanged.
"""

QUALITATIVE = r"""
### Qualitative discussion (historical saved attention figures)

The representative checkpoint was chosen on validation (seed 44). The four confident cases are extreme
examples, not a representative sample of all errors. Confidence is a softmax score, not calibrated certainty.

* **Successes #804 and #280:** both select *baby wipes* correctly for cleaning tasks (about .91).
  Pooling favours the substituted nouns. Training phrase counts recorded in the development log
  (26 correct-only versus 5 wrong-only occurrences for baby wipe(s)) are consistent with a lexical-prior
  explanation, but the maps and counts do not prove that this caused either decision.
* **Failure #660:** gold is a cooler filled with ice; the model favours a small chest of water (about .85).
  Pooling concentrates on the differing span, while the cross-solution map is diffuse and includes shared
  words such as *put the*. This suggests a failure to use temperature-related evidence effectively.
  It does not establish that the model lacks a specific fact or uses a particular shortcut internally.
* **Failure #708:** the model favours folding foil in half over crinkling it into a ball (about .83).
  The substituted phrases receive attention; *crinkle* is split into subwords. Rare-word learning and
  weak action-goal matching are plausible limitations, not proven causes.
* **Uncertain failure #1433:** both retained prefixes are identical after the 96-token cap. The distinguishing
  coffee/sugar span lies beyond it, and the displayed probabilities are .50/.50. This is direct evidence
  of information loss during preprocessing. Difference-centred truncation is future work; it was not
  evaluated and must not be described as fixing the answer or improving accuracy.

The saved aggregate statistic is .637 attention mass on differing tokens versus .248 under uniform
pooling; correct/incorrect values are .630/.647 (Mann-Whitney p=.07855). The code excludes examples
without retained differing tokens. The difference bias begins at +1 and ends near +1.019, so much of
the preference is built in before training. This is not evidence that training discovered causal importance.
Goal-attention plots show a slice of last-layer self-attention and omit other keys; cross-solution plots
omit the CLS sink. Displayed rows therefore need not sum to one. Attention weights describe model
behaviour; they are not a complete or causal explanation (Jain and Wallace, 2019,
https://aclanthology.org/N19-1357/).

The shared network swaps scores when already-encoded options are swapped. However, the historical
SequenceMatcher alignment is directional: re-encoding swapped raw options changes tags in 755/14,501
training examples. Thus the full historical input pipeline is not guaranteed permutation-equivariant.
The optional canonical alignment in the refined source is a separately tested future protocol, not the
preprocessing that produced these figures or scores.
"""

AUDIT_NOTICE = r"""
### Audit addendum (2026-09-30)

This notebook retains the original A100 **code and saved outputs**. Its narrative has been corrected after
an independent audit. The exact original file/source and SHA-256 hashes are in `evidence/historical/`.
The improved instrumentation in current `src/` is assembled separately into
`notebooks/CITS4012_69_reproducible.ipynb`; it must not inherit historical outputs. A local development smoke test
does not establish a fresh clean Colab full run. The final report is still required.

The saved historical tokenizer uses 8,000 vocabulary entries (not 8,000 merges). Raw, untruncated training
candidate lengths have median 14, p95 about 65, p99 108; the earlier table below reports *post-truncation*
lengths. One exact normalised test item (index 1545) appears twice in the training subset. No exact
train-validation triple overlap was found. Training/model choices must not be changed in response to
the test results during this audit.

The original checkpoints and prediction arrays are absent locally. Saved logs support the reported
training and validation metrics, and JSON/CSV arithmetic has been verified. Historical confidence
intervals, significance tests and full attention statistics cannot be recomputed without those artifacts.
"""


def refined_cells(old):
    """Replacement cells for a fresh run. Historical prose is explicitly separated from new results."""
    updates = {}
    updates['README'] = r'''
# Readme
*If there is something to be noted for the marker, please mention here.*

*If you are planning to implement a program with Object Oriented Programming style, please put those the bottom of this ipynb file*

**Group 69: PIQA / DACT. Reproduction candidate with improved evidence retention.**

This generated notebook has no full experimental outputs until executed. The historical executed
submission candidate is `CITS4012_69.ipynb`. Do not mix their outputs.

1. Select a Colab GPU. The original full run used an A100 40 GB. Smaller GPUs may need separately logged
   resource settings; the original B4 batch size is not guaranteed to fit every GPU.
2. Place the supplied ZIP at `MyDrive/Group69/PIQA.zip`, or place its four files under `data/piqa/`.
3. Run all. The setup cell checks/imports dependencies and installs missing ones. Exact historical
   versions are recorded in the historical notebook and `requirements-a100.txt`.
4. Results go to a fresh timestamped output directory; checkpoints and prediction rows are retained.
   Download that directory before ending the runtime. The last cell copies it to Drive when mounted.

DACT uses no pretrained weights or external corpus. B3/B4 are explicitly pretrained baselines.
All model selection uses validation. The existing split is preserved, including the disclosed
training/test duplicate; new results must retain this limitation. The test set is not an unseen benchmark
for any design revisions after the historical analysis. `symmetric_diff_tags=False` preserves historical
alignment; changing it is a separately labelled experimental protocol, not a reproduction.

For an offline development check set `PIQA_SMOKE=1` before running. It uses tiny training/validation
subsets, a validation-derived diagnostic holdout, one seed/epoch, and **skips B3/B4**. Its displayed
metrics and figures are diagnostic artifacts, never reportable full PIQA test results.
'''
    updates['S1_ENV'] = r'''
import os, sys, platform, importlib, subprocess
SMOKE = os.environ.get("PIQA_SMOKE") == "1"
packages = {"torch":"torch==2.11.0", "numpy":"numpy", "sklearn":"scikit-learn",
            "scipy":"scipy", "pandas":"pandas", "matplotlib":"matplotlib", "tokenizers":"tokenizers"}
if not SMOKE:
    packages["transformers"] = "transformers==5.16.1"
for package, requirement in packages.items():
    try:
        imported = importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install",requirement])
        imported = importlib.import_module(package)
    print(f"{package:13s}", imported.__version__)
import torch
print("python",platform.python_version())
print("GPU",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("MODE:","DEVELOPMENT ONLY; pretrained baselines skipped" if SMOKE else "FULL REPRODUCTION")
'''
    updates['S1_CFG_RUN'] = r'''
import time
CFG = Config(data_dir=os.environ.get("PIQA_DATA_DIR", "data/piqa"),
             out_dir="experiments/" + ("development/" if SMOKE else "completed/") + "outputs_reproduction_" + time.strftime("%Y%m%d_%H%M%S"))
if SMOKE:
    CFG = CFG.but(epochs=1, d_model=64, d_ff=128, n_layers=2, batch_size=32, num_workers=0)
DEVICE = get_device()
set_seed(CFG.run_seed, CFG.deterministic)
os.makedirs(CFG.out_dir, exist_ok=False)
print("device:",DEVICE,"| output:",CFG.out_dir)
'''
    updates['S1_DATA_RUN'] = r'''
DATA = prepare_everything(CFG)
if SMOKE:
    # Use validation rows for the diagnostic third split; never infer on official test in smoke mode.
    DATA["train"], DATA["val"], DATA["test"] = DATA["train"][:128], DATA["val"][:128], DATA["val"][128:256]
    for split in ("train","val","test"):
        DATA[f"{split}_ds"] = PIQADataset(DATA[split],DATA["tok"],CFG)
    print("SMOKE: 'test' below is a validation-derived diagnostic holdout, not official test.")
print({s:len(DATA[s]) for s in ("train","val","test")},"| vocab:",DATA["vocab_size"])
'''
    updates['S1_STATS'] = r'''
import numpy as np, pandas as pd, matplotlib.pyplot as plt
rows = []
for split in ("train", "val"):  # no test statistics during development
    for field in ("goal","sol1","sol2"):
        lengths = np.array([len(DATA["tok"].encode(r[field]).ids) for r in DATA[split]])
        limit = CFG.max_goal_len if field == "goal" else CFG.max_sol_len
        rows.append({"split":split,"field":field,"items":len(lengths),"mean":lengths.mean(),
                     "median":np.median(lengths),"p95":np.percentile(lengths,95),
                     "p99":np.percentile(lengths,99),"truncated":int((lengths>limit).sum())})
display(pd.DataFrame(rows).round(3))
'''
    updates['S1_STATS_MD'] = '''**Length protocol.** These statistics are before truncation. The historical limits
40/96 are retained for reproduction, with acknowledged information loss in long examples. Accuracy is
the number of correctly selected options divided by the number of examples; it directly measures the
two-choice task. Approximate label balance makes chance/majority useful reference points.'''
    updates['S1_DATA_MD'] = old['S1_DATA_MD'].replace('8k merges vocabulary','8,000-entry vocabulary').replace('is only used at the very end','is reserved for final model inference in this reproduction')
    updates['S2_MD'] = old['S2_MD'].replace('**permutation-equivariant** (swapping options swaps scores, verified below), so it cannot learn a position bias','**equivariant for encoded option pairs**; the legacy alignment is directional and does not guarantee raw-input equivariance')
    updates['S2_BASE_MD'] = old['S2_BASE_MD'].replace('Same budget, recurrent instead of our Transformer design','Same QA epoch cap, but different learning rate/dropout and no MLM; recurrent comparison')
    updates['S3_PROTOCOL'] = '''## 3.1 Experimental protocol
Training and checkpoint selection use only training/validation. B1 and B3 defer test inference until
the final evaluation cell. The supplied historical split, tokenizer training recipe and limits are retained.
The cross-split exact duplicate remains a disclosed limitation. Scratch experiments use three seeds in
full mode. CIs and paired tests refer to the best-validation seed, not the three-seed mean. Holm-adjusted
p-values cover the declared family of 12 historical comparisons (the actual available family in smoke mode).
Smoke mode metrics use a validation-derived holdout and are not official test results.'''
    updates['S3_ABL_MD'] = old['S3_ABL_MD'].replace('one change at a time','component studies; vanilla is a joint ablation').replace('options are no longer compared','cross-solution module removed; tag comparison remains').replace('explicit difference marking helps','input tag embeddings help beyond retained pooling tag bias').replace('uniform average instead of attention pooling','uniform average; removes learned scores AND tag bias')
    updates['S3_BASE'] = old['S3_BASE'].replace('DATA["val"], DATA["test"])','DATA["val"])').replace(', "held_out_test_pred": b1["test_pred"]','')
    updates['S3_B3'] = '''if SMOKE:
    print("B3 skipped: offline development mode")
else:
    B3_NAME = "FacebookAI/roberta-base"
    b3 = finetune_pretrained(B3_NAME, DATA["train"], DATA["val"], None, DEVICE, CFG.out_dir,
                             epochs=4,batch_size=32,lr=2e-5,seed=42)
    BASE["B3 RoBERTa-base (fine-tuned)"] = {"val_pred":b3["val_pred"]}
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
'''
    updates['S3_B4'] = '''if SMOKE:
    print("B4 skipped: offline development mode")
else:
    B4_NAME = "Qwen/Qwen2.5-1.5B"
    BASE["B4 Qwen2.5-1.5B (zero-shot)"] = {"val_pred":llm_zero_shot(B4_NAME,DATA["val"],DEVICE,verbose=False,out_dir=CFG.out_dir)}
for name,v in BASE.items():
    v["val_acc"] = accuracy(v["val_pred"],yva)
    save_predictions(os.path.join(CFG.out_dir,"predictions",name.split()[0]+"_val.jsonl"),DATA["val"],
                     {"pred":v["val_pred"]},name,"val")
    print(name,"validation accuracy",v["val_acc"])
'''
    updates['S3_TEST_MD'] = '## 3.2 Step 5: final inference (development holdout only in smoke mode)'
    updates['S3_TEST'] = r'''
yte = np.array([r["label"] for r in DATA["test"]])
for name,res in EXP.items():
    test_experiment(res,DATA,DEVICE)
BASE["B0 majority"]["test_pred"] = majority_baseline(DATA["train"],DATA["test"])
BASE["B1 TF-IDF + LR"]["test_pred"] = tfidf_lr_predict(b1,DATA["test"])
if not SMOKE:
    BASE["B3 RoBERTa-base (fine-tuned)"]["test_pred"] = pretrained_predict(b3,DATA["test"],DEVICE)
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    BASE["B4 Qwen2.5-1.5B (zero-shot)"]["test_pred"] = llm_zero_shot(B4_NAME,DATA["test"],DEVICE,verbose=False,out_dir=CFG.out_dir)
ref = EXP["DACT (full)"]["test_preds"]["pred"]
table = []
for name,res in EXP.items():
    pred=res["test_preds"]["pred"];lo,hi=bootstrap_ci(pred,yte)
    table.append({"model":name,"val acc":res["val"]["mean"],"val std":res["val"]["std"],
                  "test acc":res["test"]["mean"],"test std":res["test"]["std"],
                  "representative seed":res["test"]["best_val_seed"],
                  "representative test acc":res["test"]["best_val_seed_acc"],
                  "representative CI95":f"[{lo:.4f},{hi:.4f}]",
                  "McNemar p":mcnemar_exact(ref,pred,yte)[2] if name!="DACT (full)" else np.nan})
for name,v in BASE.items():
    save_predictions(os.path.join(CFG.out_dir,"predictions",name.split()[0]+"_test.jsonl"),DATA["test"],
                     {"pred":v["test_pred"]},name,"development_holdout" if SMOKE else "test")
    lo,hi=bootstrap_ci(v["test_pred"],yte)
    table.append({"model":name,"val acc":v["val_acc"],"test acc":accuracy(v["test_pred"],yte),
                  "representative CI95":f"[{lo:.4f},{hi:.4f}]","McNemar p":mcnemar_exact(ref,v["test_pred"],yte)[2]})
RESULTS = pd.DataFrame(table)
mask=RESULTS["McNemar p"].notna()
RESULTS.loc[mask,"Holm p"] = holm_adjust(RESULTS.loc[mask,"McNemar p"])
RESULTS.to_csv(os.path.join(CFG.out_dir,"results","final_results.csv"),index=False)
display(RESULTS.round(4))
'''
    updates['S3_QUANT_MD'] = '''### Interpretation required after execution
Use the newly produced tables and retained predictions to write the analysis. Do not transfer numerical
claims from the historical study or interpret smoke metrics as generalisation results. Historical audit:
full DACT did not outperform every single-component ablation on test; avoid causal claims from weak
differences. CI columns concern representative seeds, while error bars show across-seed sample SD.
The original 12-way Holm correction is exploratory; report the comparison family used in this run.'''
    updates['S3_QUAL_MD'] = '''### Interpretation required after execution
Inspect these actual figures. Include correct and failed cases, expected answers, predicted answers and
confidence. Attention slices are not renormalised; cross slices exclude the CLS sink. The positive
difference pooling bias is present before learning. Maps describe behaviour and do not prove causal
reasoning (https://aclanthology.org/N19-1357/). Do not copy historical case numbers or explanations into
a new run. Prefix truncation, directional historical alignment and rare vocabulary remain limitations.'''
    updates['S3_SAVE'] = r'''
import shutil
if not SMOKE and os.path.isdir("/content/drive/MyDrive"):
    dst = "/content/drive/MyDrive/Group69/" + os.path.basename(CFG.out_dir)
    shutil.copytree(CFG.out_dir,dst,dirs_exist_ok=False)
    print("Copied all evidence including checkpoints:",dst)
print("Retain/download:",CFG.out_dir)
'''
    return updates
