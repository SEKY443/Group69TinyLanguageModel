"""Calibration on the VALIDATION split (section 4.3 of the upgrade prompt): ECE, Brier score and NLL next to accuracy.

Reads only the saved validation predictions (outputs/predictions/*_val.jsonl.gz: DACT, ablations, B2), so no model is
re-run and no test file is opened. B1 (TF-IDF + LR) did not save validation probabilities in the A100 run; with
--data-dir it is recomputed on validation (deterministic; reproduces the logged accuracy with requirements-lock.txt).
B3/B4 need a GPU and are not included; the notebook now saves their validation probabilities for the next run.

Writes experiments/calibration/val_calibration.csv (a new time-stamped file if it exists).
Usage: python tools/calibration.py [--data-dir data/piqa]
"""
import argparse
import csv
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import fresh_path  # noqa: E402
from evaluate import calibration  # noqa: E402

NAMES = {"dact_full": "DACT (full)", "abl0": "− difference tags", "abl1": "− cross-solution attention",
         "abl2": "− diff bias in pooling", "abl3": "mean pooling", "abl4": "pointwise objective",
         "abl5": "vanilla Transformer", "abl6": "− lexical head", "abl7": "diff-bias prior 0",
         "b2_bilstm": "B2 BiLSTM + attention"}


def read_val(path):
    assert path.endswith("_val.jsonl.gz"), "only validation predictions may be read here"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        recs = sorted((json.loads(line) for line in f if line.strip()), key=lambda r: r["index"])
    assert all(r["split"] == "val" for r in recs)
    return np.array([r["probability_candidate_2"] for r in recs]), np.array([r["gold_label"] for r in recs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="PIQA folder; adds B1 recomputed on validation")
    args = ap.parse_args()
    per_exp = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "outputs", "predictions", "*_val.jsonl.gz"))):
        exp, seed = re.match(r"(.+)_seed(\d+)_val\.jsonl\.gz$", os.path.basename(path)).groups()
        if exp in NAMES:
            per_exp.setdefault(exp, []).append((int(seed), *read_val(path)))
    rows = []
    for exp, runs in per_exp.items():
        stats = [calibration(p, y) for _, p, y in runs]
        row = {"model": NAMES[exp], "source": f"outputs/predictions/{exp}_seed*_val.jsonl.gz", "n_seeds": len(runs)}
        for k in stats[0]:
            vals = [s[k] for s in stats]
            row[k] = round(float(np.mean(vals)), 4)
            row[k + "_std"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else ""
        rows.append(row)
    if args.data_dir:
        from baselines import tfidf_lr_baseline
        from config import Config
        from data import load_piqa
        train, val, _ = load_piqa(Config(data_dir=args.data_dir))     # test labels are not loaded
        b1 = tfidf_lr_baseline(train, val)
        st = calibration(b1["val_prob"], np.array([r["label"] for r in val]))
        rows.append({"model": "B1 TF-IDF + LR", "source": "recomputed on validation (tfidf_lr_baseline)", "n_seeds": 1,
                     **{k: round(v, 4) for k, v in st.items()}, **{k + "_std": "" for k in st}})
    order = list(NAMES.values()) + ["B1 TF-IDF + LR"]
    rows.sort(key=lambda r: order.index(r["model"]))
    out = fresh_path(os.path.join(ROOT, "experiments", "calibration", "val_calibration.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ["model", "n_seeds", "accuracy", "accuracy_std", "ece", "ece_std", "mean_confidence",
              "mean_confidence_std", "brier", "brier_std", "nll", "nll_std", "source"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{'model':30s} {'acc':>6s} {'ECE':>6s} {'conf':>6s} {'Brier':>6s} {'NLL':>6s}")
    for r in rows:
        print(f"{r['model']:30s} {100 * r['accuracy']:6.2f} {100 * r['ece']:6.2f} {100 * r['mean_confidence']:6.2f} "
              f"{r['brier']:6.4f} {r['nll']:6.4f}")
    print("written", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
