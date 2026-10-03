"""Re-scores SAVED test predictions with the train/test duplicate excluded (evaluation only, no retraining).

The duplicate (test index 1545, identical to training-file rows 9975 and 11667) is found by
tools/verify_audit_claims.py. B1 (TF-IDF + LR) is deterministic and its predictions were not saved, so it is
recomputed and checked against the logged accuracy before re-scoring. B3/B4 predictions were not saved; one item
can change their accuracy by at most 1/1838 = 0.054 points.

Writes outputs/results/rescore_without_duplicate.csv.
Usage: python tools/rescore_without_duplicate.py --data-dir <folder with the four PIQA files>
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
from baselines import tfidf_lr_baseline  # noqa: E402
from config import Config  # noqa: E402
from data import load_piqa, load_split  # noqa: E402

DUP_TEST = [1545]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    cfg = Config(data_dir=args.data_dir)
    train, val, test = load_piqa(cfg)
    labels = np.array([r["label"] for r in test])
    keep = np.ones(len(test), bool)
    keep[DUP_TEST] = False

    # where do the duplicated training-file rows sit after the 90/10 split?
    full = load_split(args.data_dir, "train")
    key = lambda r: (r["goal"], r["sol1"], r["sol2"])  # noqa: E731
    dup_key = key(test[DUP_TEST[0]])
    in_train = sum(key(r) == dup_key for r in train)
    in_val = sum(key(r) == dup_key for r in val)
    print(f"duplicate test item {DUP_TEST}: copies in TRAIN = {in_train}, in VAL = {in_val} "
          f"(training-file rows {[i for i, r in enumerate(full) if key(r) == dup_key]})")

    rows = []
    preds_by_exp = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "outputs", "predictions", "*_test.jsonl.gz"))):
        name = os.path.basename(path)[: -len("_test.jsonl.gz")]
        exp, seed = re.match(r"(.+)_seed(\d+)$", name).groups()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            recs = sorted((json.loads(l) for l in f if l.strip()), key=lambda r: r["index"])
        pred = np.array([r["predicted_label"] for r in recs])
        assert len(pred) == len(test) and all(r["gold_label"] == y for r, y in zip(recs, labels))
        preds_by_exp.setdefault(exp, []).append((int(seed), pred))

    b1 = tfidf_lr_baseline(train, val, test)
    b1_acc = float((b1["test_pred"] == labels).mean())
    print(f"B1 recomputed: test acc {b1_acc:.4f} (logged 0.6121)")
    assert abs(b1_acc - 0.6121) < 6e-4, "B1 does not reproduce the logged accuracy"
    preds_by_exp["B1_tfidf_lr"] = [(0, b1["test_pred"])]
    preds_by_exp["B0_majority"] = [(0, np.full(len(test), int(np.mean([r["label"] for r in train]) >= 0.5)))]

    for exp, runs in sorted(preds_by_exp.items()):
        full_acc = [float((p == labels).mean()) for _, p in runs]
        excl_acc = [float((p[keep] == labels[keep]).mean()) for _, p in runs]
        dup_correct = [int(p[DUP_TEST[0]] == labels[DUP_TEST[0]]) for _, p in runs]
        rows.append({"experiment": exp, "n_seeds": len(runs),
                     "test_acc_all_1838": round(float(np.mean(full_acc)), 6),
                     "test_acc_without_duplicate_1837": round(float(np.mean(excl_acc)), 6),
                     "difference_points": round(100 * (np.mean(excl_acc) - np.mean(full_acc)), 4),
                     "duplicate_item_correct_in_seeds": f"{sum(dup_correct)}/{len(runs)}"})
    out = os.path.join(ROOT, "outputs", "results", "rescore_without_duplicate.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['experiment']:14s} seeds {r['n_seeds']}  all {100*r['test_acc_all_1838']:.2f}  "
              f"without duplicate {100*r['test_acc_without_duplicate_1837']:.2f}  "
              f"diff {r['difference_points']:+.3f}  duplicate correct {r['duplicate_item_correct_in_seeds']}")
    print("written", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
