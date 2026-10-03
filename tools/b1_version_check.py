"""B1 (TF-IDF + LR) version sensitivity, on the VALIDATION split only (the test split is frozen).

Run the same script in two environments, e.g. one built from requirements-lock.txt and one with newer scipy/numpy,
and compare: the selected C and the validation accuracy change with the library versions. Each run appends one row
to experiments/b1_versions/b1_versions.csv (never overwritten).

Usage:  python tools/b1_version_check.py --data-dir data/piqa --label locked
"""
import argparse
import csv
import importlib.metadata
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from baselines import tfidf_lr_baseline  # noqa: E402
from config import Config  # noqa: E402
from data import load_piqa  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "b1_versions", "b1_versions.csv")
PACKAGES = ("scikit-learn", "scipy", "numpy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/piqa")
    ap.add_argument("--label", required=True, help="name of this environment, e.g. locked / newer")
    args = ap.parse_args()
    cfg = Config(data_dir=args.data_dir)
    train, val, _ = load_piqa(cfg)          # the test split is returned but never used here
    t0 = time.time()
    res = tfidf_lr_baseline(train, val)      # no test rows: only validation predictions are made
    acc = float((res["val_pred"] == np.array([r["label"] for r in val])).mean())
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label, "split": "val", "n": len(val),
           "C": res["C"], "val_acc": round(100 * acc, 4), "python": platform.python_version(),
           **{p: importlib.metadata.version(p) for p in PACKAGES}, "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.isfile(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
