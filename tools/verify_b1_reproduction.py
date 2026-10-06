"""Re-fits B1 (TF-IDF + LR) and checks it against the final run's saved evidence (verification only).

B1 is deterministic for fixed library versions (requirements-lock.txt). Its validation predictions were saved by the
run; its test predictions were not, so the report's B1 test numbers (accuracy, McNemar p against DACT, error
overlap) exist only as notebook outputs and outputs/results/*.csv. This tool:
  1. fits B1 on the training split and compares its validation predictions item by item with
     outputs/predictions/B1_tfidf_lr_val.jsonl.gz (no test data involved);
  2. reads the test labels ONCE through data.load_piqa_with_test_labels, which appends the read to
     <out-dir>/test_access.log, and recomputes B1's test accuracy, the McNemar p-value against the saved DACT
     predictions and the DACT/B1 error overlap, comparing each with final_results.csv / error_overlap.csv.
Nothing here selects or changes anything; it only re-derives already reported numbers.

Usage:  python tools/verify_b1_reproduction.py --data-dir data/piqa --out-dir experiments/final_audit_20261006
"""
import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from baselines import tfidf_lr_baseline, tfidf_lr_predict  # noqa: E402
from config import Config, environment_info  # noqa: E402
from data import load_piqa, load_piqa_with_test_labels  # noqa: E402
from evaluate import error_overlap, mcnemar_exact  # noqa: E402

OUT = os.path.join(ROOT, "outputs")


def saved_predictions(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return np.array([json.loads(line)["predicted_label"] for line in f])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/piqa")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cfg = Config(data_dir=args.data_dir)
    report = {"environment": environment_info("cpu")}

    train, val, _ = load_piqa(cfg)                              # step 1: validation only
    fitted = tfidf_lr_baseline(train, val)
    saved_val = saved_predictions(os.path.join(OUT, "predictions", "B1_tfidf_lr_val.jsonl.gz"))
    yva = np.array([r["label"] for r in val])
    report["val"] = {"selected_C": fitted["C"], "accuracy": float((fitted["val_pred"] == yva).mean()),
                     "items_differing_from_saved_predictions": int((fitted["val_pred"] != saved_val).sum())}

    _, _, test = load_piqa_with_test_labels(cfg, "verify_b1_reproduction.py: re-derive reported B1 test numbers",
                                            args.out_dir)
    yte = np.array([r["label"] for r in test])
    b1 = tfidf_lr_predict(fitted, test)
    dact = saved_predictions(os.path.join(OUT, "predictions", "dact_full_seed42_test.jsonl.gz"))  # best-val seed
    with open(os.path.join(OUT, "results", "final_results.csv"), encoding="utf-8") as f:
        logged = {r["model"]: r for r in csv.DictReader(f)}["B1 TF-IDF + LR"]
    with open(os.path.join(OUT, "results", "error_overlap.csv"), encoding="utf-8") as f:
        logged_overlap = {r[""]: r for r in csv.DictReader(f)}["B1 TF-IDF + LR"]
    overlap = error_overlap(dact, b1, yte)
    report["test"] = {
        "accuracy": float((b1 == yte).mean()), "logged_accuracy": float(logged["test acc"]),
        "mcnemar_p_vs_dact": mcnemar_exact(dact, b1, yte)[2], "logged_mcnemar_p": float(logged["McNemar p vs DACT"]),
        "overlap": overlap, "logged_overlap": {k: float(v) for k, v in logged_overlap.items() if k},
    }
    report["val_matches"] = report["val"]["items_differing_from_saved_predictions"] == 0 and \
        abs(report["val"]["accuracy"] - float(logged["val acc"])) < 1e-12
    report["test_matches"] = abs(report["test"]["accuracy"] - report["test"]["logged_accuracy"]) < 1e-12 and \
        abs(report["test"]["mcnemar_p_vs_dact"] - report["test"]["logged_mcnemar_p"]) < 1e-9 and \
        abs(overlap["only A"] - float(logged_overlap["only DACT"])) < 1e-12 and \
        abs(overlap["oracle"] - float(logged_overlap["oracle"])) < 1e-12
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "b1_reproduction.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("val", "val_matches", "test_matches")}, indent=2))
    return 0 if report["val_matches"] and report["test_matches"] else 1


if __name__ == "__main__":
    sys.exit(main())
