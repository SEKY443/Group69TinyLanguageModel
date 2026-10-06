"""Significance and capacity table of the nine ablations, from the final run's saved predictions (no retraining).

For every ablation: the exact McNemar test against the full DACT on validation and on test (representative runs =
the seed with the best validation accuracy, as in the notebook), Holm-adjusted p-values over the declared family of
the nine ablations (separately per split), whether every ablation seed lies below every full-model seed, and the
change in trainable parameters. The prediction files carry their own gold labels; test-labels.lst is not opened.

Writes outputs/results/ablation_significance.csv.   Usage: python tools/ablation_significance.py
"""
import csv
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluate import holm_adjust, mcnemar_exact  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
ABLATIONS = {"abl0": "− difference tags", "abl1": "− cross-solution attention", "abl2": "− diff bias in pooling",
             "abl3": "mean pooling", "abl4": "pointwise objective", "abl5": "vanilla Transformer",
             "abl6": "− lexical head", "abl7": "− MLM warm-up", "abl8": "diff-bias prior 0"}


def predictions(name, seed, split):
    with gzip.open(os.path.join(OUT, "predictions", f"{name}_seed{seed}_{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return np.array([r["predicted_label"] for r in rows]), np.array([r["gold_label"] for r in rows])


def summary(name):
    res = json.load(open(os.path.join(OUT, "results", f"{name}_val.json"), encoding="utf-8"))
    test = json.load(open(os.path.join(OUT, "results", f"{name}_test.json"), encoding="utf-8"))
    best = max(res["runs"], key=lambda r: r["val_acc"])
    return {"best_seed": best["seed"], "n_params": best["n_params"],
            "val": [r["val_acc"] for r in res["runs"]], "test": test["test"]["runs"]}


def main():
    full = summary("dact_full")
    rows = []
    for name, label in ABLATIONS.items():
        s = summary(name)
        row = {"experiment": name, "ablation": label, "params": s["n_params"],
               "delta_params": s["n_params"] - full["n_params"]}
        for split in ("val", "test"):
            ref, gold = predictions("dact_full", full["best_seed"], split)
            pred, _ = predictions(name, s["best_seed"], split)
            b, c, p = mcnemar_exact(ref, pred, gold)
            row.update({f"{split}_delta_points": 100 * (np.mean(s[split]) - np.mean(full[split])),
                        f"{split}_mcnemar_b_dact_only": b, f"{split}_mcnemar_c_ablation_only": c,
                        f"{split}_mcnemar_p": p,
                        f"{split}_all_seeds_below_all_full_seeds": bool(max(s[split]) < min(full[split]))})
        rows.append(row)
    for split in ("val", "test"):
        for row, adj in zip(rows, holm_adjust([r[f"{split}_mcnemar_p"] for r in rows])):
            row[f"{split}_mcnemar_p_holm"] = float(adj)
    path = os.path.join(OUT, "results", "ablation_significance.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['ablation']:28s} dParams {r['delta_params']:>8d} | val {r['val_delta_points']:+.2f} p {r['val_mcnemar_p']:.4f} "
              f"holm {r['val_mcnemar_p_holm']:.3f} | test {r['test_delta_points']:+.2f} p {r['test_mcnemar_p']:.4f} "
              f"holm {r['test_mcnemar_p_holm']:.3f} | below on both: "
              f"{r['val_all_seeds_below_all_full_seeds'] and r['test_all_seeds_below_all_full_seeds']}")
    print("written:", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
