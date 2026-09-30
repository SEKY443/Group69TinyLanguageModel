"""Evaluation metrics: accuracy, bootstrap confidence intervals, McNemar's exact test, seed aggregation."""
import math
import json
import os

import numpy as np


def save_predictions(path, rows, prediction, run, split):
    """Persist only executed predictions; probabilities are omitted if unavailable (e.g. B0)."""
    if len(rows) != len(prediction["pred"]):
        raise ValueError("Prediction count differs from dataset size")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x", encoding="utf-8") as stream:
        for i, (row, pred) in enumerate(zip(rows, prediction["pred"])):
            rec = {"run": run, "split": split, "index": i, "goal": row["goal"],
                   "candidate_1": row["sol1"], "candidate_2": row["sol2"],
                   "gold_label": row["label"], "predicted_label": int(pred),
                   "correct": int(pred) == row["label"]}
            if "prob" in prediction:
                p = float(prediction["prob"][i])
                rec.update(probability_candidate_1=1-p, probability_candidate_2=p)
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")


def holm_adjust(pvalues):
    """Holm family-wise correction; call on an explicitly declared comparison family."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def accuracy(pred, label):
    return float((np.asarray(pred) == np.asarray(label)).mean())


def bootstrap_ci(pred, label, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for accuracy (resampling test items)."""
    correct = (np.asarray(pred) == np.asarray(label)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(1)
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def mcnemar_exact(pred_a, pred_b, label):
    """Exact two-sided McNemar test on the discordant pairs (binomial with p=0.5).

    b = items A gets right and B gets wrong, c = the reverse. Returns (b, c, p_value).
    """
    ca = np.asarray(pred_a) == np.asarray(label)
    cb = np.asarray(pred_b) == np.asarray(label)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * p)


def summarise_seeds(accs):
    accs = np.asarray(accs, dtype=float)
    return {"mean": float(accs.mean()), "std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "runs": [float(a) for a in accs]}

