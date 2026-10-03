"""Checkpoint selection: maximum validation accuracy (current rule) vs minimum validation loss, from existing logs.

For every from-scratch training log of the final T4 run (outputs/logs) and the archived A100 run, it compares the
epoch each rule picks and the validation accuracy at that epoch. Validation only: no test number is read. Note that the
accuracy rule is evaluated on the same validation split it selects on, so its advantage here is optimistic by design;
the useful question is how often the two rules disagree and by how much.

Writes experiments/checkpoint_selection/selection_vs_loss.csv. Usage: python tools/selection_analysis.py
"""
import csv
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {"T4 b880108": os.path.join(ROOT, "outputs", "logs"),
        "A100 0d70cdc": os.path.join(ROOT, "experiments", "a100_run_0d70cdc", "outputs", "logs")}


def main():
    rows = []
    for run, folder in RUNS.items():
        for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
            name = os.path.basename(path)[:-6]
            if not re.fullmatch(r"(dact_full|abl\d+|hp\d+|b2_bilstm)_seed\d+", name):
                continue                                   # skip MLM warm-up logs and pretrained baselines
            ep = [json.loads(l) for l in open(path, encoding="utf-8") if '"event": "epoch"' in l]
            if not ep:
                continue
            by_acc = max(ep, key=lambda r: (r["val_acc"], -r["epoch"]))    # first epoch wins ties, as in train_qa
            by_loss = min(ep, key=lambda r: (r["val_loss"], r["epoch"]))
            rows.append({"run": run, "log": name, "epochs": len(ep),
                         "acc_rule_epoch": by_acc["epoch"], "acc_rule_val_acc": round(by_acc["val_acc"], 4),
                         "loss_rule_epoch": by_loss["epoch"], "loss_rule_val_acc": round(by_loss["val_acc"], 4),
                         "difference_points": round(100 * (by_acc["val_acc"] - by_loss["val_acc"]), 2)})
    out = os.path.join(ROOT, "experiments", "checkpoint_selection", "selection_vs_loss.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for run in RUNS:
        r = [x for x in rows if x["run"] == run]
        d = np.array([x["difference_points"] for x in r])
        same = sum(x["acc_rule_epoch"] == x["loss_rule_epoch"] for x in r)
        earlier = sum(x["loss_rule_epoch"] < x["acc_rule_epoch"] for x in r)
        print(f"{run}: {len(r)} training logs | same epoch {same} | loss rule earlier {earlier} | "
              f"val acc lost by the loss rule: mean {d.mean():.2f}, median {np.median(d):.2f}, max {d.max():.2f} points")
    print("written", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
