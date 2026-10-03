"""Pre-registered goal-matching experiment (experiments/goal_matching/PREREGISTRATION.md). VALIDATION ONLY.

Trains the main model (control) and the main model + goal-matching block in the same environment, seeds 42/43/44,
then applies the pre-registered acceptance rule. The test split is deleted from memory before any training and is
never evaluated here.

Usage (Colab GPU): python tools/goal_matching_experiment.py --out-dir experiments/goal_matching/run_<stamp>
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

import data as data_mod  # noqa: E402
from config import Config, environment_info, get_device  # noqa: E402
from data import prepare_everything  # noqa: E402
from experiments import run_experiment  # noqa: E402

SEEDS = [42, 43, 44]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data/piqa")
    args = ap.parse_args()
    if os.path.exists(args.out_dir):
        raise FileExistsError("choose a fresh output directory")
    data_mod._in_colab = lambda: True               # allow the automatic PIQA download
    device = get_device()
    # main model = the configuration selected on validation in the final run (small, lexical head, defaults)
    control = Config(d_model=128, d_ff=512, n_layers=2, data_dir=args.data_dir, out_dir=args.out_dir)
    variant = control.but(use_goal_matching=True)
    data = prepare_everything(control)
    for k in ("test", "test_ds"):                    # the test split is never touched in this experiment
        del data[k]
    print(json.dumps(environment_info(device)))
    res = {}
    for name, cfg in (("control_main_model", control), ("goal_matching", variant)):
        r = run_experiment(name, cfg, "dact", data, device, SEEDS, verbose=False)
        res[name] = [run["val_acc"] for run in r["runs"]]
        print(f"{name}: val acc per seed {np.round(res[name], 4).tolist()} | mean {np.mean(res[name]):.4f} "
              f"+- {np.std(res[name], ddof=1):.4f}", flush=True)
    c, g = np.array(res["control_main_model"]), np.array(res["goal_matching"])
    gain = 100 * (g.mean() - c.mean())
    pooled_sd = 100 * math.sqrt((c.std(ddof=1) ** 2 + g.std(ddof=1) ** 2) / 2)
    accept = gain >= 1.0 and gain > 2 * pooled_sd
    out = {"seeds": SEEDS, "val_acc": res, "gain_points": round(gain, 3), "pooled_sd_points": round(pooled_sd, 3),
           "rule": "gain >= 1.0 point AND gain > 2 x pooled seed std (validation)", "accepted": bool(accept)}
    with open(os.path.join(args.out_dir, "goal_matching_decision.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
