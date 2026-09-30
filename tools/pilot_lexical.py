"""Validation-only pilot (run with `colab run --gpu T4 tools/pilot_lexical.py`): does the lexical head help DACT? (The test split is never loaded into a model here.)"""
import os
import subprocess
import sys

subprocess.run("git clone -q -b revision-2 https://github.com/SEKY443/Group69TinyLanguageModel.git /content/repo", shell=True, check=True)
os.chdir("/content/repo")
sys.path.insert(0, "src")
print(subprocess.run("git log --oneline -1", shell=True, capture_output=True, text=True).stdout)
import numpy as np  # noqa: E402
from config import Config, get_device  # noqa: E402
from data import prepare_everything  # noqa: E402
from experiments import run_experiment  # noqa: E402

import data as data_mod  # noqa: E402
data_mod._in_colab = lambda: True   # use the automatic download path (no Drive mount needed for it)
device = get_device()
# configuration selected by the notebook's validation grid in the 2026-09-30 run: small model, no MLM warm-up
base = Config(d_model=128, d_ff=512, n_layers=2, out_dir="/content/pilot")
data = prepare_everything(base)
del data["test_ds"], data["test"]                     # make sure nothing below can touch the test split
seeds = [42, 43, 44]
rows = {}
# pilot 1 used the first three variants with the shared lr (the head was then part of the optimizer's main group);
# pilot 2 added the separate learning rate `lex_lr`. Both ran on a free Colab T4 on 2026-09-30.
for name, cfg in (("DACT", base.but(use_lexical=False)),
                  ("DACT + lexical head, lex_lr 3e-4", base.but(use_lexical=True, lex_lr=3e-4)),
                  ("DACT + lexical head, lex_lr 3e-3", base.but(use_lexical=True, lex_lr=3e-3)),
                  ("DACT + lexical head, lex_lr 1e-2", base.but(use_lexical=True, lex_lr=1e-2)),
                  ("DACT + lexical head, lex_lr 3e-2", base.but(use_lexical=True, lex_lr=3e-2))):
    res = run_experiment(name.replace(" ", "_").replace(",", "").replace("+", "p"), cfg, "dact", data, device, seeds, verbose=False)
    rows[name] = [r["val_acc"] for r in res["runs"]]
    print(f"{name:32s} val acc per seed {np.round(rows[name], 4)} | mean {np.mean(rows[name]):.4f} +- {np.std(rows[name], ddof=1):.4f}", flush=True)
print("reference: B1 TF-IDF + LR val acc 0.5924 (notebook run of 2026-09-30)")
