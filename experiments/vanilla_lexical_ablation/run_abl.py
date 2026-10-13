"""Vanilla + lexical head ablation, validation only (no test labels are read), exactly as the group's notebook runs ablations.
Variant A: vanilla + lexical head (new). Variant B: plain vanilla re-run in the same session (control)."""
import json, os, subprocess, sys, time

def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)

if not os.path.isdir("/content/g73"):
    sh("git clone -q https://github.com/SEKY443/Group69TinyLanguageModel.git /content/g73")
sh("cd /content/g73 && git checkout -q b4906d3 && git log -1 --format='commit %h'")
sh("pip -q install tokenizers==0.23.1")
os.chdir("/content/g73")
sys.path.insert(0, "/content/g73/src")

from config import Config, get_device  # noqa: E402
from data import download_piqa_release, prepare_everything  # noqa: E402
from experiments import run_experiment  # noqa: E402

CFG = Config().but(out_dir="/content/abl_out")   # fresh folder: the repo ships its own outputs/ evidence, which is never overwritten
os.makedirs(CFG.out_dir, exist_ok=True)
if not all(os.path.isfile(os.path.join(CFG.data_dir, f)) for f in
           ("train.jsonl", "train-labels.lst", "test.jsonl", "test-labels.lst")):
    assert download_piqa_release(CFG), "official PIQA release download/SHA check failed"
DATA = prepare_everything(CFG)
DEVICE = get_device()
print({s: len(DATA[s]) for s in ("train", "val", "test")}, "| device:", DEVICE, flush=True)

SEEDS = [42, 43, 44]
BEST = CFG.but(d_model=128, d_ff=512, n_layers=2, mlm_epochs=10)   # "small + MLM warm-up (10 ep)", the selected configuration
VANILLA = dict(use_diff_tags=False, use_cross_solution=False, pool_mode="mean")
t0 = time.time()
res = {}
for name, kw in (("vanilla_plus_lexical", dict(VANILLA, use_lexical=True)),
                 ("vanilla_control", dict(VANILLA, use_lexical=False))):
    res[name] = run_experiment(name, BEST.but(**kw), "dact", DATA, DEVICE, SEEDS)
    r = res[name]
    print(f"RESULT {name}: val {100*r['val']['mean']:.2f} +- {100*r['val']['std']:.2f} | "
          f"seeds {[round(100*x['val_acc'],2) for x in r['runs']]} | "
          f"params {r['runs'][0]['n_params']} | elapsed {(time.time()-t0)/60:.1f} min", flush=True)
json.dump({k: {"val": v["val"], "runs": [{kk: vv for kk, vv in x.items() if kk != 'ckpt'} for x in v["runs"]]}
           for k, v in res.items()}, open("/content/abl_summary.json", "w"), indent=2)
print("DONE", flush=True)
