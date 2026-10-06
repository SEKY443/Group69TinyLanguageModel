"""Proves that the code changes made after the final run cannot have changed its results.

The saved outputs of CITS4012_73.ipynb were produced by commit b880108. Later commits changed the module cells
(training diagnostics, an off-by-default length-bucketing sampler, a B4 out-of-memory fallback, a B3 encoding
helper). This tool trains DACT (masked-LM warm-up + QA training) and the BiLSTM with the b880108 sources and with
the current src/, on the same synthetic data, the same fixed tokenizer and the same seed, deterministically on CPU,
and compares every parameter and every validation probability bit for bit, and the per-epoch log fields
(identical except the logged training loss, whose accumulation precision changed: float32 on the device).

Usage:  python tools/verify_post_run_equivalence.py [--commit b880108] [--out <file>]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ("config", "data", "model", "baselines", "train", "evaluate", "experiments", "viz")

WORKER = r'''
import os, sys, tempfile, torch
src, tok_path, data_dir, out = sys.argv[1:5]
sys.path.insert(0, src)
from config import Config, set_seed
from data import load_piqa, make_loader, PIQADataset
from model import DACT
from baselines import BiLSTMAttention
from train import train_qa, mlm_warmup, predict
from tokenizers import Tokenizer
cfg = Config(data_dir=data_dir, out_dir=tempfile.mkdtemp(), vocab_size=300, min_frequency=1, d_model=32, n_heads=2,
             n_layers=2, d_ff=64, max_len=64, max_goal_len=16, max_sol_len=24, batch_size=32, epochs=4, patience=4,
             lex_buckets=2 ** 10, num_workers=0, amp=False, deterministic=True, mlm_epochs=2)
train, val, _ = load_piqa(cfg)
tok = Tokenizer.from_file(tok_path)
train_ds, val_ds = PIQADataset(train, tok, cfg), PIQADataset(val, tok, cfg)
dev, res = torch.device("cpu"), {}
for arch in ("dact", "bilstm"):
    c = cfg if arch == "dact" else cfg.but(lr=1e-3, dropout=0.3, mlm_epochs=0)
    set_seed(42, True)
    m = DACT(c, tok.get_vocab_size()) if arch == "dact" else BiLSTMAttention(c, tok.get_vocab_size())
    if c.mlm_epochs:
        m = mlm_warmup(m, train_ds, c, dev, tok.get_vocab_size(), arch, verbose=False)
    m, hist = train_qa(m, make_loader(train_ds, c, True, dev), make_loader(val_ds, c, False, dev), c, dev, arch, verbose=False)
    p = predict(m, make_loader(val_ds, c, False, dev), dev, c)
    res[arch] = {"state": m.state_dict(), "prob": torch.tensor(p["prob"]),
                 "history": [(h["epoch"], h["train_loss"], h["train_acc"], h["val_acc"], h["val_loss"]) for h in hist]}
torch.save(res, out)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="b880108")
    ap.add_argument("--out", default=None, help="optional JSON file for the verdict")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import torch
    from conftest import write_synthetic_piqa
    from config import Config
    from data import load_piqa, train_tokenizer

    work = tempfile.mkdtemp()
    old_src = os.path.join(work, "old_src")
    os.makedirs(old_src)
    for name in MODULES:
        code = subprocess.run(["git", "show", f"{args.commit}:src/{name}.py"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout
        open(os.path.join(old_src, f"{name}.py"), "w", encoding="utf-8").write(code)
    data_dir = write_synthetic_piqa(os.path.join(work, "data"), n_train=400, n_test=40)
    cfg = Config(data_dir=data_dir, vocab_size=300, min_frequency=1)
    tok_path = os.path.join(work, "tokenizer.json")       # one fixed tokenizer: BPE training is not deterministic
    train_tokenizer(load_piqa(cfg)[0], cfg, path=tok_path)
    worker = os.path.join(work, "worker.py")
    open(worker, "w", encoding="utf-8").write(WORKER)
    results = {}
    for label, src in (("run_commit", old_src), ("current", os.path.join(ROOT, "src"))):
        out = os.path.join(work, f"{label}.pt")
        subprocess.run([sys.executable, worker, src, tok_path, data_dir, out], check=True, cwd=work)
        results[label] = torch.load(out)
    verdict = {}
    for arch in ("dact", "bilstm"):
        a, b = results["run_commit"][arch], results["current"][arch]
        verdict[arch] = {
            "parameters_bit_identical": a["state"].keys() == b["state"].keys()
                                        and all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"]),
            "validation_probabilities_identical": torch.equal(a["prob"], b["prob"]),
            "epochs": len(a["history"]),
            # per logged field: largest absolute difference over epochs. train_loss is accumulated in float32 on the
            # device by the current code and in Python float64 by the run commit, so it may differ in the 7th digit.
            "max_abs_difference_per_logged_field": {
                field: max(abs(x[i] - y[i]) for x, y in zip(a["history"], b["history"]))
                for i, field in enumerate(("epoch", "train_loss", "train_acc", "val_acc", "val_loss"))}}
    verdict = {"run_commit": args.commit, "compared_with": "working tree src/",
               "torch": torch.__version__, "setting": "CPU, deterministic algorithms, seed 42, synthetic data",
               "models": verdict}
    verdict["trained_models_and_predictions_identical"] = all(
        m["parameters_bit_identical"] and m["validation_probabilities_identical"]
        and all(d == 0 for f, d in m["max_abs_difference_per_logged_field"].items() if f != "train_loss")
        and m["max_abs_difference_per_logged_field"]["train_loss"] < 1e-5 for m in verdict["models"].values())
    print(json.dumps(verdict, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(verdict, f, indent=2)
    return 0 if verdict["trained_models_and_predictions_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
