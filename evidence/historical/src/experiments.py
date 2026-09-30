"""Experiment orchestration: multi-seed training, validation-only selection, and a separate final test pass."""
import json
import os

import numpy as np
import torch

from baselines import BiLSTMAttention  # nb-skip
from config import Config, get_device, set_seed  # nb-skip
from data import make_loader, prepare_everything  # nb-skip
from evaluate import accuracy, bootstrap_ci, summarise_seeds  # nb-skip
from model import DACT, count_parameters  # nb-skip
from train import mlm_warmup, predict, train_qa  # nb-skip

ARCHS = {"dact": DACT, "bilstm": BiLSTMAttention}


def build_model(arch, cfg, vocab_size):
    return ARCHS[arch](cfg, vocab_size)


def run_experiment(name, cfg, arch, data, device, seeds, verbose=True):
    """Trains `arch` with each seed. Only the validation split is touched; checkpoints are kept for the test pass."""
    runs = []
    for seed in seeds:
        set_seed(seed)
        run_name = f"{name}_seed{seed}"
        model = build_model(arch, cfg, data["vocab_size"])
        if verbose and seed == seeds[0]:
            print(f"{name}: {count_parameters(model) / 1e6:.2f}M trainable parameters")
        if arch == "dact" and cfg.mlm_epochs > 0:
            model = mlm_warmup(model, data["train_ds"], cfg, device, data["vocab_size"], run_name, verbose)
        train_loader = make_loader(data["train_ds"], cfg, True, device)
        val_loader = make_loader(data["val_ds"], cfg, False, device)
        model, history = train_qa(model, train_loader, val_loader, cfg, device, run_name, verbose)
        val = predict(model, val_loader, device)
        runs.append({"seed": seed, "ckpt": os.path.join(cfg.out_dir, "checkpoints", f"{run_name}.pt"),
                     "val_acc": accuracy(val["pred"], val["label"]), "epochs": len(history)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res = {"name": name, "arch": arch, "config": cfg.to_dict(), "runs": runs,
           "val": summarise_seeds([r["val_acc"] for r in runs])}
    os.makedirs(os.path.join(cfg.out_dir, "results"), exist_ok=True)
    with open(os.path.join(cfg.out_dir, "results", f"{name}_val.json"), "w") as f:
        json.dump(res, f, indent=2)
    if verbose:
        print(f"==> {name}: val acc {res['val']['mean']:.4f} +- {res['val']['std']:.4f}")
    return res


@torch.no_grad()
def test_experiment(res, data, device):
    """FINAL evaluation on the test split: loads every saved checkpoint of an experiment and predicts once."""
    cfg = Config(**res["config"])
    test_loader = make_loader(data["test_ds"], cfg, False, device)
    preds, accs = [], []
    for r in res["runs"]:
        model = build_model(res["arch"], cfg, data["vocab_size"]).to(device)
        model.load_state_dict(torch.load(r["ckpt"], map_location=device))
        out = predict(model, test_loader, device)
        preds.append(out)
        accs.append(accuracy(out["pred"], out["label"]))
        del model
    # representative run for significance tests / qualitative analysis = best validation seed (not best test)
    best = int(np.argmax([r["val_acc"] for r in res["runs"]]))
    lo, hi = bootstrap_ci(preds[best]["pred"], preds[best]["label"])
    res["test"] = summarise_seeds(accs)
    res["test"]["best_val_seed"] = res["runs"][best]["seed"]
    res["test"]["best_val_seed_acc"] = accs[best]
    res["test"]["ci95"] = [lo, hi]
    res["test_preds"] = preds[best]
    with open(os.path.join(cfg.out_dir, "results", f"{res['name']}_test.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "test_preds"}, f, indent=2)
    return res


if __name__ == "__main__":  # nb-skip
    # Local smoke test: tiny subset, 1 epoch, to check that every component runs end-to-end.  # nb-skip
    cfg = Config(epochs=1, mlm_epochs=1, d_model=64, d_ff=128, n_layers=2, batch_size=32, out_dir="outputs_smoke")  # nb-skip
    device = get_device()  # nb-skip
    data = prepare_everything(cfg)  # nb-skip
    for k in ("train_ds", "val_ds", "test_ds"):  # nb-skip
        ds = data[k]; ds.items, ds.labels, ds.rows = ds.items[:256], ds.labels[:256], ds.rows[:256]  # nb-skip
    r = run_experiment("smoke_dact", cfg, "dact", data, device, seeds=[0])  # nb-skip
    r = test_experiment(r, data, device)  # nb-skip
    print(r["test"])  # nb-skip
    for kw in ({"use_diff_tags": False}, {"use_cross_solution": False}, {"pool_mode": "mean"}, {"pool_mode": "attn"}, {"objective": "pointwise", "mlm_epochs": 0}):  # nb-skip
        run_experiment("smoke_" + "_".join(f"{a}{b}" for a, b in kw.items()), cfg.but(**kw), "dact", data, device, seeds=[0])  # nb-skip
    b = run_experiment("smoke_bilstm", cfg, "bilstm", data, device, seeds=[0])  # nb-skip
    print(test_experiment(b, data, device)["test"])  # nb-skip
