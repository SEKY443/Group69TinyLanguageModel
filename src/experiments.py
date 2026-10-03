"""Experiment orchestration: multi-seed training, validation-only selection, and a separate final test pass."""
import glob
import json
import os
import shutil
import time

import numpy as np
import torch

from baselines import BiLSTMAttention  # nb-skip
from config import Config, get_device, set_seed, write_progress  # nb-skip
from data import make_loader, prepare_everything  # nb-skip
from evaluate import accuracy, bootstrap_ci, summarise_seeds, save_predictions  # nb-skip
from model import DACT, count_parameters  # nb-skip
from train import mlm_warmup, predict, train_qa  # nb-skip

ARCHS = {"dact": DACT, "bilstm": BiLSTMAttention}


def build_model(arch, cfg, vocab_size):
    return ARCHS[arch](cfg, vocab_size)


def _identity(config):
    """The parts of a config that define an experiment: run_seed is set per seed; out_dir and persist_dir are only
    locations (a run restored on a new VM, or into a new folder, is the same experiment)."""
    return {k: v for k, v in config.items() if k not in ("run_seed", "out_dir", "persist_dir")}


def _load_finished(name, cfg, seeds):
    """Returns the saved validation result of `name` if it finished with the same config, seeds and checkpoints."""
    path = os.path.join(cfg.out_dir, "results", f"{name}_val.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        res = json.load(f)
    same = _identity(res["config"]) == _identity(cfg.to_dict()) and [r["seed"] for r in res["runs"]] == list(seeds)
    for r in res["runs"]:  # checkpoints are found relative to the current out_dir (it may be restored on a new VM)
        r["ckpt"] = _ckpt_path(cfg, f"{name}_seed{r['seed']}")
    return res if same and all(os.path.isfile(r["ckpt"]) for r in res["runs"]) else None


def _ckpt_path(cfg, run_name):
    return os.path.join(cfg.out_dir, "checkpoints", f"{run_name}.pt")


# ---- persistence across VMs -------------------------------------------------------------------------------
def _artifacts(root, name, run_name=None):
    """Files that belong to experiment `name` (or only to one seed, `run_name`) below `root`."""
    stem = run_name or f"{name}_seed*"
    patterns = [f"results/{stem}_done.json", f"logs/{stem}.jsonl", f"logs/{stem}_mlm.jsonl",
                f"predictions/{stem}_val.jsonl", f"checkpoints/{stem}.pt", f"checkpoints/{stem}.pt.json"]
    if run_name is None:
        patterns.append(f"results/{name}_val.json")
    return sorted({p for pat in patterns for p in glob.glob(os.path.join(root, pat))})


def _copy_new(files, src_root, dst_root):
    """Copies files keeping their relative paths; an existing destination is never overwritten (evidence)."""
    for path in files:
        dst = os.path.join(dst_root, os.path.relpath(path, src_root))
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(path, dst)


def persist(cfg, name, run_name=None):
    """Copies a finished seed (or experiment) to cfg.persist_dir, e.g. Google Drive, so a lost VM loses nothing."""
    if cfg.persist_dir:
        _copy_new(_artifacts(cfg.out_dir, name, run_name), cfg.out_dir, cfg.persist_dir)


def restore(cfg, name):
    """On a new VM: copies everything already persisted for `name` back into cfg.out_dir."""
    if cfg.persist_dir and os.path.isdir(cfg.persist_dir):
        _copy_new(_artifacts(cfg.persist_dir, name), cfg.persist_dir, cfg.out_dir)


def _load_seed(cfg, run_name):
    """A seed that finished earlier (completion marker with the same config and its checkpoint), or None."""
    marker = os.path.join(cfg.out_dir, "results", f"{run_name}_done.json")
    if not os.path.isfile(marker):
        return None
    with open(marker, encoding="utf-8") as f:
        done = json.load(f)
    if _identity(done["config"]) != _identity(cfg.to_dict()):
        raise FileExistsError(f"{run_name} already finished with a different config in {cfg.out_dir}. "
                              "Use a new out_dir / RUN_ID; evidence is never overwritten.")
    done["run"]["ckpt"] = _ckpt_path(cfg, run_name)
    return done["run"] if os.path.isfile(done["run"]["ckpt"]) else None


def _check_stale_summary(cfg, name):
    """A summary that cannot be resumed: refuse if it belongs to another config, otherwise keep it renamed."""
    path = os.path.join(cfg.out_dir, "results", f"{name}_val.json")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    if _identity(old["config"]) != _identity(cfg.to_dict()):
        raise FileExistsError(f"{name} already finished with a different config in {cfg.out_dir}. "
                              "Use a new out_dir / RUN_ID; evidence is never overwritten.")
    os.replace(path, path[:-len(".json")] + f".stale-{time.strftime('%Y%m%d_%H%M%S')}.json")


def _quarantine_partial(cfg, name, run_name):
    """Renames the files of an interrupted seed to *.partial-<time>.* so training can restart without
    overwriting (or deleting) the evidence of the failed attempt."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    moved = []
    for path in _artifacts(cfg.out_dir, name, run_name):
        if path.endswith("_done.json"):
            continue
        base, ext = (path[:-len(".pt.json")], ".pt.json") if path.endswith(".pt.json") else os.path.splitext(path)
        target = f"{base}.partial-{stamp}{ext}"
        os.replace(path, target)
        moved.append(target)
    return moved


def run_experiment(name, cfg, arch, data, device, seeds, verbose=True, resume=True):
    """Trains `arch` with each seed. Only the validation split is touched; checkpoints are kept for the test pass.

    With resume=True, finished work (same config, seeds and checkpoints) is loaded instead of retrained: a whole
    experiment, or single seeds of a half-finished one. With cfg.persist_dir set (e.g. Google Drive), every
    finished seed is copied there and restored on a new VM, so a reclaimed Colab session loses at most one seed.
    The files of an interrupted seed are renamed to *.partial-<time>.* and kept as evidence.
    """
    if resume:
        restore(cfg, name)
        done = _load_finished(name, cfg, seeds)
        if done is not None:
            if verbose:
                print(f"==> {name}: already finished, loaded (val acc {done['val']['mean']:.4f})")
            return done
        _check_stale_summary(cfg, name)
    runs = []
    for seed in seeds:
        cfg = cfg.but(run_seed=seed)
        run_name = f"{name}_seed{seed}"
        if resume:
            finished = _load_seed(cfg, run_name)
            if finished is not None:
                if verbose:
                    print(f"[{run_name}] already finished, loaded (val acc {finished['val_acc']:.4f})")
                runs.append(finished)
                continue
            moved = _quarantine_partial(cfg, name, run_name)
            if moved and verbose:
                print(f"[{run_name}] interrupted earlier; kept its files as {[os.path.basename(m) for m in moved]}")
        write_progress(cfg, step="experiment", experiment=name, seed=seed)
        set_seed(seed, cfg.deterministic)
        model = build_model(arch, cfg, data["vocab_size"])
        if verbose and seed == seeds[0]:
            print(f"{name}: {count_parameters(model) / 1e6:.2f}M trainable parameters")
        if arch == "dact" and cfg.mlm_epochs > 0:
            model = mlm_warmup(model, data["train_ds"], cfg, device, data["vocab_size"], run_name, verbose)
        train_loader = make_loader(data["train_ds"], cfg, True, device)
        val_loader = make_loader(data["val_ds"], cfg, False, device)
        model, history = train_qa(model, train_loader, val_loader, cfg, device, run_name, verbose)
        val = predict(model, val_loader, device, cfg)
        save_predictions(os.path.join(cfg.out_dir, "predictions", f"{run_name}_val.jsonl"),
                         data["val_ds"].rows, val, run_name, "val")
        runs.append({"seed": seed, "ckpt": _ckpt_path(cfg, run_name),
                     "val_acc": accuracy(val["pred"], val["label"]), "val_loss": val["loss"],
                     "n_params": count_parameters(model), "epochs": len(history)})
        os.makedirs(os.path.join(cfg.out_dir, "results"), exist_ok=True)
        with open(os.path.join(cfg.out_dir, "results", f"{run_name}_done.json"), "w", encoding="utf-8") as f:
            json.dump({"run": runs[-1], "config": cfg.to_dict()}, f, indent=2)
        persist(cfg, name, run_name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res = {"name": name, "arch": arch, "config": cfg.to_dict(), "runs": runs,
           "val": summarise_seeds([r["val_acc"] for r in runs])}
    os.makedirs(os.path.join(cfg.out_dir, "results"), exist_ok=True)
    with open(os.path.join(cfg.out_dir, "results", f"{name}_val.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    persist(cfg, name)
    if verbose:
        print(f"==> {name}: val acc {res['val']['mean']:.4f} +- {res['val']['std']:.4f}")
    return res


@torch.no_grad()
def test_experiment(res, data, device):
    """FINAL evaluation on the test split: loads every saved checkpoint of an experiment and predicts once."""
    # Guard: in the notebook FINAL_EVAL is False until the final-results step, so test labels cannot be read earlier.
    # (Run as a plain module, e.g. the smoke test below, FINAL_EVAL is undefined and the check is skipped.)
    if not globals().get("FINAL_EVAL", True):
        raise RuntimeError("test_experiment called before the final-results step (FINAL_EVAL is False)")
    cfg = Config(**res["config"])
    test_loader = make_loader(data["test_ds"], cfg, False, device)
    preds, accs = [], []
    for r in res["runs"]:
        model = build_model(res["arch"], cfg, data["vocab_size"]).to(device)
        model.load_state_dict(torch.load(r["ckpt"], map_location=device, weights_only=True))
        out = predict(model, test_loader, device, cfg)
        save_predictions(os.path.join(cfg.out_dir, "predictions", f"{res['name']}_seed{r['seed']}_test.jsonl"),
                         data["test_ds"].rows, out, f"{res['name']}_seed{r['seed']}", "test")
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
    with open(os.path.join(cfg.out_dir, "results", f"{res['name']}_test.json"), "w", encoding="utf-8") as f:
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
    for kw in ({"use_diff_tags": False}, {"use_cross_solution": False}, {"pool_mode": "mean"}, {"pool_mode": "attn"}, {"objective": "pointwise", "mlm_epochs": 0}, {"tag_bias_init": 0.0}):  # nb-skip
        run_experiment("smoke_" + "_".join(f"{a}{b}" for a, b in kw.items()), cfg.but(**kw), "dact", data, device, seeds=[0])  # nb-skip
    b = run_experiment("smoke_bilstm", cfg, "bilstm", data, device, seeds=[0])  # nb-skip
    print(test_experiment(b, data, device)["test"])  # nb-skip
