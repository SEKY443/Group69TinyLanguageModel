"""End-to-end run of CITS4012_73.ipynb in smoke mode on synthetic data (CPU), with RoBERTa and Qwen stubbed.

Every code cell is executed in order in one namespace, like a kernel. Then the notebook is run again in a fresh
output folder with the same persistent run folder, as on a new Colab VM: nothing may be retrained and the final
results table must be identical.
"""
import json
import os
import shutil
import sys
import types

import numpy as np
import pytest

from conftest import write_synthetic_piqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stubs(ns):
    def finetune_pretrained(name, tr, va, te, device, out_dir, seed=42, **kw):
        rng = np.random.default_rng(seed)
        return {"val_pred": rng.integers(0, 2, len(va)), "val_prob": rng.random(len(va)),
                "test_pred": rng.integers(0, 2, len(te)), "best_val_acc": 0.5}

    def llm_zero_shot(name, rows, device, return_prob=False, **kw):
        rng = np.random.default_rng(1)
        pred = rng.integers(0, 2, len(rows))
        return (pred, rng.random(len(rows))) if return_prob else pred
    ns["finetune_pretrained"], ns["llm_zero_shot"] = finetune_pretrained, llm_zero_shot


def run_notebook(workdir, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("PIQA_SMOKE", "1")
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="stub"))
    nb = json.load(open(os.path.join(ROOT, "CITS4012_73.ipynb"), encoding="utf-8"))
    ns = {"display": lambda x: None, "__name__": "__nb__"}
    trained = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        exec(compile(src, f"cell{i}", "exec"), ns)
        if src.startswith('"""Baselines.'):
            _stubs(ns)
        if src.startswith('"""Training loops'):
            real = ns["train_qa"]
            ns["train_qa"] = lambda *a, **k: (trained.append(a[5]), real(*a, **k))[1]
    import matplotlib.pyplot as plt
    plt.close("all")
    return ns, trained


@pytest.mark.slow
def test_notebook_end_to_end_and_new_vm(tmp_path, monkeypatch):
    write_synthetic_piqa(str(tmp_path / "data" / "piqa"), n_train=300, n_test=80)
    ns, trained = run_notebook(tmp_path, monkeypatch)
    assert trained, "the first run must train"
    assert ns["FINAL_EVAL"] is True
    first = (tmp_path / "outputs_nbsmoke" / "results" / "final_results.csv").read_bytes()
    assert (tmp_path / "Group73" / "runs" / "smoke" / "run_config.json").is_file()
    for name in ("B1_tfidf_lr_val.jsonl", "B3_roberta_seed42_val.jsonl", "B4_qwen_zero_shot_val.jsonl"):
        assert (tmp_path / "outputs_nbsmoke" / "predictions" / name).is_file()

    shutil.move(str(tmp_path / "outputs_nbsmoke"), str(tmp_path / "outputs_vm1"))     # a new VM: local files gone
    ns, trained = run_notebook(tmp_path, monkeypatch)
    assert trained == [], f"retrained on the new VM: {trained}"
    assert (tmp_path / "outputs_nbsmoke" / "results" / "final_results.csv").read_bytes() == first
