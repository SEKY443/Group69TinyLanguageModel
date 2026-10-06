"""Section 2: one lock file, run_config.json, baseline validation predictions, the RoBERTa restart rule."""
import ast
import json
import os
import re

import numpy as np
import pytest

import baselines
from config import fresh_path, write_run_config
from evaluate import save_baseline_val

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORT_NAME = {"scikit-learn": "sklearn"}


def _lock_file():
    pins = {}
    for line in open(os.path.join(ROOT, "requirements-lock.txt"), encoding="utf-8"):
        m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line.strip())
        if m:
            pins[IMPORT_NAME.get(m[1], m[1])] = m[2]
    return pins


def _notebook_lock():
    nb = json.load(open(os.path.join(ROOT, "CITS4012_73.ipynb"), encoding="utf-8"))
    for c in nb["cells"]:
        src = "".join(c["source"])
        m = re.search(r"^LOCK = (\{.*?\})$", src, re.S | re.M)
        if c["cell_type"] == "code" and m:
            return ast.literal_eval(m[1])
    raise AssertionError("no LOCK dict in the notebook")


def test_notebook_checks_exactly_the_lock_file():
    assert _notebook_lock() == _lock_file()


def test_fresh_path_never_overwrites(tmp_path):
    p = tmp_path / "a.json"
    assert fresh_path(str(p)) == str(p)
    p.write_text("{}")
    assert fresh_path(str(p)) != str(p) and fresh_path(str(p)).endswith(".json")


def test_run_config(tiny_setup, tmp_path):
    cfg, _ = tiny_setup
    c = cfg.but(out_dir=str(tmp_path / "out"), persist_dir=str(tmp_path / "drive"))
    os.makedirs(c.out_dir)
    json.dump({"files": {"train.jsonl": "abc"}}, open(os.path.join(c.out_dir, "data_manifest.json"), "w"))
    rec = write_run_config(c, "cpu", "test-run", version_mismatch={})
    saved = json.load(open(tmp_path / "out" / "run_config.json"))
    assert saved["config"]["lr"] == c.lr and saved["data"]["files"]["train.jsonl"] == "abc"
    assert {"commit", "environment", "run_id", "time"} <= set(saved)
    assert (tmp_path / "drive" / "run_config.json").is_file()
    write_run_config(c, "cpu", "test-run")          # a second run keeps the first file
    assert len(list((tmp_path / "out").glob("run_config*.json"))) == 2
    assert rec["path"].endswith("run_config.json")


def test_baseline_val_predictions(tiny_setup, tmp_path):
    cfg, data = tiny_setup
    c = cfg.but(out_dir=str(tmp_path / "out"), persist_dir=str(tmp_path / "drive"))
    b1 = baselines.tfidf_lr_baseline(data["train"], data["val"])
    assert np.array_equal((b1["val_prob"] > 0.5).astype(int), b1["val_pred"])
    save_baseline_val(c, "B1_tfidf_lr", data["val"], b1["val_pred"], b1["val_prob"])
    save_baseline_val(c, "B1_tfidf_lr", data["val"], b1["val_pred"], b1["val_prob"])   # re-run: kept, not overwritten
    files = sorted((tmp_path / "out" / "predictions").glob("B1_tfidf_lr_val*.jsonl"))
    assert len(files) == 2
    recs = [json.loads(line) for line in open(files[0], encoding="utf-8")]
    assert len(recs) == len(data["val"]) and recs[0]["split"] == "val" and "probability_candidate_2" in recs[0]
    assert (tmp_path / "drive" / "predictions" / "B1_tfidf_lr_val.jsonl").is_file()


def test_restart_rule(monkeypatch):
    calls = []

    def fake(model_name, tr, va, te, device, out_dir, seed=42, **kw):
        calls.append(seed)
        if len(calls) == 1:
            raise baselines.AtChance(model_name, seed, 0.5)
        return {"val_pred": np.zeros(2)}
    monkeypatch.setattr(baselines, "finetune_pretrained", fake)
    out = baselines.finetune_with_restart("m", [], [], None, "cpu", "x", seed=43, chance_threshold=0.52)
    assert calls == [43, 1043] and out["seed"] == 1043 and out["restarted_from"] == [43]


def test_restart_rule_gives_up(monkeypatch):
    def always_chance(*a, seed=42, **kw):
        raise baselines.AtChance("m", seed, 0.5)
    monkeypatch.setattr(baselines, "finetune_pretrained", always_chance)
    with pytest.raises(RuntimeError, match="42, 1042, 2042"):
        baselines.finetune_with_restart("m", [], [], None, "cpu", "x", seed=42)
