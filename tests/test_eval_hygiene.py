"""Section 4: test labels only through final_eval (logged), unrounded deltas, calibration metric."""
import json
import os

import numpy as np
import pytest
import torch

import experiments
from data import HIDDEN_LABEL, load_piqa, load_piqa_with_test_labels, load_split
from evaluate import calibration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_test_labels_are_not_loaded(tiny_setup):
    cfg, data = tiny_setup
    assert all(r["label"] == HIDDEN_LABEL for r in data["test"])
    assert set(data["test_ds"].labels) == {HIDDEN_LABEL}
    assert data["test_labels_loaded"] is False
    real = load_split(cfg.data_dir, "test")
    assert len(set(r["label"] for r in real)) == 2, "the synthetic test file does have both labels"


def test_test_experiment_needs_final_eval(tiny_setup, monkeypatch):
    cfg, data = tiny_setup
    monkeypatch.setitem(experiments.__dict__, "FINAL_EVAL", True)      # even with the flag forced open
    with pytest.raises(RuntimeError, match="final_eval"):
        experiments.test_experiment({"config": cfg.to_dict(), "runs": []}, data, torch.device("cpu"))


def test_final_eval_loads_labels_and_logs(tiny_setup, tmp_path, monkeypatch):
    cfg, _ = tiny_setup
    from data import prepare_everything
    c = cfg.but(out_dir=str(tmp_path / "out"), persist_dir=str(tmp_path / "drive"))
    data = prepare_everything(c)
    monkeypatch.setitem(experiments.__dict__, "FINAL_EVAL", False)
    yte = experiments.final_eval(data, c, reason="unit test")
    real = [r["label"] for r in load_split(cfg.data_dir, "test")]
    assert list(yte) == real and data["test_ds"].labels == real and data["test_labels_loaded"]
    assert experiments.FINAL_EVAL is True
    for root in ("out", "drive"):
        lines = (tmp_path / root / "test_access.log").read_text().splitlines()
        rec = json.loads(lines[-1])
        assert rec["reason"] == "unit test" and rec["n_items"] == len(real) and "commit" in rec and "time" in rec


def test_tools_read_test_labels_only_with_a_log(tiny_setup, tmp_path):
    cfg, _ = tiny_setup
    _, _, hidden = load_piqa(cfg)
    assert all(r["label"] == HIDDEN_LABEL for r in hidden)
    _, _, test = load_piqa_with_test_labels(cfg, "tool test", str(tmp_path))
    assert [r["label"] for r in test] == [r["label"] for r in load_split(cfg.data_dir, "test")]
    assert json.loads((tmp_path / "test_access.log").read_text())["reason"] == "tool test"


def test_no_test_access_log_in_repository():
    """Deliverable 7: no new test evaluation was made in the upgrade (the repository has no test_access.log)."""
    assert not os.path.exists(os.path.join(ROOT, "outputs", "test_access.log"))


def test_calibration_metric():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)               # perfectly calibrated by construction
    assert calibration(p, y)["ece"] < 0.02
    over = np.where(p > 0.5, 0.99, 0.01)                   # same decisions, far too confident
    c = calibration(over, y)
    assert c["ece"] > 0.2 and c["mean_confidence"] == pytest.approx(0.99)
    assert calibration(p, y)["accuracy"] == c["accuracy"]


def test_report_deltas_use_unrounded_means():
    """Table 2 must use differences of the unrounded means (the rounded convention gave -1.0 here, not -0.9)."""
    import csv
    fr = {r["model"]: r for r in csv.DictReader(open(os.path.join(ROOT, "outputs", "results", "final_results.csv"), encoding="utf-8"))}
    d = round(100 * (float(fr["− difference tags"]["test acc"]) - float(fr["DACT (full)"]["test acc"])), 1)
    tex = open(os.path.join(ROOT, "report", "CITS4012_69.tex"), encoding="utf-8").read()
    assert f"$-$ difference tags & $-$1.3 & $-${abs(d):.1f}" in tex and d == -0.9
