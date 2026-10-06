"""The saved result tables are recomputed from the saved per-item predictions of the final run (outputs/).

Every accuracy, seed mean/std, bootstrap CI and McNemar p-value in outputs/results/final_results.csv for the
from-scratch models (DACT, its nine ablations, the BiLSTM) must follow from outputs/predictions/*.jsonl.gz.
The prediction files carry their own gold labels, so test-labels.lst is never opened here.
The repository's own checkers (notebook = src/, no early test-label access, report numbers = result files)
are run as tests too.
"""
import csv
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from evaluate import bootstrap_ci, holm_adjust, mcnemar_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
SCRATCH = {"DACT (full)": "dact_full", "− difference tags": "abl0", "− cross-solution attention": "abl1",
           "− diff bias in pooling": "abl2", "mean pooling": "abl3", "pointwise objective": "abl4",
           "vanilla Transformer": "abl5", "− lexical head": "abl6", "− MLM warm-up": "abl7",
           "diff-bias prior 0": "abl8", "B2 BiLSTM + attention": "b2_bilstm"}
SEEDS = (42, 43, 44)


def _predictions(name, seed, split):
    with gzip.open(os.path.join(OUT, "predictions", f"{name}_seed{seed}_{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    pred = np.array([r["predicted_label"] for r in rows])
    gold = np.array([r["gold_label"] for r in rows])
    assert [r["index"] for r in rows] == list(range(len(rows)))
    assert all(r["correct"] == (r["predicted_label"] == r["gold_label"]) for r in rows)
    return pred, gold


@pytest.fixture(scope="module")
def final_results():
    with open(os.path.join(OUT, "results", "final_results.csv"), encoding="utf-8") as f:
        return {r["model"]: r for r in csv.DictReader(f)}


@pytest.mark.parametrize("model", list(SCRATCH))
def test_seed_accuracies_recompute_from_predictions(model, final_results):
    name = SCRATCH[model]
    val = json.load(open(os.path.join(OUT, "results", f"{name}_val.json"), encoding="utf-8"))
    test = json.load(open(os.path.join(OUT, "results", f"{name}_test.json"), encoding="utf-8"))
    accs = {split: [float((p == g).mean()) for p, g in (_predictions(name, s, split) for s in SEEDS)]
            for split in ("val", "test")}
    assert accs["val"] == pytest.approx([r["val_acc"] for r in val["runs"]], abs=1e-12)
    assert accs["test"] == pytest.approx(test["test"]["runs"], abs=1e-12)
    row = final_results[model]
    for split in ("val", "test"):
        assert float(row[f"{split} acc"]) == pytest.approx(np.mean(accs[split]), abs=1e-12)
        assert float(row[f"{split} std"]) == pytest.approx(np.std(accs[split], ddof=1), abs=1e-12)
    best = SEEDS[int(np.argmax(accs["val"]))]                    # representative run = best validation seed
    assert best == test["test"]["best_val_seed"]
    pred, gold = _predictions(name, best, "test")
    lo, hi = bootstrap_ci(pred, gold)
    assert row["test 95% CI"] == f"[{lo:.3f}, {hi:.3f}]"
    if model != "DACT (full)":
        ref = _predictions("dact_full", 42, "test")[0]
        assert float(row["McNemar p vs DACT"]) == pytest.approx(mcnemar_exact(ref, pred, gold)[2], abs=1e-12)


def test_error_overlap_recomputes_from_predictions():
    ref, gold = _predictions("dact_full", 42, "test")
    with open(os.path.join(OUT, "results", "error_overlap.csv"), encoding="utf-8") as f:
        rows = {r[""]: r for r in csv.DictReader(f)}
    for model, name, seed in (("B2 BiLSTM + attention", "b2_bilstm", 43), ("vanilla Transformer", "abl5", 42)):
        other = _predictions(name, seed, "test")[0]
        ca, cb = ref == gold, other == gold
        assert float(rows[model]["only DACT"]) == pytest.approx((ca & ~cb).mean(), abs=1e-12)
        assert float(rows[model]["oracle"]) == pytest.approx((ca | cb).mean(), abs=1e-12)


def test_holm_adjustment_of_the_nine_ablation_tests(final_results):
    """The report's statement on multiple testing: only the vanilla ablation has p < 0.05 on test, and it does
    not stay below 0.05 after Holm correction over the nine single-model ablation comparisons."""
    names = [m for m in SCRATCH if m not in ("DACT (full)", "B2 BiLSTM + attention")]
    p = np.array([float(final_results[m]["McNemar p vs DACT"]) for m in names])
    adjusted = dict(zip(names, holm_adjust(p)))
    assert [m for m in names if float(final_results[m]["McNemar p vs DACT"]) < 0.05] == ["vanilla Transformer"]
    assert adjusted["vanilla Transformer"] == pytest.approx(0.0608, abs=5e-4)
    assert min(adjusted.values()) > 0.05


def test_mcnemar_exact_known_values():
    y = np.zeros(20, dtype=int)
    a = np.zeros(20, dtype=int)
    b = np.r_[np.ones(6, dtype=int), np.zeros(14, dtype=int)]   # 6 discordant, all in A's favour
    assert mcnemar_exact(a, b, y) == (6, 0, pytest.approx(2 / 2 ** 6))
    assert mcnemar_exact(a, a, y) == (0, 0, 1.0)
    assert holm_adjust([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.parametrize("tool", [["tools/sync_notebook.py", "--check"], ["tools/check_test_access.py"],
                                  ["tools/check_report_numbers.py"]])
def test_repository_checker_passes(tool):
    out = subprocess.run([sys.executable, *tool], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
