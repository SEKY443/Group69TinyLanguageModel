"""Unit tests of the model and data pipeline (section 3.1 of the upgrade prompt). CPU only, synthetic data."""
import json
import os
import sys

import numpy as np
import pytest
import torch

import experiments
from config import Config
from data import TAG_DIFF, TAG_SHARED, PAD, collate, diff_tags, encode_example, truncate_around_diff
from evaluate import save_predictions
from model import DACT
from train import make_optimizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _batch(data, n=8):
    ds = data["val_ds"]
    return collate([ds[i] for i in range(n)])


def _model(cfg, data, **kw):
    torch.manual_seed(0)
    return DACT(cfg.but(**kw), data["vocab_size"]).eval()


# ---------------------------------------------------------------- masks and padding
def test_padding_masks(tiny_setup):
    cfg, data = tiny_setup
    b = _batch(data)
    assert torch.equal(b["attn_mask"], b["input_ids"] != PAD)
    assert not (b["sol_mask"] & ~b["attn_mask"]).any(), "solution positions must be real tokens"
    assert (b["segment"][b["sol_mask"]] == 1).all()


def test_padding_does_not_change_scores(tiny_setup):
    cfg, data = tiny_setup
    model = _model(cfg, data)
    ds = data["val_ds"]
    alone = model(collate([ds[0]]))["logits"]
    longest = max(range(len(ds)), key=lambda i: max(len(s[0]) for s in ds.items[i]))
    padded = model(collate([ds[0], ds[longest]]))["logits"][:1]
    assert torch.allclose(alone, padded, atol=1e-5)


# ---------------------------------------------------------------- swap symmetry
def test_swapping_options_swaps_scores(tiny_setup):
    """With symmetric alignment the model is exactly equivariant to the option order."""
    cfg, data = tiny_setup
    c = cfg.but(symmetric_diff_tags=True)
    model = _model(c, data, symmetric_diff_tags=True)
    rows = data["val"][:8]
    swapped = [{**r, "sol1": r["sol2"], "sol2": r["sol1"], "label": 1 - r["label"]} for r in rows]
    enc = lambda rs: collate([(encode_example(r, data["tok"], c), r["label"], i) for i, r in enumerate(rs)])  # noqa: E731
    a, b = model(enc(rows))["logits"], model(enc(swapped))["logits"]
    assert torch.allclose(a, b.flip(-1), atol=1e-5)


def test_symmetric_diff_tags_are_order_free():
    a, b = [5, 6, 7, 8, 9], [5, 9, 7, 8, 6]
    ta, tb = diff_tags(a, b, symmetric=True)
    tb2, ta2 = diff_tags(b, a, symmetric=True)
    assert (ta, tb) == (ta2, tb2)


# ---------------------------------------------------------------- truncation
def test_truncate_keeps_short_sequences():
    ids, tags = list(range(10)), [TAG_SHARED] * 10
    assert truncate_around_diff(ids, tags, 96) == (ids, tags)


def test_truncate_cobbler_case_difference_past_96():
    """Two long, near-identical solutions whose only difference lies after token 96 (cf. test item 1433)."""
    common = list(range(10, 130))
    a, b = common + [500, 501] + [7] * 5, common + [600, 601] + [7] * 5
    ta, tb = diff_tags(a, b)
    a_t, ta_t = truncate_around_diff(a, ta, 96)
    b_t, tb_t = truncate_around_diff(b, tb, 96)
    assert len(a_t) == 96 and 500 in a_t and 600 in b_t
    assert TAG_DIFF in ta_t and a_t != b_t
    assert a[:96] == b[:96], "plain prefix truncation would make the two inputs identical"


def test_truncate_without_difference_is_prefix():
    ids, tags = list(range(200)), [TAG_SHARED] * 200
    assert truncate_around_diff(ids, tags, 96) == (ids[:96], tags[:96])


# ---------------------------------------------------------------- lexical head
def test_lexical_bucket_zero_stays_zero(tiny_setup):
    """Bucket 0 (non-solution positions) gets no gradient, so it keeps its initial weight 0 through training
    (padding_idx only blocks the gradient; the forward pass does read the bucket)."""
    cfg, data = tiny_setup
    model = _model(cfg, data).train()
    opt = make_optimizer(model, lr=3e-4, weight_decay=0.05, lex_lr=1e-2)
    b = _batch(data)
    for _ in range(3):
        loss = torch.nn.functional.cross_entropy(model(b)["logits"], b["labels"])
        opt.zero_grad()
        loss.backward()
        assert model.lex_w.grad[0].item() == 0.0
        opt.step()
    assert model.lex_w[0].item() == 0.0 and torch.count_nonzero(model.lex_w) > 0
    ids, sol = b["input_ids"].flatten(0, 1), b["sol_mask"].flatten(0, 1)
    assert torch.equal(model.lexical_score(ids, torch.zeros_like(sol)), torch.zeros(len(ids)))


def test_lexical_head_has_its_own_learning_rate(tiny_setup):
    cfg, data = tiny_setup
    model = _model(cfg, data)
    opt = make_optimizer(model, lr=3e-4, weight_decay=0.05, lex_lr=1e-2)
    lex_groups = [g for g in opt.param_groups if any(p is model.lex_w for p in g["params"])]
    assert len(lex_groups) == 1 and lex_groups[0]["lr"] == 1e-2 and len(lex_groups[0]["params"]) == 1
    assert all(g["lr"] == 3e-4 for g in opt.param_groups if g is not lex_groups[0])


def test_lexical_head_starts_at_zero(tiny_setup):
    cfg, data = tiny_setup
    assert torch.count_nonzero(_model(cfg, data).lex_w) == 0


# ---------------------------------------------------------------- FINAL_EVAL guard
def test_test_experiment_refuses_before_final_step(tiny_setup, monkeypatch):
    cfg, data = tiny_setup
    monkeypatch.setitem(experiments.__dict__, "FINAL_EVAL", False)
    with pytest.raises(RuntimeError, match="FINAL_EVAL"):
        experiments.test_experiment({"config": cfg.to_dict(), "runs": []}, data, torch.device("cpu"))


# ---------------------------------------------------------------- save_predictions
def test_save_predictions_format_and_no_overwrite(tiny_setup, tmp_path):
    cfg, data = tiny_setup
    rows = data["val"][:5]
    pred = {"pred": np.array([0, 1, 1, 0, 1]), "prob": np.array([.2, .8, .6, .4, .9])}
    path = tmp_path / "p" / "x_val.jsonl"
    save_predictions(str(path), rows, pred, "x", "val")
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert [r["predicted_label"] for r in recs] == [0, 1, 1, 0, 1]
    assert recs[1]["probability_candidate_2"] == pytest.approx(0.8)
    assert recs[1]["probability_candidate_1"] == pytest.approx(0.2)
    assert all(r["correct"] == (r["predicted_label"] == r["gold_label"]) for r in recs)
    with pytest.raises(FileExistsError):
        save_predictions(str(path), rows, pred, "x", "val")
    with pytest.raises(ValueError):
        save_predictions(str(tmp_path / "y.jsonl"), rows, {"pred": np.zeros(3)}, "y", "val")


# ---------------------------------------------------------------- duplicate re-scoring
def test_rescore_excludes_only_the_duplicate():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from rescore_without_duplicate import rescore
    labels = np.array([1, 0, 1, 1, 0])
    pred = np.array([1, 0, 0, 1, 1])
    r = rescore([(42, pred)], labels, exclude=[4])
    assert r["test_acc_all"] == pytest.approx(3 / 5)
    assert r["test_acc_without_duplicate"] == pytest.approx(3 / 4)
    assert r["duplicate_item_correct_in_seeds"] == "0/1"
