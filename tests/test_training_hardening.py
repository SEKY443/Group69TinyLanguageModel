"""Training hardening (TRAINING_UPGRADE_PROMPT.md): bucketing flag, masking rules, OOM fallback, training diagnostics."""
import json

import numpy as np
import pytest
import torch

import baselines
from data import (BucketBatchSampler, PAD, collate, item_lengths, make_loader, padding_share)
from model import MultiHeadAttention
from train import N_SPECIAL, _mlm_collate, train_qa


# ---------------------------------------------------------------- length bucketing (opt-in new protocol)
def test_bucketing_is_off_by_default(tiny_setup):
    cfg, data = tiny_setup
    assert cfg.length_bucketing is False
    loader = make_loader(data["train_ds"], cfg, True, torch.device("cpu"))
    assert not isinstance(loader.batch_sampler, BucketBatchSampler)


def test_bucket_sampler_covers_every_item_once_and_reshuffles(tiny_setup):
    cfg, data = tiny_setup
    lengths = item_lengths(data["train_ds"])
    sampler = BucketBatchSampler(lengths, batch_size=16, chunk=4)
    torch.manual_seed(0)
    epoch1 = list(sampler)
    epoch2 = list(sampler)
    for epoch in (epoch1, epoch2):
        flat = sorted(i for b in epoch for i in b)
        assert flat == list(range(len(lengths)))                  # every item exactly once, never split or packed
        assert all(len(b) <= 16 for b in epoch) and len(epoch) == len(sampler)
    assert epoch1 != epoch2                                       # a new random order every epoch
    torch.manual_seed(0)
    assert list(sampler) == epoch1                                # reproducible under the seed


def test_bucketing_reduces_padding(tiny_setup):
    cfg, data = tiny_setup
    ds = data["train_ds"]
    torch.manual_seed(1)
    random_batches = [b.tolist() for b in torch.randperm(len(ds)).split(16)]
    torch.manual_seed(1)
    bucketed = list(BucketBatchSampler(item_lengths(ds), 16, chunk=50))
    assert padding_share(ds, bucketed) < padding_share(ds, random_batches)


def test_bucketing_flag_switches_the_loader(tiny_setup):
    cfg, data = tiny_setup
    c = cfg.but(length_bucketing=True)
    assert isinstance(make_loader(data["train_ds"], c, True, torch.device("cpu")).batch_sampler, BucketBatchSampler)
    # evaluation loaders keep the dataset order (predictions are matched to items by index)
    assert not isinstance(make_loader(data["val_ds"], c, False, torch.device("cpu")).batch_sampler, BucketBatchSampler)


# ---------------------------------------------------------------- masking rules
def test_padding_never_receives_attention():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 2, 0.0).eval()
    x = torch.randn(1, 6, 16)
    mask = torch.tensor([[True, True, True, True, False, False]])
    garbage = x.clone()
    garbage[:, 4:] = 1e3 * torch.randn(1, 2, 16)                  # anything in padded positions
    out = mha(x, x, mask)[0][:, :4]
    out_garbage = mha(garbage, garbage, mask)[0][:, :4]
    assert torch.allclose(out, out_garbage, atol=1e-5)


def test_mlm_masks_only_real_tokens(tiny_setup):
    cfg, data = tiny_setup
    seqs = [seq for item in data["train_ds"].items[:40] for seq in item]
    torch.manual_seed(0)
    batch = _mlm_collate(seqs, data["vocab_size"], mask_prob=0.5)
    chosen = batch["labels"] != -100
    original = collate([data["train_ds"][i] for i in range(40)])["input_ids"].view(80, -1)
    L = min(original.size(1), batch["labels"].size(1))
    assert chosen.any()
    assert (batch["labels"][chosen] >= N_SPECIAL).all()           # never [PAD], [CLS], [SEP], [MASK], [UNK]
    assert not (chosen & ~batch["attn_mask"]).any()               # never padding
    assert torch.equal(batch["labels"][:, :L][chosen[:, :L]], original[:, :L][chosen[:, :L]])


def test_b4_scores_only_the_continuation():
    chunk = [([1, 2, 3, 4, 5], 2), ([1, 2, 9], 1)]                 # prompt + continuation, n continuation tokens
    ids, attn, cont = baselines.lm_batch(chunk, pad_id=0)
    assert ids.tolist() == [[1, 2, 3, 4, 5], [1, 2, 9, 0, 0]]
    assert attn.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert cont.tolist() == [[False, False, False, True, True], [False, False, True, False, False]]


def test_b3_encodes_goal_solution_pairs():
    calls = {}

    def fake_tok(goals, sols, truncation, max_length, padding, return_tensors):
        calls["pairs"] = list(zip(goals, sols))
        n = len(goals)
        return {"input_ids": torch.arange(n * max_length).view(n, max_length),
                "attention_mask": torch.ones(n, max_length, dtype=torch.long)}
    rows = [{"goal": "g1", "sol1": "a", "sol2": "b", "label": 1}, {"goal": "g2", "sol1": "c", "sol2": "d", "label": 0}]
    enc, y = baselines.encode_multiple_choice(fake_tok, rows, 8)
    assert calls["pairs"] == [("g1", "a"), ("g1", "b"), ("g2", "c"), ("g2", "d")]
    assert enc["input_ids"].shape == (2, 2, 8) and enc["attention_mask"].shape == (2, 2, 8)
    assert y.tolist() == [1, 0]


# ---------------------------------------------------------------- B4 out-of-memory fallback
class _FakeLM(torch.nn.Module):
    """Deterministic 'language model': logits depend only on each token, so batching cannot change scores."""

    def __init__(self, fail_above=None):
        super().__init__()
        self.emb = torch.nn.Embedding(20, 20)
        self.fail_above, self.calls = fail_above, []

    def forward(self, input_ids, attention_mask):
        self.calls.append(input_ids.size(0))
        if self.fail_above is not None and input_ids.size(0) > self.fail_above:
            raise torch.cuda.OutOfMemoryError("simulated")
        return type("Out", (), {"logits": self.emb(input_ids)})()


def test_oom_fallback_halves_the_batch_and_gives_the_same_scores():
    torch.manual_seed(0)
    seqs = [([1, 2, 3, 4], 2), ([1, 5, 6], 1), ([2, 3, 7, 8, 9], 3), ([4, 4], 1), ([5, 6, 7], 2)]
    ok = _FakeLM()
    flaky = _FakeLM(fail_above=2)
    flaky.load_state_dict(ok.state_dict())
    ref, oom_ref = baselines.score_continuations(ok, seqs, 4, 0, torch.device("cpu"))
    got, oom = baselines.score_continuations(flaky, seqs, 4, 0, torch.device("cpu"))
    assert oom_ref == [] and oom == [2]
    assert torch.allclose(ref, got, atol=1e-6) and len(got) == len(seqs)


def test_oom_at_batch_one_is_raised():
    with pytest.raises(torch.cuda.OutOfMemoryError):
        baselines.score_continuations(_FakeLM(fail_above=0), [([1, 2], 1)], 1, 0, torch.device("cpu"))


# ---------------------------------------------------------------- training diagnostics in the log
def test_train_qa_logs_gradient_norm_and_step_time(tiny_setup, tmp_path):
    cfg, data = tiny_setup
    from model import DACT
    c = cfg.but(out_dir=str(tmp_path), epochs=1)
    torch.manual_seed(0)
    model = DACT(c, data["vocab_size"])
    dev = torch.device("cpu")
    train_qa(model, make_loader(data["train_ds"], c, True, dev), make_loader(data["val_ds"], c, False, dev), c, dev,
             "diag_seed0", verbose=False)
    ep = [json.loads(l) for l in open(tmp_path / "logs" / "diag_seed0.jsonl") if '"epoch"' in l and '"event": "epoch"' in l][0]
    assert ep["grad_norm_mean"] > 0 and ep["grad_norm_max"] >= ep["grad_norm_mean"]
    assert ep["step_ms"] > 0 and ep["max_memory_mb"] is None          # CPU: no GPU memory figure
    assert np.isfinite(ep["train_loss"]) and 0 <= ep["train_acc"] <= 1
