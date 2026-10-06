"""Properties of the submitted main model that the report relies on.

* DACT is built and trained from scratch: no pretrained weights or Hugging Face loaders in its code path.
* Attention is a proper distribution over the keys it may use (never over padding or forbidden positions), the
  explicit implementation used for the figures equals the fused kernel used for training, and the QA loss sends
  gradient through every attention component (so attention is connected to the prediction).
* Model inputs never depend on the gold label, the tokenizer only sees the training split, a checkpoint round
  trip is exact, the same seed gives the same initialisation, and the model can fit a tiny training set.
"""
import ast
import os

import pytest
import torch

from config import Config, set_seed
from data import PAD, TAG_DIFF, collate, encode_example
from model import DACT
from train import make_optimizer, qa_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_MODEL_MODULES = ("config", "data", "model", "train", "evaluate", "experiments", "viz")


def _batch(tiny_setup, n=6):
    cfg, data = tiny_setup
    return collate([data["train_ds"][i] for i in range(n)])


def _model(tiny_setup, **kw):
    cfg, data = tiny_setup
    set_seed(0, True)
    return DACT(cfg.but(**kw), data["vocab_size"]).eval()


def test_main_model_code_path_loads_no_pretrained_weights():
    """Only baselines.py (B3/B4) may import transformers or call from_pretrained; the DACT path never does."""
    for name in MAIN_MODEL_MODULES:
        tree = ast.parse(open(os.path.join(ROOT, "src", f"{name}.py"), encoding="utf-8").read())
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert "transformers" not in imported, f"{name}.py imports transformers"
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "from_pretrained" not in calls, f"{name}.py calls from_pretrained"
    from experiments import ARCHS
    assert set(ARCHS) == {"dact", "bilstm"}, "only the two from-scratch architectures may be trained by run_experiment"


def test_dact_starts_from_random_initialisation(tiny_setup):
    cfg, data = tiny_setup
    set_seed(1, True)
    a = DACT(cfg, data["vocab_size"]).state_dict()
    set_seed(2, True)
    b = DACT(cfg, data["vocab_size"]).state_dict()
    set_seed(1, True)
    c = DACT(cfg, data["vocab_size"]).state_dict()
    assert not torch.equal(a["tok.weight"], b["tok.weight"]), "different seeds must give different random weights"
    assert all(torch.equal(a[k], c[k]) for k in a), "the same seed must give the same initialisation"
    assert float(a["tok.weight"].std()) == pytest.approx(0.02, rel=0.2), "token embeddings use N(0, 0.02) init"


def test_explicit_attention_equals_fused_kernel(tiny_setup):
    """The return_attn=True path (figures) and the SDPA path (training/evaluation) give the same scores."""
    m, batch = _model(tiny_setup), _batch(tiny_setup)
    with torch.no_grad():
        fused, explicit = m(batch)["logits"], m(batch, return_attn=True)["logits"]
    assert torch.allclose(fused, explicit, atol=1e-5)


def test_attention_weights_are_distributions_over_allowed_keys(tiny_setup):
    m, batch = _model(tiny_setup), _batch(tiny_setup)
    with torch.no_grad():
        out = m(batch, return_attn=True)
    attn_mask, sol_mask = batch["attn_mask"], batch["sol_mask"]
    for w in out["self_attn"]:                                   # [B, 2, heads, Lq, Lk]
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
        assert float(w.masked_select(~attn_mask[:, :, None, None, :].expand_as(w)).abs().max()) == 0.0
    allowed = sol_mask.clone()
    allowed[:, :, 0] = True                                      # rival solution tokens + rival [CLS] sink
    for w, rival in zip(out["cross_attn"], (1, 0)):              # [B, heads, Lq, Lk]
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
        forbidden = ~allowed[:, rival][:, None, None, :].expand_as(w)
        assert float(w.masked_select(forbidden).abs().max()) == 0.0
    pool = out["pool"]                                           # [B, 2, L]
    assert torch.allclose(pool.sum(-1), torch.ones_like(pool.sum(-1)), atol=1e-5)
    assert float(pool.masked_select(~sol_mask).abs().max()) == 0.0, "pooling only over solution tokens"


def test_qa_loss_sends_gradient_through_every_attention_component(tiny_setup):
    cfg, _ = tiny_setup
    m, batch = _model(tiny_setup), _batch(tiny_setup)
    m.train()
    qa_loss(m(batch)["logits"], batch["labels"], cfg).backward()
    must_learn = ["tag.weight", "layers.0.attn.q.weight", "layers.0.attn.k.weight", "layers.0.attn.v.weight",
                  "cross.attn.q.weight", "cross.attn.k.weight", "cross.attn.v.weight", "cross.fuse.0.weight",
                  "pool.proj.weight", "pool.v.weight", "pool.tag_bias", "scorer.0.weight", "lex_w"]
    grads = dict(m.named_parameters())
    for name in must_learn:
        g = grads[name].grad
        assert g is not None and float(g.abs().sum()) > 0, f"no QA gradient reaches {name}"
    # the masked-LM head is used only by the optional warm-up, never by the QA prediction
    assert grads["mlm_bias"].grad is None


def test_pooling_attention_changes_the_prediction(tiny_setup):
    """Attention is not decorative: replacing the learned pooling weights by uniform weights changes the scores."""
    m, batch = _model(tiny_setup), _batch(tiny_setup)
    with torch.no_grad():
        m.pool.v.weight.normal_(0, 1.0)                           # make the pooling scores non-trivial
        attn = m(batch)["logits"]
        m.pool.mode = "mean"
        mean = m(batch)["logits"]
    assert not torch.allclose(attn, mean, atol=1e-4)


def test_model_inputs_do_not_depend_on_the_label(tiny_setup):
    cfg, data = tiny_setup
    row = dict(data["train"][0])
    a = encode_example({**row, "label": 0}, data["tok"], cfg)
    b = encode_example({**row, "label": 1}, data["tok"], cfg)
    assert a == b
    items = [data["train_ds"][i] for i in range(8)]
    flipped = [(enc, 1 - y, i) for enc, y, i in items]
    x, y = collate(items), collate(flipped)
    for key in ("input_ids", "segment", "tags", "sol_mask", "attn_mask"):
        assert torch.equal(x[key], y[key]), f"{key} changes with the label"


def test_tokenizer_is_learned_from_the_training_split_only(tiny_setup, monkeypatch):
    import data as data_module
    cfg, data = tiny_setup
    seen = []
    real = data_module.train_tokenizer
    monkeypatch.setattr(data_module, "train_tokenizer", lambda rows, c, path=None: (seen.append(rows), real(rows, c, path))[1])
    out = data_module.prepare_everything(cfg.but(out_dir=cfg.out_dir + "_tokcheck"))
    assert len(seen) == 1 and seen[0] == out["train"]
    assert not {id(r) for r in seen[0]} & {id(r) for r in out["val"] + out["test"]}


def test_checkpoint_round_trip_is_exact(tiny_setup, tmp_path):
    cfg, data = tiny_setup
    m, batch = _model(tiny_setup), _batch(tiny_setup)
    path = tmp_path / "ckpt.pt"
    torch.save(m.state_dict(), path)
    m2 = DACT(cfg, data["vocab_size"]).eval()
    m2.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        assert torch.equal(m(batch)["logits"], m2(batch)["logits"])


def test_dact_can_fit_a_tiny_training_set(tiny_setup):
    """Optimisation sanity check: 64 training items are fitted almost perfectly."""
    cfg, data = tiny_setup
    set_seed(0, True)
    c = cfg.but(dropout=0.0, attn_dropout=0.0, label_smoothing=0.0)
    m = DACT(c, data["vocab_size"])
    batch = collate([data["train_ds"][i] for i in range(64)])
    opt = make_optimizer(m, 3e-3, 0.0, 3e-2)
    m.train()
    for _ in range(150):
        loss = qa_loss(m(batch)["logits"], batch["labels"], c)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = float((m(batch)["logits"].argmax(-1) == batch["labels"]).float().mean())
    assert acc >= 0.95 and float(loss) < 0.2, (acc, float(loss))


def test_padding_tokens_get_no_difference_tag(tiny_setup):
    batch = _batch(tiny_setup)
    pad = batch["input_ids"] == PAD
    assert not (batch["tags"][pad] == TAG_DIFF).any()
    assert not batch["sol_mask"][pad].any()
