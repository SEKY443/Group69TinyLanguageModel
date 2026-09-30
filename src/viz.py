"""Attention visualisation and quantitative attention statistics for DACT."""
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import textwrap
from matplotlib.gridspec import GridSpec

from config import set_seed  # nb-skip
from data import TAG_DIFF, collate, encode_example, token_strings  # nb-skip
from model import DACT  # nb-skip
from train import to_device  # nb-skip


@torch.no_grad()
def attention_for_example(model, row, tok, cfg, device):
    """Runs one PIQA item through DACT and extracts every attention map needed for the plots (head-averaged)."""
    model.eval()
    enc = encode_example(row, tok, cfg)
    batch = to_device(collate([(enc, row["label"], 0)]), device)
    out = model(batch, return_attn=True)
    probs = out["logits"].softmax(-1)[0].cpu().numpy()
    info = {"probs": probs, "pred": int(probs.argmax()), "label": row["label"],
            "goal": row["goal"], "candidate_1": row["sol1"], "candidate_2": row["sol2"],
            "attention_note": "Head-averaged slices; goal slice excludes other keys; cross slice omits CLS sink; rows need not sum to one.",
            "options": []}
    for k in range(2):
        ids, seg, tags, sol = enc[k]
        sol_pos = [i for i, m in enumerate(sol) if m]
        goal_pos = [i for i, s in enumerate(seg) if s == 0][1:-1]  # drop [CLS] and [SEP]
        info["options"].append({
            "tokens": token_strings(tok, ids), "sol_pos": sol_pos, "goal_pos": goal_pos,
            "diff": [tags[i] == TAG_DIFF for i in sol_pos],
            "pool": out["pool"][0, k].float().cpu().numpy()[sol_pos],
            # last encoder layer: attention of solution tokens (queries) over goal tokens (keys)
            "sol2goal": out["self_attn"][-1][0, k].float().mean(0).cpu().numpy()[np.ix_(sol_pos, goal_pos)],
        })
    if "cross_attn" in out:
        w12 = out["cross_attn"][0][0].float().mean(0).cpu().numpy()
        o1, o2 = info["options"]
        info["cross12"] = w12[np.ix_(o1["sol_pos"], o2["sol_pos"])]
    return info


def _label_ticks(ax, tokens, diff, axis):
    set_ticks = ax.set_xticks if axis == "x" else ax.set_yticks
    set_labels = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    set_ticks(range(len(tokens)))
    labels = set_labels(tokens, rotation=90 if axis == "x" else 0, fontsize=8)
    for lab, d in zip(labels, diff):
        if d:
            lab.set_color("crimson")
            lab.set_fontweight("bold")


def plot_attention_case(model, row, tok, cfg, device, title="", save_path=None):
    """Four panels: pooling weights for each option, sol1->sol2 cross-attention, sol->goal self-attention.

    Tokens that differ between the two options are shown in bold red.
    """
    info = attention_for_example(model, row, tok, cfg, device)
    o1, o2 = info["options"]
    goal_tokens = [o1["tokens"][i] for i in o1["goal_pos"]]
    longest = max(len(o1["sol_pos"]), len(o2["sol_pos"]))
    fig = plt.figure(figsize=(max(15, min(28, longest * .32)), max(10, min(26, longest * .27))))
    gs = GridSpec(3, 2, height_ratios=[0.35, 1, 1], figure=fig, hspace=0.9, wspace=0.3)
    for k, o in enumerate((o1, o2)):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(o["pool"][None], aspect="auto", cmap="viridis", vmin=0,
                       vmax=max(float(o1["pool"].max()), float(o2["pool"].max()), 1e-8))
        fig.colorbar(im, ax=ax, fraction=.025)
        _label_ticks(ax, [o["tokens"][i] for i in o["sol_pos"]], o["diff"], "x")
        ax.set_yticks([])
        tag = " (gold)" if info["label"] == k else ""
        ax.set_title(f"Option {k + 1}{tag}: pooling attention, p={info['probs'][k]:.2f}", fontsize=10)
    if "cross12" in info:
        ax = fig.add_subplot(gs[1:, 0])
        im = ax.imshow(info["cross12"], aspect="auto", cmap="magma")
        _label_ticks(ax, [o2["tokens"][i] for i in o2["sol_pos"]], o2["diff"], "x")
        _label_ticks(ax, [o1["tokens"][i] for i in o1["sol_pos"]], o1["diff"], "y")
        ax.set_title("Cross-solution attention (rows: option 1 queries, cols: option 2 keys)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03)
    for k, o in enumerate((o1, o2)):
        ax = fig.add_subplot(gs[1 + k, 1])
        im = ax.imshow(o["sol2goal"], aspect="auto", cmap="Blues")
        _label_ticks(ax, goal_tokens, [False] * len(goal_tokens), "x")
        _label_ticks(ax, [o["tokens"][i] for i in o["sol_pos"]], o["diff"], "y")
        ax.set_title(f"Option {k + 1}: last-layer attention to goal tokens", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03)
    verdict = "CORRECT" if info["pred"] == info["label"] else "WRONG"
    fig.suptitle(textwrap.fill(f"{title} [{verdict}] Gold: option {info['label']+1}; predicted: option {info['pred']+1}. Goal: {row['goal']}", 130), fontsize=12)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        with open(str(save_path) + ".json", "w", encoding="utf-8") as stream:
            json.dump(info, stream, indent=2, ensure_ascii=False,
                      default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    plt.show()
    return info


@torch.no_grad()
def diff_attention_mass(model, loader, device):
    """Share of pooling attention that falls on difference-tagged tokens, per item (mean of the two options)."""
    model.eval()
    mass, correct = [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        is_diff = (batch["tags"] == TAG_DIFF).float()
        m = (out["pool"].float() * is_diff).sum(-1).mean(-1)  # [B]
        has_diff = is_diff.sum((-1, -2)) > 0
        mass.append(m[has_diff].cpu())
        correct.append((out["logits"].argmax(-1) == batch["labels"])[has_diff].cpu())
    return torch.cat(mass).numpy(), torch.cat(correct).numpy()


@torch.no_grad()
def diff_attention_controls(model, cfg, vocab_size, loader, device, seed=0):
    """Separates learned from built-in attention on differing tokens (mean pooling mass per condition).

    trained               : the trained model, as used for prediction
    trained, tag bias = 0 : same weights with the tag bias removed -> focus coming from learned token content only
    untrained             : a freshly initialised model -> focus produced by the initial tag bias alone
    """
    no_bias = copy.deepcopy(model)
    no_bias.pool.tag_bias.zero_()
    set_seed(seed)
    untrained = DACT(cfg, vocab_size).to(device)
    conditions = {"trained": model, "trained, tag bias = 0": no_bias, "untrained (initialisation)": untrained}
    out = {name: float(diff_attention_mass(m, loader, device)[0].mean()) for name, m in conditions.items()}
    del no_bias, untrained
    return out


@torch.no_grad()
def pooling_entropy(model, loader, device):
    """Normalised entropy of the pooling weights over each option's solution tokens (1 = uniform, 0 = one token).

    Returns per-item values (mean of the two options) and whether the item was predicted correctly.
    """
    model.eval()
    ent, correct = [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        w = out["pool"].float().clamp_min(1e-12)
        n = batch["sol_mask"].sum(-1).float()
        h = -(w * w.log() * batch["sol_mask"]).sum(-1) / n.clamp(min=2).log()
        keep = (n > 1).all(-1)
        ent.append(h.mean(-1)[keep].cpu())
        correct.append((out["logits"].argmax(-1) == batch["labels"])[keep].cpu())
    return torch.cat(ent).numpy(), torch.cat(correct).numpy()


@torch.no_grad()
def _layer_states(model, batch):
    """Hidden states after the embedding and after every encoder layer, for both options: list of [N, L, d]."""
    ids, seg, tags, mask = (batch[k].flatten(0, 1) for k in ("input_ids", "segment", "tags", "attn_mask"))
    pos = torch.arange(ids.size(1), device=ids.device)[None]
    x = model.tok(ids) + model.pos(pos) + model.seg(seg)
    if model.tag is not None:
        x = x + model.tag(tags)
    x = model.emb_ln(x)
    states = [x]
    for layer in model.layers:
        x, _ = layer(x, mask)
        states.append(x)
    return states


def diff_probe_by_layer(model, train_loader, eval_loader, device, max_tokens=40000, seed=0):
    """Linear probe: can a logistic regression tell differing from shared solution tokens at each layer?

    Trained on solution tokens from `train_loader` and scored (balanced accuracy, 0.5 = chance) on `eval_loader`.
    For a model without tag embeddings this shows whether the encoder works out the difference by itself; for DACT
    it shows whether the injected tag survives through the layers or fades.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    model.eval()

    def collect(loader):
        per_layer, labels, n = [[] for _ in range(len(model.layers) + 1)], [], 0
        for batch in loader:
            batch = to_device(batch, device)
            sol = batch["sol_mask"].flatten(0, 1)
            for store, s in zip(per_layer, _layer_states(model, batch)):
                store.append(s[sol].float().cpu())
            labels.append((batch["tags"].flatten(0, 1) == TAG_DIFF)[sol].cpu())
            n += int(sol.sum())
            if n >= max_tokens:
                break
        return [torch.cat(s).numpy() for s in per_layer], torch.cat(labels).numpy()

    Xtr, ytr = collect(train_loader)
    Xte, yte = collect(eval_loader)
    scores = []
    for layer, (a, b) in enumerate(zip(Xtr, Xte)):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(a, ytr)
        scores.append({"layer": "embedding" if layer == 0 else f"encoder {layer}",
                       "balanced acc": float(balanced_accuracy_score(yte, clf.predict(b)))})
    return scores


def diff_token_share(loader):
    """Fraction of solution tokens that are difference-tagged, per item (the uniform-attention reference)."""
    shares = []
    for batch in loader:
        is_diff = (batch["tags"] == TAG_DIFF).float()
        sol = batch["sol_mask"].float()
        s = (is_diff.sum(-1) / sol.sum(-1).clamp(min=1)).mean(-1)
        has_diff = is_diff.sum((-1, -2)) > 0
        shares.append(s[has_diff])
    return torch.cat(shares).numpy()


def select_cases(pred, prob, label, k=2):
    """Picks the most confident correct, most confident wrong and most uncertain items."""
    pred, prob, label = map(np.asarray, (pred, prob, label))
    conf = np.abs(prob - 0.5)
    correct = np.where(pred == label)[0]
    wrong = np.where(pred != label)[0]
    return {
        "confident_correct": correct[np.argsort(-conf[correct])][:k].tolist(),
        "confident_wrong": wrong[np.argsort(-conf[wrong])][:k].tolist(),
        "uncertain": np.argsort(conf)[:k].tolist(),
    }
