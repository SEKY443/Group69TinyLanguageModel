"""Baselines. Every number reported for these is produced by running this code (no numbers copied from papers).

B0  majority class
B1  TF-IDF of solution text + logistic regression (tests shallow lexical cues)
B2  BiLSTM + additive attention, trained from scratch with the same tokenizer (non-Transformer comparison)
B3  fine-tuned pretrained encoder (RoBERTa) with a multiple-choice head
B4  zero-shot open LLM (Qwen2.5), length-normalised log-likelihood of each solution
"""
import copy
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

from config import Config, JsonlLogger, amp_dtype, set_seed, environment_info  # nb-skip
from data import PAD  # nb-skip
from train import autocast_ctx  # nb-skip


# ---------------------------------------------------------------- B0
def majority_baseline(train_rows, rows):
    majority = int(np.mean([r["label"] for r in train_rows]) >= 0.5)
    return np.full(len(rows), majority)


# ---------------------------------------------------------------- B1
def tfidf_lr_baseline(train_rows, val_rows, test_rows=None, Cs=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0)):
    """Feature vector = tfidf(sol2) - tfidf(sol1); trained on both option orders; C picked on validation."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    vec.fit([r[k] for r in train_rows for k in ("goal", "sol1", "sol2")])

    def feats(rows):
        return vec.transform([r["sol2"] for r in rows]) - vec.transform([r["sol1"] for r in rows])

    Xtr = feats(train_rows)
    ytr = np.array([r["label"] for r in train_rows])
    X_aug, y_aug = vstack([Xtr, -Xtr]), np.concatenate([ytr, 1 - ytr])
    Xva, yva = feats(val_rows), np.array([r["label"] for r in val_rows])
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000).fit(X_aug, y_aug)
        acc = float((clf.predict(Xva) == yva).mean())
        if best is None or acc > best[0]:
            best = (acc, C, clf)
    _, C, clf = best
    result = {"C": C, "val_pred": clf.predict(Xva), "vectorizer": vec, "classifier": clf}
    if test_rows is not None:
        result["test_pred"] = clf.predict(feats(test_rows))
    return result


def tfidf_lr_predict(fitted, rows):
    vec = fitted["vectorizer"]
    x = vec.transform([r["sol2"] for r in rows]) - vec.transform([r["sol1"] for r in rows])
    return fitted["classifier"].predict(x)


# ---------------------------------------------------------------- B2
class BiLSTMAttention(nn.Module):
    """Each option ([CLS] goal [SEP] sol [SEP]) -> BiLSTM -> additive attention pooling -> score."""

    def __init__(self, cfg: Config, vocab_size, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD)
        self.lstm = nn.LSTM(cfg.d_model, hidden, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=cfg.dropout)
        self.att_w = nn.Linear(2 * hidden, 2 * hidden)
        self.att_v = nn.Linear(2 * hidden, 1, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.scorer = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                    nn.Linear(hidden, 1))

    def forward(self, batch, return_attn=False):
        ids, mask = batch["input_ids"], batch["attn_mask"]
        B, _, L = ids.shape
        ids, mask = ids.view(B * 2, L), mask.view(B * 2, L)
        lengths = mask.sum(-1).cpu()
        x = self.drop(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=L)
        s = self.att_v(torch.tanh(self.att_w(h))).squeeze(-1).float().masked_fill(~mask, float("-inf"))
        w = s.softmax(-1)
        pooled = torch.einsum("nl,nld->nd", w.to(h.dtype), h)
        return {"logits": self.scorer(self.drop(pooled)).view(B, 2).float(), "pool": w.view(B, 2, L)}


# ---------------------------------------------------------------- B3
def finetune_pretrained(model_name, train_rows, val_rows, test_rows, device, out_dir, epochs=4, lr=1e-5,
                        batch_size=16, max_len=128, seed=42, verbose=True, revision=None):
    """Fine-tunes a pretrained encoder with a multiple-choice head; model selection on validation only."""
    from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForMultipleChoice.from_pretrained(model_name, revision=revision).to(device)
    logger = JsonlLogger(f"{out_dir}/logs/B3_{model_name.split('/')[-1]}_seed{seed}.jsonl")
    if os.path.exists(logger.path):
        raise FileExistsError(logger.path)
    logger.log(event="start", model=model_name, requested_revision=revision,
               resolved_revision=getattr(model.config, "_commit_hash", None), seed=seed,
               epochs=epochs, lr=lr, batch_size=batch_size, max_len=max_len,
               n_params=sum(p.numel() for p in model.parameters()), environment=environment_info(device))

    def encode(rows):
        goals = [r["goal"] for r in rows for _ in range(2)]
        sols = [r[k] for r in rows for k in ("sol1", "sol2")]
        enc = tok(goals, sols, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        enc = {k: v.view(len(rows), 2, -1) for k, v in enc.items()}
        return enc, torch.tensor([r["label"] for r in rows])

    def batches(enc, y, shuffle):
        order = torch.randperm(len(y)) if shuffle else torch.arange(len(y))
        for i in range(0, len(y), batch_size):
            idx = order[i:i + batch_size]
            yield {k: v[idx].to(device) for k, v in enc.items()}, y[idx].to(device)

    @torch.no_grad()
    def predict(enc, y):
        model.eval()
        preds = []
        for xb, _ in batches(enc, y, False):
            with autocast_ctx(device):
                preds.append(model(**xb).logits.argmax(-1).cpu())
        return torch.cat(preds).numpy()

    tr, ytr = encode(train_rows)
    va, yva = encode(val_rows)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(ytr) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype(device) == torch.float16)
    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for xb, yb in batches(tr, ytr, True):
            with autocast_ctx(device):
                loss = F.cross_entropy(model(**xb).logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        val_acc = float((predict(va, yva) == yva.numpy()).mean())
        rec = logger.log(event="epoch", epoch=epoch, train_loss=float(np.mean(losses)), val_acc=val_acc,
                         seconds=round(time.time() - t0, 1))
        if verbose:
            print(f"[B3 {model_name}] ep {epoch} loss {rec['train_loss']:.4f} val_acc {val_acc:.4f} ({rec['seconds']}s)")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    checkpoint = f"{out_dir}/checkpoints/B3_{model_name.split('/')[-1]}_seed{seed}"
    model.save_pretrained(checkpoint)
    tok.save_pretrained(checkpoint)
    out = {"val_pred": predict(va, yva), "best_val_acc": best_acc, "checkpoint": checkpoint,
           "max_len": max_len, "batch_size": batch_size}
    if test_rows is not None:
        te, yte = encode(test_rows)
        out["test_pred"] = predict(te, yte)
    logger.log(event="end", best_val_acc=best_acc)
    del model
    return out


@torch.no_grad()
def pretrained_predict(fitted, rows, device):
    """Separate final evaluation of the locally saved validation-selected checkpoint."""
    from transformers import AutoModelForMultipleChoice, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(fitted["checkpoint"])
    model = AutoModelForMultipleChoice.from_pretrained(fitted["checkpoint"]).to(device).eval()
    predictions = []
    for start in range(0, len(rows), fitted["batch_size"]):
        chunk = rows[start:start + fitted["batch_size"]]
        goals = [r["goal"] for r in chunk for _ in range(2)]
        sols = [r[k] for r in chunk for k in ("sol1", "sol2")]
        enc = tok(goals, sols, truncation=True, max_length=fitted["max_len"], padding="max_length", return_tensors="pt")
        enc = {k:v.view(len(chunk),2,-1).to(device) for k,v in enc.items()}
        with autocast_ctx(device):
            predictions.append(model(**enc).logits.argmax(-1).cpu())
    return torch.cat(predictions).numpy()


# ---------------------------------------------------------------- B4
@torch.no_grad()
def llm_zero_shot(model_name, rows, device, batch_size=32, verbose=True, revision=None, out_dir=None):
    """Chooses the solution with the higher mean token log-probability given a goal prompt (no training)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    dtype = amp_dtype(device) or torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, revision=revision).to(device).eval()
    if out_dir is not None:
        JsonlLogger(f"{out_dir}/logs/B4_zero_shot.jsonl").log(event="inference", model=model_name,
            requested_revision=revision, resolved_revision=getattr(model.config, "_commit_hash", None),
            n_items=len(rows), batch_size=batch_size, continuation_limit=128,
            environment=environment_info(device))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    seqs = []  # (input ids, number of continuation tokens)
    for r in rows:
        prompt = tok(f"Goal: {r['goal']}\nSolution:", add_special_tokens=True).input_ids
        for k in ("sol1", "sol2"):
            cont = tok(" " + r[k], add_special_tokens=False).input_ids[:128]
            seqs.append((prompt + cont, len(cont)))
    scores = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        L = max(len(s) for s, _ in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        cont_mask = torch.zeros((len(chunk), L), dtype=torch.bool)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (s, n) in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s)
            attn[j, :len(s)] = 1
            cont_mask[j, len(s) - n:len(s)] = True
        ids, attn, cont_mask = ids.to(device), attn.to(device), cont_mask.to(device)
        logp = model(input_ids=ids, attention_mask=attn).logits.float().log_softmax(-1)
        tok_lp = logp[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
        m = cont_mask[:, 1:].float()
        scores.append(((tok_lp * m).sum(-1) / m.sum(-1).clamp(min=1)).cpu())
        if verbose and (i // batch_size) % 50 == 0:
            print(f"[B4 {model_name}] {i}/{len(seqs)}")
    scores = torch.cat(scores).view(-1, 2)
    del model
    return scores.argmax(-1).numpy()
