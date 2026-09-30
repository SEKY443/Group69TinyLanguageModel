"""PIQA loading, train/validation split, BPE tokenizer, difference tags and batching."""
import difflib
import glob
import json
import os
import zipfile
import hashlib

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset

from config import Config  # nb-skip

EXPECTED_FILES = ("train.jsonl", "train-labels.lst", "test.jsonl", "test-labels.lst")
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD, UNK, CLS, SEP, MASK = range(len(SPECIALS))

# difference-tag vocabulary
TAG_OTHER, TAG_SHARED, TAG_DIFF = 0, 1, 2   # goal/special tokens, token shared by both solutions, token unique to this solution


# ----------------------------------------------------------------------------------------------
# 1. Locating and extracting the data
# ----------------------------------------------------------------------------------------------
def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _find_zip(cfg: Config):
    if _in_colab():
        from google.colab import drive
        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        if os.path.isfile(cfg.drive_zip):
            return cfg.drive_zip
        raise FileNotFoundError(f"PIQA zip not found at {cfg.drive_zip}. Update Config.drive_zip.")
    for path in [cfg.drive_zip, *sorted(glob.glob(cfg.local_zip_glob))]:
        if os.path.isfile(path) and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                names = {os.path.basename(n) for n in zf.namelist()}
            if set(EXPECTED_FILES) <= names:
                return path
    raise FileNotFoundError("No zip containing the PIQA files was found.")


def prepare_data(cfg: Config):
    """Extract only the four expected PIQA files into cfg.data_dir (no arbitrary paths are written)."""
    if all(os.path.isfile(os.path.join(cfg.data_dir, f)) for f in EXPECTED_FILES):
        return cfg.data_dir
    zpath = _find_zip(cfg)
    os.makedirs(cfg.data_dir, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        members = {os.path.basename(n): n for n in zf.namelist() if not n.endswith("/")}
        for fname in EXPECTED_FILES:
            with zf.open(members[fname]) as src, open(os.path.join(cfg.data_dir, fname), "wb") as dst:
                dst.write(src.read())
    print(f"Extracted {EXPECTED_FILES} from {zpath} -> {cfg.data_dir}")
    return cfg.data_dir


def load_split(data_dir, split):
    with open(os.path.join(data_dir, f"{split}.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(data_dir, f"{split}-labels.lst"), encoding="utf-8") as f:
        labels = [int(line) for line in f if line.strip()]
    if len(rows) != len(labels):
        raise ValueError(f"{split}: {len(rows)} rows vs {len(labels)} labels")
    for i, (r, y) in enumerate(zip(rows, labels)):
        if y not in (0, 1):
            raise ValueError(f"{split} row {i}: label must be 0 or 1")
        r["label"] = y
        for k in ("goal", "sol1", "sol2"):
            if not isinstance(r.get(k), str) or not r[k].strip():
                raise ValueError(f"{split} row {i}: {k} must be a nonempty string")
            r[k] = " ".join(r[k].split())  # collapse irregular whitespace
    return rows


def load_piqa(cfg: Config):
    """Returns train / validation (held out from the official training file) / test.

    The test split is returned separately and must only be used for the final evaluation.
    """
    data_dir = prepare_data(cfg)
    full_train = load_split(data_dir, "train")
    test = load_split(data_dir, "test")
    idx = np.arange(len(full_train))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg.val_frac, random_state=cfg.seed,
                                      stratify=[r["label"] for r in full_train])
    train = [full_train[i] for i in tr_idx]
    val = [full_train[i] for i in va_idx]
    return train, val, test


# ----------------------------------------------------------------------------------------------
# 2. Tokenizer (BPE learned from the training split only)
# ----------------------------------------------------------------------------------------------
def train_tokenizer(train_rows, cfg: Config, path=None):
    if path and os.path.exists(path):
        raise FileExistsError(f"Tokenizer evidence already exists: {path}. Use a fresh output directory.")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)])
    trainer = trainers.BpeTrainer(vocab_size=cfg.vocab_size, min_frequency=cfg.min_frequency,
                                  special_tokens=SPECIALS, continuing_subword_prefix="##")
    corpus = (text for r in train_rows for text in (r["goal"], r["sol1"], r["sol2"]))
    tok.train_from_iterator(corpus, trainer=trainer, length=3 * len(train_rows))
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tok.save(path)
    return tok


def diff_tags(a, b, symmetric=False):
    """Legacy alignment by default; canonical option order is an opt-in new preprocessing protocol.

    SequenceMatcher tie-breaking is directional. Canonical ordering makes alignment equivariant
    without consulting labels. Historical results must not be attributed to symmetric=True.
    """
    if symmetric and tuple(a) > tuple(b):
        tb, ta = diff_tags(b, a)
        return ta, tb
    ta, tb = [TAG_DIFF] * len(a), [TAG_DIFF] * len(b)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ta[i1:i2] = [TAG_SHARED] * (i2 - i1)
            tb[j1:j2] = [TAG_SHARED] * (j2 - j1)
    return ta, tb


def encode_example(row, tok, cfg: Config):
    """Encodes one PIQA item into two sequences [CLS] goal [SEP] sol_i [SEP] plus side information."""
    g = tok.encode(row["goal"]).ids[: cfg.max_goal_len]
    s1 = tok.encode(row["sol1"]).ids[: cfg.max_sol_len]
    s2 = tok.encode(row["sol2"]).ids[: cfg.max_sol_len]
    t1, t2 = diff_tags(s1, s2, cfg.symmetric_diff_tags)
    out = []
    for s, t in ((s1, t1), (s2, t2)):
        ids = [CLS] + g + [SEP] + s + [SEP]
        seg = [0] * (len(g) + 2) + [1] * (len(s) + 1)
        tags = [TAG_OTHER] * (len(g) + 2) + t + [TAG_OTHER]
        sol = [0] * (len(g) + 2) + [1] * len(s) + [0]   # positions that belong to the solution text
        out.append((ids, seg, tags, sol))
    return out


class PIQADataset(Dataset):
    """Pre-tokenises everything once so the GPU is not starved by Python-side work."""

    def __init__(self, rows, tok, cfg: Config):
        self.rows = rows
        self.items = [encode_example(r, tok, cfg) for r in rows]
        self.labels = [r["label"] for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.labels[i], i


def collate(batch):
    """Pads a batch into tensors of shape [B, 2, L]."""
    B = len(batch)
    L = max(len(seq[0]) for item, _, _ in batch for seq in item)
    ids = torch.full((B, 2, L), PAD, dtype=torch.long)
    seg = torch.zeros((B, 2, L), dtype=torch.long)
    tags = torch.zeros((B, 2, L), dtype=torch.long)
    sol = torch.zeros((B, 2, L), dtype=torch.bool)
    for b, (item, _, _) in enumerate(batch):
        for k, (i_, s_, t_, m_) in enumerate(item):
            n = len(i_)
            ids[b, k, :n] = torch.tensor(i_)
            seg[b, k, :n] = torch.tensor(s_)
            tags[b, k, :n] = torch.tensor(t_)
            sol[b, k, :n] = torch.tensor(m_, dtype=torch.bool)
    return {
        "input_ids": ids, "segment": seg, "tags": tags, "sol_mask": sol,
        "attn_mask": ids != PAD,
        "labels": torch.tensor([y for _, y, _ in batch], dtype=torch.long),
        "index": torch.tensor([i for _, _, i in batch], dtype=torch.long),
    }


def make_loader(ds, cfg: Config, shuffle, device):
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=cfg.num_workers if device.type == "cuda" else 0,
                      pin_memory=device.type == "cuda", persistent_workers=False)


def prepare_everything(cfg: Config):
    """Loads the splits, learns the BPE tokenizer on the training split and pre-tokenises every split."""
    train, val, test = load_piqa(cfg)
    tok = train_tokenizer(train, cfg, path=os.path.join(cfg.out_dir, "tokenizer.json"))
    manifest = {"split_seed": cfg.seed, "val_frac": cfg.val_frac, "files": {}, "splits": {}}
    for fname in EXPECTED_FILES:
        with open(os.path.join(cfg.data_dir, fname), "rb") as stream:
            manifest["files"][fname] = hashlib.sha256(stream.read()).hexdigest()
    for name, rows in (("train", train), ("val", val), ("test", test)):
        payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
        manifest["splits"][name] = {"size": len(rows), "ordered_sha256": hashlib.sha256(payload).hexdigest()}
    manifest["tokenizer_sha256"] = hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()
    with open(os.path.join(cfg.out_dir, "data_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return {"train": train, "val": val, "test": test, "tok": tok, "vocab_size": tok.get_vocab_size(),
            "train_ds": PIQADataset(train, tok, cfg), "val_ds": PIQADataset(val, tok, cfg),
            "test_ds": PIQADataset(test, tok, cfg)}


def token_strings(tok, ids):
    return [tok.id_to_token(int(i)) for i in ids]
