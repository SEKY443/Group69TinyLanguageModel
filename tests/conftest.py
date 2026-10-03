"""Shared fixtures: src/ on the import path and a small synthetic PIQA-format dataset (no real data needed)."""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

WORDS = ("water pan heat oven tape paper bowl spoon ice salt cut stir melt cool put add bottle cloth glue "
         "knife fork board wood nail screw rope bag box lid cup towel soap brush").split()


def _row(rng):
    goal = " ".join(rng.choices(WORDS, k=rng.randint(4, 9)))
    sol = rng.choices(WORDS, k=rng.randint(6, 14))
    other = list(sol)
    for i in rng.sample(range(len(other)), k=min(2, len(other))):
        other[i] = rng.choice(WORDS)
    return {"goal": goal, "sol1": " ".join(sol), "sol2": " ".join(other)}


def write_synthetic_piqa(folder, n_train=240, n_test=60, seed=0):
    """Writes the four PIQA files; the label is learnable from whether 'water' appears in the correct option."""
    rng = random.Random(seed)
    os.makedirs(folder, exist_ok=True)
    for split, n in (("train", n_train), ("test", n_test)):
        rows, labels = [], []
        for _ in range(n):
            r, y = _row(rng), rng.randint(0, 1)
            key = ("sol1", "sol2")[y]
            r[key] = "water " + r[key]
            rows.append(r)
            labels.append(y)
        with open(os.path.join(folder, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(os.path.join(folder, f"{split}-labels.lst"), "w", encoding="utf-8") as f:
            f.writelines(f"{y}\n" for y in labels)
    return folder


@pytest.fixture(scope="session")
def tiny_setup(tmp_path_factory):
    """A tiny CPU config and the prepared synthetic data (tokenizer learned once for the whole session)."""
    from config import Config
    from data import prepare_everything

    root = tmp_path_factory.mktemp("piqa")
    cfg = Config(data_dir=write_synthetic_piqa(str(root / "data")), out_dir=str(root / "prep"), vocab_size=300,
                 min_frequency=1, d_model=32, n_heads=2, n_layers=1, d_ff=64, max_len=64, max_goal_len=16,
                 max_sol_len=24, batch_size=32, epochs=2, patience=2, lex_buckets=2 ** 10, num_workers=0,
                 amp=False, deterministic=True)
    return cfg, prepare_everything(cfg)
