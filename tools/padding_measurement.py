"""Measures the padding share of the DACT training loader with and without length bucketing (no training).

Uses the real training split and the saved tokenizer of the final run (outputs/tokenizer.json); the test split is
not used. Usage: python tools/padding_measurement.py --data-dir data/piqa
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from config import Config, set_seed  # noqa: E402
from data import BucketBatchSampler, PIQADataset, item_lengths, load_piqa, padding_share  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
args = ap.parse_args()
cfg = Config(data_dir=args.data_dir)
train, _, _ = load_piqa(cfg)
ds = PIQADataset(train, Tokenizer.from_file(os.path.join(ROOT, "outputs", "tokenizer.json")), cfg)
for seed in (42, 43, 44):
    set_seed(seed)
    random_batches = torch.randperm(len(ds)).split(cfg.batch_size)
    set_seed(seed)
    bucket_batches = list(BucketBatchSampler(item_lengths(ds), cfg.batch_size, cfg.bucket_chunk))
    print(f"seed {seed}: padding share random batches {padding_share(ds, [b.tolist() for b in random_batches]):.3f} | "
          f"bucketed {padding_share(ds, bucket_batches):.3f} | batches {len(random_batches)} vs {len(bucket_batches)}")
