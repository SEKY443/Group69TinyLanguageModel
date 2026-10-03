"""Reproduces the audit's claims with the CURRENT pipeline (no training, no test labels used for decisions).

1. exact train/test duplicates (whitespace-normalised goal, sol1, sol2; also with the options swapped)
2. test items whose difference tags change when the two solutions are swapped
3. train vs val accuracy per epoch of the final run (from outputs/logs)

Usage: python tools/verify_audit_claims.py --data-dir <folder with the four PIQA files>
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from tokenizers import Tokenizer  # noqa: E402

from config import Config  # noqa: E402
from data import encode_example, load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    train_full = load_split(args.data_dir, "train")       # the whole training file (TRAIN + VAL)
    test = load_split(args.data_dir, "test", with_labels=False)   # labels are not needed for this audit

    # ---- 1. duplicates ----------------------------------------------------------------------------
    key = lambda r: (r["goal"], r["sol1"], r["sol2"])  # noqa: E731
    seen = {}
    for i, r in enumerate(train_full):
        seen.setdefault(key(r), []).append(i)
    within_train = sum(len(v) - 1 for v in seen.values() if len(v) > 1)
    exact = [(j, seen[key(r)]) for j, r in enumerate(test) if key(r) in seen]
    swapped = [(j, seen[(r["goal"], r["sol2"], r["sol1"])]) for j, r in enumerate(test)
               if (r["goal"], r["sol2"], r["sol1"]) in seen]
    print("1. DUPLICATES (whitespace-normalised, exact text)")
    print(f"   duplicate rows within the training file (beyond the first occurrence): {within_train}")
    print(f"   test items identical to a training-file item: {len(exact)}")
    for j, idx in exact:
        r = test[j]
        print(f"     test index {j} == train-file index {idx} | label test={r['label']} train={[train_full[i]['label'] for i in idx]}")
        print(f"       goal: {r['goal']}\n       sol1: {r['sol1']}\n       sol2: {r['sol2']}")
    print(f"   test items identical to a training-file item with the options swapped: {len(swapped)}")

    # ---- 2. order-dependent difference tags -------------------------------------------------------
    tok = Tokenizer.from_file(os.path.join(ROOT, "outputs", "tokenizer.json"))
    cfg = Config()                                          # current defaults: diff-aware truncation on, symmetric off
    changed, identical = 0, 0
    for r in test:
        a = encode_example(r, tok, cfg)
        b = encode_example({**r, "sol1": r["sol2"], "sol2": r["sol1"]}, tok, cfg)
        if a[0][2] != b[1][2] or a[1][2] != b[0][2]:      # tags of the same solution differ by position
            changed += 1
        if a[0][0] == a[1][0]:
            identical += 1
    sym = Config(symmetric_diff_tags=True)
    changed_sym = sum(1 for r in test
                      if encode_example(r, tok, sym)[0][2] != encode_example({**r, "sol1": r["sol2"], "sol2": r["sol1"]}, tok, sym)[1][2])
    print("\n2. ORDER-DEPENDENT DIFFERENCE TAGS (current pipeline, test file)")
    print(f"   items whose tags change when the options are swapped: {changed} / {len(test)} "
          f"(symmetric_diff_tags=True: {changed_sym})")
    print(f"   items whose two encoded inputs are identical: {identical}")

    # ---- 3. overfitting ------------------------------------------------------------------------------
    print("\n3. TRAIN vs VAL ACCURACY PER EPOCH (final run, outputs/logs/dact_full_seed*.jsonl)")
    rows = [{"check": "duplicate_rows_within_training_file", "value": within_train},
            {"check": "test_items_identical_to_training_item", "value": len(exact)},
            {"check": "duplicate_test_index", "value": ";".join(str(j) for j, _ in exact)},
            {"check": "test_items_tags_change_when_swapped_current_pipeline", "value": changed},
            {"check": "test_items_tags_change_when_swapped_symmetric", "value": changed_sym},
            {"check": "test_items_identical_encoded_inputs", "value": identical}]
    for path in sorted(p for p in glob.glob(os.path.join(ROOT, "outputs", "logs", "dact_full_seed*.jsonl"))
                       if re.search(r"dact_full_seed\d+\.jsonl$", p)):   # not the MLM warm-up logs (*_mlm.jsonl)
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        ep = [r for r in recs if r.get("event") == "epoch"]
        end = [r for r in recs if r.get("event") == "end"]
        seed = os.path.basename(path).split("seed")[1].split(".")[0]
        print(f"   {os.path.basename(path)}: best val {end[0]['best_val_acc']:.4f} at epoch {end[0]['best_epoch']}" if end else os.path.basename(path))
        for r in ep:
            print(f"     ep {r['epoch']:2d}  train {r['train_acc']:.3f}  val {r['val_acc']:.3f}  gap {r['train_acc'] - r['val_acc']:+.3f}")
        best = next(r for r in ep if r["epoch"] == end[0]["best_epoch"])
        rows += [{"check": f"dact_seed{seed}_best_epoch", "value": end[0]["best_epoch"]},
                 {"check": f"dact_seed{seed}_train_acc_at_best_epoch", "value": round(best["train_acc"], 4)},
                 {"check": f"dact_seed{seed}_val_acc_at_best_epoch", "value": round(best["val_acc"], 4)},
                 {"check": f"dact_seed{seed}_train_acc_last_epoch", "value": round(ep[-1]["train_acc"], 4)}]
    import csv
    out = os.path.join(ROOT, "outputs", "results", "audit_checks.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "value"])
        w.writeheader()
        w.writerows(rows)
    print("\nwritten", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
