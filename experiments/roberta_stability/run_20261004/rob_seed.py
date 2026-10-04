"""Colab VM: one seed of one arm of the pre-registered RoBERTa test (env ARM, LR, SEED). Validation only."""
import json
import os
import sys

sys.path.insert(0, "/content/rob/src")
os.environ["GROUP69_COMMIT"] = "2db3f51"
from baselines import finetune_with_restart  # noqa: E402
from config import Config, get_device  # noqa: E402
from data import load_piqa  # noqa: E402

arm, lr, seed = os.environ["ARM"], float(os.environ["LR"]), int(os.environ["SEED"])
train, val, test = load_piqa(Config(data_dir="/content/piqa"))
assert all(r["label"] == 0 for r in test)            # hidden placeholder labels; the test split is not used
del test
out = f"/content/roberta_exp/{arm}"
r = finetune_with_restart("FacebookAI/roberta-base", train, val, None, get_device(), out, seed=seed, epochs=4,
                          batch_size=32, lr=lr, chance_threshold=0.52, verbose=True)
ep = [json.loads(l) for l in open(f"{out}/logs/B3_roberta-base_seed{r['seed']}.jsonl") if '"event": "epoch"' in l]
res = {"arm": arm, "lr": lr, "seed_requested": seed, "seed_used": r["seed"], "restarted_from": r["restarted_from"],
       "best_val_acc": r["best_val_acc"], "learned": len(ep) >= 2 and ep[1]["train_loss"] < 0.68,
       "train_loss_by_epoch": [round(e["train_loss"], 4) for e in ep], "val_acc_by_epoch": [round(e["val_acc"], 4) for e in ep]}
os.system(f"rm -rf {out}/checkpoints")
json.dump(res, open(f"{out}/result_seed{seed}.json", "w"), indent=1)
print("RESULT " + json.dumps(res), flush=True)
