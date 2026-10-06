"""Global configuration, reproducibility and device helpers."""
import json
import os
import random
import time
import platform
import importlib.metadata
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch


@dataclass
class Config:
    # ---- paths ----
    # PIQA sub-folder of the unit's shared A2 dataset folder; on Colab the data is downloaded from here automatically
    data_folder_url: str = "https://drive.google.com/drive/folders/1lxEsHLbRsgOHh8rOAd75QHUZds7ybKtT"
    drive_zip: str = "/content/drive/MyDrive/Group73/PIQA.zip"  # Colab location of the provided data
    local_zip_glob: str = "*.zip"                               # fallback when running outside Colab
    data_dir: str = "data/piqa"
    out_dir: str = "outputs"
    persist_dir: str = ""       # e.g. MyDrive/Group73/runs/<run_id>: finished experiments are copied here and restored on a new VM

    # ---- data ----
    seed: int = 42              # seed for the train/validation split (kept fixed across runs)
    val_frac: float = 0.10
    vocab_size: int = 8000
    min_frequency: int = 2
    max_goal_len: int = 40      # BPE tokens
    max_sol_len: int = 96
    diff_aware_truncation: bool = True  # over-long solutions keep the window around the differing span, not the prefix

    # ---- model (DACT) ----
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.2
    attn_dropout: float = 0.1
    max_len: int = 160

    # ---- ablation switches ----
    use_diff_tags: bool = True        # difference-tag embedding
    use_cross_solution: bool = True   # sol1 <-> sol2 contrastive cross-attention
    pool_mode: str = "diff_attn"      # "diff_attn" | "attn" | "mean"
    tag_bias_init: float = 1.0        # initial pooling bias for difference-tagged tokens (0.0 = no built-in prior)
    use_lexical: bool = True          # "wide" head: learned weights of hashed solution unigrams/bigrams added to the score
    use_goal_matching: bool = False   # pre-registered experiment: solution tokens attend to goal tokens (gated fusion)
    lex_buckets: int = 2 ** 18        # hash buckets of the lexical head
    lex_lr: float = 1e-2              # learning rate of the lexical head (the rest of the model uses `lr`)
    objective: str = "pairwise"       # "pairwise" (softmax over the two options) | "pointwise" (independent BCE)
    mlm_epochs: int = 0               # in-domain masked-LM warm-up on the training split only

    # ---- optimisation ----
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 20
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.1
    patience: int = 6
    grad_clip: float = 1.0
    mlm_lr: float = 5e-4
    mlm_prob: float = 0.15
    amp: bool = True
    num_workers: int = 2
    run_seed: int = 42         # model/data-loader RNG; seed above remains the fixed split seed
    deterministic: bool = False  # historical CUDA runs used fast, nondeterministic kernels
    symmetric_diff_tags: bool = False  # opt-in NEW protocol; historical results use False
    length_bucketing: bool = False     # opt-in NEW protocol: training batches of similar length (less padding)
    bucket_chunk: int = 50             # bucketing sorts chunks of bucket_chunk x batch_size shuffled items

    def to_dict(self):
        return asdict(self)

    def but(self, **kw):
        """Return a copy with some fields overridden (used for ablations / seeds)."""
        return replace(self, **kw)


def set_seed(seed: int, deterministic=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def environment_info(device=None):
    """Runtime provenance; recorded values describe this run, never the historical run."""
    versions = {}
    for package in ("torch", "numpy", "scikit-learn", "scipy", "pandas", "matplotlib", "tokenizers", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "device": str(device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def get_device():
    if torch.cuda.is_available():
        # TF32 matmuls: large speed-up on Ampere+ GPUs (A100 / L4) with negligible accuracy impact
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not torch.are_deterministic_algorithms_enabled()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device):
    """bf16 autocast on GPUs with native bf16 (Ampere+), fp16 on older CUDA GPUs, disabled elsewhere.

    Only *native* support counts: a T4 reports bf16 as supported through emulation, which trains 3.5x slower than fp16.
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def git_commit():
    """Short commit of the code being run, or 'unknown' (e.g. a notebook uploaded without its repository)."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or os.environ.get("GROUP73_COMMIT", "unknown")
    except Exception:
        return os.environ.get("GROUP73_COMMIT", "unknown")


def write_progress(cfg, **fields):
    """Heartbeat: overwrites <out_dir>/progress.json (and the persistent copy) so a run can be monitored cheaply."""
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": git_commit(), **fields}
    for root in filter(None, (cfg.out_dir, cfg.persist_dir)):
        try:
            os.makedirs(root, exist_ok=True)
            tmp = os.path.join(root, "progress.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=1)
            os.replace(tmp, os.path.join(root, "progress.json"))     # atomic: a reader never sees half a file
        except OSError:
            pass                                                     # monitoring must never break training
    return record


def fresh_path(path):
    """`path`, or a time-stamped sibling if it exists, so a re-run never overwrites earlier evidence."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    return f"{base}.{time.strftime('%Y%m%d_%H%M%S')}{ext}"


def log_test_access(roots, reason, n_items):
    """Appends one line (time, commit, reason) to <root>/test_access.log for every root: the audit trail of every
    read of the test labels. Called by experiments.final_eval and data.load_piqa_with_test_labels."""
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": git_commit(), "reason": reason, "n_items": n_items}
    for root in filter(None, roots):
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "test_access.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    return record


def write_run_config(cfg, device, run_id, **extra):
    """One file per run with everything needed to reproduce it: config, seeds, commit, environment, data checksums
    (the data manifest written by prepare_everything). Saved to out_dir and to the persistent run folder."""
    manifest = os.path.join(cfg.out_dir, "data_manifest.json")
    record = {"run_id": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": git_commit(),
              "config": cfg.to_dict(), "environment": environment_info(device),
              "data": json.load(open(manifest, encoding="utf-8")) if os.path.isfile(manifest) else None, **extra}
    for root in filter(None, (cfg.out_dir, cfg.persist_dir)):
        os.makedirs(root, exist_ok=True)
        path = fresh_path(os.path.join(root, "run_config.json"))
        with open(path, "x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        record.setdefault("path", path)
    return record


class JsonlLogger:
    """Append-only JSON-lines logger so every training/evaluation run leaves a persistent log file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def log(self, **record):
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record
