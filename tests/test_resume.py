"""Section 1.4 acceptance test: an interrupted run, restarted on the same VM and on a new VM, loads finished seeds
instead of retraining them, keeps the files of the interrupted seed, and ends with the same table as an
uninterrupted run."""
import glob
import json
import os

import pytest
import torch

import experiments
import train

SEEDS = [1, 2]


class VMReclaimed(Exception):
    """Stands in for Colab taking the VM away in the middle of a seed."""


def _run(cfg, data, out_dir, persist_dir):
    c = cfg.but(out_dir=str(out_dir), persist_dir=str(persist_dir))
    return experiments.run_experiment("exp", c, "dact", data, torch.device("cpu"), SEEDS, verbose=False)


def _table(res):
    return [(r["seed"], r["val_acc"], r["val_loss"], r["epochs"]) for r in res["runs"]], res["val"]


def _count_training(monkeypatch):
    calls = []
    real = experiments.train_qa

    def spy(*args, **kwargs):
        calls.append(args[5])   # run_name
        return real(*args, **kwargs)
    monkeypatch.setattr(experiments, "train_qa", spy)
    return calls


def test_interrupt_and_resume(tiny_setup, tmp_path, monkeypatch):
    cfg, data = tiny_setup

    # 1. reference: an uninterrupted run
    reference = _run(cfg, data, tmp_path / "ref_out", tmp_path / "ref_drive")

    # 2. the same run, with the VM lost after epoch 1 of seed 2 (log written, checkpoint not yet)
    real_progress = train.write_progress

    def reclaim(c, **fields):
        if fields.get("run") == "exp_seed2":
            raise VMReclaimed
        return real_progress(c, **fields)
    monkeypatch.setattr(train, "write_progress", reclaim)
    with pytest.raises(VMReclaimed):
        _run(cfg, data, tmp_path / "vm1", tmp_path / "drive")
    monkeypatch.setattr(train, "write_progress", real_progress)
    drive = tmp_path / "drive"
    assert (drive / "results" / "exp_seed1_done.json").is_file(), "finished seed must be persisted at once"
    assert (drive / "checkpoints" / "exp_seed1.pt").is_file()
    assert not (drive / "results" / "exp_seed2_done.json").exists()
    assert json.load(open(drive / "progress.json"))["seed"] == 2   # heartbeat shows where the run stopped

    # 3. restart on the same VM: seed 1 is loaded, the partial log of seed 2 is kept under a new name
    calls = _count_training(monkeypatch)
    same_vm = _run(cfg, data, tmp_path / "vm1", drive)
    assert calls == ["exp_seed2"]
    partial = glob.glob(str(tmp_path / "vm1" / "logs" / "exp_seed2.partial-*.jsonl"))
    assert len(partial) == 1 and open(partial[0]).read().count('"event": "epoch"') == 1
    assert _table(same_vm) == _table(reference)

    # 4. a new VM: empty out_dir, everything comes back from the persistent folder, nothing is retrained
    calls.clear()
    new_vm = _run(cfg, data, tmp_path / "vm2", drive)
    assert calls == []
    assert _table(new_vm) == _table(reference)
    assert all(os.path.isfile(r["ckpt"]) and str(tmp_path / "vm2") in r["ckpt"] for r in new_vm["runs"])


def test_restart_after_interrupt_on_new_vm_trains_only_missing_seed(tiny_setup, tmp_path, monkeypatch):
    cfg, data = tiny_setup
    reference = _run(cfg, data, tmp_path / "ref_out", tmp_path / "ref_drive")
    real_progress = train.write_progress
    monkeypatch.setattr(train, "write_progress",
                        lambda c, **f: (_ for _ in ()).throw(VMReclaimed) if f.get("run") == "exp_seed2" else real_progress(c, **f))
    with pytest.raises(VMReclaimed):
        _run(cfg, data, tmp_path / "vm1", tmp_path / "drive")
    monkeypatch.setattr(train, "write_progress", real_progress)
    calls = _count_training(monkeypatch)
    resumed = _run(cfg, data, tmp_path / "vm2", tmp_path / "drive")
    assert calls == ["exp_seed2"]
    assert _table(resumed) == _table(reference)


def test_persist_never_overwrites(tiny_setup, tmp_path):
    cfg, data = tiny_setup
    _run(cfg, data, tmp_path / "out", tmp_path / "drive")
    marker = tmp_path / "drive" / "results" / "exp_seed1_done.json"
    before = marker.read_bytes()
    c = cfg.but(out_dir=str(tmp_path / "out"), persist_dir=str(tmp_path / "drive"))
    (tmp_path / "out" / "results" / "exp_seed1_done.json").write_text('{"tampered": true}')
    experiments.persist(c, "exp")
    assert marker.read_bytes() == before


def test_different_config_is_refused(tiny_setup, tmp_path):
    cfg, data = tiny_setup
    _run(cfg, data, tmp_path / "out", tmp_path / "drive")
    with pytest.raises(FileExistsError):
        _run(cfg.but(lr=1e-3), data, tmp_path / "out", tmp_path / "drive")


def test_persist_dir_is_not_part_of_identity(tiny_setup, tmp_path, monkeypatch):
    cfg, data = tiny_setup
    _run(cfg, data, tmp_path / "out", tmp_path / "drive_a")
    calls = _count_training(monkeypatch)
    _run(cfg, data, tmp_path / "out", tmp_path / "drive_b")
    assert calls == []
