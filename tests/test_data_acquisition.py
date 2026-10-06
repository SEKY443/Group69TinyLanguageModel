"""Data acquisition for a clean runtime: the official-release fallback is used only when it is byte-identical to the
unit's files, and every source is checked against the SHA-256 of the data behind the reported results (offline)."""
import hashlib
import os
import shutil

import pytest

import data
from config import Config
from conftest import write_synthetic_piqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_release(src_dir):
    """A stand-in for urllib.request.urlretrieve that serves the files of src_dir under the release's names."""
    reverse = {remote: local for local, remote in data.PIQA_RELEASE_NAMES.items()}

    def retrieve(url, target):
        shutil.copy(os.path.join(src_dir, reverse[url.rsplit("/", 1)[1]]), target)
    return retrieve


def test_expected_hashes_are_the_ones_recorded_by_the_final_run():
    import json
    manifest = json.load(open(os.path.join(ROOT, "outputs", "data_manifest.json"), encoding="utf-8"))
    assert manifest["files"] == data.EXPECTED_SHA256


def test_release_with_different_bytes_is_rejected(tmp_path, monkeypatch):
    src = write_synthetic_piqa(str(tmp_path / "other"))
    monkeypatch.setattr("urllib.request.urlretrieve", _fake_release(src))
    cfg = Config(data_dir=str(tmp_path / "piqa"))
    assert data.download_piqa_release(cfg) is False
    assert not os.path.exists(cfg.data_dir)


def test_release_with_identical_bytes_is_used_and_verified(tmp_path, monkeypatch, capsys):
    src = write_synthetic_piqa(str(tmp_path / "release"))
    digests = {f: hashlib.sha256(open(os.path.join(src, f), "rb").read()).hexdigest() for f in data.EXPECTED_FILES}
    monkeypatch.setattr(data, "EXPECTED_SHA256", digests)
    monkeypatch.setattr("urllib.request.urlretrieve", _fake_release(src))
    monkeypatch.setattr(data, "_in_colab", lambda: False)
    cfg = Config(data_dir=str(tmp_path / "piqa"))
    assert data.prepare_data(cfg) == cfg.data_dir
    assert data.verify_piqa_files(cfg.data_dir)
    assert "SHA-256 identical" in capsys.readouterr().out


def test_failed_download_falls_back_to_the_zip(tmp_path, monkeypatch):
    def offline(url, target):
        raise OSError("offline")
    monkeypatch.setattr("urllib.request.urlretrieve", offline)
    monkeypatch.setattr(data, "_in_colab", lambda: False)
    called = []
    monkeypatch.setattr(data, "extract_zip", lambda cfg: (called.append(cfg.data_dir),
                                                          write_synthetic_piqa(cfg.data_dir))[1])
    cfg = Config(data_dir=str(tmp_path / "piqa"))
    data.prepare_data(cfg)
    assert called == [cfg.data_dir]


def test_mismatching_local_files_are_flagged(tmp_path, capsys):
    folder = write_synthetic_piqa(str(tmp_path / "piqa"))
    assert data.verify_piqa_files(folder) is False
    assert "SHA-256 mismatch" in capsys.readouterr().out


@pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, "data", "piqa", "train.jsonl")),
                    reason="the real PIQA files are not in data/piqa")
def test_local_piqa_copy_is_the_reported_data():
    assert data.verify_piqa_files(os.path.join(ROOT, "data", "piqa"))
