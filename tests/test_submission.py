"""The two submitted files: format checks of the brief, and byte identity with their sources (submission/SHA256SUMS)."""
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
HAVE_PDF_TEXT = shutil.which("pdftotext") is not None or importlib.util.find_spec("pypdf") is not None
HAVE_NBFORMAT = importlib.util.find_spec("nbformat") is not None


@pytest.mark.skipif(not (HAVE_PDF_TEXT and HAVE_NBFORMAT), reason="needs pdftotext or pypdf, and nbformat")
def test_submission_format_checks_pass():
    out = subprocess.run([sys.executable, "tools/build_submission.py", "--check-only"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.skipif(not os.path.isdir(SUB), reason="submission/ not built")
def test_submitted_files_are_the_repository_files():
    sums = dict(line.split()[::-1] for line in open(os.path.join(SUB, "SHA256SUMS"), encoding="utf-8") if line.strip())
    assert set(sums) == {"CITS4012_73.pdf", "CITS4012_73.ipynb"}
    sources = {"CITS4012_73.pdf": os.path.join(ROOT, "report", "CITS4012_69.pdf"),
               "CITS4012_73.ipynb": os.path.join(ROOT, "CITS4012_69.ipynb")}
    for name, digest in sums.items():
        for path in (os.path.join(SUB, name), sources[name]):
            assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest, f"{path} differs from SHA256SUMS"
