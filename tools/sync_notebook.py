"""Copies the modules in src/ into their code cells in CITS4012_69.ipynb.

Each module cell is found by the module docstring on its first line. Lines ending in `# nb-skip` (imports between
modules and the local smoke test) are dropped, because in the notebook every module shares one namespace.
Cells whose code changes have their stale outputs cleared; all other cells, including their outputs, are untouched.

Usage:  python tools/sync_notebook.py [--check]
        --check only reports which cells are out of date (exit code 1 if any).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "CITS4012_69.ipynb"
MODULES = ["config", "data", "model", "baselines", "train", "evaluate", "experiments", "viz"]


def notebook_source(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    text = "\n".join(line for line in lines if not line.rstrip().endswith("# nb-skip"))
    return re.sub(r"\n{4,}", "\n\n\n", text).strip()   # removed imports must not leave extra blank lines


def main(check_only=False):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    stale = []
    for name in MODULES:
        src = notebook_source(ROOT / "src" / f"{name}.py")
        first_line = src.splitlines()[0]
        matches = [c for c in cells if "".join(c["source"]).startswith(first_line)]
        if len(matches) != 1:
            sys.exit(f"{name}.py: expected exactly one notebook cell starting with {first_line!r}, found {len(matches)}")
        cell = matches[0]
        if "".join(cell["source"]).strip() != src:
            stale.append(name)
            if not check_only:
                cell["source"] = src
                cell["outputs"], cell["execution_count"] = [], None
    if check_only:
        print("out of date:", ", ".join(stale) if stale else "none")
        return 1 if stale else 0
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("updated:", ", ".join(stale) if stale else "nothing (already in sync)")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
