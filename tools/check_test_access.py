"""Proof of test-set discipline: lists every notebook line that touches test data and checks the ordering rules.

Rules checked:
- test LABELS (`yte`, `.labels` of the test split, `test_experiment`) are only used at or after the cell that sets
  `FINAL_EVAL = True`;
- before that cell, test data may only be loaded/tokenised or passed to baselines for held-out predictions.
Exit code 1 if a rule is broken.

Usage: python tools/check_test_access.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(open(os.path.join(ROOT, "CITS4012_69.ipynb"), encoding="utf-8"))
TOUCH = re.compile(r'DATA\["test(_ds)?"\]|test_ds|\byte\b|test_pred|test_experiment\(|\b_te\b|test_loader|FINAL_EVAL')
LABELS = re.compile(r'\byte\b|test_experiment\(|final_eval\(|load_labels\(|test-labels|for split in \("train", "val", "test"\).*labels|DATA\[f"\{split\}_ds"\]\.labels')

final_cell = None
problems = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if src.startswith('"""'):                         # module definitions (src/*.py): calls are what matters
        continue
    if re.search(r"^FINAL_EVAL = True|^yte = final_eval\(", src, re.M):
        final_cell = i
    lines = src.splitlines()
    hits = [(n + 1, l) for n, l in enumerate(lines) if TOUCH.search(l)]
    if not hits:
        continue
    stage = "FINAL/after" if final_cell is not None and i >= final_cell else "before final"
    print(f"cell {i} [{stage}]")
    for n, l in hits:
        print(f"   {n:3d}: {l.strip()[:120]}")
        label_use = LABELS.search(l) and "FINAL_EVAL" not in l
        guarded = "if (split != \"test\" or FINAL_EVAL)" in l or "if (split != \"test\" or DATA[\"test_labels_loaded\"])" in l
        if stage == "before final" and label_use and not guarded:
            problems.append(f"cell {i} line {n}: test labels used before FINAL_EVAL = True")

print("\nfinal-results cell (sets FINAL_EVAL = True):", final_cell)
if problems:
    print("PROBLEMS:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("OK: no test labels are read before the final-results cell; earlier cells only load the test data or "
      "compute held-out baseline predictions without labels.")
