"""Exports the analysis numbers that the executed notebook PRINTS (but does not save as CSV) to
outputs/results/analysis_numbers.csv, so that every number in the report is traceable to a results file.

Sources: the saved outputs of CITS4012_73.ipynb (attention controls, entropy, items only DACT solves) and the
attention data saved next to the case figures (outputs/figures/*.png.json). Nothing is recomputed.
Usage: python tools/export_analysis_numbers.py
"""
import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(open(os.path.join(ROOT, "CITS4012_73.ipynb"), encoding="utf-8"))
text = "\n".join("".join(o.get("text", "")) or "".join(o.get("data", {}).get("text/plain", ""))
                 for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))

PATTERNS = {
    "items_only_dact_solves_vs_tfidf": r"(\d+) test items are solved by DACT but not by TF-IDF",
    "tag_bias_other_shared_diff": r"learned pooling tag bias \[other, shared, diff\]: \[([^\]]+)\]",
    "attention_mass_differing_trained": r"mean attention mass on differing tokens: ([\d.]+)",
    "attention_mass_uniform_reference": r"uniform-attention reference ([\d.]+)",
    "attention_mass_correct_predictions": r"correct predictions: ([\d.]+)",
    "attention_mass_wrong_predictions": r"wrong predictions: ([\d.]+)",
    "attention_mass_correct_vs_wrong_mannwhitney_p": r"Mann-Whitney U \(correct vs wrong\): p = ([\d.e-]+)",
    "attention_mass_trained_tag_bias_0": r"trained, tag bias = 0\s+([\d.]+)",
    "attention_mass_untrained": r"untrained \(initialisation\)\s+([\d.]+)",
    "pooling_entropy_mean_correct_wrong_p": r"normalised pooling entropy \(1 = uniform\): mean ([\d.]+) \| correct ([\d.]+) \| wrong ([\d.]+) \| Mann-Whitney p = ([\d.]+)",
}
rows = []
for name, pat in PATTERNS.items():
    m = re.search(pat, text)
    if not m:
        raise SystemExit(f"pattern not found in notebook outputs: {name}")
    rows.append({"quantity": name, "value": " ".join(" ".join(m.groups()).split()), "source": "CITS4012_73.ipynb outputs"})
for f in sorted(glob.glob(os.path.join(ROOT, "outputs", "figures", "*.png.json"))):
    d = json.load(open(f, encoding="utf-8"))
    rows.append({"quantity": f"case_{os.path.basename(f)[:-9]}_probabilities_option1_option2",
                 "value": " ".join(f"{p:.4f}" for p in d["probs"]), "source": os.path.relpath(f, ROOT)})
out = os.path.join(ROOT, "outputs", "results", "analysis_numbers.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["quantity", "value", "source"])
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['quantity']:60s} {r['value']}")
print("written", os.path.relpath(out, ROOT))
