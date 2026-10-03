"""Checks every number in the report's main text against logged result files. Exit code 1 on any mismatch.

A number in report/CITS4012_69.tex (abstract to the end of the Limitations; citations, labels and figure paths are
ignored) passes if it equals, at the precision written in the report, one of:
  - a value in outputs/results/*.csv or *.json (the final run), as stored or x100 (percentages);
  - a value in the archived earlier runs (experiments/*/outputs/results, evidence/historical/a100_outputs/results)
    or in experiments/*/*.csv (pilot and timing records) -- used by the development-history section;
  - a best-epoch validation accuracy in outputs/logs/B3_*.jsonl (RoBERTa seed range);
  - a difference between two models of outputs/results/final_results.csv (val or test, x100), or between a model
    and its value re-scored without the duplicate;
  - an entry of DOCUMENTED below: numbers that are not results (dataset sizes, fixed settings, literature figures).
Usage: python tools/check_report_numbers.py
"""
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "report", "CITS4012_69.tex")

# Numbers that are not experimental results; each has a stated source.
DOCUMENTED = {
    # dataset and split (README / notebook dataset cell)
    "16113": "PIQA training file size", "14501": "TRAIN size", "1612": "VAL size", "1838": "TEST (dev) size",
    "14.5": "TRAIN size in thousands (14,501)",
    "103": "order-dependent test items in the ORIGINAL pipeline (docs/PROJECT_AUDIT.md, audit track)",
    "10": "validation share 10 %", "50": "chance level 50 %", "15": "median share of differing tokens (notebook statistics)",
    "1.6": "share of test solutions longer than 96 BPE tokens (DEVELOPMENT_LOG section 5)",
    "40": "goal truncation (Config.max_goal_len)", "96": "solution truncation (Config.max_sol_len)",
    "97": "token position of the cobbler item's difference (DEVELOPMENT_LOG section 16)",
    "98": "token position of the cobbler item's difference (DEVELOPMENT_LOG section 16)",
    # configuration (src/config.py; Step 1 grid)
    "1.9": "parameters in millions (1,927,492, notebook sanity cell)", "1.93": "parameters in millions",
    "262": "lexical-head parameters in thousands (2^18 = 262,144)", "18": "2^18 hash buckets",
    "165": "goal-matching block parameters in thousands (165,504; experiments/goal_matching/PREREGISTRATION.md)",
    "128": "d_model / B3 max length / batch size", "256": "base d_model", "512": "d_ff (small)",
    "4": "layers (base) / heads / B3 epochs", "2": "layers (small) / BiLSTM layers / two options",
    "0.98": "AdamW beta2", "0.05": "weight decay", "0.2": "dropout", "0.1": "label smoothing",
    "1.0": "gradient clipping / initial tag bias", "6": "warm-up % and patience / six duplicate rows",
    "20": "maximum epochs", "3": "seeds / learning-rate exponent", "5": "B3 lr exponent", "42": "seed", "44": "seed",
    "2000": "bootstrap resamples", "95": "confidence level", "1.5": "Qwen2.5-1.5B model size",
    "5.1": "BiLSTM parameters in millions", "8000": "BPE vocabulary", "8": "BPE vocabulary in thousands",
    "1": "one item / one seed", "0": "zero / initial bias 0", "7": "seven single-component comparisons",
    # literature (Bisk et al. 2020, Table 2)
    "77.1": "RoBERTa-large on PIQA (Bisk et al., 2020)", "94.9": "human accuracy on PIQA (Bisk et al., 2020)",
    # development history (DEVELOPMENT_LOG sections 24-26)
    "35": "free-tier session lifetime in minutes (section 24)", "0.694": "chance-level loss ln 2 of the failed RoBERTa seed",
    "1545": "index of the duplicated test item (audit_checks.csv)", "26": "count from the baby-wipes analysis (section 16)",
    "100": "probe accuracy 100 % / percentages",
}


def collect_values():
    vals = []
    files = glob.glob(os.path.join(ROOT, "outputs", "results", "*.csv"))
    files += glob.glob(os.path.join(ROOT, "experiments", "**", "results", "*.csv"), recursive=True)
    files += glob.glob(os.path.join(ROOT, "experiments", "*", "*.csv"))
    files += glob.glob(os.path.join(ROOT, "evidence", "historical", "a100_outputs", "results", "*.csv"))
    for f in files:
        for row in csv.reader(open(f, encoding="utf-8")):
            for cell in row:
                vals += [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", cell)]
    jfiles = glob.glob(os.path.join(ROOT, "outputs", "results", "*.json"))
    jfiles += glob.glob(os.path.join(ROOT, "experiments", "**", "results", "*.json"), recursive=True)
    for f in jfiles:
        vals += [float(x) for x in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", open(f, encoding="utf-8").read())]
    for f in glob.glob(os.path.join(ROOT, "outputs", "logs", "B3_*.jsonl")):
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        vals += [r["val_acc"] for r in recs if "val_acc" in r]
    # differences between models of the final results table, and duplicate re-scoring differences
    fr = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "results", "final_results.csv"), encoding="utf-8")))
    for col in ("val acc", "test acc"):
        xs = [float(r[col]) for r in fr if r[col]]
        vals += [abs(a - b) for a in xs for b in xs]
    return vals


def matches(text, vals):
    x = float(text)
    dec = len(text.split(".")[1]) if "." in text else 0
    for v in vals:
        for cand in (v, 100 * v):
            if round(abs(cand), dec) == x or round(cand, dec) == x:
                return True
    return False


TABLE1 = {"B0 majority": "B0 majority", "B1 TF-IDF + LR": "B1 TF-IDF + LR",
          "B2 BiLSTM + attention": "B2 BiLSTM + attention", "(ours)": "DACT (full)",
          "B3 RoBERTa-base, fine-tuned": "B3 RoBERTa-base (fine-tuned)", "B4 Qwen2.5-1.5B, zero-shot": "B4 Qwen2.5-1.5B (zero-shot)"}
TABLE2 = {"difference tags": "− difference tags", "lexical head": "− lexical head",
          "pointwise": "pointwise objective", "mean instead": "mean pooling",
          "cross-solution attention": "− cross-solution attention", "tag bias in pooling": "− diff bias in pooling",
          "tag bias initialised at 0": "diff-bias prior 0", "vanilla": "vanilla Transformer"}


def strict_tables(s):
    """Row-by-row check of Tables 1 and 2 and the headline p-value against final_results.csv (semantic check)."""
    fr = {r["model"]: r for r in csv.DictReader(open(os.path.join(ROOT, "outputs", "results", "final_results.csv"), encoding="utf-8"))}
    pct = lambda x: f"{100 * float(x):.1f}"  # noqa: E731
    errors, checked = [], 0
    rows = [l for l in s.splitlines() if "&" in l and "\\\\" in l]
    for label, model in TABLE1.items():
        line = next(l for l in rows if label in l)
        cells = re.findall(r"\d+\.\d", line.split("&", 1)[1])        # value cells only, not the model name
        r = fr[model]
        expect = [pct(r["val acc"])] + ([pct(r["val std"])] if r["val std"] else []) + \
                 [pct(r["test acc"])] + ([pct(r["test std"])] if r["test std"] else [])
        checked += len(expect)
        if cells != expect:
            errors.append(f"Table 1 '{label}': report {cells} vs results {expect}")
    d = fr["DACT (full)"]
    for label, model in TABLE2.items():
        line = next(l for l in rows if label in l and "$\\Delta$" not in l)
        cells = [float(x.replace("$-$", "-").replace("+", "")) for x in re.findall(r"(?:\$-\$|\+)?\d+\.\d", line)]
        r = fr[model]
        # same convention as the notebook's key-numbers cell: difference of the means rounded to 0.1 points
        expect = [round(round(100 * float(r[c]), 1) - round(100 * float(d[c]), 1), 1) for c in ("val acc", "test acc")]
        expect = [0.0 if e == 0 else e for e in expect]
        checked += 2
        if [round(c, 1) for c in cells] != expect:
            errors.append(f"Table 2 '{label}': report {cells} vs results {expect}")
    p = round(float(fr["B1 TF-IDF + LR"]["McNemar p vs DACT"]), 2)
    checked += 1
    if f"p{{=}}{p:.2f}" not in s:
        errors.append(f"headline McNemar p vs TF-IDF should be {p:.2f}")
    return checked, errors


def main():
    s = open(TEX, encoding="utf-8").read()
    checked, errors = strict_tables(s)
    print(f"STRICT check (Tables 1-2 row by row, headline p-value): {checked} values, {len(errors)} errors")
    for e in errors:
        print("   ", e)
    body = s[s.index(r"\begin{abstract}"):s.index(r"\section*{Team Contributions}")]
    body = re.sub(r"\\(citep|citet|citealp|cite|ref|label|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", body)
    body = re.sub(r"\\begin\{tabular\}\{[^}]*\}", " ", body)
    nums = re.findall(r"(?<![\w.\\])(\d+(?:\{,\}\d+)*(?:\.\d+)?)(?![\w])", body)
    vals = collect_values()
    ok_result, ok_doc, bad = set(), set(), []
    for raw in nums:
        n = raw.replace("{,}", "")
        if n in ok_result or n in ok_doc:
            continue
        if "." in n and matches(n, vals):
            ok_result.add(n)
        elif n in DOCUMENTED:
            ok_doc.add(n)
        elif "." not in n and matches(n, vals):
            ok_result.add(n)
        else:
            bad.append(n)
    print(f"numbers in the main text: {len(nums)} occurrences, {len(set(x.replace('{,}', '') for x in nums))} distinct")
    print(f"  traced to logged result files: {len(ok_result)} distinct")
    print(f"  documented non-result numbers: {len(ok_doc)} distinct")
    bad = sorted(set(bad), key=float)
    print(f"  MISMATCHES (not traceable): {len(bad)}" + (f" -> {bad}" if bad else ""))
    print("  note: the traceability check is loose (a random 1-decimal number matches some logged value about half the "
          "time); the STRICT check above is the semantic verification of the result tables.")
    sys.exit(1 if (bad or errors) else 0)


if __name__ == "__main__":
    main()
