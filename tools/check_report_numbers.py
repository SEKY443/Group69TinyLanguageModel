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
    "10": "validation share 10 %", "50": "chance level 50 %", "50.0": "TRAIN label-1 share (notebook statistics cell)",
    "0.3": "dropout of the grid candidate 'base, dropout 0.3' (notebook Step 1)", "15": "median share of differing tokens (notebook statistics)",
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
    "33": "training runs with logs in the final run (notebook consistency-check cell: '33 training runs with logs')",
    "69.6": "RoBERTa validation re-run, seed 42 (experiments/roberta_stability; checked as a strict claim)",
    "3.6": "section number (Section 3.6, Experimental Protocol)",
    "125": "RoBERTa-base parameters in millions (124,646,401 in outputs/logs/B3_*.jsonl)",
    "800": "parameter ratio Qwen2.5-1.5B / DACT, about 1.5e9 / 1.93e6 (model name; checked as a strict claim)",
    "65": "parameter ratio RoBERTa / DACT, 124.6 M / 1.93 M (checked as a strict claim)",
    "52": "RoBERTa restart threshold, 52 % validation accuracy after epoch 1 (declared rule, notebook Step 4)",
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
          "B2 BiLSTM + attention": "B2 BiLSTM + attention", "(main model)": "DACT (full)",
          "B3 RoBERTa-base, fine-tuned": "B3 RoBERTa-base (fine-tuned)", "B4 Qwen2.5-1.5B, zero-shot": "B4 Qwen2.5-1.5B (zero-shot)"}
TABLE2 = {"difference tags": "− difference tags", "lexical head": "− lexical head",
          "pointwise": "pointwise objective", "mean instead": "mean pooling",
          "cross-solution attention": "− cross-solution attention", "tag bias in pooling": "− diff bias in pooling",
          "tag bias initialised at 0": "diff-bias prior 0", "$-$ MLM warm-up": "− MLM warm-up",
          "vanilla": "vanilla Transformer"}


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
        # same convention as the notebook's key-numbers cell: difference of the UNROUNDED means, rounded once
        expect = [round(100 * (float(r[c]) - float(d[c])), 1) for c in ("val acc", "test acc")]
        expect = [0.0 if e == 0 else e for e in expect]
        checked += 2
        if [round(c, 1) for c in cells] != expect:
            errors.append(f"Table 2 '{label}': report {cells} vs results {expect}")
    # (Table 3, the development history, was folded into Section 4.5; its numbers are prose claims now)
    p = round(float(fr["B1 TF-IDF + LR"]["McNemar p vs DACT"]), 2)
    checked += 1
    if f"p{{=}}{p:.2f}" not in s:
        errors.append(f"headline McNemar p vs TF-IDF should be {p:.2f}")
    return checked, errors


# ---------------------------------------------------------------- strict check of the prose (section 3.4)
def _csv(path, key):
    return {r[key]: r for r in csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8"))}


def _json(path):
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def p1(x):
    """A fraction as a percentage with one decimal, as written in the report."""
    return f"{100 * float(x):.1f}"


def prose_claims():
    """Every result number in the report's prose, as the exact LaTeX text it must appear in, built from one named
    result-file cell. A changed result or a typo in the report makes the snippet disappear -> error."""
    fr = _csv("outputs/results/final_results.csv", "model")
    rs = _csv("outputs/results/rescore_without_duplicate.csv", "experiment")
    an = {k: v["value"] for k, v in _csv("outputs/results/analysis_numbers.csv", "quantity").items()}
    eo = _csv("outputs/results/error_overlap.csv", "")
    probe = _csv("outputs/results/diff_probe_by_layer.csv", "layer")
    goal = _csv("experiments/goal_matching/results.csv", "arm")
    hp = [_json(f"outputs/results/hp{i}_val.json")["val"]["mean"] for i in range(5)]
    v = lambda m: p1(fr[m]["val acc"])    # noqa: E731
    t = lambda m: p1(fr[m]["test acc"])   # noqa: E731
    D, B1, B2 = "DACT (full)", "B1 TF-IDF + LR", "B2 BiLSTM + attention"

    def delta(m, col, ref=D, sign=False):
        # difference of the unrounded means, as in the notebook's key-numbers cell and Table 2
        d = round(100 * (float(fr[m][col]) - float(fr[ref][col])), 1)
        d = 0.0 if d == 0 else d
        return f"{d:+.1f}" if sign else f"{abs(d):.1f}"

    def cost(m):
        return f"{delta(m, 'val acc')} / {delta(m, 'test acc')}"
    pent = an["pooling_entropy_mean_correct_wrong_p"].split()
    case = {re.search(r"_(\d+)_prob", k)[1]: max(map(float, val.split())) for k, val in an.items() if k.startswith("case_")}
    # run 2's logs were not archived; its RoBERTa seeds are recorded in the development log (section 24)
    log = open(os.path.join(ROOT, "docs", "DEVELOPMENT_LOG.md"), encoding="utf-8").read()
    run2_b3 = min(float(x) for x in re.search(r"RoBERTa seeds\.\*\* Validation accuracy was ([\d.]+) / \*\*([\d.]+)\*\* / ([\d.]+)", log).groups()) / 100
    run4 = _csv("experiments/a100_run_0d70cdc/outputs/results/final_results.csv", "model")
    moved = max(abs(round(100 * float(run4[m][c]), 1) - round(100 * float(fr[m][c]), 1))
                for m in fr if m != D and m in run4 and fr[m]["from scratch"] == "True"
                for c in ("val acc", "test acc") if fr[m][c] and run4[m][c])
    b3 = "B3 RoBERTa-base (fine-tuned)"
    b3_seeds = {}
    for f in glob.glob(os.path.join(ROOT, "outputs", "logs", "B3_*.jsonl")):
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if any(r.get("event") == "end" for r in recs):
            b3_seeds[f] = max(r["val_acc"] for r in recs if r.get("event") == "epoch")
    b3_ci = [float(x) for x in re.findall(r"[\d.]+", fr[b3]["test 95% CI"])]
    pval = lambda m: float(fr[m]["McNemar p vs DACT"])  # noqa: E731
    rerun = [100 * _json(f"experiments/roberta_stability/run_20261004/control_lr2e-5_seed{s}.json")["best_val_acc"]
             for s in (42, 43, 44)]
    rerun_mean = sum(rerun) / 3
    rerun_std = (sum((x - rerun_mean) ** 2 for x in rerun) / 2) ** 0.5
    n_dact = _json("outputs/results/dact_full_val.json")["runs"][0]["n_params"]
    n_roberta = next(json.loads(l)["n_params"] for l in open(os.path.join(ROOT, "outputs", "logs", "B3_roberta-base_seed43.jsonl"))
                     if '"event": "start"' in l)
    ratio_roberta = round(n_roberta / n_dact)                    # Qwen2.5-1.5B: about 1.5e9 / 1.93e6, from its name
    ratio_qwen = round(1.5e9 / n_dact, -2)
    run4_vanilla = [round(100 * (float(run4["vanilla Transformer"][c]) - float(run4[D][c])), 1) for c in ("val acc", "test acc")]
    hp_top3 = sorted(hp, reverse=True)[:3]
    llm_gap = float(delta('B4 Qwen2.5-1.5B (zero-shot)', 'test acc'))
    roberta_gap = rerun_mean - 100 * float(fr[D]["val acc"])
    p_wrong = float(an["attention_mass_correct_vs_wrong_mannwhitney_p"])
    tags_test = delta("− difference tags", "test acc")
    return {
        "abstract: DACT val": f"among the models trained from scratch ({v(D)}\\%)",
        "abstract: DACT vs TF-IDF test, p": f"({t(D)} vs.\\ {t(B1)}\\%, McNemar $p{{=}}{pval(B1):.2f}$)",
        "conclusion: parameter ratios and accuracies": f"models with {ratio_roberta}--{ratio_qwen:.0f} times more parameters and pretraining reach {rerun_mean:.0f}--{100 * float(fr['B4 Qwen2.5-1.5B (zero-shot)']['test acc']):.0f}\\%",
        "duplicate: re-scoring change": f"changes accuracy by about {abs(float(rs['dact_full']['difference_points'])):.2f} points",
        "duplicate: DACT": f"{100 * float(rs['dact_full']['test_acc_all_1838']):.2f}\\,$\\to$\\,{100 * float(rs['dact_full']['test_acc_without_duplicate_1837']):.2f}\\%",
        "duplicate: TF-IDF": f"{100 * float(rs['B1_tfidf_lr']['test_acc_all_1838']):.2f}\\,$\\to$\\,{100 * float(rs['B1_tfidf_lr']['test_acc_without_duplicate_1837']):.2f}\\%",
        "data: val / test label-1 share (B0 accuracy)": f"/ {p1(fr['B0 majority']['val acc'])} / {p1(fr['B0 majority']['test acc'])}\\% label 1",
        "grid: base": f"4 layers) {p1(hp[0])}\\%",
        "grid: small": f"4 heads) {p1(hp[1])}\\%",
        "grid: dropout 0.3": f"dropout 0.3 {p1(hp[2])}\\%",
        "grid: MLM warm-up (selected)": f"{p1(hp[3])} / \\textbf{{{p1(hp[4])}\\%}}",
        "grid: selection margin": f"{100 * (hp[4] - hp[0]):.2f} points ahead of base",
        "params: DACT": f"{_json('outputs/results/dact_full_val.json')['runs'][0]['n_params'] / 1e6:.2f}\\,M parameters",
        "params: BiLSTM": f"({_json('outputs/results/b2_bilstm_val.json')['runs'][0]['n_params'] / 1e6:.1f}\\,M parameters)",
        "vs BiLSTM": f"+{delta(B2, 'val acc')} / +{delta(B2, 'test acc')} points over the BiLSTM (McNemar $p{{=}}{pval(B2):.2f}$ on test)",
        "tie in both runs": f"detected ($p{{=}}{pval(B1):.2f}$; $p{{=}}{float(run4[B1]['McNemar p vs DACT']):.2f}$ in the A100 run)",
        "vs TF-IDF": f"+{delta(B1, 'val acc')} / +{delta(B1, 'test acc')} over TF-IDF",
        "pretraining gap (test)": f"the zero-shot LLM (about {ratio_qwen:.0f} times more parameters) is {delta('B4 Qwen2.5-1.5B (zero-shot)', 'test acc')} points above",
        "RoBERTa gap (validation re-run)": f"RoBERTa ({ratio_roberta} times more parameters), when its fine-tuning succeeds, is about {roberta_gap:.0f} points above on validation ({rerun_mean:.1f} vs.\\ {v(D)}\\%)",
        "grid: top-3 spread and seed std": f"lie within {hp_top3[0] * 100 - hp_top3[2] * 100:.2f} points, less than one seed std of the final model ({100 * float(fr[D]['val std']):.1f})",
        "B3 working seed (caption)": f"the working seed reaches {p1(max(b3_seeds.values()))}\\% (test CI {100 * b3_ci[0]:.1f}--{100 * b3_ci[1]:.1f})",
        "B3 A100 reference": f"An A100 run gave {p1(run4[b3]['val acc'])} / {p1(run4[b3]['test acc'])}",
        "B3 re-run seeds (caption)": "all three seeds trained ({:.1f} / {:.1f} / {:.1f})".format(*rerun),
        "B3 re-run table row": f"validation re-run$^\\ddagger$ & {rerun_mean:.1f}\\,$\\pm$\\,{rerun_std:.1f} & --",
        "ablation: vanilla, both runs": f"reduced accuracy by {cost('vanilla Transformer')} points, the largest drop and the only significant one on test ($p{{=}}{pval('vanilla Transformer'):.3f}$); in the A100 run the same comparison cost {abs(run4_vanilla[0]):.1f} / {abs(run4_vanilla[1]):.1f}",
        "ablation: lexical": f"(removing it costs {cost('− lexical head')} points)",
        "ablation: pointwise": f"(independent binary scoring: {cost('pointwise objective')})",
        "ablation: tags": f"The difference tags help on test only ({cost('− difference tags')})",
        "ablation: cross": f"cross-solution attention (removing it gives {delta('− cross-solution attention', 'val acc', sign=True)} / $-${delta('− cross-solution attention', 'test acc')})",
        "ablation: mean pooling": f"attention pooling ({delta('mean pooling', 'test acc', sign=True)} on test with mean pooling)",
        "ablation: MLM warm-up": f"the MLM warm-up ({delta('− MLM warm-up', 'val acc', sign=True)} / $-${delta('− MLM warm-up', 'test acc')})",
        "errors: agreement": f"agree on only {p1(eo[B1]['agreement'])}\\% of test items",
        "conclusion: vanilla": f"the clearest improvement ({cost('vanilla Transformer')} points over a vanilla Transformer)",
        "errors: only DACT": f"alone solves {p1(eo[B1]['only DACT'])}\\% ({an['items_only_dact_solves_vs_tfidf']} items)",
        "errors: only TF-IDF / oracle": f"TF-IDF alone {p1(eo[B1]['only other'])}\\%, for an oracle accuracy of {p1(eo[B1]['oracle'])}\\%",
        "errors: oracle vs RoBERTa": f"the oracle rises to {p1(eo[b3]['oracle'])}\\%",
        "attention: trained / share": f"puts {p1(an['attention_mass_differing_trained'])}\\% of its pooling attention on differing tokens, which make up {p1(an['attention_mass_uniform_reference'])}\\%",
        "attention: untrained": f"untrained model reaches only {p1(an['attention_mass_untrained'])}\\%",
        "attention: bias 0": f"set to 0 still reaches {p1(an['attention_mass_trained_tag_bias_0'])}\\%",
        "attention: bias moved": f"(1.0\\,$\\to$\\,{an['tag_bias_other_shared_diff'].split()[-1]})",
        "entropy, p": f"entropy is {float(pent[0]):.2f} (1 = uniform), for correct and wrong predictions alike ($p{{=}}{float(pent[3]):.2f}$)",
        "probe: vanilla embedding / encoder": f"reaches {p1(probe['embedding']['vanilla (no tags)'])}\\% balanced accuracy on the vanilla Transformer's embeddings and only {p1(probe['encoder 2']['vanilla (no tags)'])}\\%",
        "attention: wrong vs correct": f"({p1(an['attention_mass_wrong_predictions'])} vs.\\ {p1(an['attention_mass_correct_predictions'])}\\%, $p{{=}}{p_wrong:.2f}$)" if p_wrong >= 0.001 else "p < 0.001",
        "case: lotion bars (747), caption": f"an incorrect prediction (p = {case['747']:.2f} for option 1; option 2 is correct)",
        "case: lotion bars (747), text": f"the model prefers this option with p = {case['747']:.2f}",
        "case: small cheap home (1516)": f"the most uncertain item (p = {case['1516']:.2f}; option 1 is correct)",
        "case: LED (993)": f"wins with p = {case['993']:.2f} against",
        "case: exotic trip (841)": f"(p = {case['841']:.2f} for a strict itinerary",
        "history: goal matching": f"it lost {100 * (float(goal['control_main_model']['val_mean']) - float(goal['goal_matching']['val_mean'])):.1f} points on validation",
        "history: run 4 vs final spread": f"moved by up to {moved:.1f} points",
        "tags help on test (consistency)": f"$-$ difference tags & $-${delta('− difference tags', 'val acc')} & $-${tags_test}",
    }


def strict_prose(s):
    claims = prose_claims()
    errors = [f"{label}: expected text not found: {snippet}" for label, snippet in claims.items() if snippet not in s]
    # coverage: every decimal number of the prose must sit inside a claim's text or be a documented non-result
    body = s[s.index(r"\begin{abstract}"):s.index(r"\section*{Team Contributions}")]
    body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", " ", body, flags=re.S)
    body = re.sub(r"\\includegraphics\[[^\]]*\]", " ", body)
    covered = [(m.start(), m.end()) for snippet in claims.values() for m in re.finditer(re.escape(snippet), body)]
    for m in re.finditer(r"(?<![\w.\\])(\d+\.\d+)(?![\w])", body):
        inside = any(a <= m.start() and m.end() <= b for a, b in covered)
        if not inside and m.group(1) not in DOCUMENTED:
            ctx = body[max(0, m.start() - 40):m.end() + 10].replace("\n", " ")
            errors.append(f"result number {m.group(1)} is not covered by a claim: ...{ctx}...")
    return len(claims), errors


def main():
    s = open(TEX, encoding="utf-8").read()
    checked, errors = strict_tables(s)
    print(f"STRICT check (Tables 1-2 row by row, headline p-value): {checked} values, {len(errors)} errors")
    for e in errors:
        print("   ", e)
    n_prose, prose_errors = strict_prose(s)
    print(f"STRICT check (result numbers in the prose, each built from a named result-file cell): {n_prose} claims, "
          f"{len(prose_errors)} errors")
    for e in prose_errors:
        print("   ", e)
    errors += prose_errors
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
