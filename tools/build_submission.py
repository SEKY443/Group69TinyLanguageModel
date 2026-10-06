"""Builds and checks the two files the brief requires: submission/CITS4012_<ID>.pdf and submission/CITS4012_<ID>.ipynb.

The working files keep their repository names (report/CITS4012_69.pdf, CITS4012_69.ipynb; "Group69" is the
repository name). The group's assigned ID is 73, so the submitted copies are named CITS4012_73.*. Before copying,
this tool checks the format rules of the brief that can be checked mechanically:
  notebook: valid nbformat, the template's three section titles unchanged, no error outputs, module cells equal src/,
            the group ID in the title cell;
  report:   \\usepackage[final]{acl}, \\author{Group <ID>} as the only author, main content (abstract to the end of
            the Conclusion) within six pages, i.e. Team Contributions starts on page 7 at the latest and nothing of
            the main text follows it.
It then writes submission/SHA256SUMS and submission/MANIFEST.json (freeze record: commits, hashes, environments).

Usage:  python tools/build_submission.py [--group 73] [--check-only]
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "CITS4012_69.ipynb")
TEX = os.path.join(ROOT, "report", "CITS4012_69.tex")
PDF = os.path.join(ROOT, "report", "CITS4012_69.pdf")
OUT = os.path.join(ROOT, "submission")
TEMPLATE_SECTIONS = ("# 1.Dataset Processing", "# 2. Model Implementation", "# 3.Testing and Evaluation")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pdf_pages_text(path):
    """Text of every page (poppler's pdftotext, else pypdf)."""
    try:
        n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", path], capture_output=True, text=True,
                                                             check=True).stdout)[1])
        return [subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), path, "-"], capture_output=True, text=True,
                               check=True).stdout for p in range(1, n + 1)]
    except (OSError, subprocess.CalledProcessError):
        from pypdf import PdfReader
        return [page.extract_text() or "" for page in PdfReader(path).pages]


def check_notebook(group):
    import nbformat
    errors = []
    nb = nbformat.read(NOTEBOOK, as_version=4)
    try:
        nbformat.validate(nb)
    except nbformat.ValidationError as e:
        errors.append(f"notebook does not validate: {str(e).splitlines()[0]}")
    headers = [c.source.strip() for c in nb.cells if c.cell_type == "markdown"]
    for title in TEMPLATE_SECTIONS:
        if not any(h.splitlines()[0].strip() == title for h in headers if h):
            errors.append(f"template section title missing or modified: {title!r}")
    n_err = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error")
    if n_err:
        errors.append(f"{n_err} error outputs in the notebook")
    if f"Group {group}" not in nb.cells[0].source:
        errors.append(f"title cell does not name Group {group}")
    sync = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "sync_notebook.py"), "--check"],
                          capture_output=True, text=True)
    if sync.returncode != 0:
        errors.append("notebook module cells differ from src/: " + sync.stdout.strip())
    stats = {"cells": len(nb.cells), "code_cells": sum(c.cell_type == "code" for c in nb.cells),
             "code_cells_with_outputs": sum(c.cell_type == "code" and bool(c.outputs) for c in nb.cells),
             "error_outputs": n_err}
    return errors, stats


def check_report(group):
    errors = []
    tex = open(TEX, encoding="utf-8").read()
    if "\\usepackage[final]{acl}" not in tex or "[review]{acl}" in tex:
        errors.append("report must use \\usepackage[final]{acl}")
    authors = re.findall(r"\\author\{([^}]*)\}", tex)
    if authors != [f"Group {group}"]:
        errors.append(f"author field must be exactly 'Group {group}', found {authors}")
    pages = pdf_pages_text(PDF)
    tc = [i for i, t in enumerate(pages, 1) if "Team Contributions" in t]
    concl = [i for i, t in enumerate(pages, 1) if re.search(r"\n\s*\d\s+Conclusion\s*\n", "\n" + t)]
    if not tc:
        errors.append("no Team Contributions section found in the PDF")
    else:
        first_tc = tc[0]
        before_tc = pages[first_tc - 1].split("Team Contributions")[0].strip()
        content_end = first_tc if before_tc else first_tc - 1      # last page that holds main content
        if content_end > 6:
            errors.append(f"main content runs to page {content_end} (limit: 6 pages)")
        if concl and concl[0] > 6:
            errors.append(f"Conclusion starts on page {concl[0]}")
    stats = {"pdf_pages": len(pages), "team_contributions_page": tc[0] if tc else None,
             "main_content_last_page": (content_end if tc else None)}
    return errors, stats


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="73")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    nb_errors, nb_stats = check_notebook(args.group)
    rep_errors, rep_stats = check_report(args.group)
    errors = nb_errors + rep_errors
    print("notebook:", nb_stats)
    print("report:  ", rep_stats)
    for e in errors:
        print("ERROR:", e)
    if errors:
        return 1
    if args.check_only:
        print("OK: all submission checks pass")
        return 0
    os.makedirs(OUT, exist_ok=True)
    targets = {f"CITS4012_{args.group}.pdf": PDF, f"CITS4012_{args.group}.ipynb": NOTEBOOK}
    for name, src in targets.items():
        shutil.copyfile(src, os.path.join(OUT, name))
    sums = {name: sha256(os.path.join(OUT, name)) for name in targets}
    with open(os.path.join(OUT, "SHA256SUMS"), "w", encoding="utf-8") as f:
        f.writelines(f"{h}  {name}\n" for name, h in sums.items())
    run = json.load(open(os.path.join(ROOT, "outputs", "run_config.json"), encoding="utf-8"))
    manifest = {
        "group": args.group, "built": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source_commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_clean_for_sources": git("status", "--porcelain", "--", "CITS4012_69.ipynb", "report", "src") == "",
        "files": {name: {"sha256": h, "source": os.path.relpath(targets[name], ROOT), "bytes": os.path.getsize(targets[name])}
                  for name, h in sums.items()},
        "report_source_sha256": sha256(TEX), "notebook": nb_stats, "report": rep_stats,
        "reported_run": {"commit": run["commit"], "time": run["time"], "environment": run["environment"],
                         "data_sha256": run["data"]["files"] if run.get("data") else None,
                         "tokenizer_sha256": run["data"].get("tokenizer_sha256") if run.get("data") else None},
        "results_sha256": {os.path.relpath(p, ROOT): sha256(p) for p in sorted(
            os.path.join(ROOT, "outputs", "results", f) for f in os.listdir(os.path.join(ROOT, "outputs", "results"))
            if f.endswith((".csv", ".json")) and not f.endswith("_done.json"))},
        "build_environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("written:", ", ".join(f"submission/{n} ({h[:12]})" for n, h in sums.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
