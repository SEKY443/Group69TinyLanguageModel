# Report revision for review: `CITS4012_73_revised.tex`

**Status: draft for group review. Nothing in the submission has changed.**
`report/CITS4012_73.tex`, `report/CITS4012_73.pdf` and `submission/` are untouched. The revised wording lives only in
`report/CITS4012_73_revised.tex`, next to the original, so both are easy to compare and edit.

## What was changed and why
The prose was rewritten to read more naturally: active voice ("we"), shorter sentences, fewer stacked parentheses and
fewer repeated connectors ("therefore" appeared eight times). The content is the same:

- every number, citation, equation, table, figure and section is unchanged;
- no claim was strengthened or weakened (an independent second-model review compared the two versions paragraph by
  paragraph; the precision issues it found were fixed);
- Team Contributions is unchanged.

## How to review
1. See every change side by side (word level):
   ```bash
   git diff --no-index --word-diff=color report/CITS4012_73.tex report/CITS4012_73_revised.tex
   ```
   Any editor works for edits. To read it as a PDF, upload the `report/` folder to Overleaf with
   `CITS4012_73_revised.tex` as the main file, or build it locally (step 4 below) without replacing anything.
2. Edit `CITS4012_73_revised.tex` freely, with one rule: **sentences that carry result numbers must keep their exact
   wording around the numbers.** `tools/check_report_numbers.py` searches the report for 53 exact snippets, e.g.
   `the BiLSTM by +2.0 / +3.1 points (McNemar $p{=}0.09$ on the evaluation split)`; the full list is the dictionary
   at the end of `prose_claims()` in that file. Changing the words around a snippet is fine; changing the snippet
   itself makes the check fail.
3. Keep the length about the same. The main content must end on page 6, and the revised version ends there with
   about the same slack as the original.

## After the group has agreed
Only then replace the original and export the PDF (the brief requires LaTeX with the official ACL template, so the
PDF must come from `report/build.sh`, not from Word or another editor):
```bash
cp report/CITS4012_73_revised.tex report/CITS4012_73.tex
sh report/build.sh                              # pdflatex + BibTeX: expect "pages: 8", no warnings
python tools/check_report_numbers.py            # expect 0 errors and 0 mismatches
python tools/build_submission.py                # rebuilds submission/ with new SHA256SUMS and MANIFEST.json
python -m pytest                                # test_submission.py checks the frozen copies
git rm report/CITS4012_73_revised.tex report/REVISION_REVIEW.md
```

## Checks already run on the revised version (Google Colab, 2026-10-07)
Run on a clean Colab VM with the exact versions of `requirements-lock.txt` and TeX Live (pdflatex), with the revised
file temporarily in place of the original in a separate copy of the repository:

| Check | Result |
|---|---|
| `tools/check_report_numbers.py` | 46 table values and 53 prose claims, 0 errors; 0 untraceable numbers |
| `sh report/build.sh` | 8 pages, 0 overfull boxes, no LaTeX warnings |
| `tools/build_submission.py --check-only` | all checks pass: main content ends on page 6, Team Contributions on page 7 |

The same run also re-checked the repository as it is: `python -m pytest` 79 passed, notebook = `src/`, no test-label
access before the final-results cell, and the original report passes the same three checks.
