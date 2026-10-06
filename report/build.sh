#!/usr/bin/env sh
# Builds report/CITS4012_69.pdf with pdflatex + BibTeX (TeX Live; ACL template files are in this folder).
# Usage (from the repository root or from report/):  sh report/build.sh
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error CITS4012_69.tex > /dev/null
bibtex CITS4012_69 > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error CITS4012_69.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error CITS4012_69.tex > /dev/null
grep -q "Rerun to get" CITS4012_69.log && pdflatex -interaction=nonstopmode -halt-on-error CITS4012_69.tex > /dev/null
echo "pages: $(pdfinfo CITS4012_69.pdf 2>/dev/null | awk '/^Pages/ {print $2}')"
echo "overfull boxes: $(grep -c 'Overfull' CITS4012_69.log || true)"
grep -E "LaTeX Warning|Citation .* undefined" CITS4012_69.log || echo "no LaTeX warnings"
