"""Builds the report figures from the saved outputs of the final run (run from the repository root)."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

OUT = "report/figures"
PREVIEW = os.environ.get("FIGURE_PREVIEW_DIR")   # optional PNG previews for visual checks
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


# ---------------------------------------------------------------- 1. architecture diagram
def box(ax, x, y, w, h, text, fc, bold=False, size=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333",
                                                                    shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, ha="center", va="bottom", fontsize=5.8, style="italic")


fig = plt.figure(figsize=(7.0, 2.45))
ax = fig.add_axes([0, 0, 1, 1])                          # the whole figure is drawing area: 1 unit = 7/16.6 inch
ax.set_xlim(0, 16.6)
ax.set_ylim(0.0, 6.0)
ax.axis("off")
NEW, STD, IN = "#fde2c8", "#dfe9f5", "#f2f2f2"
H = 0.95
rows = {1: 3.55, 2: 1.0}                                  # y of the two candidate rows
for k, y in rows.items():
    box(ax, 0.05, y, 2.05, H, rf"[CLS] goal [SEP]" + "\n" + rf"sol$_{k}$ [SEP]", IN)
    box(ax, 2.75, y, 2.2, H, "embedding\n+ difference tag", NEW, bold=True)
    box(ax, 5.3, y, 2.05, H, "Transformer\nencoder (shared)", STD)
    box(ax, 10.15, y, 2.3, H, "difference-guided\npooling " + r"$\to p_{%d}$" % k, NEW, bold=True)
    box(ax, 12.8, y, 1.75, H, r"scorer" + "\n" + r"$[p_{%d}; \tilde h_{\mathrm{CLS}}] \to s_{%d}$" % (k, k), STD, size=6.2)
    arrow(ax, 2.1, y + H / 2, 2.75, y + H / 2)
    arrow(ax, 4.95, y + H / 2, 5.3, y + H / 2)
    arrow(ax, 7.35, y + H / 2, 7.75, y + H / 2)
    arrow(ax, 9.75, y + H / 2, 10.15, y + H / 2)
    arrow(ax, 12.45, y + H / 2, 12.8, y + H / 2)
    arrow(ax, 14.55, y + H / 2, 14.85, 2.5 + (0.45 if k == 1 else -0.45))
# alignment of the two solutions -> difference tags of both candidates
box(ax, 0.05, 2.45, 2.05, 0.75, "difflib alignment\n" + r"of sol$_1$ and sol$_2$", "#fff7ec", size=6.2)
arrow(ax, 1.075, rows[1], 1.075, 3.2)
arrow(ax, 1.075, rows[2] + H, 1.075, 2.45)
arrow(ax, 2.1, 2.95, 2.75, rows[1] + 0.2)
arrow(ax, 2.1, 2.7, 2.75, rows[2] + H - 0.2)
ax.text(2.2, 2.83, "tags", fontsize=6.6, style="italic", ha="left", va="center")
# cross-solution attention spans both rows
box(ax, 7.75, 1.0, 2.0, 3.5, "cross-solution\nattention\n(ESIM fusion)", NEW, bold=True)
# lexical head and the pairwise softmax
box(ax, 2.75, 4.95, 11.8, 0.7, r"lexical head: learned weights of hashed solution unigrams + bigrams $\to \ell_1, \ell_2$",
    NEW, bold=True)
box(ax, 14.85, 1.75, 1.7, 1.5, "softmax over\n" + r"$z_k = s_k + \ell_k$", STD, size=6.4)
ax.plot([14.55, 15.7], [5.3, 5.3], color="#333333", lw=0.8)
arrow(ax, 15.7, 5.3, 15.7, 3.25)
ax.text(0.05, 0.3, "orange: PIQA-specific components (ablated in Table 2); blue: standard components; "
        "both candidates share all weights", fontsize=7.2, color="#333333")
fig.savefig(f"{OUT}/architecture.pdf", bbox_inches="tight")
if PREVIEW:
    fig.savefig(f"{PREVIEW}/architecture.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. pooling attention of two test cases
# Drawn from the attention data the notebook saves next to each figure (outputs/figures/*.png.json), as
# highlighted text: each solution token is shaded by its pooling weight, differing tokens are bold.
import json  # noqa: E402

CASES = [("confident_correct_1300", "Correct prediction"), ("confident_wrong_747", "Incorrect prediction")]
WIDTH = 7.0                          # inches (full text width)
LINE = 0.19                          # inches per text line
FONT = 7.4                           # token font size (pt)
cmap = plt.get_cmap("Oranges")


def option_lines(fig, ax, tokens, weights, diff, x0, max_x):
    """Lays out tokens left to right with wrapping; returns the drawn rows as lists of (x, token, w, bold)."""
    renderer = fig.canvas.get_renderer()
    rows, row, x = [], [], x0
    for tok, w, d in zip(tokens, weights, diff):
        text = tok[2:] if tok.startswith("##") else tok
        glue = tok.startswith("##") or text in ".,;:!?)"
        t = ax.text(0, 0, text, fontsize=FONT, weight="bold" if d else "normal", family="serif")
        tw = t.get_window_extent(renderer).width / fig.dpi
        t.remove()
        gap = 0.0 if (glue or not row) else 0.045
        if row and x + gap + tw > max_x:
            rows.append(row)
            row, x, gap = [], x0, 0.0
        row.append((x + gap, text, w, d, tw))
        x += gap + tw
    rows.append(row)
    return rows


blocks = []
for name, goal in CASES:
    d = json.load(open(f"outputs/figures/{name}.png.json", encoding="utf-8"))
    for k, o in enumerate(d["options"]):
        toks = [o["tokens"][i] for i in o["sol_pos"]]
        tag = " (gold)" if d["label"] == k else ""
        head = f"{goal} (gold: option {d['label'] + 1}, predicted: option {d['pred'] + 1}). Goal: {d['goal']}"
        blocks.append((head if k == 0 else None, f"Option {k + 1}{tag}, p = {d['probs'][k]:.2f}", toks, o["pool"], o["diff"]))

fig = plt.figure(figsize=(WIDTH, 4.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WIDTH)
ax.axis("off")
laid = [option_lines(fig, ax, b[2], b[3], b[4], 1.45, WIDTH - 0.05) for b in blocks]
n_lines = sum(len(rows) for rows in laid) + 2 * len(CASES)
height = n_lines * LINE + 0.1
fig.set_size_inches(WIDTH, height)
ax.set_ylim(height, 0)
y = 0.05
for (goal, label, *_), rows in zip(blocks, laid):
    if goal:
        ax.text(0.02, y + 0.14, goal, fontsize=FONT + 0.3, style="italic", family="serif")
        y += LINE + 0.04
    ax.text(0.02, y + 0.14, label, fontsize=FONT, family="serif", color="#333333")
    for row in rows:
        for x, text, w, bold, tw in row:
            shade = 0.04 + 0.86 * min(1.0, w / 0.15)
            ax.add_patch(plt.Rectangle((x - 0.012, y + 0.015), tw + 0.024, LINE - 0.03, color=cmap(shade), lw=0))
            ax.text(x, y + 0.14, text, fontsize=FONT, family="serif", weight="bold" if bold else "normal",
                    color="white" if shade > 0.6 else "black")
        y += LINE
    y += 0.06
fig.savefig(f"{OUT}/attention_cases.pdf", bbox_inches="tight")
if PREVIEW:
    fig.savefig(f"{PREVIEW}/attention_cases.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# sanity: the numbers used above must match the saved result files
with open("outputs/results/final_results.csv", encoding="utf-8") as f:
    rows = {r["model"]: r for r in csv.DictReader(f)}
assert abs(float(rows["DACT (full)"]["test acc"]) - 0.6186) < 1e-3   # the final T4 run of b880108
print("figures written to", OUT, sorted(os.listdir(OUT)))
