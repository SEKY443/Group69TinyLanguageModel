"""Builds the report figures from the saved outputs of the final run (run from the repository root)."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


# ---------------------------------------------------------------- 1. architecture diagram
def box(ax, x, y, w, h, text, fc, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.4, weight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))


fig, ax = plt.subplots(figsize=(7.0, 2.2))
ax.set_xlim(0, 14.4)
ax.set_ylim(0, 4.8)
ax.axis("off")
NEW, STD = "#fde2c8", "#dfe9f5"
for k, y in ((1, 3.0), (2, 0.8)):
    box(ax, 0.05, y, 2.2, 0.8, rf"goal $\oplus$ sol$_{k}$", "#f2f2f2")
    box(ax, 2.55, y, 2.3, 0.8, "embedding\n+ difference tag", NEW, bold=True)
    box(ax, 5.15, y, 2.2, 0.8, "Transformer\nencoder (shared)", STD)
    box(ax, 9.95, y, 1.85, 0.8, "diff-guided\npooling", NEW, bold=True)
    box(ax, 12.1, y, 1.6, 0.8, rf"scorer $\to s_{k}$", STD)
    for x0, x1 in ((2.25, 2.55), (4.85, 5.15), (11.8, 12.1)):
        arrow(ax, x0, y + 0.4, x1, y + 0.4)
    arrow(ax, 7.35, y + 0.4, 7.7, y + 0.4 + (-0.5 if k == 1 else 0.5))
    arrow(ax, 9.55, y + 0.4 + (-0.5 if k == 1 else 0.5), 9.95, y + 0.4)
box(ax, 7.7, 1.5, 1.85, 1.9, "cross-solution\nattention\n(ESIM fusion)", NEW, bold=True)
box(ax, 2.55, 2.0, 4.8, 0.6, "difflib alignment: shared / differing tokens", "#fff7ec")
box(ax, 0.05, 4.1, 11.75, 0.55, r"lexical head: learned weights of hashed sol$_k$ unigrams + bigrams $\to \ell_k$", NEW, bold=True)
ax.text(13.0, 4.37, r"$z_k = s_k + \ell_k$", ha="center", va="center", fontsize=7.5)
ax.text(13.0, 2.2, "softmax\n" + r"$(z_1, z_2)$", ha="center", va="center", fontsize=7.5)
arrow(ax, 11.8, 4.37, 12.35, 4.37)
ax.text(0.05, 0.2, "orange: our PIQA-specific components (ablated in Table 3); blue: standard components",
        fontsize=6.5, color="#555555")
fig.savefig(f"{OUT}/architecture.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2. attention controls
labels = ["uniform\n(reference)", "untrained\n(init. bias)", "trained,\ntag bias = 0", "trained\n(full)"]
values = [24.8, 38.9, 73.2, 84.9]   # printed by the notebook (Section 3.3) in the final run
fig, ax = plt.subplots(figsize=(3.1, 1.55))
bars = ax.bar(range(4), values, color=["#bbbbbb", "#9ecae1", "#fdae6b", "#e6550d"], width=0.62)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel("pooling mass on\ndiffering tokens (%)", fontsize=7)
ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"{OUT}/attention_controls.pdf", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- 3. pooling strips of two test cases
def crop_top(src, dst):
    """Keeps the two pooling strips (top of the saved figure) and trims the white margins."""
    from PIL import ImageOps
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im = im.crop((0, int(0.035 * h), w, int(0.30 * h)))
    im.crop(ImageOps.invert(im).getbbox()).save(dst)


crop_top("outputs/figures/confident_wrong_747.png", f"{OUT}/case_747_pooling.png")
crop_top("outputs/figures/uncertain_1258.png", f"{OUT}/case_1258_pooling.png")

# sanity: the numbers used above must match the saved result files
with open("outputs/results/final_results.csv", encoding="utf-8") as f:
    rows = {r["model"]: r for r in csv.DictReader(f)}
assert abs(float(rows["DACT (full)"]["test acc"]) - 0.6159) < 1e-3
print("figures written to", OUT, sorted(os.listdir(OUT)))
