#!/usr/bin/env python3
"""Regenerate Figure 1 (pipeline overview).

Final analysis: two cohorts, 178 common predictors, two missingness mechanisms,
two intensities, eight imputers, one reference arm, and five endpoints.

Layout notes
------------
Every box is sized to its own contents: line widths are *measured* with the real font at
the real point size (`_measure`), so a box is never wider or taller than the text it holds
and the text block sits centred in it.  Nothing here is hand-tuned to a particular string,
which is what made the first version of this script overflow when the wording changed.

Flow, expressed by the arrows:
    Datasets -> Missingness Simulation -> Imputation Methods -> { Data Fidelity,
                                                                 Outcomes Prediction }

Font: Nimbus Roman, a metrically-exact Times clone.  Note that `main.tex` declares no font
package, so the manuscript body is actually Computer Modern rather than Times; matplotlib
ships only a partial CM subset (cmr10, no bold text face and thin glyph coverage), so a
full-coverage Times is the closest practical match.  Set FONT below to change it.

Run:  python -m cvd.reporting.figures.schematic
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from cvd.conf import active

_cfg = active()

OUT = Path(str(_cfg["reporting.figures.Figure_1.output"]))

FONT = "Nimbus Roman"
matplotlib.rcParams["font.family"] = FONT
matplotlib.rcParams["mathtext.fontset"] = "stix"

C = dict(orange="#E0712B", cyan="#1D9FD8", green="#3AA535",
         purple="#9C2489", blue="#0F6CB4")

TITLE_FS, BODY_FS = 15.0, 12.0
DPI = 300

# geometry, all in inches
LINE_H   = 0.215     # body line pitch
HEAD_H   = 0.34      # coloured header bar height
PAD_X    = 0.13      # body inset left/right of the text block
PAD_Y    = 0.11      # body inset above first / below last line
BULLET_W = 0.13      # width reserved for the bullet glyph
SUB_IND  = 0.13      # extra indent for continuation ("plain") lines
COL_GAP  = 0.62      # horizontal gap between the two columns (room for arrows)
ROW_GAP  = 0.30      # vertical gap between stacked boxes


# ── text measurement ──────────────────────────────────────────────────────
_probe_fig = plt.figure(figsize=(4, 4), dpi=DPI)
_probe_ax = _probe_fig.add_axes([0, 0, 1, 1])
_renderer = _probe_fig.canvas.get_renderer()


def _measure(txt: str, size: float, bold: bool) -> float:
    """Rendered width of `txt` in inches, using the real font at the real size."""
    t = _probe_ax.text(0, 0, txt, fontsize=size,
                       fontweight=("bold" if bold else "normal"))
    w = t.get_window_extent(_renderer).width / DPI
    t.remove()
    return w


def _segments(txt: str, bold: bool) -> list[tuple[str, bool]]:
    """Split ``a **b** c`` into weighted runs, so one word can be bold inside a
    line. Bold text is wider than normal at the same size, so the pieces have to
    be measured and laid out separately rather than measured as one string."""
    out, flag = [], bold
    for i, piece in enumerate(txt.split("**")):
        if piece:
            out.append((piece, flag if i % 2 == 0 else not bold))
    return out or [("", bold)]


def _measure_rich(txt: str, size: float, bold: bool) -> float:
    return sum(_measure(t, size, b) for t, b in _segments(txt, bold))


def _draw_rich(ax, X, Y, x, y, txt, size, bold, fw):
    """Draw the weighted runs of `txt` left to right starting at x inches."""
    for piece, piece_bold in _segments(txt, bold):
        ax.text(X(x), Y(y), piece, fontsize=size, va="center", zorder=4,
                fontweight=("bold" if piece_bold else "normal"))
        x += _measure(piece, size, piece_bold)


def content_size(title: str, items) -> tuple[float, float]:
    """(width, height) in inches of a panel holding `items` under `title`."""
    widest = 0.0
    for txt, kind in items:
        indent = 0.0 if kind == "head" else BULLET_W if kind == "bullet" else BULLET_W + SUB_IND
        widest = max(widest, indent + _measure_rich(txt, BODY_FS, kind == "head"))
    # the header title must also fit
    widest = max(widest, _measure(max(title.split("\n"), key=len), TITLE_FS, True) + 0.22)
    n_title_lines = title.count("\n") + 1
    head = HEAD_H * n_title_lines
    return widest + 2 * PAD_X, head + 2 * PAD_Y + len(items) * LINE_H


# ── drawing ───────────────────────────────────────────────────────────────
def panel(ax, x0, y0, w, h, title, colour, items, fw, fh):
    """Draw a panel whose lower-left corner is (x0, y0), all in inches."""
    def X(v): return v / fw
    def Y(v): return v / fh

    n_title_lines = title.count("\n") + 1
    head = HEAD_H * n_title_lines
    body_h = h - head
    aspect = fw / fh

    ax.add_patch(FancyBboxPatch((X(x0), Y(y0)), X(w), Y(body_h),
                                boxstyle="round,pad=0,rounding_size=0.012",
                                linewidth=1.5, edgecolor="#3B3B3B", facecolor="white",
                                zorder=2, mutation_aspect=aspect))
    ax.add_patch(FancyBboxPatch((X(x0), Y(y0 + body_h)), X(w), Y(head),
                                boxstyle="round,pad=0,rounding_size=0.012",
                                linewidth=0, facecolor=colour, zorder=3,
                                mutation_aspect=aspect))
    ax.text(X(x0 + w / 2), Y(y0 + body_h + head / 2), title,
            ha="center", va="center", fontsize=TITLE_FS, fontweight="bold",
            color="white", zorder=4, linespacing=1.15)

    # centre the text block vertically in the body, and horizontally as a block
    block_h = len(items) * LINE_H
    y = y0 + body_h - (body_h - block_h) / 2 - LINE_H / 2
    xt = x0 + PAD_X
    for txt, kind in items:
        if kind == "head":
            _draw_rich(ax, X, Y, xt, y, txt, BODY_FS, True, fw)
        elif kind == "bullet":
            ax.text(X(xt + 0.012), Y(y), "▪", fontsize=8.5, va="center",
                    color="#3B3B3B", zorder=4)
            _draw_rich(ax, X, Y, xt + BULLET_W, y, txt, BODY_FS, False, fw)
        else:
            _draw_rich(ax, X, Y, xt + BULLET_W + SUB_IND, y, txt, BODY_FS, False, fw)
        y -= LINE_H


# ── content ───────────────────────────────────────────────────────────────
DATA = ("Input Data", C["orange"], [
    ("Adult cardiac surgery datasets:", "head"),
    ("Study window: 2014–2025", "plain"),
    ("Primary cohort: **Cohort 1 (Cohort1)**", "bullet"),
    ("10,159 patients", "plain"),
    ("External cohort: **Cohort2 Hospital**", "bullet"),
    ("3,875 patients", "plain"),
    ("Comparable datasets:", "head"),
    ("178 common variables", "bullet"),
    ("Shared variable taxonomy", "bullet"),
    ("Identical preprocessing rules", "bullet"),
    ("Comparable value ranges", "bullet")])

# The registry's own missingness is the thing being imputed; the two mechanisms
# below only decide which OBSERVED cells are additionally held out so there is a
# ground truth to score against. Naming the box "Extra" keeps that distinction.
MISS = ("Extra Missingness Simulation", C["cyan"], [
    ("Native missingness retained", "plain"),
    ("Two extra masking mechanisms:", "head"),
    ("MCAR: uniform observed-cell draw", "bullet"),
    ("MNAR: value-dependent weights", "bullet"),
    ("Different intensity levels:", "head"),
    ("5% held-out set ⊂ 20% held-out set", "plain"),
    ("Masks frozen across all imputers", "plain")])

IMP = ("Imputation Methods", C["purple"], [
    ("Classical:", "head"),
    ("Mean / Mode substitution", "bullet"),
    ("KNN (k-nearest neighbours, k = 3)", "bullet"),
    ("MICE (chained equations)", "bullet"),
    ("MissRanger (random forest)", "bullet"),
    ("Generative AI:", "head"),
    ("MIWAE (importance-weighted autoencoder)", "bullet"),
    ("GAIN (adversarial network)", "bullet"),
    ("MIRACLE (causally aware)", "bullet"),
    ("ReMasker (transformer)", "bullet"),
    ("Ensemble K = 2 for the six stochastic methods", "plain"),
    ("Mean/Mode and KNN are deterministic: K = 1", "plain"),
    ("All post-operative targets held out when imputing", "head"),
    ])

# Accuracy and macro F1 are computed and stored per variable but feed no figure,
# table or test, so they are not shown here. Balanced accuracy and standardized
# RMSE are the only two metrics the analysis uses.
FID = ("Data Reconstruction Fidelity", C["green"], [
    ("Imputed vs. held-out ground truth,", "plain"),
    ("scored per variable at masked cells only", "plain"),
    ("Binary, nominal and ordinal:", "head"),
    ("Balanced accuracy (higher better)", "bullet"),
    ("Continuous and count:", "head"),
    ("Standardized RMSE (lower better)", "bullet"),
    ("Inference:", "head"),
    ("Friedman omnibus over cohort ×", "bullet"),
    ("mechanism × intensity × variable blocks", "plain"),
    ("Mean ranks, 1 = best of 8 imputers", "bullet"),
    ("No-imputation arm excluded: it", "plain"),
    ("produces no value to score", "plain")])

PRED = ("Outcomes Prediction", C["blue"], [
    ("Nine arms compared:", "head"),
    ("Eight imputed matrices", "bullet"),
    ("+ Reference arm (no imputation):", "bullet"),
    ("XGBoost native missing-value handling", "plain"),
    ("Binary outcomes (3):", "head"),
    ("Atrial fibrillation (afibproc)", "bullet"),
    ("Complications (complics)", "bullet"),
    ("Prolonged ventilation (cpvntlng)", "bullet"),
    ("Continuous outcomes (2):", "head"),
    ("Initial ICU hours (icuinhrs)", "bullet"),
    ("Post-operative ejection fraction (ppef)", "bullet")])

# ── layout ────────────────────────────────────────────────────────────────
# Datasets and Missingness sit side by side; Imputation Methods sits directly below
# Missingness (so its inbound arrow is a clean vertical); the two evaluation panels
# stack in a right-hand column.  Stacking all five vertically instead makes the figure
# portrait and nearly a full page tall at journal width, which is why it is not done.
sz = {k: content_size(v[0], v[2]) for k, v in
      dict(DATA=DATA, MISS=MISS, IMP=IMP, FID=FID, PRED=PRED).items()}

row1_h = max(sz["DATA"][1], sz["MISS"][1])
_miss_x = sz["DATA"][0] + COL_GAP
# Imputation is centred under Missingness, so it can overhang the row-1 extent; the
# column width has to be derived from where it actually lands, not assumed.
_imp_x = max(0.0, _miss_x + (sz["MISS"][0] - sz["IMP"][0]) / 2)
left_w = max(_miss_x + sz["MISS"][0], _imp_x + sz["IMP"][0])
right_w = max(sz["FID"][0], sz["PRED"][0])
left_h = row1_h + ROW_GAP + sz["IMP"][1]
right_h = sz["FID"][1] + ROW_GAP + sz["PRED"][1]

FIG_W = left_w + COL_GAP + right_w
FIG_H = max(left_h, right_h)

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

pos = {}


def place(key, spec, x0, y0):
    w, h = sz[key]
    panel(ax, x0, y0, w, h, *spec, FIG_W, FIG_H)
    pos[key] = (x0, y0, w, h)


top_y = FIG_H
# row 1: Missingness is the anchor; Datasets to its left, both centred in the row band
place("MISS", MISS, _miss_x, top_y - row1_h + (row1_h - sz["MISS"][1]) / 2)
place("DATA", DATA, 0.0, top_y - row1_h + (row1_h - sz["DATA"][1]) / 2)
# row 2: Imputation Methods, centred under Missingness
place("IMP", IMP, _imp_x, top_y - row1_h - ROW_GAP - sz["IMP"][1])

# right column, vertically centred against the figure
xr = left_w + COL_GAP
y = FIG_H - (FIG_H - right_h) / 2
for key, spec in (("FID", FID), ("PRED", PRED)):
    place(key, spec, xr + (right_w - sz[key][0]) / 2, y - sz[key][1])
    y -= sz[key][1] + ROW_GAP

# ── arrows ────────────────────────────────────────────────────────────────
def X(v): return v / FIG_W
def Y(v): return v / FIG_H


# NOTE: the arrowstyle's head_width/head_length are multiplied by mutation_scale, so
# leaving mutation_scale at its default of 1 renders a head a fraction of a point wide
# -- i.e. an invisible arrowhead on a visible line.
A = dict(arrowstyle="-|>,head_width=0.30,head_length=0.62", linewidth=1.9,
         color="#4A4A4A", zorder=1, shrinkA=0, shrinkB=0,
         mutation_scale=26)


def route(pts):
    """Orthogonal polyline through `pts` (inches), arrowhead on the final segment.

    Drawn as plain segments plus one FancyArrowPatch for the last leg. matplotlib's
    "angle" connectionstyle produces a single corner and orients the head along the
    approach angle, which put the head parallel to the target's edge rather than into
    it for the two-corner routes below.
    """
    for (x1, y1), (x2, y2) in zip(pts, pts[1:-1]):
        ax.plot([X(x1), X(x2)], [Y(y1), Y(y2)], color=A["color"],
                linewidth=A["linewidth"], solid_capstyle="butt", zorder=1)
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    ax.add_patch(FancyArrowPatch((X(xa), Y(ya)), (X(xb), Y(yb)), **A))


def cx(k): x0, y0, w, h = pos[k]; return x0 + w / 2
def cy(k): x0, y0, w, h = pos[k]; return y0 + h / 2
def top(k): x0, y0, w, h = pos[k]; return y0 + h
def bot(k): return pos[k][1]
def rgt(k): x0, y0, w, h = pos[k]; return x0 + w
def lft(k): return pos[k][0]


# 1. Datasets -> Missingness Simulation (horizontal)
route([(rgt("DATA"), cy("DATA")), (lft("MISS"), cy("DATA"))])
# 2. Missingness Simulation -> Imputation Methods (straight down)
route([(cx("MISS"), bot("MISS")), (cx("MISS"), top("IMP"))])
# 3. Imputation Methods -> Data Fidelity: right, up through the inter-column gap,
#    then right into the left edge.  The gap is the only clear vertical corridor --
#    running up inside the right column would cross the Outcomes Prediction box.
#    It leaves IMP ABOVE its own centre, midway to the Outcomes arrow's height:
#    the two arrows both exit IMP's right edge, and when the boxes happen to size
#    such that cy(IMP) ~ cy(PRED) they render as one forked line.
x_j = (rgt("IMP") + lft("FID")) / 2
y_fid = (cy("PRED") + top("IMP")) / 2
route([(rgt("IMP"), y_fid), (x_j, y_fid),
       (x_j, cy("FID")), (lft("FID"), cy("FID"))])
# 4. Imputation Methods -> Outcomes Prediction (right into the left edge)
route([(rgt("IMP"), cy("PRED")), (lft("PRED"), cy("PRED"))])

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print(f"wrote {OUT}")
print(f"  figure {FIG_W:.2f} x {FIG_H:.2f} in; left col {left_w:.2f} in, right col {right_w:.2f} in")
for k, (w, h) in sz.items():
    print(f"  {k:5s} {w:5.2f} x {h:5.2f} in")
