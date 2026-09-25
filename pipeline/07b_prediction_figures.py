#!/usr/bin/env python
"""Figures 5, 6, 8 and 9 from the v2 prediction tables, with ranking rows.

Binary and continuous endpoints are rendered by one script so both get the
same layout and the same Friedman ranking row as Figures 2 and 3 -- rank
position over mean rank, best two in bold, one caption naming the test.

Reuses `_draw_compact_relative_boxplot` and `annotate_rank_row` from
`cvd.reporting.figures.prediction`, so the boxes are drawn by the published
code and only the composition is new.

    python pipeline/07b_prediction_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvd import conf  # noqa: E402
from cvd.reporting.figures.prediction import (  # noqa: E402
    _add_target_baseline_delta, _add_target_baseline_ratio,
    _draw_compact_relative_boxplot, _ordered_method_values, annotate_rank_row,
)

# The ranking row is read from the canonical table rather than recomputed, so
# the caption under a panel cannot drift from the supplementary table. See
# pipeline/06b_rank.py.
RANKINGS = "prediction_rankings.csv"


def ranking(table, cohort, endpoint_type, metric):
    """Rank row, bolding rule, per-panel lines and the figure-level verdict.

    Bold marks *significantly* best, so it is withheld wherever the omnibus does
    not reject after Holm across the four families the paper reports. A panel
    that bolds its top two where the arms are statistically indistinguishable
    would assert a difference the data does not carry.

    The text is split by what varies. The test statistic and its corrected
    p-value are per panel; the verdict they support is a statement about the
    cohort, and where both metrics reach the same one it is printed once for
    the figure rather than twice in small type under each panel.
    """
    hit = table[(table.cohort == cohort)
                & (table.endpoint_type == endpoint_type)
                & (table.metric == metric)]
    if hit.empty:
        return None, 0, [], ""
    r = hit.iloc[0]
    row = pd.Series({
        "mean_ranks": r.mean_ranks,
        "friedman_chi2": r.friedman_chi2,
        "friedman_p": r.friedman_p,
        "n_complete_blocks": r.n_complete_blocks,
    })
    holm = float(r.friedman_p_holm)
    if bool(r.omnibus_significant):
        pairs = int(r.n_pairwise_significant)
        pair_note = (f"{pairs} of {int(r.n_pairwise_tested)} pairs separable"
                     if pairs else
                     f"no individual pair separable ({int(r.n_pairwise_tested)} tested)")
        lines = [f"Holm-adjusted $p$={holm:.1e} across the four reported tests",
                 f"all-pairs Wilcoxon signed-rank with Holm: {pair_note}"]
        verdict = (f"The omnibus rejects: the arms are not all alike, "
                   f"{r.rank1} ranks first and {r.rank2} second.")
        return row, 2, lines, verdict
    lines = [f"Holm-adjusted $p$={holm:.2f} across the four reported tests"]
    verdict = ("The omnibus does not reject: no method is significantly best, "
               "and the ranking does not separate the methods in this cohort.")
    return row, 0, lines, verdict


TITLE_FS = 19
PANEL_CAPTION_FS = 12
SHARED_CAPTION_FS = 13


def panel_figure(df, specs, dest):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9.2))
    shared, artists = [], []
    for idx, (ax, (value_col, ylabel, ref, lower, star, title, rk, bold, lines, verdict)) in enumerate(zip(axes, specs)):
        _draw_compact_relative_boxplot(
            ax, df, value_col, ylabel, reference=ref,
            lower_is_better=lower, star_label=star, font_scale=1.25,
        )
        ax.set_title(title, fontsize=TITLE_FS)
        # Panel letter, journal house style: the legend names (a) and (b)
        # rather than "left"/"right", which does not survive reflow.
        ax.text(
            -0.07, 1.02, f"({chr(ord('a') + idx)})", transform=ax.transAxes,
            fontsize=TITLE_FS, fontweight="bold", va="bottom", ha="left",
        )
        if rk is None:
            continue
        caption = "\n".join(
            [f"$\\chi^2$={float(rk['friedman_chi2']):.2f}, "
             f"$p$={float(rk['friedman_p']):.2e}"] + list(lines))
        prefix, artist = annotate_rank_row(
            ax, _ordered_method_values(df, value_col)[0], rk,
            bold_top_n=bold, y_caption=-0.40,
            caption=caption, fontsize=PANEL_CAPTION_FS)
        shared.append(f"{prefix}\n{verdict}")
        artists.append(artist)
    fig.tight_layout(h_pad=3.0)

    # Hoist the shared half only when the two panels really do agree on it; if a
    # figure ever pairs metrics with different block counts, bolding rules or
    # verdicts, print it under each panel rather than asserting a match.
    if len(shared) == 2 and shared[0] == shared[1]:
        # Sit it just under the lowest per-panel caption. A fixed figure
        # fraction cannot do this: tight_layout rescales the axes as the
        # below-axis text grows, so the gap has to be measured after drawing.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lowest = min(a.get_window_extent(renderer).y0 for a in artists if a)
        y = lowest / fig.bbox.height - 0.035
        fig.text(0.5, y, shared[0], ha="center", va="top",
                 fontsize=SHARED_CAPTION_FS, color="#555555", linespacing=1.4)
    else:
        for ax, text in zip(axes, shared):
            ax.text(0.5, -0.62, text, transform=ax.transAxes, ha="center",
                    va="top", fontsize=PANEL_CAPTION_FS, color="#555555",
                    linespacing=1.4)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {dest}")


def main() -> int:
    cfg = conf.load()
    tables = Path(str(cfg["masking.output_root"])) / "prediction_tables"
    out = Path(str(cfg["paths.paper.figures_v2_dir"]))
    rk_table = pd.read_csv(tables / RANKINGS)

    b = pd.read_csv(tables / "binary_prediction_results.csv", low_memory=False)
    c = pd.read_csv(tables / "continuous_prediction_results.csv", low_memory=False)
    b = b[b.status == "ok"].copy()
    c = c[c.status == "ok"].copy()

    for cohort, fig_b, fig_c in (("Cohort1", "Figure_5.png", "Figure_8.png"),
                                 ("Cohort2", "Figure_6.png", "Figure_9.png")):
        db = b[b.cohort == cohort].copy()
        db = _add_target_baseline_delta(db, "balanced_accuracy", "balanced_accuracy_vs_baseline")
        db = _add_target_baseline_delta(db, "au_prc", "au_prc_vs_baseline")
        n = len(db) // max(db.method.nunique(), 1)
        print(f"\n{cohort} binary: {db.target.nunique()} endpoints, {n} results/method")
        panel_figure(db, [
            ("balanced_accuracy_vs_baseline",
             "Balanced Accuracy minus target baseline median", 0.0, False,
             "Higher than baseline median", "Balanced Accuracy",
             *ranking(rk_table, cohort, "binary", "balanced_accuracy")),
            ("au_prc_vs_baseline", "AU-PRC minus target baseline median", 0.0, False,
             "Higher than baseline median", "AU-PRC",
             *ranking(rk_table, cohort, "binary", "au_prc")),
        ], out / fig_b)

        dc = c[c.cohort == cohort].copy()
        dc = _add_target_baseline_ratio(dc, "rmse_z", "rmse_z_vs_target_baseline")
        dc = _add_target_baseline_delta(dc, "r2", "r2_vs_target_baseline")
        n = len(dc) // max(dc.method.nunique(), 1)
        print(f"{cohort} continuous: {dc.target.nunique()} endpoints, {n} results/method")
        panel_figure(dc, [
            ("rmse_z_vs_target_baseline",
             "Standardized RMSE / target baseline median",
             1.0, True, "Lower than baseline median", "Standardized RMSE",
             *ranking(rk_table, cohort, "continuous", "rmse_z")),
            ("r2_vs_target_baseline",
             "R² minus target baseline median", 0.0, False,
             "Higher than baseline median", "R²",
             *ranking(rk_table, cohort, "continuous", "r2")),
        ], out / fig_c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
