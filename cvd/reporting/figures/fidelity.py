"""
Compact paper figures for imputation fidelity results.

Load the version-2 long-form fidelity table and render one figure per cohort.
Rows show categorical balanced accuracy and standardized numeric RMSE. Points
and solid lines mark medians, dashed lines means, and shading the IQR across
feature--mechanism--intensity observations.

"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fontconfig-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "mean": "Mean/Mode",
    "KNN": "KNN",
    "k-Nearest-Neighbor": "KNN",
    "MICE": "MICE",
    "MissRanger": "MissRanger",
    "MissForest": "MissRanger",
    "MissForest (ensemble tree-based)": "MissRanger",
    "MIWAE (VAE-based)": "MIWAE",
    "GAIN (GAN-based)": "GAIN",
    "ReMasker (transformer-based)": "ReMasker",
    "MIRACLE (diffusion/energy-based)": "MIRACLE",
}
METHOD_ORDER = (
    "Mean/Mode",
    "KNN",
    "MICE",
    "MissRanger",
    "MIWAE",
    "GAIN",
    "ReMasker",
    "MIRACLE",
)
METHOD_GROUPS = {
    "Classic imputers": ("Mean/Mode", "KNN", "MICE", "MissRanger"),
    "GenAI imputers": ("MIWAE", "GAIN", "ReMasker", "MIRACLE"),
}
METHOD_RANKS = {name: idx for idx, name in enumerate(METHOD_ORDER)}

METRIC_ALIASES = {
    "rmse": ("Continuous", "RMSE"),
    "RMSE": ("Continuous", "RMSE"),
    "balanced_acc": ("Categorical", "Balanced accuracy"),
    "Balanced Accuracy": ("Categorical", "Balanced accuracy"),
}
VARIABLE_TYPE_ORDER = ("Categorical", "Continuous")
COHORT_ORDER = ("Cohort1", "Cohort2")
DIRECTION_BY_VARIABLE_TYPE = {
    "Categorical": "higher",
    "Continuous": "lower",
}

METHOD_COLORS = {
    "Mean/Mode": "#2F6F9F",
    "KNN": "#4C78A8",
    "MICE": "#72A6C9",
    "MissRanger": "#A6CEE3",
    "MIWAE": "#B23A48",
    "GAIN": "#D95F02",
    "ReMasker": "#E76F51",
    "MIRACLE": "#F4A261",
}
IQR_FILL_ALPHA = 0.16
INTERVAL_LINE_ALPHA = 0.34
POINT_ALPHA = 0.82
MEAN_LINE_ALPHA = 0.75
MEDIAN_LINE_ALPHA = 0.82
# The two count variables are numeric and scored exactly as the continuous ones,
# so the panel is titled for what it measures rather than for the taxonomy.
VARIABLE_TYPE_TITLES = {
    "Categorical": "Categorical Variables",
    "Continuous": "Continuous Variables",
}

FONT_SCALE = 1.5625
AXIS_LABEL_SIZE = 10 * FONT_SCALE
TICK_LABEL_SIZE = 10 * FONT_SCALE
PANEL_TITLE_SIZE = 12 * FONT_SCALE
SUPTITLE_SIZE = 15 * FONT_SCALE
ANNOTATION_SIZE = 9 * FONT_SCALE


def project_root() -> Path:
    """Return the repository root for the installed source tree."""
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: Path | str, root: Optional[Path] = None) -> Path:
    """Resolve a user path relative to the project root when needed."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (root or project_root()) / resolved
    return resolved


def _normalize_method(method: object) -> str:
    method_text = str(method)
    return METHOD_LABELS.get(method_text, method_text)


def _metric_info(metric: object) -> Optional[Tuple[str, str]]:
    return METRIC_ALIASES.get(str(metric))


def load_fidelity_values(
    data_dir: Path | str = "results_v2/fidelity_tables",
) -> pd.DataFrame:
    """Read the table written by pipeline/07a_fidelity_figures.py."""
    path = _resolve_path(data_dir)
    if path.is_dir():
        path = path / "fidelity_long.csv"
    values = pd.read_csv(path)
    required = {"cohort", "variable_type", "metric", "simulation", "method",
                "feature", "value"}
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f"{path}: missing fidelity columns {sorted(missing)}")
    if values.empty:
        raise ValueError(f"{path}: no fidelity values")
    return values


def summarize_fidelity(values_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature-level values across all selected mechanisms and intensities."""
    grouped = (
        values_df.groupby(
            ["cohort", "variable_type", "metric", "method"],
            observed=True,
        )["value"]
        .agg(
            mean="mean",
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            n_values="count",
        )
        .reset_index()
    )
    grouped["method_rank"] = grouped["method"].map(
        METHOD_RANKS
    )
    return grouped.sort_values(
        ["variable_type", "cohort", "method_rank"],
        na_position="last",
    ).reset_index(drop=True)


def _format_method_series(series: pd.Series, methods: Sequence[str]) -> str:
    """Format method-indexed values for compact CSV output."""
    parts = []
    for method in methods:
        if method not in series.index or pd.isna(series.loc[method]):
            continue
        parts.append(f"{method}={float(series.loc[method]):.6g}")
    return ";".join(parts)


def run_overall_fidelity_friedman_tests(
    values_df: pd.DataFrame,
    methods: Sequence[str] = METHOD_ORDER,
    cohorts: Sequence[str] = COHORT_ORDER,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run Friedman omnibus tests across all selected mechanisms and intensities and cohorts.

    The paired blocks are cohort, mechanism--intensity simulation, and feature. Separate tests
    are run for categorical balanced accuracy and continuous RMSE because the
    metrics have different scales and directions.
    """
    from scipy.stats import friedmanchisquare

    rows: List[Dict[str, object]] = []
    scoped = values_df.loc[values_df["cohort"].isin(cohorts)].copy()
    block_cols = ["cohort", "simulation", "feature"]

    for variable_type in VARIABLE_TYPE_ORDER:
        panel = scoped.loc[scoped["variable_type"] == variable_type]
        direction = DIRECTION_BY_VARIABLE_TYPE[variable_type]
        metric = str(panel["metric"].dropna().iloc[0]) if not panel.empty else ""
        matrix = panel.pivot_table(
            index=block_cols,
            columns="method",
            values="value",
            aggfunc="mean",
        )
        available = [method for method in methods if method in matrix.columns]
        missing = [method for method in methods if method not in available]
        complete = matrix[available].dropna() if available else pd.DataFrame()

        base_row: Dict[str, object] = {
            "scope": "All cohorts" if len(cohorts) > 1 else "Within cohort",
            "cohorts": ";".join(cohorts),
            "variable_type": variable_type,
            "metric": metric,
            "direction": direction,
            "block_columns": ";".join(block_cols),
            "methods_tested": ";".join(available),
            "missing_methods": ";".join(missing),
            "n_methods": len(available),
            "n_blocks_total": int(len(matrix)),
            "n_complete_blocks": int(len(complete)),
            "alpha": alpha,
            "friedman_chi2": np.nan,
            "friedman_p": np.nan,
            "reject": False,
            "best_median_method": "",
            "best_median_value": np.nan,
            "best_mean_rank_method": "",
            "best_mean_rank": np.nan,
            "method_medians": "",
            "method_q1": "",
            "method_q3": "",
            "method_mean_ranks": "",
            "status": "ok",
        }

        if missing:
            rows.append({**base_row, "status": "missing_methods"})
            continue
        if len(available) < 3 or len(complete) < 2:
            rows.append({**base_row, "status": "insufficient_complete_blocks"})
            continue

        medians = complete[available].median(axis=0)
        if direction == "higher":
            best_median_method = str(medians.idxmax())
            ranks = complete[available].rank(axis=1, ascending=False, method="average")
        else:
            best_median_method = str(medians.idxmin())
            ranks = complete[available].rank(axis=1, ascending=True, method="average")
        mean_ranks = ranks.mean(axis=0)
        best_rank_method = str(mean_ranks.idxmin())

        try:
            friedman = friedmanchisquare(
                *[complete[method].to_numpy(dtype=float) for method in available]
            )
            friedman_chi2 = float(friedman.statistic)
            friedman_p = float(friedman.pvalue)
            reject = bool(np.isfinite(friedman_p) and friedman_p <= alpha)
            status = "ok"
        except ValueError as exc:
            friedman_chi2 = np.nan
            friedman_p = np.nan
            reject = False
            status = f"friedman_error: {exc}"

        rows.append(
            {
                **base_row,
                "friedman_chi2": friedman_chi2,
                "friedman_p": friedman_p,
                "reject": reject,
                "best_median_method": best_median_method,
                "best_median_value": float(medians.loc[best_median_method]),
                "best_mean_rank_method": best_rank_method,
                "best_mean_rank": float(mean_ranks.loc[best_rank_method]),
                "method_medians": _format_method_series(medians, available),
                "method_q1": _format_method_series(complete[available].quantile(0.25), available),
                "method_q3": _format_method_series(complete[available].quantile(0.75), available),
                "method_mean_ranks": _format_method_series(mean_ranks, available),
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def _axis_limit(summary_df: pd.DataFrame, variable_type: str) -> Tuple[float, float]:
    panel = summary_df.loc[summary_df["variable_type"] == variable_type]
    if variable_type == "Categorical":
        return 0.0, 1.05

    upper = float(panel["q3"].max())
    if not np.isfinite(upper) or upper <= 0:
        upper = 1.0
    return 0.0, upper * 1.12


def _plot_overall_method_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    methods: Sequence[str] = METHOD_ORDER,
) -> None:
    """Draw median points, mean/median lines, and shaded IQR intervals."""
    method_panel = panel.set_index("method").reindex(methods)
    x_positions = np.arange(len(methods))

    for x_position, method in zip(x_positions, methods):
        row = method_panel.loc[method]
        if pd.isna(row["median"]):
            continue

        median = float(row["median"])
        mean = float(row["mean"])
        q1 = float(row["q1"])
        q3 = float(row["q3"])
        color = METHOD_COLORS[method]

        ax.fill_between(
            [x_position - 0.32, x_position + 0.32],
            [q1, q1],
            [q3, q3],
            color=color,
            alpha=IQR_FILL_ALPHA,
            linewidth=0,
        )
        ax.vlines(
            x_position,
            q1,
            q3,
            color=color,
            linewidth=1.0,
            alpha=INTERVAL_LINE_ALPHA,
        )
        ax.hlines(
            mean,
            x_position - 0.32,
            x_position + 0.32,
            color=color,
            linewidth=1.5,
            alpha=MEAN_LINE_ALPHA,
            linestyle=(0, (3, 2)),
            zorder=3,
        )
        ax.scatter(
            x_position,
            median,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=POINT_ALPHA,
            s=48,
            zorder=3,
        )
        ax.hlines(
            median,
            x_position - 0.20,
            x_position + 0.20,
            color=color,
            linewidth=2.0,
            alpha=MEDIAN_LINE_ALPHA,
            zorder=4,
        )


def _style_axis(
    ax: plt.Axes,
    variable_type: str,
    y_min: float,
    y_max: float,
) -> None:
    """Apply shared axis styling."""
    x_positions = np.arange(len(METHOD_ORDER))

    ax.set_xticks(x_positions, METHOD_ORDER)
    ax.set_xlim(-0.55, len(METHOD_ORDER) - 0.45)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color="#D0D0D0", linestyle="--", linewidth=0.6)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", labelrotation=25)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")

    if variable_type == "Categorical":
        ax.axhline(
            0.5,
            color="#777777",
            linewidth=0.8,
            linestyle=":",
            zorder=0,
        )
        ylabel = "Balanced accuracy (higher better)"
    else:
        ylabel = "Standardized RMSE (lower better)"

    ax.axvline(3.5, color="#BBBBBB", linewidth=0.8)
    ax.text(
        1.5,
        1.015,
        "Classic imputers",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_SIZE,
        color="#555555",
    )
    ax.text(
        5.5,
        1.015,
        "GenAI imputers",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_SIZE,
        color="#555555",
    )
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("", fontsize=AXIS_LABEL_SIZE)


# Sized against TICK_LABEL_SIZE (10 * FONT_SCALE): the rank sits directly
# under the method name it belongs to and should read as the same tier of
# information, with the explanatory line a step down.
RANK_LABEL_SIZE = 9.0 * FONT_SCALE
RANK_VALUE_SIZE = 10.5 * FONT_SCALE
# How many top-ranked methods to set in bold in each panel's ranking row.
BOLD_TOP_N = 2


def parse_method_series(text: object) -> Dict[str, float]:
    """Invert :func:`_format_method_series` back into a mapping."""
    if not isinstance(text, str) or not text:
        return {}
    out: Dict[str, float] = {}
    for item in text.split(";"):
        if "=" not in item:
            continue
        method, _, value = item.partition("=")
        try:
            out[method] = float(value)
        except ValueError:
            continue
    return out


def _annotate_ranking(
    ax: plt.Axes,
    row: pd.Series,
    methods: Sequence[str] = METHOD_ORDER,
    y_values: float = -0.215,
    y_caption: float = -0.465,
) -> None:
    """Write the Friedman ranking underneath one panel.

    Two lines below the axis: the ordinal rank and mean rank for each method,
    aligned to its x position, and a single caption line naming the test. The
    top ``BOLD_TOP_N`` methods in the panel are set in bold.

    Ordinal rank position orders the average within-block ranks; it need not
    match the ordering of the pooled medians drawn above.
    """
    mean_ranks = parse_method_series(row.get("method_mean_ranks"))
    if not mean_ranks:
        return
    order = sorted(mean_ranks, key=mean_ranks.get)
    ordinal = {method: idx + 1 for idx, method in enumerate(order)}

    for x_position, method in enumerate(methods):
        if method not in mean_ranks:
            continue
        ax.text(
            x_position,
            y_values,
            f"{ordinal[method]}\n({mean_ranks[method]:.2f})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=RANK_VALUE_SIZE,
            color="#333333",
            linespacing=1.25,
            fontweight="bold" if ordinal[method] <= BOLD_TOP_N else "normal",
        )

    chi2 = row.get("friedman_chi2")
    pvalue = row.get("friedman_p")
    blocks = row.get("n_complete_blocks")
    # A leading blank line, not a larger y offset. The offsets are fractions of
    # the AXES height and tight_layout shrinks the axes as the block below it
    # grows, so lowering y_caption moves both the rank row and the caption and
    # leaves the gap between them unchanged. One empty line is a fixed
    # typographic separation that survives any resize.
    caption = (f"\n"
               f"rank position and (Friedman mean rank), 1 = best, "
               f"best {BOLD_TOP_N} in bold\n")
    caption += (
        f"within this cohort, over {int(blocks)} mechanism × intensity × feature blocks"
    )
    if pd.notna(chi2) and pd.notna(pvalue):
        caption += f"\n$\\chi^2$={float(chi2):.2f}, $p$={float(pvalue):.2e}"
    ax.text(
        0.5,
        y_caption,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=RANK_LABEL_SIZE,
        color="#555555",
        linespacing=1.4,
    )


def plot_cohort_fidelity_figure(
    summary_df: pd.DataFrame,
    cohort: str,
    output_path: Path | str,
    dpi: int = 300,
    ranking_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Write one cohort-specific paper figure and return its path.

    *ranking_df* is the output of :func:`run_overall_fidelity_friedman_tests`
    run on THIS cohort alone. When given, each panel gets a ranking row
    underneath it. Pass the pooled two-cohort table here and both figures
    would carry the same row, which is not what either one shows.
    """
    output = _resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort_df = summary_df.loc[summary_df["cohort"] == cohort]
    if cohort_df.empty:
        raise ValueError(f"No summary rows found for cohort: {cohort}")

    annotate = ranking_df is not None and not ranking_df.empty
    # The ranking rows need vertical room, and each panel now carries its own
    # method labels: with sharex the top panel had none, so a rank row under it
    # would have sat above nothing that named the methods.
    fig, axes = plt.subplots(
        len(VARIABLE_TYPE_ORDER),
        1,
        # Taller when annotating, so the panels keep their original plot
        # height. The ranking rows are drawn in axis coordinates below the
        # axis, which tight_layout does not reserve room for, so the space has
        # to come from the figure height and h_pad rather than from the axes.
        figsize=(13.8, 15.2 if annotate else 8.6),
        sharex=not annotate,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    for row_idx, variable_type in enumerate(VARIABLE_TYPE_ORDER):
        y_min, y_max = _axis_limit(cohort_df, variable_type)
        ax = axes[row_idx]
        panel = cohort_df.loc[cohort_df["variable_type"] == variable_type]
        _plot_overall_method_panel(ax, panel)
        ax.set_title(
            VARIABLE_TYPE_TITLES[variable_type],
            fontsize=PANEL_TITLE_SIZE,
            y=1.075,
            pad=0,
        )
        # Panel letter, journal house style: the legend refers to (a) and (b)
        # rather than to "upper"/"lower", which a reader cannot check against
        # the artwork once the figure is reflowed onto a page.
        ax.text(
            -0.055, 1.075, f"({chr(ord('a') + row_idx)})",
            transform=ax.transAxes, fontsize=PANEL_TITLE_SIZE,
            fontweight="bold", va="bottom", ha="left",
        )
        _style_axis(
            ax,
            variable_type,
            y_min,
            y_max,
        )
        if annotate:
            match = ranking_df.loc[ranking_df["variable_type"] == variable_type]
            if not match.empty:
                # The offsets are fractions of the AXES height, so a tall panel
                # pushes the rank row further down in inches and opens a band of
                # dead space between the caption and the next panel's title.
                # These values are tuned against the figsize above; change one
                # and the other has to move with it.
                # Scaled with the figure height above: the offsets are axes
                # fractions, so a taller panel needs smaller fractions to keep
                # the rank row the same physical distance below the axis.
                _annotate_ranking(ax, match.iloc[0],
                                  y_values=-0.205, y_caption=-0.355)

    fig.tight_layout(rect=(0.03, 0.02, 1.0, 1.0), h_pad=0.8 if annotate else 2.9)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_compact_fidelity_figures(
    summary_df: pd.DataFrame,
    output_dir: Path | str = "paper/figures",
    output_prefix: str = "fig_imputation_fidelity",
    cohorts: Sequence[str] = COHORT_ORDER,
    dpi: int = 300,
) -> Dict[str, Path]:
    """Write one compact paper figure per cohort."""
    output_base = _resolve_path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for cohort in cohorts:
        safe_cohort = cohort.replace(" ", "_")
        output_path = output_base / f"{output_prefix}_{safe_cohort}.png"
        outputs[cohort] = plot_cohort_fidelity_figure(
            summary_df=summary_df,
            cohort=cohort,
            output_path=output_path,
            dpi=dpi,
        )
    return outputs


def build_compact_fidelity_figure(
    data_dir: Path | str = "results_v2/fidelity_tables",
    output_dir: Path | str = "paper/figures",
    output_prefix: str = "fig_imputation_fidelity",
    summary_csv: Optional[Path | str] = None,
    friedman_csv: Optional[Path | str] = None,
    dpi: int = 300,
) -> Tuple[Dict[str, Path], pd.DataFrame, pd.DataFrame]:
    """
    Load data, summarize it, write cohort figures, and optionally save CSVs.

    The returned Friedman table tests method differences across all selected mechanisms,
    intensities and both cohorts, separately for categorical and continuous
    variables.
    """
    values_df = load_fidelity_values(data_dir=data_dir)
    summary_df = summarize_fidelity(values_df)
    friedman_df = run_overall_fidelity_friedman_tests(values_df)
    outputs = plot_compact_fidelity_figures(
        summary_df=summary_df,
        output_dir=output_dir,
        output_prefix=output_prefix,
        dpi=dpi,
    )

    if summary_csv is not None:
        summary_path = _resolve_path(summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)

    if friedman_csv is not None:
        friedman_path = _resolve_path(friedman_csv)
        friedman_path.parent.mkdir(parents=True, exist_ok=True)
        friedman_df.to_csv(friedman_path, index=False)

    return outputs, summary_df, friedman_df


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one compact paper figure from the Cohort1 and Cohort2 "
            "imputation fidelity CSV files."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="results_v2/fidelity_tables",
        help="Directory containing fidelity_long.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/figures",
        help="Directory for the two cohort-specific compact figures.",
    )
    parser.add_argument(
        "--output-prefix",
        default="fig_imputation_fidelity",
        help="Filename prefix for cohort-specific figures.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional path for the median/IQR summary CSV.",
    )
    parser.add_argument(
        "--friedman-csv",
        default=None,
        help=(
            "Optional path for the all-cohort Friedman omnibus summary CSV."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    outputs, summary_df, friedman_df = build_compact_fidelity_figure(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        summary_csv=args.summary_csv,
        friedman_csv=args.friedman_csv,
        dpi=args.dpi,
    )
    for cohort, output in outputs.items():
        print(f"Wrote {cohort} compact fidelity figure: {output}")
    print(f"Summarized {len(summary_df)} cohort/type/method rows.")
    for _, row in friedman_df.iterrows():
        print(
            "Friedman "
            f"{row['variable_type']}: p={row['friedman_p']:.4g}, "
            f"best median={row['best_median_method']}, "
            f"best mean rank={row['best_mean_rank_method']}."
        )


if __name__ == "__main__":
    main()
