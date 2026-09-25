"""Prediction figures: Figures 4-9 and Supplementary Figures S1-S6.

Paper sections: Results 2.3 (binary), 2.4 (continuous), 2.6 (Cohort2 replication).

Four generations of plotting code accumulated here (v1 through the "compact v4"
variants). They are kept because the published figures were produced by
specific ones, and which generation produced which figure is recorded in
``conf/reporting.yaml`` -- previously it could only be recovered from file
mtimes, and three different scripts each claimed to produce ``Figure_8.png``.

Compact figures plot each method's deviation from the target-specific Baseline
median, which is why they are near-invariant to which targets are included
while the absolute values are not.

Note for captions: ``notch`` is never set, so the boxes are NOT notched.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cvd.prep.frames import _target_series, load_clean
from cvd.reporting.text import _format_plot_value, _tex_escape
from cvd.prediction.constants import _method_sort_key, BASE_DIR, BINARY_TARGETS, COMPACT_LEGEND_STAR_MARKER_SIZE, COMPACT_STAR_MARKER_SIZE, COMPACT_V4_FONT_SCALE, CONTINUOUS_TARGETS, INTENSITIES, METHOD_COLORS, METHOD_ORDER, RMSLE_TARGETS, TARGET_LABELS
from cvd.prediction.cohort_paths import CohortConfig

logger = logging.getLogger(__name__)
def _distribution_title(cohort_name: str, target_kind: str) -> str:
    return f"{cohort_name} Cohort: Distribution of {target_kind} Clinical Targets"

def _format_percent(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}%"
    return f"{value:.2f}%"

def _style_distribution_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#111111")

def _wrap_title(label: str, width: int = 26) -> str:
    """Break a panel title onto as few lines as fit `width` characters.

    Not ``textwrap.wrap``: that treats every space alike and splits
    "Prolonged Ventilation (>24 h)" after "(>24", which reads as a truncation
    rather than a wrap. Spaces inside a parenthetical are protected first, so a
    bracketed qualifier always travels as one token.
    """
    protected = re.sub(r"\(([^)]*)\)",
                       lambda m: "(" + m.group(1).replace(" ", "\u00a0") + ")",
                       label)
    lines: list[str] = []
    current = ""
    for word in protected.split(" "):
        trial = f"{current} {word}".strip()
        if current and len(trial) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return "\n".join(lines).replace("\u00a0", " ")


def plot_binary_target_distribution(
    clean: pd.DataFrame,
    output_dir: Path,
    cohort_name: str,
    targets: Iterable[str] = BINARY_TARGETS,
) -> None:
    target_list = [target for target in targets if target in clean.columns]
    if not target_list:
        return

    fig, axes = plt.subplots(1, len(target_list), figsize=(4.7 * len(target_list), 6.2), sharey=True)
    if len(target_list) == 1:
        axes = np.asarray([axes])

    colors = {0: "#A1D99B", 1: "#F8766D"}
    for ax, target in zip(axes, target_list):
        y = _target_series(clean, target, "binary")[target].dropna()
        values = y.astype(int)
        total = max(1, int(values.shape[0]))
        counts = values.value_counts().reindex([0, 1], fill_value=0)
        percents = counts / total * 100.0
        bars = ax.bar(
            [0, 1],
            percents.to_numpy(dtype=float),
            color=[colors[0], colors[1]],
            edgecolor="black",
            linewidth=1.4,
            width=0.72,
        )
        for bar, pct in zip(bars, percents):
            height = float(bar.get_height())
            if height <= 0:
                continue
            y_text = height * 0.62 if height >= 18 else height + 2.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                _format_percent(float(pct)),
                ha="center",
                va="center" if height >= 18 else "bottom",
                fontsize=17,
                color="black",
            )
        # Clinical names are long enough to run past a panel at this font size,
        # so wrap rather than shrink -- the three panels must stay comparable.
        ax.set_title(_wrap_title(TARGET_LABELS.get(target, target)),
                     fontsize=20, pad=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"], fontsize=18)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylim(0, 105)
        _style_distribution_axis(ax)

    axes[0].set_ylabel("Percentage (%)", fontsize=20)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(output_dir / "binary_clinical_target_distribution_v4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def _standardized_target_values(clean: pd.DataFrame, target: str) -> np.ndarray:
    values = pd.to_numeric(_target_series(clean, target, "continuous")[target], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    std = float(np.std(values, ddof=0))
    if std <= 0 or not np.isfinite(std):
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std

def plot_continuous_target_distribution(
    clean: pd.DataFrame,
    output_dir: Path,
    cohort_name: str,
    targets: Iterable[str] = CONTINUOUS_TARGETS,
) -> None:
    target_list = [target for target in targets if target in clean.columns]
    if not target_list:
        return

    # One row when three or fewer targets are drawn. The main-text figure now
    # shows only the three analysed post-operative durations, and a fixed
    # 2-column grid put those three in a 2x2 block with an empty quadrant.
    ncols = min(3, len(target_list))
    nrows = int(math.ceil(len(target_list) / ncols))
    # Width scales with the column count instead of a fixed 14.5 in. At three
    # columns the fixed canvas gave each panel 7.25/2 in, which is narrower
    # than these outcome names need: "Length of Stay After Surgery (y-log
    # scale)" ran into its neighbour's title.
    #
    # The per-panel size is chosen so this figure lands on the page at the same
    # scale as the binary distributions (Figure 4): two panels here against
    # three there gives 14.2 in versus 14.1 in of canvas, so at equal
    # \includegraphics widths the two render at the same apparent font size.
    # They are wider and shorter than Figure 4's panels because a histogram
    # reads better landscape than a two-bar chart does.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.1 * ncols, 4.3 * nrows), squeeze=False
    )
    colors = ["#2C7FB8", "#F28E2B", "#4DAF4A", "#D62728", "#9467BD"]
    log_y_targets = {"losas", "icuinhrs", "icuadhrs"}

    for ax, target, color in zip(axes.ravel(), target_list, colors):
        values = _standardized_target_values(clean, target)
        if values.size == 0:
            ax.set_axis_off()
            continue
        ax.hist(values, bins=42, color=color, edgecolor="black", linewidth=0.8, alpha=0.9)
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        ax.axvline(mean_value, color="black", linestyle="--", linewidth=2.2, label="Mean")
        ax.axvline(median_value, color="red", linestyle=":", linewidth=2.2, label="Median")
        title = _wrap_title(TARGET_LABELS.get(target, target))
        if target in log_y_targets:
            # Second line, not a suffix: appending it kept the title one long
            # run of text that no panel width accommodated.
            title = f"{title}\n(y-log scale)"
            ax.set_yscale("log")
        ax.set_title(title, fontsize=20, pad=10)
        ax.set_xlabel("Standardized value", fontsize=17)
        ax.set_ylabel("Count", fontsize=17)
        ax.tick_params(axis="both", labelsize=15)
        ax.legend(loc="best", fontsize=15, frameon=True)
        _style_distribution_axis(ax)

    for ax in axes.ravel()[len(target_list) :]:
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0, w_pad=1.4)
    fig.savefig(output_dir / "continuous_clinical_target_distribution_standardized_v4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_target_distribution_figures(
    config: CohortConfig,
    binary_targets: Iterable[str] = BINARY_TARGETS,
    continuous_targets: Iterable[str] = CONTINUOUS_TARGETS,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    clean = load_clean(config)
    plot_binary_target_distribution(clean, config.output_dir, config.name, binary_targets)
    plot_continuous_target_distribution(clean, config.output_dir, config.name, continuous_targets)

def plot_binary_figures(binary: pd.DataFrame, output_dir: Path, cohort_name: str) -> None:
    ok = binary[binary["status"] == "ok"].copy()
    if ok.empty:
        return
    for target, target_df in ok.groupby("target"):
        label = TARGET_LABELS.get(target, target)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, intensity in zip(axes, INTENSITIES):
            sub = target_df[target_df["intensity"] == intensity].copy()
            summary = (
                sub.groupby("method", observed=False)[["balanced_accuracy", "au_prc"]]
                .mean()
                .reindex(METHOD_ORDER)
                .dropna(how="all")
            )
            x = np.arange(len(summary))
            width = 0.38
            ax.bar(x - width / 2, summary["balanced_accuracy"], width, label="Balanced Accuracy", color="#4C78A8")
            ax.bar(x + width / 2, summary["au_prc"], width, label="AU-PRC", color="#F58518")
            ax.set_title(f"{intensity}% MCAR")
            ax.set_xticks(x)
            ax.set_xticklabels(summary.index, rotation=35, ha="right")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        axes[0].set_ylabel("Score")
        axes[-1].legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / f"binary_detail_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        summary = (
            target_df.groupby("method", observed=False)[["balanced_accuracy", "au_prc"]]
            .agg(["mean", "std"])
            .reindex(METHOD_ORDER)
            .dropna(how="all")
        )
        x = np.arange(len(summary))
        fig, ax = plt.subplots(figsize=(11, 5))
        width = 0.38
        ax.bar(
            x - width / 2,
            summary[("balanced_accuracy", "mean")],
            width,
            yerr=summary[("balanced_accuracy", "std")].fillna(0),
            capsize=3,
            label="Balanced Accuracy",
            color="#4C78A8",
        )
        ax.bar(
            x + width / 2,
            summary[("au_prc", "mean")],
            width,
            yerr=summary[("au_prc", "std")].fillna(0),
            capsize=3,
            label="AU-PRC",
            color="#F58518",
        )
        ax.set_title(f"{cohort_name}: {label} Mean Across Missingness")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"binary_summary_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_binary_summary_v2(target_df, output_dir, cohort_name, target)
    plot_binary_all_summary_v2(ok, output_dir, cohort_name)
    plot_binary_compact_summary_v3(ok, output_dir, cohort_name)

def _continuous_error_metric_for_plot(target_df: pd.DataFrame, target: str) -> tuple[str, str]:
    if "mphe" in target_df.columns and pd.to_numeric(target_df["mphe"], errors="coerce").notna().any():
        return "mphe", "MLPE (z-scored)"
    if target in RMSLE_TARGETS and "rmsle" in target_df.columns and pd.to_numeric(target_df["rmsle"], errors="coerce").notna().any():
        return "rmsle", "RMSLE"
    return "rmse_z", "RMSE (z-scored)"

def plot_continuous_figures(continuous: pd.DataFrame, output_dir: Path, cohort_name: str) -> None:
    ok = continuous[continuous["status"] == "ok"].copy()
    if ok.empty:
        return
    for target, target_df in ok.groupby("target"):
        label = TARGET_LABELS.get(target, target)
        metric, metric_label = _continuous_error_metric_for_plot(target_df, str(target))
        summary = (
            target_df.groupby("method", observed=False)[metric]
            .agg(["mean", "std"])
            .reindex(METHOD_ORDER)
            .dropna(how="all")
        )
        x = np.arange(len(summary))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x, summary["mean"], yerr=summary["std"].fillna(0), capsize=3, color="#54A24B")
        ax.set_title(f"{cohort_name}: {label} {metric_label} Across Missingness")
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(output_dir / f"continuous_summary_V2_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_continuous_summary_v2(target_df, output_dir, cohort_name, target)
    plot_continuous_all_summary_v2(ok, output_dir, cohort_name)
    plot_continuous_compact_summary_v3(ok, output_dir, cohort_name)
    plot_continuous_compact_all_V2(ok, output_dir, cohort_name)

def _ordered_method_values(df: pd.DataFrame, metric: str) -> tuple[list[str], list[np.ndarray]]:
    methods: list[str] = []
    values: list[np.ndarray] = []
    for method in METHOD_ORDER:
        vals = pd.to_numeric(df.loc[df["method"] == method, metric], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            continue
        methods.append(method)
        values.append(vals)
    return methods, values

def _metric_ylim(values: list[np.ndarray], *, clamp_binary: bool = False) -> tuple[float, float]:
    merged = np.concatenate(values) if values else np.array([0.0, 1.0])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(finite.min())
    hi = float(finite.max())
    span = hi - lo
    pad = max(span * 0.18, 0.01 if clamp_binary else 0.02)
    if span <= 1e-12:
        pad = max(abs(hi) * 0.05, 0.02)
    ymin, ymax = lo - pad, hi + pad
    if clamp_binary:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)
        if ymax - ymin < 0.05:
            center = (ymin + ymax) / 2
            ymin = max(0.0, center - 0.025)
            ymax = min(1.0, center + 0.025)
    return ymin, ymax

def _draw_metric_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    clamp_binary: bool = False,
) -> None:
    methods, values = _ordered_method_values(df, metric)
    if not values:
        ax.set_axis_off()
        return

    positions = np.arange(1, len(methods) + 1)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops={"color": "#222222", "linewidth": 1.5},
        meanprops={"color": "#B23A48", "linewidth": 1.3},
        boxprops={"facecolor": "#D8E2DC", "edgecolor": "#333333", "linewidth": 1.0},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#6B7280", "markeredgecolor": "#6B7280"},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#D8E2DC"))
        patch.set_alpha(0.78)

    intensity_offsets = {"5": -0.18, "10": 0.0, "20": 0.18}
    intensity_colors = {"5": "#4C78A8", "10": "#F58518", "20": "#54A24B"}
    for pos, method in zip(positions, methods):
        method_df = df[df["method"] == method].copy()
        method_df["intensity"] = method_df["intensity"].astype(str)
        for _, row in method_df.iterrows():
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            intensity = str(row["intensity"])
            x = pos + intensity_offsets.get(intensity, 0.0)
            ax.scatter(
                x,
                float(value),
                s=38,
                color=intensity_colors.get(intensity, "#6B7280"),
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
                label=f"{intensity}% MCAR",
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8, frameon=True)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*_metric_ylim(values, clamp_binary=clamp_binary))
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

def plot_binary_summary_v2(target_df: pd.DataFrame, output_dir: Path, cohort_name: str, target: str) -> None:
    label = TARGET_LABELS.get(target, target)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    _draw_metric_boxplot(axes[0], target_df, "balanced_accuracy", "Balanced Accuracy", clamp_binary=True)
    _draw_metric_boxplot(axes[1], target_df, "au_prc", "AU-PRC", clamp_binary=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"binary_summary_v2_{target}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_continuous_summary_v2(target_df: pd.DataFrame, output_dir: Path, cohort_name: str, target: str) -> None:
    label = TARGET_LABELS.get(target, target)
    metric, metric_label = _continuous_error_metric_for_plot(target_df, str(target))
    fig, ax = plt.subplots(figsize=(13, 5.8))
    _draw_metric_boxplot(ax, target_df, metric, metric_label)
    ax.set_title(f"{cohort_name}: {label} {metric_label} Across Missingness - V2")
    fig.tight_layout()
    fig.savefig(output_dir / f"continuous_summary_v2_V2_{target}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def _target_order(df: pd.DataFrame, preferred: tuple[str, ...]) -> list[str]:
    present = set(df["target"].dropna().astype(str))
    ordered = [target for target in preferred if target in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered

def _metric_xlim(values: list[np.ndarray], *, clamp_binary: bool = False) -> tuple[float, float]:
    ymin, ymax = _metric_ylim(values, clamp_binary=clamp_binary)
    return ymin, ymax

def _draw_horizontal_metric_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    xlabel: str,
    *,
    clamp_binary: bool = False,
    show_method_labels: bool = True,
    show_reference_label: bool = False,
) -> None:
    methods, values = _ordered_method_values(df, metric)
    if not values:
        ax.set_axis_off()
        return

    positions = np.arange(1, len(methods) + 1)
    box = ax.boxplot(
        values,
        positions=positions,
        vert=False,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops={"color": "#222222", "linewidth": 1.5},
        meanprops={"color": "#333333", "linewidth": 1.1, "linestyle": ":"},
        boxprops={"edgecolor": "#333333", "linewidth": 1.0},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#6B7280", "markeredgecolor": "#6B7280"},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#D8E2DC"))
        patch.set_alpha(0.80)

    intensity_offsets = {"5": -0.16, "10": 0.0, "20": 0.16}
    for pos, method in zip(positions, methods):
        method_df = df[df["method"] == method].copy()
        method_df["intensity"] = method_df["intensity"].astype(str)
        for _, row in method_df.iterrows():
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            y = pos + intensity_offsets.get(str(row["intensity"]), 0.0)
            ax.scatter(
                float(value),
                y,
                s=22,
                color="#111827",
                edgecolor="white",
                linewidth=0.35,
                alpha=0.85,
                zorder=3,
            )

    baseline_values = pd.to_numeric(df.loc[df["method"] == "Baseline", metric], errors="coerce").dropna()
    if not baseline_values.empty:
        baseline_median = float(baseline_values.median())
        ax.axvline(
            baseline_median,
            color="#7A7A7A",
            linestyle="--",
            linewidth=1.2,
            label="Baseline median" if show_reference_label else None,
            zorder=1,
        )
        if show_reference_label:
            ax.legend(loc="best", fontsize=8, frameon=True)

    ax.set_xlabel(xlabel)
    ax.set_xlim(*_metric_xlim(values, clamp_binary=clamp_binary))
    ax.set_yticks(positions)
    ax.set_yticklabels(methods if show_method_labels else [])
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

def plot_binary_all_summary_v2(binary: pd.DataFrame, output_dir: Path, cohort_name: str) -> None:
    ok = binary[binary["status"] == "ok"].copy()
    targets = _target_order(ok, BINARY_TARGETS)
    if not targets:
        return

    fig, axes = plt.subplots(len(targets), 1, figsize=(14, max(3.4 * len(targets), 8)), squeeze=False)
    for row_idx, target in enumerate(targets):
        target_df = ok[ok["target"] == target]
        label = TARGET_LABELS.get(target, target)
        _draw_horizontal_metric_boxplot(
            axes[row_idx, 0],
            target_df,
            "balanced_accuracy",
            "Balanced Accuracy",
            clamp_binary=True,
            show_method_labels=True,
            show_reference_label=(row_idx == 0),
        )
        axes[row_idx, 0].set_title(f"{label} - Balanced Accuracy")

    fig.tight_layout()
    fig.savefig(output_dir / "binary_summary_all_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Axis and panel wording for the explicit-metric path. "Standardized RMSE" is
# the phrase the manuscript, the section heading and both captions use; the
# auto-detect path's "RMSE (z-scored)" means the same thing but reads as a
# different metric next to them.
_METRIC_LABELS = {
    "rmse_z": "Standardized RMSE",
    "rmsle": "RMSLE",
    "mphe": "MLPE (z-scored)",
}


def plot_continuous_all_summary_v2(
    continuous: pd.DataFrame,
    output_dir: Path,
    cohort_name: str,
    metric: str | None = None,
) -> None:
    """Per-target continuous panels for one cohort.

    *metric* names the column to draw. Pass it: the fallback
    ``_continuous_error_metric_for_plot`` picks per target and prefers
    ``rmsle`` for the three post-operative endpoints, which is how
    Supplementary Figures S4/S5 came to be titled RMSLE while their registry
    entry, section heading, captions and the Friedman statistics quoted in
    those captions all described standardized RMSE.
    """
    ok = continuous[continuous["status"] == "ok"].copy()
    targets = _target_order(ok, CONTINUOUS_TARGETS)
    if not targets:
        return

    fig, axes = plt.subplots(len(targets), 1, figsize=(14, max(3.4 * len(targets), 8)), squeeze=False)
    for row_idx, target in enumerate(targets):
        ax = axes[row_idx, 0]
        target_df = ok[ok["target"] == target]
        label = TARGET_LABELS.get(target, target)
        if metric is None:
            metric_col, metric_label = _continuous_error_metric_for_plot(target_df, str(target))
        else:
            metric_col, metric_label = metric, _METRIC_LABELS.get(metric, metric)
        _draw_horizontal_metric_boxplot(
            ax,
            target_df,
            metric_col,
            metric_label,
            show_method_labels=True,
            show_reference_label=(row_idx == 0),
        )
        ax.set_title(f"{label} - {metric_label}")

    fig.tight_layout()
    fig.savefig(output_dir / "continuous_summary_all_v2_V2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def _add_target_baseline_delta(df: pd.DataFrame, metric: str, output_col: str) -> pd.DataFrame:
    out = df.copy()
    baseline = (
        out.loc[out["method"] == "Baseline"]
        .groupby("target", observed=False)[metric]
        .median()
        .rename("target_baseline_median")
    )
    out = out.merge(baseline, left_on="target", right_index=True, how="left")
    out[output_col] = pd.to_numeric(out[metric], errors="coerce") - pd.to_numeric(
        out["target_baseline_median"], errors="coerce"
    )
    return out.drop(columns=["target_baseline_median"])

def _add_target_baseline_ratio(df: pd.DataFrame, metric: str, output_col: str) -> pd.DataFrame:
    out = df.copy()
    baseline = (
        out.loc[out["method"] == "Baseline"]
        .groupby("target", observed=False)[metric]
        .median()
        .replace(0, np.nan)
        .rename("target_baseline_median")
    )
    out = out.merge(baseline, left_on="target", right_index=True, how="left")
    out[output_col] = pd.to_numeric(out[metric], errors="coerce") / pd.to_numeric(
        out["target_baseline_median"], errors="coerce"
    )
    return out.drop(columns=["target_baseline_median"])

def annotate_rank_row(
    ax,
    methods,
    row,
    *,
    bold_top_n: int = 2,
    y_values: float = -0.215,
    y_caption: float = -0.40,
    block_label: str = "design x intensity x endpoint x split blocks",
    note: str | None = None,
    caption: str | None = None,
    fontsize: float = 9.0,
) -> tuple[str, object]:
    """Friedman ranking underneath a prediction panel.

    Same presentation as the fidelity figures -- ordinal rank over the mean
    rank, best `bold_top_n` in bold, one caption naming the test -- but the
    positions are 1-indexed here, because `_draw_compact_relative_boxplot`
    places its boxes at `arange(1, n+1)` while the fidelity panels start at 0.
    Using the fidelity annotator directly would shift every label one method to
    the left.

    Pass `caption` to replace the auto-built caption: in a two-panel figure the
    lines saying what the numbers are and how many blocks they cover are
    identical under both panels, and repeating them twice in 9pt is just noise.
    Returns `(shared prefix, caption artist)` so the caller can draw the shared
    half once for the whole figure and position it under both panels.
    """
    import pandas as _pd

    raw = row.get("mean_ranks") if hasattr(row, "get") else None
    if not raw:
        return "", None
    ranks = {}
    for part in str(raw).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                ranks[k.strip()] = float(v)
            except ValueError:
                pass
    if not ranks:
        return "", None
    order = sorted(ranks, key=ranks.get)
    # Tied mean ranks share an ordinal position (standard competition ranking:
    # 1, 2, 2, 4). Numbering them sequentially asserted an ordering the
    # statistic does not contain -- Cohort2 continuous puts GAIN, MIRACLE and
    # MIWAE at exactly 4.917 and they were drawn as 3rd, 4th and 5th, an
    # artefact of the sort rather than a result.
    ordinal: dict[str, int] = {}
    position = 0
    for i, m in enumerate(order):
        if i == 0 or ranks[m] != ranks[order[i - 1]]:
            position = i + 1
        ordinal[m] = position

    for pos, method in enumerate(methods, start=1):
        if method not in ranks:
            continue
        ax.text(
            pos, y_values, f"{ordinal[method]}\n({ranks[method]:.2f})",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=10.5, color="#333333", linespacing=1.25,
            fontweight="bold" if ordinal[method] <= bold_top_n else "normal",
        )

    chi2, pval = row.get("friedman_chi2"), row.get("friedman_p")
    blocks = row.get("n_complete_blocks")

    shared = "rank position and (Friedman mean rank), 1 = best"
    if bold_top_n:
        shared += f", best {bold_top_n} in bold"
    if blocks is not None and _pd.notna(blocks):
        shared += f"\nwithin this cohort, over {int(blocks)} {block_label}"

    if caption is None:
        cap = shared
        if chi2 is not None and pval is not None and _pd.notna(chi2) and _pd.notna(pval):
            cap += f"\n$\\chi^2$={float(chi2):.2f}, $p$={float(pval):.2e}"
        if note:
            cap += f"\n{note}"
    else:
        cap = caption

    artist = None
    if cap:
        artist = ax.text(0.5, y_caption, cap, transform=ax.transAxes,
                         ha="center", va="top", fontsize=fontsize,
                         color="#555555", linespacing=1.4)
    return shared, artist


def _draw_compact_relative_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    *,
    reference: float,
    lower_is_better: bool,
    star_label: str,
    font_scale: float = 1.0,
) -> None:
    methods, values = _ordered_method_values(df, value_col)
    if not values:
        ax.set_axis_off()
        return

    positions = np.arange(1, len(methods) + 1)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanline=True,
        medianprops={"color": "#111827", "linewidth": 1.5},
        meanprops={"color": "#333333", "linewidth": 1.1, "linestyle": ":"},
        boxprops={"edgecolor": "#333333", "linewidth": 1.0},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#374151", "markeredgecolor": "#374151"},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#D8E2DC"))
        patch.set_alpha(0.82)

    visible_parts: list[np.ndarray] = []
    for artist_key in ("boxes", "whiskers", "caps", "medians", "means"):
        for artist in box.get(artist_key, []):
            if hasattr(artist, "get_ydata"):
                y_data = np.asarray(artist.get_ydata(), dtype=float)
            else:
                y_data = np.asarray(artist.get_path().vertices[:, 1], dtype=float)
            y_data = y_data[np.isfinite(y_data)]
            if y_data.size:
                visible_parts.append(y_data)

    if not visible_parts:
        ax.set_axis_off()
        return
    visible_y = np.concatenate(visible_parts)
    data_min = min(float(visible_y.min()), reference)
    data_max = max(float(visible_y.max()), reference)
    span = max(data_max - data_min, 1e-6)
    star_ys: list[float] = []
    has_star = False

    for pos, method, vals in zip(positions, methods, values):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        if method == "Baseline":
            continue
        median = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        better = median < reference if lower_is_better else median > reference
        if better:
            star_y = float(q1 if lower_is_better else q3)
            star_ys.append(star_y)
            has_star = True
            ax.scatter(
                pos,
                star_y,
                marker="*",
                s=COMPACT_STAR_MARKER_SIZE,
                color="#111827",
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
                clip_on=False,
            )

    ax.axhline(reference, color="#7A7A7A", linestyle="--", linewidth=1.3, label="Baseline reference", zorder=1)
    if has_star:
        ax.scatter(
            [],
            [],
            marker="*",
            s=COMPACT_LEGEND_STAR_MARKER_SIZE,
            color="#111827",
            label=star_label,
            clip_on=False,
        )
    ax.legend(loc="best", fontsize=8 * font_scale, frameon=True)

    all_y = np.concatenate([visible_y, np.asarray(star_ys + [reference], dtype=float)])
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    y_span = max(y_max - y_min, 1e-6)
    y_pad = max(y_span * 0.055, 0.01)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_ylabel(ylabel, fontsize=12 * font_scale)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.tick_params(axis="both", labelsize=10 * font_scale)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

@dataclass(frozen=True)
class CompactCaptionEntry:
    filename: str
    metric_title: str
    caption: str
    label: str

def _compact_plot_title(cohort_name: str, target_kind: str, version: str = "v3") -> str:
    return f"{cohort_name} Cohort: Prediction Performance on {target_kind} Targets"

def _compact_font_scale(version: str) -> float:
    return COMPACT_V4_FONT_SCALE if version == "v4" else 1.0

def _save_compact_relative_figure(
    df: pd.DataFrame,
    output_dir: Path,
    cohort_name: str,
    target_kind: str,
    filename: str,
    metric_title: str,
    value_col: str,
    ylabel: str,
    *,
    reference: float,
    lower_is_better: bool,
    star_label: str,
    version: str = "v3",
) -> None:
    font_scale = _compact_font_scale(version)
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    _draw_compact_relative_boxplot(
        ax,
        df,
        value_col,
        ylabel,
        reference=reference,
        lower_is_better=lower_is_better,
        star_label=star_label,
        font_scale=font_scale,
    )
    ax.set_title(metric_title, fontsize=12 * font_scale)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_method_list(methods: list[str]) -> str:
    if not methods:
        return "none"
    if len(methods) == 1:
        return methods[0]
    if len(methods) == 2:
        return f"{methods[0]} and {methods[1]}"
    return f"{', '.join(methods[:-1])}, and {methods[-1]}"


def _binary_metric_extra_note(df: pd.DataFrame, metric_slug: str) -> str:
    if metric_slug == "balanced-accuracy":
        genai_methods = ["MIWAE", "GAIN", "ReMasker", "MIRACLE"]
        medians = (
            df.groupby("method", observed=False)["balanced_accuracy_vs_baseline"]
            .median()
            .reindex(genai_methods)
            .dropna()
        )
        better_genai = [method for method, value in medians.items() if value > 0]
        if len(better_genai) == len(genai_methods):
            genai_note = (
                "All four GenAI methods have median balanced accuracy above the baseline median in this cohort."
            )
        elif better_genai:
            genai_note = (
                f"GenAI methods with median balanced accuracy above baseline are {_format_method_list(better_genai)}."
            )
        else:
            genai_note = "No GenAI method has median balanced accuracy above baseline in this cohort."
        return (
            "Mode imputation for binary variables can show an upper whisker above baseline, "
            "but this is misleading for highly unbalanced targets because mode filling is effectively filling 0; "
            "its lower whisker remains low and its AU--PRC median is not above baseline. "
            f"{genai_note}"
        )

    if metric_slug == "au-prc":
        method_stats = (
            df[df["method"] != "Baseline"]
            .groupby("method", observed=False)["au_prc_vs_baseline"]
            .agg(
                median="median",
                q1=lambda s: float(np.percentile(pd.to_numeric(s, errors="coerce").dropna(), 25)),
                q3=lambda s: float(np.percentile(pd.to_numeric(s, errors="coerce").dropna(), 75)),
            )
            .dropna()
        )
        if method_stats.empty:
            return "AU--PRC is emphasized because missing a positive case is high risk."
        best_method = str(method_stats["median"].idxmax())
        best = method_stats.loc[best_method]
        if best_method == "GAIN":
            bound_note = (
                "GAIN has the highest median AU--PRC delta; its interquartile range is "
                f"{_format_plot_value(float(best['q1']), 3, signed=True)} to "
                f"{_format_plot_value(float(best['q3']), 3, signed=True)} relative to baseline."
            )
        else:
            bound_note = (
                f"{best_method} has the highest median AU--PRC delta; GAIN is not the AU--PRC median leader in this cohort."
            )
        return (
            "AU--PRC is emphasized for clinical positives because missing a positive case is high risk. "
            f"{bound_note}"
        )

    return ""

def _compact_caption(
    df: pd.DataFrame,
    value_col: str,
    cohort_name: str,
    target_kind: str,
    metric_tex: str,
    value_tex: str,
    *,
    reference: float,
    lower_is_better: bool,
    decimals: int,
    extra_note: str = "",
) -> str:
    methods, values = _ordered_method_values(df, value_col)
    rows: list[dict[str, object]] = []
    for method, vals in zip(methods, values):
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        q1, q3 = np.percentile(finite, [25, 75])
        rows.append(
            {
                "method": method,
                "median": float(np.median(finite)),
                "q1": float(q1),
                "q3": float(q3),
                "n": int(finite.size),
            }
        )

    non_baseline = [row for row in rows if row["method"] != "Baseline"]
    if not non_baseline:
        return (
            f"{cohort_name} {metric_tex} compact summary for {target_kind.lower()} targets. "
            f"Values are {value_tex}; the dashed line is the target-specific baseline median."
        )

    best = min(non_baseline, key=lambda row: float(row["median"])) if lower_is_better else max(
        non_baseline, key=lambda row: float(row["median"])
    )
    better = [
        row
        for row in non_baseline
        if (float(row["median"]) < reference if lower_is_better else float(row["median"]) > reference)
    ]
    better = sorted(better, key=lambda row: float(row["median"]), reverse=not lower_is_better)

    counts = sorted({int(row["n"]) for row in rows})
    n_splits = int(df["split_repeat"].nunique()) if "split_repeat" in df.columns else 1
    split_text = (
        f" across {n_splits} independently randomized train/validation/test splits"
        if n_splits > 1
        else ""
    )
    if len(counts) == 1:
        count_text = f"Each box pools {counts[0]} target-intensity-split results{split_text}."
    else:
        count_text = f"Boxes pool {counts[0]} to {counts[-1]} target-intensity-split results per method{split_text}."

    direction = "lower" if lower_is_better else "higher"
    signed = not math.isclose(reference, 1.0)
    best_text = (
        f"Among non-baseline methods, {best['method']} has the best median "
        f"({_format_plot_value(float(best['median']), decimals, signed=signed)}; "
        f"IQR {_format_plot_value(float(best['q1']), decimals, signed=signed)} to "
        f"{_format_plot_value(float(best['q3']), decimals, signed=signed)})."
    )
    if better:
        star_text = (
            f"Stars mark methods with median {direction} than the baseline median; "
            f"here this includes {_format_method_list([str(row['method']) for row in better])}."
        )
    else:
        star_text = f"No non-baseline method has a median {direction} than the baseline median, so no star is shown."

    ref_text = _format_plot_value(reference, decimals, signed=signed)
    return (
        f"{cohort_name} {metric_tex} compact summary for {target_kind.lower()} targets. "
        f"Values are {value_tex}; the dashed line marks the target-specific baseline median ({ref_text}). "
        f"{count_text} {best_text} {star_text}{(' ' + extra_note) if extra_note else ''}"
    )

def _caption_label(cohort_name: str, section_key: str, metric_slug: str, version: str = "v3") -> str:
    cohort_slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in cohort_name).strip("-")
    return f"fig:{cohort_slug}-{section_key}-{metric_slug}-compact-{version}"

def _write_compact_caption_section(
    output_dir: Path,
    cohort_name: str,
    section_key: str,
    section_title: str,
    entries: list[CompactCaptionEntry],
    version: str = "v3",
) -> None:
    if not entries:
        return

    if version == "v3":
        start = f"% BEGIN {section_key.upper()} CAPTIONS"
        end = f"% END {section_key.upper()} CAPTIONS"
    else:
        start = f"% BEGIN {version.upper()} {section_key.upper()} CAPTIONS"
        end = f"% END {version.upper()} {section_key.upper()} CAPTIONS"
    section_lines = [start, f"\\subsection*{{{cohort_name}: {section_title}}}"]
    for entry in entries:
        section_lines.extend(
            [
                "",
                "\\begin{figure}[htbp]",
                "\\centering",
                f"\\includegraphics[width=0.86\\linewidth]{{\\detokenize{{{entry.filename}}}}}",
                f"\\caption{{{entry.caption}}}",
                f"\\label{{{entry.label}}}",
                "\\end{figure}",
            ]
        )
    section_lines.append(end)
    section_text = "\n".join(section_lines)

    caption_path = output_dir / f"compact_summary_{version}_captions.tex"
    if caption_path.exists():
        text = caption_path.read_text(encoding="utf-8")
    else:
        text = "% Auto-generated captions for compact prediction performance figures.\n"

    start_at = text.find(start)
    end_at = text.find(end)
    if start_at >= 0 and end_at >= start_at:
        end_at += len(end)
        text = f"{text[:start_at].rstrip()}\n\n{section_text}\n\n{text[end_at:].lstrip()}"
    else:
        text = f"{text.rstrip()}\n\n{section_text}\n"
    caption_path.write_text(text, encoding="utf-8")

def plot_binary_compact_summary_v3(
    binary: pd.DataFrame,
    output_dir: Path,
    cohort_name: str,
    version: str = "v3",
    rankings: dict | None = None,
) -> None:
    ok = binary[binary["status"] == "ok"].copy()
    if ok.empty:
        return
    ok = _add_target_baseline_delta(ok, "balanced_accuracy", "balanced_accuracy_vs_baseline")
    ok = _add_target_baseline_delta(ok, "au_prc", "au_prc_vs_baseline")
    font_scale = _compact_font_scale(version)

    # Taller when a ranking row is drawn: the rows sit below the axis in axis
    # coordinates, which tight_layout does not reserve space for.
    fig, axes = plt.subplots(1, 2, figsize=(16, 9.2 if rankings else 6.5), sharey=False)
    _draw_compact_relative_boxplot(
        axes[0],
        ok,
        "balanced_accuracy_vs_baseline",
        "Balanced Accuracy minus target baseline median",
        reference=0.0,
        lower_is_better=False,
        star_label="Higher than baseline median",
        font_scale=font_scale,
    )
    _draw_compact_relative_boxplot(
        axes[1],
        ok,
        "au_prc_vs_baseline",
        "AU-PRC minus target baseline median",
        reference=0.0,
        lower_is_better=False,
        star_label="Higher than baseline median",
        font_scale=font_scale,
    )
    axes[0].set_title("Balanced Accuracy", fontsize=12 * font_scale)
    axes[1].set_title("AU-PRC", fontsize=12 * font_scale)
    if rankings:
        for ax, key in zip(axes, ("balanced_accuracy", "au_prc")):
            if rankings.get(key) is not None:
                annotate_rank_row(ax, _ordered_method_values(ok, f"{key}_vs_baseline")[0],
                                  rankings[key])
    fig.tight_layout(h_pad=3.0)
    fig.savefig(output_dir / f"binary_summary_compact_{version}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metric_specs = [
        {
            "filename": f"binary_summary_compact_{version}_balanced_accuracy.png",
            "metric_title": "Balanced Accuracy",
            "value_col": "balanced_accuracy_vs_baseline",
            "ylabel": "Balanced Accuracy minus target baseline median",
            "metric_tex": "balanced accuracy",
            "value_tex": "balanced accuracy differences from each target-specific baseline median",
            "metric_slug": "balanced-accuracy",
            "decimals": 3,
        },
        {
            "filename": f"binary_summary_compact_{version}_au_prc.png",
            "metric_title": "AU-PRC",
            "value_col": "au_prc_vs_baseline",
            "ylabel": "AU-PRC minus target baseline median",
            "metric_tex": "AU--PRC",
            "value_tex": "AU--PRC differences from each target-specific baseline median",
            "metric_slug": "au-prc",
            "decimals": 3,
        },
    ]
    caption_entries: list[CompactCaptionEntry] = []
    for spec in metric_specs:
        _save_compact_relative_figure(
            ok,
            output_dir,
            cohort_name,
            "Binary",
            str(spec["filename"]),
            str(spec["metric_title"]),
            str(spec["value_col"]),
            str(spec["ylabel"]),
            reference=0.0,
            lower_is_better=False,
            star_label="Higher than baseline median",
            version=version,
        )
        note = _binary_metric_extra_note(ok, str(spec["metric_slug"])) if version == "v4" else ""
        caption_entries.append(
            CompactCaptionEntry(
                filename=str(spec["filename"]),
                metric_title=str(spec["metric_title"]),
                caption=_compact_caption(
                    ok,
                    str(spec["value_col"]),
                    cohort_name,
                    "binary",
                    str(spec["metric_tex"]),
                    str(spec["value_tex"]),
                    reference=0.0,
                    lower_is_better=False,
                    decimals=int(spec["decimals"]),
                    extra_note=note,
                ),
                label=_caption_label(cohort_name, "binary", str(spec["metric_slug"]), version=version),
            )
        )
    _write_compact_caption_section(
        output_dir,
        cohort_name,
        "binary",
        "Prediction Performance: Binary Targets",
        caption_entries,
        version=version,
    )

def plot_binary_compact_summary_v4(binary: pd.DataFrame, output_dir: Path, cohort_name: str) -> None:
    plot_binary_compact_summary_v3(binary, output_dir, cohort_name, version="v4")
    plot_binary_false_negative_v4(binary, output_dir, cohort_name)

def _save_tex_table(path: Path, rows: list[list[object]], columns: list[str], caption: str, label: str) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(_tex_escape(col) for col in columns) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_escape(value) for value in row) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")

def _write_binary_class_count_tables(binary: pd.DataFrame, output_dir: Path, cohort_name: str) -> None:
    ok = binary[binary["status"] == "ok"].copy()
    required = {
        "target",
        "split_repeat",
        "split_seed",
        "n_train_negative_before_downsampling",
        "n_train_positive_before_downsampling",
        "n_train_negative",
        "n_train_positive",
        "n_test_negative",
        "n_test_positive",
    }
    if ok.empty or not required.issubset(ok.columns):
        return

    count_cols = [
        "target",
        "split_repeat",
        "split_seed",
        "n_train_negative_before_downsampling",
        "n_train_positive_before_downsampling",
        "n_train_negative",
        "n_train_positive",
        "n_test_negative",
        "n_test_positive",
    ]
    counts = ok[count_cols].drop_duplicates().sort_values(["target", "split_repeat"])
    counts.to_csv(output_dir / "binary_class_counts_v4.csv", index=False)

    rows = []
    for _, row in counts.iterrows():
        rows.append(
            [
                TARGET_LABELS.get(str(row["target"]), str(row["target"])),
                int(row["split_repeat"]),
                int(row["split_seed"]),
                int(row["n_train_negative_before_downsampling"]),
                int(row["n_train_positive_before_downsampling"]),
                int(row["n_train_negative"]),
                int(row["n_train_positive"]),
                int(row["n_test_negative"]),
                int(row["n_test_positive"]),
            ]
        )
    _save_tex_table(
        output_dir / "binary_class_counts_v4.tex",
        rows,
        [
            "Target",
            "Split",
            "Seed",
            "Train 0 before",
            "Train 1 before",
            "Train 0 used",
            "Train 1 used",
            "Test 0",
            "Test 1",
        ],
        (
            f"{cohort_name} binary target class counts for the three independently randomized "
            "train/validation/test splits. Training counts are shown before and after train-fold-only "
            "downsampling of class 0."
        ),
        _caption_label(cohort_name, "binary", "class-counts", version="v4"),
    )

def _aggregate_binary_false_negatives(binary: pd.DataFrame) -> pd.DataFrame:
    ok = binary[binary["status"] == "ok"].copy()
    required = {"method", "intensity", "split_repeat", "split_seed", "false_negative", "n_test_positive"}
    if ok.empty or not required.issubset(ok.columns):
        return pd.DataFrame()

    ok["false_negative"] = pd.to_numeric(ok["false_negative"], errors="coerce")
    ok["n_test_positive"] = pd.to_numeric(ok["n_test_positive"], errors="coerce")
    ok = ok.dropna(subset=["false_negative", "n_test_positive"])
    if ok.empty:
        return pd.DataFrame()

    agg = (
        ok.groupby(["method", "intensity", "split_repeat", "split_seed"], observed=False)
        .agg(
            false_negative=("false_negative", "sum"),
            n_test_positive=("n_test_positive", "sum"),
            n_targets=("target", "nunique"),
        )
        .reset_index()
    )
    agg["false_negative_rate"] = np.where(
        agg["n_test_positive"] > 0,
        agg["false_negative"] / agg["n_test_positive"],
        np.nan,
    )
    return agg

def _write_binary_false_negative_summary(binary: pd.DataFrame, output_dir: Path, cohort_name: str) -> None:
    aggregated = _aggregate_binary_false_negatives(binary)
    if aggregated.empty:
        return

    grouped = (
        aggregated.groupby(["method"], observed=False)
        .agg(
            median_false_negative=("false_negative", "median"),
            q1_false_negative=("false_negative", lambda s: float(np.percentile(pd.to_numeric(s, errors="coerce").dropna(), 25))),
            q3_false_negative=("false_negative", lambda s: float(np.percentile(pd.to_numeric(s, errors="coerce").dropna(), 75))),
            min_false_negative=("false_negative", "min"),
            max_false_negative=("false_negative", "max"),
            median_false_negative_rate=("false_negative_rate", "median"),
            median_test_positive=("n_test_positive", "median"),
            median_targets=("n_targets", "median"),
        )
        .reset_index()
    )
    grouped["method_order"] = grouped["method"].map(_method_sort_key)
    grouped = grouped.sort_values(["method_order"]).drop(columns=["method_order"])
    grouped.to_csv(output_dir / "binary_false_negative_summary_v4.csv", index=False)

    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            [
                row["method"],
                _format_plot_value(float(row["median_false_negative"]), 1),
                _format_plot_value(float(row["q1_false_negative"]), 1),
                _format_plot_value(float(row["q3_false_negative"]), 1),
                int(row["min_false_negative"]),
                int(row["max_false_negative"]),
                _format_plot_value(float(row["median_false_negative_rate"]), 3),
                int(round(float(row["median_test_positive"]))),
                int(round(float(row["median_targets"]))),
            ]
        )
    _save_tex_table(
        output_dir / "binary_false_negative_summary_v4.tex",
        rows,
        [
            "Method",
            "Median total FN",
            "Q1 total FN",
            "Q3 total FN",
            "Min total FN",
            "Max total FN",
            "Median FN rate",
            "Median total test 1",
            "Targets pooled",
        ],
        (
            f"{cohort_name} aggregate positive-case misses by imputation method. "
            "FN denotes positive test cases predicted as 0 after summing across binary targets "
            "within each missingness-intensity and split run."
        ),
        _caption_label(cohort_name, "binary", "false-negative-summary", version="v4"),
    )

def _binary_class_count_caption_text(binary: pd.DataFrame) -> str:
    ok = binary[binary["status"] == "ok"].copy()
    required = {
        "target",
        "n_train_negative_before_downsampling",
        "n_train_positive_before_downsampling",
        "n_train_negative",
        "n_train_positive",
        "n_test_negative",
        "n_test_positive",
    }
    if ok.empty or not required.issubset(ok.columns):
        return ""

    pieces = []
    counts = (
        ok[
            [
                "target",
                "n_train_negative_before_downsampling",
                "n_train_positive_before_downsampling",
                "n_train_negative",
                "n_train_positive",
                "n_test_negative",
                "n_test_positive",
            ]
        ]
        .drop_duplicates()
        .sort_values("target")
    )
    for target, target_df in counts.groupby("target", observed=False):
        label = TARGET_LABELS.get(str(target), str(target))
        unique = target_df.drop(columns=["target"]).drop_duplicates()
        if len(unique) == 1:
            row = unique.iloc[0]
            pieces.append(
                f"{label}: train before downsampling 0={int(row['n_train_negative_before_downsampling'])}, "
                f"1={int(row['n_train_positive_before_downsampling'])}; train used 0={int(row['n_train_negative'])}, "
                f"1={int(row['n_train_positive'])}; test 0={int(row['n_test_negative'])}, "
                f"1={int(row['n_test_positive'])}"
            )
        else:
            pieces.append(f"{label}: exact per-split class counts are listed in binary_class_counts_v4.tex")
    return "; ".join(pieces) + "."

def plot_binary_false_negative_v4(binary: pd.DataFrame, output_dir: Path, cohort_name: str) -> None:
    ok = binary[binary["status"] == "ok"].copy()
    if ok.empty or "false_negative" not in ok.columns:
        return

    aggregated = _aggregate_binary_false_negatives(ok)
    if aggregated.empty:
        return

    font_scale = _compact_font_scale("v4")
    methods, values = _ordered_method_values(aggregated, "false_negative")
    if not values:
        return

    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_title("Aggregated Across Binary Targets", fontsize=12 * font_scale)

    positions = np.arange(1, len(methods) + 1)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanline=True,
        medianprops={"color": "#111827", "linewidth": 1.5},
        meanprops={"color": "#333333", "linewidth": 1.1, "linestyle": ":"},
        boxprops={"edgecolor": "#333333", "linewidth": 1.0},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#D8E2DC"))
        patch.set_alpha(0.82)

    baseline = pd.to_numeric(aggregated.loc[aggregated["method"] == "Baseline", "false_negative"], errors="coerce").dropna()
    reference = float(baseline.median()) if not baseline.empty else float("nan")
    star_ys: list[float] = []
    if np.isfinite(reference):
        ax.axhline(reference, color="#7A7A7A", linestyle="--", linewidth=1.3, label="Baseline median", zorder=1)
    for pos, method, vals in zip(positions, methods, values):
        finite = vals[np.isfinite(vals)]
        if method == "Baseline" or finite.size == 0 or not np.isfinite(reference):
            continue
        median = float(np.median(finite))
        if median < reference:
            q1 = float(np.percentile(finite, 25))
            star_ys.append(q1)
            ax.scatter(
                pos,
                q1,
                marker="*",
                s=COMPACT_STAR_MARKER_SIZE,
                color="#111827",
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
    if star_ys:
        ax.scatter([], [], marker="*", s=COMPACT_LEGEND_STAR_MARKER_SIZE, color="#111827", label="Fewer missed positives than baseline median")
    ax.legend(loc="best", fontsize=8 * font_scale, frameon=True)
    ax.set_ylabel("Positive test cases predicted as 0", fontsize=12 * font_scale)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.tick_params(axis="both", labelsize=10 * font_scale)
    all_vals = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
    if all_vals.size:
        y_min = min(float(all_vals.min()), reference if np.isfinite(reference) else float(all_vals.min()))
        y_max = max(float(all_vals.max()), reference if np.isfinite(reference) else float(all_vals.max()))
        y_span = max(y_max - y_min, 1.0)
        ax.set_ylim(max(0.0, y_min - y_span * 0.08), y_max + y_span * 0.12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "binary_false_negative_v4.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "binary_false_negative_by_target_v4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_binary_class_count_tables(ok, output_dir, cohort_name)
    _write_binary_false_negative_summary(ok, output_dir, cohort_name)
    class_count_text = _binary_class_count_caption_text(ok)
    total_positive = pd.to_numeric(aggregated["n_test_positive"], errors="coerce").dropna()
    total_positive_text = ""
    if not total_positive.empty:
        total_positive_text = (
            f" The aggregate denominator is {int(round(float(total_positive.median())))} positive test cases "
            "per method-intensity-split run after pooling the binary targets."
        )
    caption = (
        f"{cohort_name} aggregate false-negative counts for binary targets. Each plotted value is the sum of "
        "positive test cases predicted as 0 across all binary targets within one method, missingness intensity, "
        "and split. Each box pools 9 aggregated intensity-split results across 3 independently randomized "
        "train/validation/test splits. Lower values are clinically preferred because each false negative is a "
        f"missed positive case.{total_positive_text} {class_count_text} Exact per-method aggregate "
        "false-negative summaries are written to "
        "binary_false_negative_summary_v4.tex."
    )
    _write_compact_caption_section(
        output_dir,
        cohort_name,
        "binary_false_negative",
        "Positive Cases Predicted as 0",
        [
            CompactCaptionEntry(
                filename="binary_false_negative_v4.png",
                metric_title="False Negatives",
                caption=caption,
                label=_caption_label(cohort_name, "binary", "false-negative", version="v4"),
            )
        ],
        version="v4",
    )

def plot_continuous_compact_all_V2(
    continuous: pd.DataFrame,
    output_dir: Path,
    cohort_name: str,
    version: str = "v3",
) -> None:
    ok = continuous[continuous["status"] == "ok"].copy()
    if ok.empty:
        return

    def _pick_primary(row: pd.Series) -> float:
        mphe_value = pd.to_numeric(row.get("mphe"), errors="coerce")
        if pd.notna(mphe_value):
            return float(mphe_value)
        return pd.to_numeric(row.get("rmsle"), errors="coerce")

    ok["primary_metric"] = ok.apply(_pick_primary, axis=1)
    if ok["primary_metric"].isna().all():
        return

    ok = _add_target_baseline_ratio(ok, "primary_metric", "error_vs_baseline")
    font_scale = _compact_font_scale(version)

    # collect ordered methods and their error_vs_baseline values
    methods_present: list[str] = []
    vals_list: list[np.ndarray] = []
    for method in METHOD_ORDER:
        vals = pd.to_numeric(
            ok.loc[ok["method"] == method, "error_vs_baseline"], errors="coerce"
        ).dropna().to_numpy()
        if vals.size == 0:
            continue
        methods_present.append(method)
        vals_list.append(vals)

    if not methods_present:
        return

    # CV (%) = 100 * std / |mean| for each method's error_vs_baseline distribution
    cv_vals: list[float] = []
    for vals in vals_list:
        mean_ = float(np.mean(vals))
        std_ = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        cv_vals.append(100.0 * std_ / abs(mean_) if abs(mean_) > 1e-12 else float("nan"))

    fig, ax_main = plt.subplots(figsize=(12, 6.5))
    _draw_compact_relative_boxplot(
        ax_main,
        ok,
        "error_vs_baseline",
        "MLPE / Baseline median (lower is better)",
        reference=1.0,
        lower_is_better=True,
        star_label="Lower error than baseline median",
        font_scale=font_scale,
    )
    ax_main.set_title(
        "Error (MLPE for all continuous targets)",
        fontsize=10 * font_scale,
    )

    fig.tight_layout()
    fname = f"continuous_compact_all_error_V2_{version}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_continuous_compact_summary_v3(
    continuous: pd.DataFrame,
    output_dir: Path,
    cohort_name: str,
    version: str = "v3",
) -> None:
    ok = continuous[continuous["status"] == "ok"].copy()
    if ok.empty:
        return
    font_scale = _compact_font_scale(version)
    caption_entries: list[CompactCaptionEntry] = []

    # RMSLE group
    rmsle_ok = ok[ok["target"].isin(RMSLE_TARGETS)].copy()
    if not rmsle_ok.empty and "rmsle" in rmsle_ok.columns and rmsle_ok["rmsle"].notna().any():
        rmsle_ok = _add_target_baseline_ratio(rmsle_ok, "rmsle", "rmsle_vs_target_baseline")
        has_r2_rmsle = "r2" in rmsle_ok.columns and pd.to_numeric(rmsle_ok["r2"], errors="coerce").notna().any()
        if has_r2_rmsle:
            rmsle_ok = _add_target_baseline_delta(rmsle_ok, "r2", "r2_vs_target_baseline_rmsle")
            fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), sharey=False)
            rmsle_ax, r2_ax = axes
        else:
            fig, rmsle_ax = plt.subplots(figsize=(12, 6.5))
            r2_ax = None
        _draw_compact_relative_boxplot(
            rmsle_ax, rmsle_ok, "rmsle_vs_target_baseline",
            "RMSLE divided by target baseline median (lower is better)",
            reference=1.0, lower_is_better=True, star_label="Lower RMSLE than baseline median", font_scale=font_scale,
        )
        rmsle_ax.set_title("RMSLE", fontsize=12 * font_scale)
        if r2_ax is not None:
            _draw_compact_relative_boxplot(
                r2_ax, rmsle_ok, "r2_vs_target_baseline_rmsle",
                "R² minus target baseline median (higher is better)",
                reference=0.0, lower_is_better=False, star_label="Higher R² than baseline median", font_scale=font_scale,
            )
            r2_ax.set_title("R²", fontsize=12 * font_scale)
        fig.tight_layout()
        rmsle_fig_name = f"continuous_summary_compact_{version}_rmsle_V2.png"
        fig.savefig(output_dir / rmsle_fig_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        caption_entries.append(CompactCaptionEntry(
            filename=rmsle_fig_name,
            metric_title="RMSLE",
            caption=_compact_caption(
                rmsle_ok, "rmsle_vs_target_baseline", cohort_name, "continuous (RMSLE targets)",
                "RMSLE", "RMSLE divided by each target-specific baseline median",
                reference=1.0, lower_is_better=True, decimals=4,
            ),
            label=_caption_label(cohort_name, "continuous", "rmsle", version=version),
        ))

    # MLPE group
    mphe_ok = ok.copy()
    if "mphe" in mphe_ok.columns:
        mphe_ok["mphe"] = pd.to_numeric(mphe_ok["mphe"], errors="coerce")
        mphe_ok = mphe_ok[mphe_ok["mphe"].notna()].copy()
    else:
        mphe_ok = pd.DataFrame()
    if not mphe_ok.empty and "mphe" in mphe_ok.columns and mphe_ok["mphe"].notna().any():
        mphe_ok = _add_target_baseline_ratio(mphe_ok, "mphe", "mphe_vs_target_baseline")
        has_r2_mphe = "r2" in mphe_ok.columns and pd.to_numeric(mphe_ok["r2"], errors="coerce").notna().any()
        if has_r2_mphe:
            mphe_ok = _add_target_baseline_delta(mphe_ok, "r2", "r2_vs_target_baseline")
            fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), sharey=False)
            mphe_ax, r2_ax = axes
        else:
            fig, mphe_ax = plt.subplots(figsize=(12, 6.5))
            r2_ax = None
        _draw_compact_relative_boxplot(
            mphe_ax, mphe_ok, "mphe_vs_target_baseline",
            "MLPE divided by target baseline median (lower is better)",
            reference=1.0, lower_is_better=True, star_label="Lower MLPE than baseline median", font_scale=font_scale,
        )
        mphe_ax.set_title("MLPE (z-scored)", fontsize=12 * font_scale)
        if r2_ax is not None:
            _draw_compact_relative_boxplot(
                r2_ax, mphe_ok, "r2_vs_target_baseline",
                "R² minus target baseline median (higher is better)",
                reference=0.0, lower_is_better=False, star_label="Higher R² than baseline median", font_scale=font_scale,
            )
            r2_ax.set_title("R²", fontsize=12 * font_scale)
        fig.tight_layout()
        mphe_fig_name = f"continuous_summary_compact_{version}_mphe_V2.png"
        fig.savefig(output_dir / mphe_fig_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        caption_entries.append(CompactCaptionEntry(
            filename=mphe_fig_name,
            metric_title="MLPE",
            caption=_compact_caption(
                mphe_ok, "mphe_vs_target_baseline", cohort_name, "continuous (MLPE targets)",
                "MLPE", "MLPE divided by each target-specific baseline median",
                reference=1.0, lower_is_better=True, decimals=4,
            ),
            label=_caption_label(cohort_name, "continuous", "mphe", version=version),
        ))

    if caption_entries:
        _write_compact_caption_section(
            output_dir, cohort_name, "continuous",
            "Prediction Performance: Continuous Targets (V2)",
            caption_entries, version=version,
        )

def plot_continuous_compact_summary_v4(continuous: pd.DataFrame, output_dir: Path, cohort_name: str) -> None:
    plot_continuous_compact_summary_v3(continuous, output_dir, cohort_name, version="v4")

def render_cont_figures_v2_from_csv(cohorts: list[CohortConfig]) -> None:
    for config in cohorts:
        csv_path = config.output_dir / "continuous_prediction_results_V2.csv"
        if not csv_path.exists():
            print(f"Not found: {csv_path.relative_to(BASE_DIR)} — skipping {config.name}")
            continue
        continuous = pd.read_csv(csv_path)
        if continuous.empty:
            print(f"Empty CSV for {config.name} — skipping")
            continue
        config.output_dir.mkdir(parents=True, exist_ok=True)
        plot_continuous_compact_all_V2(continuous, config.output_dir, config.name)
        plot_continuous_all_summary_v2(continuous, config.output_dir, config.name)
        for target, target_df in continuous[continuous["status"] == "ok"].groupby("target"):
            plot_continuous_summary_v2(target_df, config.output_dir, config.name, str(target))
        print(f"Wrote V2 continuous figures under {config.output_dir.relative_to(BASE_DIR)}")

def render_allmphe_figures_from_csv(cohorts: list[CohortConfig]) -> None:
    cont_suffix = "_allmphe_V2"
    for config in cohorts:
        csv_path = config.output_dir / f"continuous_prediction_results{cont_suffix}.csv"
        if not csv_path.exists():
            print(f"Not found: {csv_path.relative_to(BASE_DIR)} — skipping {config.name}")
            continue
        continuous = pd.read_csv(csv_path)
        if continuous.empty:
            print(f"Empty CSV for {config.name} — skipping")
            continue
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_compact_continuous_csv(continuous, config.output_dir, config.name, cont_suffix=cont_suffix)
        plot_continuous_compact_all_V2(continuous, config.output_dir, config.name)
        print(f"Wrote all-MLPE compact figure and CSV under {config.output_dir.relative_to(BASE_DIR)}")


# ── companion CSV for the compact continuous figures ──────────────────────


def write_compact_continuous_csv(continuous: pd.DataFrame, output_dir: Path, cohort_name: str, cont_suffix: str = "_V2") -> None:
    ok = continuous[continuous["status"] == "ok"].copy() if "status" in continuous.columns else continuous.copy()
    if ok.empty:
        return
    ok["mphe"] = pd.to_numeric(ok.get("mphe", np.nan), errors="coerce")
    ok["r2"] = pd.to_numeric(ok.get("r2", np.nan), errors="coerce")
    rows = []
    for method in METHOD_ORDER:
        sub = ok[ok["method"] == method]
        if sub.empty:
            continue
        mphe_vals = sub["mphe"].dropna()
        r2_vals = sub["r2"].dropna()
        rows.append({
            "cohort": cohort_name,
            "method": method,
            "n_obs": len(sub),
            "median_mphe": float(mphe_vals.median()) if not mphe_vals.empty else float("nan"),
            "q1_mphe": float(mphe_vals.quantile(0.25)) if not mphe_vals.empty else float("nan"),
            "q3_mphe": float(mphe_vals.quantile(0.75)) if not mphe_vals.empty else float("nan"),
            "median_r2": float(r2_vals.median()) if not r2_vals.empty else float("nan"),
            "q1_r2": float(r2_vals.quantile(0.25)) if not r2_vals.empty else float("nan"),
            "q3_r2": float(r2_vals.quantile(0.75)) if not r2_vals.empty else float("nan"),
        })
    pd.DataFrame(rows).to_csv(output_dir / f"continuous_compact_summary{cont_suffix}.csv", index=False)
