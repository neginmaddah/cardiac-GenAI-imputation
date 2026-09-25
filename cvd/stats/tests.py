"""Friedman omnibus and pairwise Wilcoxon with Holm-Bonferroni correction.

Paper section: Methods, embedded at the end of "Post-Operative Adverse Outcome
Prediction".

This is the single implementation. Four existed before this refactor
(``prediction._significance_tests``, ``paper_prediction_review_metrics.
holm_bonferroni``, ``paper_fidelity_figures.run_overall_fidelity_friedman_
tests`` and ``bootstrap_fidelity_analysis.run_method_statistics``) and they did
not agree: one omitted Holm's running-maximum step and so could report
adjusted p-values that decreased as raw p-values increased.

Three properties are enforced rather than left to the caller, because each
corresponds to an error that reached a submitted draft:

1. **Complete blocks only.** Friedman and Wilcoxon both assume the same units
   across arms. Filtering to blocks observed for every arm is done here, so a
   caller cannot forget it and silently compare different sets of units.
2. **Block counts are reported.** ``n_blocks_total`` and ``n_complete_blocks``
   are always returned. A block count is the cheapest available tripwire: 54
   versus 180 was visible immediately once printed.
3. **Effect size travels with the p-value.** A tiny p-value is not a large
   effect; chi2=152.24, p=6.7e-29 sat on a best-to-worst spread of 0.42%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from cvd.conf import active

__all__ = [
    "holm_bonferroni",
    "paired_matrix",
    "friedman_test",
    "pairwise_wilcoxon",
    "compare_methods",
    "FriedmanResult",
]


# ── multiplicity correction ────────────────────────────────────────────────

def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Includes the running maximum, which makes the adjusted values monotone in
    the raw values, and clips at 1.0. NaNs pass through unchanged so a pair
    that could not be tested does not silently become significant.
    """
    values = np.asarray(p_values, dtype=float)
    finite = np.flatnonzero(~np.isnan(values))
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if finite.size == 0:
        return adjusted.tolist()

    order = finite[np.argsort(values[finite], kind="stable")]
    m = order.size
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * values[idx]
        running = max(running, scaled)          # <- the step that was missing
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


# ── block construction ─────────────────────────────────────────────────────

def paired_matrix(
    df: pd.DataFrame,
    *,
    block_keys: Sequence[str],
    arm_key: str,
    value_key: str,
    arms: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Pivot to a ``blocks x arms`` matrix of complete blocks.

    Returns ``(matrix, n_blocks_total)``. Rows with a missing value for any arm
    are dropped, which is what makes the paired tests valid.
    """
    missing = [c for c in (*block_keys, arm_key, value_key) if c not in df.columns]
    if missing:
        raise KeyError(f"missing column(s) for pairing: {missing}")

    wide = df.pivot_table(
        index=list(block_keys), columns=arm_key, values=value_key, aggfunc="mean",
    )
    if arms is not None:
        present = [a for a in arms if a in wide.columns]
        wide = wide.reindex(columns=present)
    n_total = len(wide)
    return wide.dropna(axis=0, how="any"), n_total


# ── omnibus ────────────────────────────────────────────────────────────────

@dataclass
class FriedmanResult:
    """Omnibus outcome plus everything needed to interpret it."""
    metric: str
    direction: str
    arms: list[str]
    n_blocks_total: int
    n_complete_blocks: int
    chi2: float = float("nan")
    p_value: float = float("nan")
    reject: bool = False
    status: str = "ok"
    mean_ranks: dict[str, float] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    best_median_arm: str | None = None
    best_mean_rank_arm: str | None = None
    spread_pct: float = float("nan")

    def as_row(self) -> dict:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "arms_tested": ";".join(self.arms),
            "n_arms": len(self.arms),
            "n_blocks_total": self.n_blocks_total,
            "n_complete_blocks": self.n_complete_blocks,
            "friedman_chi2": self.chi2,
            "friedman_p": self.p_value,
            "reject": self.reject,
            "best_median_arm": self.best_median_arm,
            "best_mean_rank_arm": self.best_mean_rank_arm,
            "spread_pct": self.spread_pct,
            "mean_ranks": ";".join(f"{k}={v:.4g}" for k, v in self.mean_ranks.items()),
            "medians": ";".join(f"{k}={v:.6g}" for k, v in self.medians.items()),
            "status": self.status,
        }


def _rank_frame(matrix: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Within-block ranks, 1 = best under ``direction``."""
    ascending = direction == "lower_is_better"
    return matrix.rank(axis=1, ascending=ascending)


def friedman_test(
    matrix: pd.DataFrame,
    *,
    metric: str,
    direction: str,
    n_blocks_total: int | None = None,
    alpha: float | None = None,
) -> FriedmanResult:
    """Friedman omnibus over a complete ``blocks x arms`` matrix."""
    alpha = active()["stats.alpha"] if alpha is None else alpha
    arms = list(matrix.columns)
    result = FriedmanResult(
        metric=metric,
        direction=direction,
        arms=arms,
        n_blocks_total=n_blocks_total if n_blocks_total is not None else len(matrix),
        n_complete_blocks=len(matrix),
    )
    if len(arms) < 3:
        result.status = f"skipped: needs >=3 arms, got {len(arms)}"
        return result
    if len(matrix) < 3:
        result.status = f"skipped: needs >=3 complete blocks, got {len(matrix)}"
        return result

    result.chi2, result.p_value = (
        float(v) for v in friedmanchisquare(*[matrix[a].to_numpy() for a in arms])
    )
    result.reject = bool(result.p_value < alpha)

    ranks = _rank_frame(matrix, direction)
    result.mean_ranks = {a: float(ranks[a].mean()) for a in arms}
    result.medians = {a: float(matrix[a].median()) for a in arms}
    better = min if direction == "lower_is_better" else max
    result.best_median_arm = better(result.medians, key=result.medians.get)
    result.best_mean_rank_arm = min(result.mean_ranks, key=result.mean_ranks.get)

    # Effect size, reported alongside the p-value by construction.
    values = np.array(list(result.medians.values()), dtype=float)
    reference = np.nanmedian(np.abs(values))
    if reference > 0:
        result.spread_pct = float((values.max() - values.min()) / reference * 100.0)
    return result


# ── pairwise ───────────────────────────────────────────────────────────────

def pairwise_wilcoxon(
    matrix: pd.DataFrame,
    *,
    direction: str,
    alpha: float | None = None,
    min_nonzero: int = 1,
) -> pd.DataFrame:
    """All-pairs Wilcoxon signed-rank with Holm-Bonferroni within the family."""
    alpha = active()["stats.alpha"] if alpha is None else alpha
    rows: list[dict] = []
    for a, b in combinations(matrix.columns, 2):
        xa, xb = matrix[a].to_numpy(float), matrix[b].to_numpy(float)
        diff = xa - xb
        n_nonzero = int(np.count_nonzero(diff))
        statistic, p_value = float("nan"), float("nan")
        if n_nonzero >= max(1, min_nonzero):
            try:
                statistic, p_value = (float(v) for v in wilcoxon(xa, xb))
            except ValueError:
                pass
        median_a, median_b = float(np.median(xa)), float(np.median(xb))
        # Direction must come from the paired differences the test is computed
        # on, not from the marginal medians: blocks differ in scale, so the
        # difference of medians can point the other way from the median of the
        # within-block differences.
        delta = float(np.median(diff)) if diff.size else float("nan")
        if direction == "lower_is_better":
            better, worse = (a, b) if delta < 0 else (b, a)
            n_favour_a = int(np.count_nonzero(diff < 0))
        else:
            better, worse = (a, b) if delta > 0 else (b, a)
            n_favour_a = int(np.count_nonzero(diff > 0))
        rows.append({
            "arm_a": a, "arm_b": b,
            "n_complete_blocks": len(matrix),
            "median_a": median_a, "median_b": median_b,
            "median_diff_a_minus_b": delta,
            "n_blocks_favouring_a": n_favour_a,
            "better_arm": better if delta != 0 else None,
            "worse_arm": worse if delta != 0 else None,
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "n_nonzero_differences": n_nonzero,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm_bonferroni(out["p_value"].tolist())
    out["reject_holm"] = out["p_holm"] < alpha
    out["alpha"] = alpha
    return out


# ── the one entry point callers should use ─────────────────────────────────

def compare_methods(
    df: pd.DataFrame,
    *,
    block_keys: Sequence[str],
    arm_key: str = "method",
    value_key: str,
    metric: str,
    direction: str,
    arms: Sequence[str] | None = None,
    alpha: float | None = None,
    expected_blocks: int | None = None,
) -> tuple[FriedmanResult, pd.DataFrame]:
    """Omnibus then, only if it rejects, the pairwise family.

    ``expected_blocks`` asserts the complete-block count. Pass it whenever the
    number is known from the design -- it is the tripwire that catches a
    silently-dropped arm or target.
    """
    matrix, n_total = paired_matrix(
        df, block_keys=block_keys, arm_key=arm_key, value_key=value_key, arms=arms,
    )
    if expected_blocks is not None and len(matrix) != expected_blocks:
        raise AssertionError(
            f"expected {expected_blocks} complete blocks for {metric!r}, got "
            f"{len(matrix)} (of {n_total} total). A changed arm or target list "
            f"is the usual cause; verify before reporting."
        )

    omnibus = friedman_test(
        matrix, metric=metric, direction=direction,
        n_blocks_total=n_total, alpha=alpha,
    )
    gate = active()["stats.pairwise.gate_on_omnibus"]
    if gate and not omnibus.reject:
        return omnibus, pd.DataFrame()
    return omnibus, pairwise_wilcoxon(matrix, direction=direction, alpha=alpha)
