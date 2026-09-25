"""Stage 02 (v2) -- nested, type-aware missingness injection.

Paper section: Methods 4.2, Missing-Data Simulation.

This replaces :mod:`cvd.missingness.designs` for the v2 re-run. The v1 module is
left untouched so the frozen v1 artifacts stay reproducible.

Three defects in v1 are fixed here.

**Nesting** (the re-run plan). v1 called ``default_rng(seed)`` afresh for
every intensity and drew independently, so the 5% held-out set was *not* a
subset of the 10% set -- measured on the frozen files, 11,153 of 12,199 Cohort1
cells at 5% were absent from the 10% mask. Intensities were therefore three
unrelated experiments rather than a dose-response curve.

Here every column is sampled **once**, in a random order that respects the
design's weights, and each intensity takes a *prefix* of that order. Nesting is
then true by construction, not by luck.

The order comes from the Gumbel top-k trick (equivalently Efraimidis-Spirakis
A-Res): draw ``key_i = log(w_i) + Gumbel_i`` and sort descending. The top *k*
by key is distributed exactly as a weight-proportional sample without
replacement, *for every k simultaneously* -- which is precisely the property
prefix-nesting needs. Taking a prefix of ``rng.choice(..., replace=False, p=w)``
would not be safe: numpy draws that in batches with rejection, so its output
order is not a successive-sampling order.

**Type-aware MNAR** (the re-run plan). v1's MNAR branched on
``pd.api.types.is_numeric_dtype``. Every categorical in these matrices is stored
as an integer level code, so *every* variable took the continuous
tail-quantile path and the rare-category branch was dead code. For a 0/1 binary
the 30th and 70th percentiles often coincide, making the weights uniform -- MNAR
silently degenerated to MCAR on two of the three variable types. Here the branch
is on the **declared** taxonomy from ``conf/variable_types.yaml``.

**Per-unit seeds.** v1 used one global ``seed: 2025`` for every unit, so at a
given intensity the three designs started from the same RNG state. Seeds are now
derived from ``(cohort, design)``.

Unchanged from v1, deliberately: the per-column budget
``n_mask = floor(N * alpha * m0(col))``, which makes injected missingness
proportional to each column's *existing* missingness, and the restriction of
eligible cells to those actually observed. A column with no registry
missingness receives no injected missingness.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "STATUS_OBSERVED",
    "STATUS_REGISTRY_MISSING",
    "STATUS_HELD_OUT",
    "NestedMasks",
    "unit_seed",
    "inject_nested",
]

STATUS_OBSERVED = 0
STATUS_REGISTRY_MISSING = 1
STATUS_HELD_OUT = 2

_CATEGORICAL_KINDS = {"binary", "ordinal", "nominal"}
_NUMERIC_KINDS = {"continuous", "count"}


@dataclass
class NestedMasks:
    """Masked frames and status frames, keyed by intensity."""

    masked: dict[float, pd.DataFrame]
    status: dict[float, pd.DataFrame]
    budget: dict[str, dict[float, int]]

    @property
    def alphas(self) -> list[float]:
        return sorted(self.masked)


def unit_seed(base_seed: int, cohort: str, design: str) -> int:
    """Derive a stable per-unit seed.

    Intensities deliberately share a seed -- they are prefixes of one draw.
    Cohort and design do not, so the MCAR/MAR/MNAR masks at a given intensity
    are independent rather than three views of the same RNG stream.
    """
    digest = hashlib.sha256(f"{base_seed}|{cohort}|{design}".encode()).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# per-design cell weights
# ---------------------------------------------------------------------------


def _mcar_weights(n: int) -> np.ndarray:
    """Every eligible cell equally likely."""
    return np.ones(n, dtype=float)


def _mar_weights(
    drivers: np.ndarray, rng: np.random.Generator, params: Mapping[str, Any]
) -> np.ndarray:
    """Missingness depends on OTHER, fully observed variables.

    ``drivers`` is already z-scored, shape ``(n_eligible, k)``. Each target
    column gets its **own** random coefficient vector
    ``beta ~ N(0, 1/sqrt(k))``, so different columns go missing for different
    reasons -- which is what "at random given the observed data" is supposed to
    mean.

    v1 instead took an unweighted sum of the first five columns of the frame and
    reused that single propensity direction for every one of 100+ targets, so
    the same patients were selected again and again, and the sigmoid of a
    sum-of-five z-scores saturated (‖z‖ frequently > 5, pinning probabilities at
    0 or 1).
    """
    k = drivers.shape[1]
    if k == 0:
        return np.ones(drivers.shape[0], dtype=float)
    beta = rng.normal(0.0, 1.0 / np.sqrt(k), size=k)
    logits = drivers @ beta
    scale = float(params.get("logit_scale", 1.0))
    logits = np.clip(logits * scale, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _rare_level_weights(values: pd.Series, weight: float) -> np.ndarray:
    """Up-weight the least-frequent half of the levels.

    Rank-based, not median-based: with two levels a median split leaves *both*
    "not rare" whenever their frequencies straddle the median symmetrically,
    which is how v1's binary MNAR collapsed to MCAR. The least-frequent half of
    a level set is always non-empty.
    """
    freq = values.value_counts(normalize=True)
    if len(freq) < 2:
        return np.ones(len(values), dtype=float)
    order = freq.sort_values()
    n_rare = max(1, len(order) // 2)
    rare = set(order.index[:n_rare])
    return np.where(values.isin(rare).to_numpy(dtype=bool), weight, 1.0)


def _mnar_weights(
    values: pd.Series, kind: str, params: Mapping[str, Any]
) -> np.ndarray:
    """Missingness depends on the cell's OWN value. Branches on declared type.

    .. important::

       Declared type alone is not sufficient. A *numeric* column with very few
       distinct values -- ``distvein`` takes 8 values, ``totalnodistanastartcond``
       6 -- has ``q30 == q70`` in practice, so "in either tail" is true for
       **100%** of its cells and every weight comes out equal. That is the exact
       shape of the v1 defect, just relocated from binary to count columns. Such
       columns are routed to the rare-level branch instead, and a final guard
       catches any remaining degenerate split.
    """
    tail_weight = float(params.get("tail_weight", 3.0))
    rare_weight = float(params.get("rare_category_weight", 3.0))
    min_distinct = int(params.get("min_distinct_for_quantile", 12))

    if kind in _CATEGORICAL_KINDS or values.nunique(dropna=True) < min_distinct:
        return _rare_level_weights(values, rare_weight)

    if kind in _NUMERIC_KINDS:
        tail_q = float(params.get("tail_quantile", 0.30))
        lo = values.quantile(tail_q)
        hi = values.quantile(1.0 - tail_q)
        in_tail = ((values <= lo) | (values >= hi)).to_numpy(dtype=bool)
        share = float(in_tail.mean())
        if share > 0.95 or share < 0.05:
            # Ties have collapsed the tails; fall back to frequency weighting so
            # the mechanism still depends on the value.
            return _rare_level_weights(values, rare_weight)
        return np.where(in_tail, tail_weight, 1.0)

    return np.ones(len(values), dtype=float)


def _ordered_draw(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return eligible-cell positions in weighted random order (Gumbel top-k).

    Sorting by ``log(w) + Gumbel`` yields an order whose every prefix is a
    weight-proportional sample without replacement. That is what makes the
    intensities nested.
    """
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-12)
    gumbel = -np.log(-np.log(rng.random(w.shape[0])))
    keys = np.log(w) + gumbel
    return np.argsort(-keys, kind="stable")


# ---------------------------------------------------------------------------
# driver matrix for MAR
# ---------------------------------------------------------------------------


def _driver_matrix(
    df: pd.DataFrame, exclude: set[str], params: Mapping[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """Z-scored matrix of the always-observed numeric columns."""
    min_observed = float(params.get("predictor_min_observed", 0.99))
    pool = [
        c
        for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().mean() >= min_observed
    ]
    max_drivers = int(params.get("max_drivers", 40))
    if len(pool) > max_drivers:
        # Deterministic: the most complete columns, ties broken by name.
        pool = sorted(pool, key=lambda c: (-df[c].notna().mean(), c))[:max_drivers]
        pool = sorted(pool)
    if not pool:
        return np.zeros((len(df), 0)), []
    sub = df[pool].astype(float)
    sub = sub.fillna(sub.mean())
    std = sub.std(ddof=0).replace(0, 1.0)
    return ((sub - sub.mean()) / std).to_numpy(dtype=float), pool


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def inject_nested(
    df: pd.DataFrame,
    *,
    design: str,
    alphas: Sequence[float],
    seed: int,
    kinds: Mapping[str, str],
    id_column: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> NestedMasks:
    """Produce nested masks for every intensity in one pass.

    Parameters
    ----------
    df
        The clean analysis matrix, already encoded.
    design
        ``mcar``, ``mar`` or ``mnar``.
    alphas
        Intensities, e.g. ``[0.05, 0.10, 0.20]``. Returned masks satisfy
        ``held_out(0.05) ⊂ held_out(0.10) ⊂ held_out(0.20)`` exactly.
    kinds
        Column -> declared taxonomy kind, from the harmonise codebook. MNAR
        branches on this rather than on storage dtype.
    """
    design = design.lower()
    if design not in {"mcar", "mar", "mnar"}:
        raise ValueError(f"unknown design {design!r}")
    params = dict(params or {})
    alphas = sorted(float(a) for a in alphas)
    rng = np.random.default_rng(seed)
    n_rows = len(df)

    base_status = pd.DataFrame(
        STATUS_OBSERVED, index=df.index, columns=df.columns, dtype="int8"
    )
    base_status[df.isna()] = STATUS_REGISTRY_MISSING

    masked = {a: df.copy() for a in alphas}
    status = {a: base_status.copy() for a in alphas}
    budget: dict[str, dict[float, int]] = {}

    m0 = df.isna().mean()
    skip = {c for c in df.columns if m0[c] <= 0}
    if id_column:
        skip.add(id_column)
    columns = [c for c in df.columns if c not in skip]

    drivers_all: np.ndarray | None = None
    driver_names: list[str] = []
    if design == "mar":
        drivers_all, driver_names = _driver_matrix(
            df, skip | {id_column or ""}, params
        )
        logger.info("[mar] %d always-observed driver column(s)", len(driver_names))

    for col in columns:
        eligible_mask = df[col].notna().to_numpy()
        eligible_pos = np.flatnonzero(eligible_mask)
        if eligible_pos.size == 0:
            continue

        counts = {a: int(n_rows * a * m0[col]) for a in alphas}
        n_max = max(counts.values())
        if n_max <= 0 or eligible_pos.size < n_max:
            continue

        values = df[col].iloc[eligible_pos]
        if design == "mcar":
            weights = _mcar_weights(eligible_pos.size)
        elif design == "mar":
            assert drivers_all is not None
            sub = drivers_all[eligible_pos]
            if col in driver_names:  # never let a column drive its own holdout
                j = driver_names.index(col)
                sub = np.delete(sub, j, axis=1)
            weights = _mar_weights(sub, rng, params)
        else:
            weights = _mnar_weights(values, str(kinds.get(col, "continuous")), params)

        order = _ordered_draw(weights, rng)
        ranked = eligible_pos[order]

        budget[col] = counts
        for a in alphas:
            k = counts[a]
            if k <= 0:
                continue
            chosen = ranked[:k]  # nested by construction
            idx = df.index[chosen]
            masked[a].loc[idx, col] = np.nan
            status[a].loc[idx, col] = STATUS_HELD_OUT

    for a in alphas:
        n = int((status[a] == STATUS_HELD_OUT).to_numpy().sum())
        logger.info("[%s] alpha=%.2f held out %d cell(s)", design, a, n)

    return NestedMasks(masked=masked, status=status, budget=budget)
