"""Stage 00 -- harmonise the two raw registry exports into one comparable schema.

Paper section: Methods 4.1, "After cleaning and harmonization".

This module is the single producer of the analysis matrices. It replaces the
two per-cohort preprocessing notebooks used during development, whose output
was audited on 2026-09-10 and retired. The notebooks are not distributed.

.. important::

   **Never assign a column across frames by position.** The defect that forced
   this rewrite was a single line in the Cohort2 notebook::

       clean_df[col] = df_og[col]

   ``df_og`` had 3,876 rows and ``clean_df`` 3,875, because one record had
   already been dropped. pandas aligns on the integer index, so every row after
   the dropped one received the *next* patient's value -- 1,329 patients
   (34.3% of the cohort) across 11 columns, including sex. Every cross-frame
   operation in this module goes through :func:`_align_on_id`, which joins on
   ``recordid`` and asserts a one-to-one match.

The transformation order is fixed and load-bearing:

1. normalise column names, then apply per-cohort :func:`apply_column_aliases`
2. :func:`apply_missing_tokens` -- registry "not recorded" strings become NaN
3. :func:`apply_label_aliases` -- category domains are made identical
4. :func:`apply_structural_na` + :func:`apply_graft_ladder` -- blanks that mean
   "did not apply" become real levels, BEFORE any missingness is measured
5. :func:`apply_implausible` -- physiologically impossible values become NaN
6. :func:`select_columns` -- strict intersection, missingness filter last
7. :func:`build_codebook` + :func:`encode` -- one encoding for both cohorts

Steps 2-5 must precede step 6: v1 measured missingness *before* the structural
fill and so deleted whole column families whose blanks were never missing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CohortFrame",
    "Codebook",
    "normalise_column_names",
    "apply_column_aliases",
    "apply_missing_tokens",
    "apply_label_aliases",
    "apply_structural_na",
    "apply_graft_ladder",
    "apply_implausible",
    "missingness_table",
    "select_columns",
    "build_codebook",
    "encode",
]

NOT_APPLICABLE = "NotApplicable"

# Binary columns whose positive class is not "Yes". Anything not listed and not
# {No, Yes} falls back to sorted order, recorded in the codebook either way.
_POSITIVE_CLASS: dict[str, str] = {
    "gender": "Male",
    "mt30stat": "Dead",
    "infendty": "Active",
    "coprebldtim": "Acute",
}


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


@dataclass
class CohortFrame:
    """A cohort's frame plus the audit trail of what was done to it."""

    name: str
    df: pd.DataFrame
    id_column: str = "recordid"
    report: dict[str, Any] = field(default_factory=dict)

    def note(self, step: str, payload: Any) -> None:
        self.report.setdefault(step, {})
        if isinstance(payload, Mapping):
            self.report[step].update(payload)
        else:  # pragma: no cover - defensive
            self.report[step] = payload


@dataclass
class Codebook:
    """The label <-> code mapping, shared by both cohorts.

    ``kinds`` maps column -> one of ``binary``/``ordinal``/``nominal``/
    ``continuous``/``count``. ``levels`` maps column -> ordered label list; the
    code of a label is its index in that list.
    """

    kinds: dict[str, str]
    levels: dict[str, list[str]]

    def code_map(self, column: str) -> dict[str, int]:
        return {lab: i for i, lab in enumerate(self.levels[column])}

    def decode_map(self, column: str) -> dict[int, str]:
        return {i: lab for i, lab in enumerate(self.levels[column])}


# ---------------------------------------------------------------------------
# 0. alignment helper -- the rule that prevents the v1 Cohort2 defect
# ---------------------------------------------------------------------------


def _align_on_id(
    target: pd.DataFrame, source: pd.DataFrame, columns: Sequence[str], *, id_column: str
) -> pd.DataFrame:
    """Copy *columns* from *source* into *target*, joined on ``id_column``.

    This exists so that no caller is tempted to write ``target[c] = source[c]``.
    Raises when the join is not one-to-one.
    """
    if id_column not in target.columns or id_column not in source.columns:
        raise KeyError(f"{id_column!r} must be present in both frames")
    if not source[id_column].is_unique or not target[id_column].is_unique:
        raise ValueError(f"{id_column!r} is not unique; refusing to align")

    indexed = source.set_index(id_column)
    out = target.copy()
    for col in columns:
        if col not in indexed.columns:
            continue
        out[col] = out[id_column].map(indexed[col])
    return out


# ---------------------------------------------------------------------------
# 1. names
# ---------------------------------------------------------------------------


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, then collapse every run of non-alphanumeric characters to ``_``.

    Punctuation must go, not just whitespace. Cohort2's length-of-stay column is
    literally ``losas (days from admission to discharge)``; a whitespace-only
    rule leaves the parentheses in place, the alias below silently fails to
    match, and the column is dropped as cohort-specific. That happened once --
    hence :func:`apply_column_aliases` now raises on an unmatched alias.
    """

    def norm(name: str) -> str:
        n = str(name).strip().lower()
        n = re.sub(r"[^0-9a-z]+", "_", n)
        return re.sub(r"_+", "_", n).strip("_")

    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    dupes = [c for c in set(out.columns) if list(out.columns).count(c) > 1]
    if dupes:
        raise ValueError(f"column-name normalisation collided: {sorted(dupes)}")
    return out


def apply_column_aliases(
    df: pd.DataFrame, aliases: Mapping[str, str], *, cohort: str = "?"
) -> pd.DataFrame:
    """Rename this cohort's columns onto the canonical names.

    Every declared alias source must exist in this cohort. A silent no-op here
    means a variable quietly stops being shared and drops out of the strict
    intersection, which is exactly how ``losas`` went missing.
    """
    missing = [src for src in aliases if src not in df.columns]
    if missing:
        raise KeyError(
            f"[{cohort}] column_aliases names {len(missing)} column(s) that do not "
            f"exist after normalisation: {sorted(missing)}"
        )
    clashes = [dst for dst in aliases.values() if dst in df.columns]
    if clashes:
        raise ValueError(f"[{cohort}] alias target already exists: {sorted(clashes)}")
    if aliases:
        logger.info("[%s] renamed %d column(s)", cohort, len(aliases))
    return df.rename(columns=dict(aliases))


# ---------------------------------------------------------------------------
# 2-3. values
# ---------------------------------------------------------------------------


def apply_missing_tokens(
    df: pd.DataFrame, tokens: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace registry "not recorded" strings with NaN, everywhere."""
    token_set = {str(t) for t in tokens}
    out = df.copy()
    counts: dict[str, int] = {}
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        as_str = out[col].astype("string").str.strip()
        hit = as_str.isin(token_set)
        n = int(hit.sum())
        if n:
            out[col] = as_str.mask(hit, other=pd.NA)
            counts[col] = n
    return out, counts


def apply_label_aliases(
    df: pd.DataFrame, groups: Sequence[Mapping[str, Any]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the declarative label maps from ``conf/label_aliases.yaml``."""
    out = df.copy()
    counts: dict[str, int] = {}
    for group in groups:
        mapping = dict(group.get("map") or {})
        if not mapping:
            continue
        for col in group.get("columns", []):
            if col not in out.columns:
                continue
            as_str = out[col].astype("string").str.strip()
            hit = as_str.isin(set(mapping))
            n = int(hit.sum())
            if n:
                out[col] = as_str.replace(mapping)
                counts[col] = counts.get(col, 0) + n
    return out, counts


# ---------------------------------------------------------------------------
# 4. structural "not applicable"
# ---------------------------------------------------------------------------


def _condition_mask(
    df: pd.DataFrame, when: Mapping[str, Sequence[Any]], *, mode: str = "any"
) -> pd.Series:
    """Boolean mask for a ``{column: [values]}`` condition block."""
    parts: list[pd.Series] = []
    for col, values in when.items():
        if col not in df.columns:
            # A gate column that does not exist cannot license a fill.
            return pd.Series(False, index=df.index)
        parts.append(df[col].astype("string").isin({str(v) for v in values}))
    if not parts:
        return pd.Series(False, index=df.index)
    out = parts[0]
    for p in parts[1:]:
        out = (out & p) if mode == "all" else (out | p)
    return out.fillna(False)


def apply_structural_na(
    df: pd.DataFrame, rules: Sequence[Mapping[str, Any]]
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Fill blanks that exist because the question did not apply.

    Only cells that are currently null AND satisfy the rule's condition are
    written, so a recorded value is never overwritten.
    """
    out = df.copy()
    report: dict[str, dict[str, int]] = {}
    for rule in rules:
        name = str(rule.get("name", "?"))
        mask = _condition_mask(
            out, rule.get("when") or {}, mode=str(rule.get("when_mode", "any"))
        )
        if not bool(mask.any()):
            continue
        filled: dict[str, int] = {}

        for col in rule.get("categorical") or []:
            if col not in out.columns:
                continue
            target = mask & out[col].isna()
            n = int(target.sum())
            if not n:
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                # A column read as float64 with no observed values is an empty
                # column, not a numeric one -- pandas simply had nothing to
                # infer from. Casting is safe. A numeric column that DOES hold
                # values is a config error: it belongs in `numeric_zero`.
                if out[col].notna().any():
                    raise TypeError(
                        f"structural_na rule {name!r} lists {col!r} as categorical "
                        f"but it holds numeric values; move it to `numeric_zero`"
                    )
                out[col] = out[col].astype("object")
            out.loc[target, col] = str(rule.get("fill", NOT_APPLICABLE))
            filled[col] = n

        for col in rule.get("numeric_zero") or []:
            if col not in out.columns:
                continue
            target = mask & out[col].isna()
            n = int(target.sum())
            if n:
                out.loc[target, col] = 0
                filled[col] = n

        if filled:
            report[name] = filled
            logger.info(
                "[structural_na] %s: filled %d cell(s) across %d column(s)",
                name,
                sum(filled.values()),
                len(filled),
            )
    return out, report


def apply_graft_ladder(
    df: pd.DataFrame,
    *,
    families: Sequence[str],
    indices: Sequence[str],
    fill: str = NOT_APPLICABLE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill repeated-measure ladders past each patient's last recorded rung.

    For a family ``cabconduit01..06``, a patient with three grafts has values at
    01-03 and blanks at 04-06. Those trailing blanks mean "this patient did not
    have a fourth graft", not "the fourth graft's conduit was not recorded".

    The last recorded rung is derived **per row from the family itself** rather
    than from a separate count column. v1 used a count column that a later
    notebook cell dropped, which is why the fill silently stopped working.
    """
    out = df.copy()
    counts: dict[str, int] = {}
    for stem in families:
        cols = [f"{stem}{i}" for i in indices if f"{stem}{i}" in out.columns]
        if len(cols) < 2:
            continue
        present = out[cols].notna()
        # position (1-based) of the last non-null rung; 0 when the row is empty
        last = present.values * np.arange(1, len(cols) + 1)
        last_rung = last.max(axis=1)
        for pos, col in enumerate(cols, start=1):
            target = (pos > last_rung) & out[col].isna().values
            n = int(target.sum())
            if n:
                out.loc[target, col] = fill
                counts[col] = n
    if counts:
        logger.info(
            "[graft_ladder] filled %d cell(s) across %d column(s)",
            sum(counts.values()),
            len(counts),
        )
    return out, counts


def apply_implausible(
    df: pd.DataFrame,
    bounds: Mapping[str, Mapping[str, float]],
    sentinels: Mapping[str, Sequence[float]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Blank out sentinel codes and physiologically impossible values.

    Set to NaN rather than clipped: a clipped value is an invented measurement,
    whereas NaN is honest and the imputation pipeline is built to handle it.
    """
    out = df.copy()
    counts: dict[str, int] = {}
    for col, values in sentinels.items():
        if col not in out.columns:
            continue
        hit = out[col].isin(list(values))
        n = int(hit.sum())
        if n:
            out.loc[hit, col] = np.nan
            counts[f"{col}:sentinel"] = n
    for col, spec in bounds.items():
        if col not in out.columns:
            continue
        num = pd.to_numeric(out[col], errors="coerce")
        hit = (num < float(spec["min"])) | (num > float(spec["max"]))
        hit = hit.fillna(False)
        n = int(hit.sum())
        if n:
            out.loc[hit, col] = np.nan
            counts[f"{col}:out_of_range"] = n
    return out, counts


# ---------------------------------------------------------------------------
# 5-6. selection
# ---------------------------------------------------------------------------


def missingness_table(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-column missing fraction in each cohort, for the Check-stop-1a review."""
    shared = sorted(set.intersection(*(set(f.columns) for f in frames.values())))
    rows = []
    for col in shared:
        row: dict[str, Any] = {"column": col}
        for name, f in frames.items():
            row[f"miss_{name}"] = float(f[col].isna().mean())
        row["miss_max"] = max(v for k, v in row.items() if k.startswith("miss_"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("miss_max").reset_index(drop=True)


def select_columns(
    frames: Mapping[str, pd.DataFrame],
    *,
    n_predictors: int | None = None,
    max_missing_fraction: float | None = None,
    always_keep: Sequence[str] = (),
    always_drop: Sequence[str] = (),
    held_out: Sequence[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Strict intersection: retained in BOTH cohorts, filtered jointly.

    v1 applied the filter per cohort and *before* the structural fill, leaving
    two matrices 192 and 207 columns wide with only 180 in common. Filtering
    jointly, after the fill, is what makes the two schemas identical.

    Either ``n_predictors`` (keep the N least-missing) or
    ``max_missing_fraction`` (keep everything under a cutoff) must be given.
    ``n_predictors`` is preferred: the count is the quantity that was decided at
    Check stop 1a, and the equivalent cutoff is merely a consequence of it.

    ``held_out`` names columns that are written to a separate targets file and
    must not consume predictor slots.

    Returns ``(columns, info)`` where ``info`` records the realised cutoff.
    """
    if (n_predictors is None) == (max_missing_fraction is None):
        raise ValueError("pass exactly one of n_predictors or max_missing_fraction")

    drop = set(always_drop)
    keep = list(always_keep)
    reserved = set(keep) | set(held_out) | drop
    shared = sorted(set.intersection(*(set(f.columns) for f in frames.values())))

    scored = sorted(
        (
            (max(float(f[col].isna().mean()) for f in frames.values()), col)
            for col in shared
            if col not in reserved
        )
    )

    if n_predictors is not None:
        if n_predictors > len(scored):
            raise ValueError(
                f"asked for {n_predictors} predictors but only {len(scored)} shared "
                f"columns are available"
            )
        chosen = scored[:n_predictors]
        cutoff = chosen[-1][0] if chosen else 0.0
        # Refuse to split a tie: including one column and excluding an equally
        # observed one would make the schema depend on sort order.
        tied = [c for m, c in scored[n_predictors:] if m == cutoff]
        if tied:
            raise ValueError(
                f"cutoff {cutoff:.4f} ties with {len(tied)} excluded column(s) "
                f"{tied[:5]}; choose a different n_predictors"
            )
    else:
        chosen = [(m, c) for m, c in scored if m <= float(max_missing_fraction)]
        cutoff = float(max_missing_fraction)

    selected = [c for _, c in chosen]
    info = {
        "n_predictors": len(selected),
        "realised_cutoff": cutoff,
        "n_shared_candidates": len(scored),
        "n_held_out": len([c for c in held_out if c in shared]),
        "excluded_next": [
            {"column": c, "miss_max": m} for m, c in scored[len(selected) : len(selected) + 5]
        ],
    }
    return keep + sorted(selected), info


# ---------------------------------------------------------------------------
# 7. encoding
# ---------------------------------------------------------------------------


def _observed_levels(frames: Mapping[str, pd.DataFrame], column: str) -> list[str]:
    """Union of non-null labels across cohorts -- the shared category domain."""
    seen: set[str] = set()
    for f in frames.values():
        if column in f.columns:
            seen |= {str(v) for v in f[column].dropna().unique()}
    return sorted(seen)


def build_codebook(
    frames: Mapping[str, pd.DataFrame],
    columns: Sequence[str],
    types_cfg: Mapping[str, Any],
    *,
    id_column: str = "recordid",
) -> Codebook:
    """Decide every column's type and its label order, once, for both cohorts.

    Ordinal orders come from ``conf/variable_types.yaml`` and are never sorted.
    Nominal orders *are* sorted -- there is no order to destroy, and sorting is
    what makes the two cohorts agree on the code of a label.
    """
    declared_ordinal: dict[str, list[str]] = {
        k: [str(v) for v in vs] for k, vs in (types_cfg.get("ordinal") or {}).items()
    }
    declared_nominal = set(types_cfg.get("nominal") or [])
    declared_count = set(types_cfg.get("count") or [])
    rules = types_cfg.get("rules") or {}
    max_auto = int(rules.get("auto_nominal_max_levels", 12))

    kinds: dict[str, str] = {}
    levels: dict[str, list[str]] = {}
    problems: list[str] = []

    for col in columns:
        if col == id_column:
            continue
        numeric = all(
            pd.api.types.is_numeric_dtype(f[col]) for f in frames.values() if col in f.columns
        )
        if col in declared_count:
            kinds[col] = "count"
            continue
        if numeric and col not in declared_ordinal:
            kinds[col] = "continuous"
            continue

        observed = _observed_levels(frames, col)

        if col in declared_ordinal:
            order = list(declared_ordinal[col])
            unexpected = [lab for lab in observed if lab not in order and lab != NOT_APPLICABLE]
            if unexpected:
                problems.append(
                    f"{col}: observed level(s) {unexpected} are not in the declared "
                    f"ordinal order {order}"
                )
            # NotApplicable is off the clinical scale; it sits below it.
            if NOT_APPLICABLE in observed:
                order = [NOT_APPLICABLE] + order
            kinds[col] = "ordinal"
            levels[col] = [lab for lab in order if lab in set(observed)]
            continue

        if len(observed) == 2:
            kinds[col] = "binary"
            levels[col] = _binary_order(col, observed)
            continue

        if col in declared_nominal or len(observed) <= max_auto:
            kinds[col] = "nominal"
            levels[col] = observed
            continue

        problems.append(
            f"{col}: {len(observed)} levels exceeds auto_nominal_max_levels="
            f"{max_auto} and it is not declared in conf/variable_types.yaml"
        )

    if problems:
        raise ValueError(
            "variable taxonomy is incomplete:\n  - " + "\n  - ".join(problems)
        )
    return Codebook(kinds=kinds, levels=levels)


def _binary_order(column: str, observed: Sequence[str]) -> list[str]:
    """Return ``[negative, positive]`` so the positive class encodes to 1."""
    obs = list(observed)
    positive = _POSITIVE_CLASS.get(column)
    if positive is None and "Yes" in obs:
        positive = "Yes"
    if positive is not None and positive in obs:
        negative = [lab for lab in obs if lab != positive]
        return [negative[0], positive]
    return sorted(obs)


def encode(df: pd.DataFrame, codebook: Codebook, *, id_column: str = "recordid") -> pd.DataFrame:
    """Apply the shared codebook. Unknown labels raise rather than become NaN."""
    out = pd.DataFrame(index=df.index)
    if id_column in df.columns:
        out[id_column] = df[id_column]
    for col, kind in codebook.kinds.items():
        if col not in df.columns:
            continue
        if kind in {"continuous", "count"}:
            out[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        mapping = codebook.code_map(col)
        as_str = df[col].astype("string")
        unknown = sorted(set(as_str.dropna().unique()) - set(mapping))
        if unknown:
            raise ValueError(f"{col}: label(s) absent from the codebook: {unknown}")
        out[col] = as_str.map(mapping).astype("Float64")
    return out
