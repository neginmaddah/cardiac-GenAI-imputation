"""Taxonomy-aware encoding of a masked unit into what each imputer family wants.

Paper section: Methods 4.3, Imputation of Masked Data.

Replaces the ad-hoc encoding that lived inside ``cvd.imputation.registry`` and
``cvd.prep.transforms``. Three differences, each fixing a defect rather than
restyling working code:

**The level set comes from the codebook, not from the data.**
``transforms.one_hot_with_nans`` calls ``pd.get_dummies`` on the observed
values, so the width of the encoded matrix depends on which levels survived
masking. Cohort1 and Cohort2 handed the generative imputers matrices of different
widths, and so did 5% and 20% of the same cohort -- which quietly undermines
the claim that Cohort2 is external validation of an identically parameterised
model. Here every block is sized by ``codebook["levels"]``, so the width is a
property of the schema.

**Ordinal is a real type.** v1 had only binary / multiclass / continuous, so
ordered scales (``status``: Elective < Urgent < Emergent < Emergent Salvage)
were one-hot'ed and their order thrown away -- while ``snap_multiclass``
snapped predictions by numeric distance, which is only defensible if the codes
*are* ordered. The two halves of the pipeline disagreed. Ordinals now stay
ordered end to end.

**Everything the generative models see is in [0, 1].** v1 mixed quantile-
transformed continuous columns in [0, 1] with raw integer codes up to 14 in one
tensor, so a single nominal column could dominate a reconstruction loss.

The encoding table itself is declarative, in ``conf/inputs.yaml``; this module
only implements the primitives it names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

__all__ = [
    "CanonicalScaler",
    "GenaiEncoding",
    "fit_canonical",
    "apply_canonical",
    "invert_canonical",
    "fit_genai",
    "apply_genai",
    "invert_genai",
]


# ── canonical space ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CanonicalScaler:
    """z-score constants for the numeric columns, fitted on ONE masked matrix.

    Fitted on the masked matrix and nothing else: a lower-intensity unit must
    not be able to observe the statistics of the higher-intensity unit built on
    top of it (the re-run plan).
    """

    columns: tuple[str, ...]
    means: Mapping[str, float]
    scales: Mapping[str, float]
    degenerate: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "means": {k: float(v) for k, v in self.means.items()},
            "scales": {k: float(v) for k, v in self.scales.items()},
            "degenerate": list(self.degenerate),
        }

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "CanonicalScaler":
        return cls(
            columns=tuple(blob["columns"]),
            means=dict(blob["means"]),
            scales=dict(blob["scales"]),
            degenerate=tuple(blob.get("degenerate", ())),
        )


def fit_canonical(
    masked: pd.DataFrame,
    kinds: Mapping[str, str],
    *,
    standardize_kinds: Sequence[str],
    ddof: int = 0,
    min_scale: float = 1e-9,
) -> CanonicalScaler:
    targets = [c for c in masked.columns if kinds.get(c) in set(standardize_kinds)]
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    degenerate: list[str] = []
    for col in targets:
        observed = pd.to_numeric(masked[col], errors="coerce").dropna()
        mu = float(observed.mean()) if len(observed) else 0.0
        sd = float(observed.std(ddof=ddof)) if len(observed) else 0.0
        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(sd) or sd < min_scale:
            # A constant column would divide by zero and emit inf, which then
            # propagates silently through every downstream metric.
            sd = 1.0
            degenerate.append(col)
        means[col] = mu
        scales[col] = sd
    return CanonicalScaler(
        columns=tuple(targets), means=means, scales=scales,
        degenerate=tuple(degenerate),
    )


def apply_canonical(df: pd.DataFrame, scaler: CanonicalScaler) -> pd.DataFrame:
    out = df.copy()
    for col in scaler.columns:
        if col not in out.columns:
            raise KeyError(f"canonical scaler expects column {col!r}, absent from frame")
        values = pd.to_numeric(out[col], errors="coerce").astype("float64")
        out[col] = (values - scaler.means[col]) / scaler.scales[col]
    return out


def invert_canonical(df: pd.DataFrame, scaler: CanonicalScaler) -> pd.DataFrame:
    out = df.copy()
    for col in scaler.columns:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce").astype("float64")
        out[col] = values * scaler.scales[col] + scaler.means[col]
    return out


# ── generative-model space ──────────────────────────────────────────────────

@dataclass
class GenaiEncoding:
    """How one canonical matrix maps onto the tensor a neural imputer sees.

    ``columns`` is the encoded column order and is deliberately derived from
    the canonical column order, with each nominal column replaced in place by
    its dummy block. Two cohorts with the same schema therefore produce the
    same order without any sorting step to get wrong.
    """

    source_columns: tuple[str, ...]
    columns: tuple[str, ...]
    dummy_map: dict[str, list[str]] = field(default_factory=dict)
    ordinal_k: dict[str, int] = field(default_factory=dict)
    binary_columns: tuple[str, ...] = ()
    qt_columns: tuple[str, ...] = ()
    qt: QuantileTransformer | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "source_columns": list(self.source_columns),
            "columns": list(self.columns),
            "dummy_map": {k: list(v) for k, v in self.dummy_map.items()},
            "ordinal_k": dict(self.ordinal_k),
            "binary_columns": list(self.binary_columns),
            "qt_columns": list(self.qt_columns),
        }


def _dummy_names(template: str, column: str, k: int) -> list[str]:
    return [template.format(column=column, index=i) for i in range(k)]


def fit_genai(
    canonical_masked: pd.DataFrame,
    kinds: Mapping[str, str],
    levels: Mapping[str, Sequence[Any]],
    *,
    feature_columns: Sequence[str],
    one_hot_template: str = "{column}__{index}",
    quantile_kwargs: Mapping[str, Any] | None = None,
) -> GenaiEncoding:
    """Fit the generative-family encoding on ONE canonical masked matrix."""
    dummy_map: dict[str, list[str]] = {}
    ordinal_k: dict[str, int] = {}
    binary_columns: list[str] = []
    qt_columns: list[str] = []
    encoded_order: list[str] = []

    for col in feature_columns:
        kind = kinds.get(col)
        if kind is None:
            raise KeyError(f"column {col!r} has no declared kind in the codebook")
        if kind == "nominal":
            k = len(levels[col])
            if k < 2:
                raise ValueError(f"nominal column {col!r} declares {k} level(s)")
            names = _dummy_names(one_hot_template, col, k)
            dummy_map[col] = names
            encoded_order.extend(names)
        elif kind == "ordinal":
            k = len(levels[col])
            if k < 2:
                raise ValueError(f"ordinal column {col!r} declares {k} level(s)")
            ordinal_k[col] = k
            encoded_order.append(col)
        elif kind == "binary":
            binary_columns.append(col)
            encoded_order.append(col)
        elif kind in ("continuous", "count"):
            qt_columns.append(col)
            encoded_order.append(col)
        else:
            raise ValueError(f"column {col!r} has unhandled kind {kind!r}")

    qt = None
    if qt_columns:
        kwargs = dict(quantile_kwargs or {})
        n_obs = int(canonical_masked[qt_columns].notna().sum().min())
        kwargs["n_quantiles"] = max(2, min(int(kwargs.get("n_quantiles", 1000)), n_obs))
        qt = QuantileTransformer(**kwargs)
        qt.fit(canonical_masked[qt_columns].astype("float64"))

    return GenaiEncoding(
        source_columns=tuple(feature_columns),
        columns=tuple(encoded_order),
        dummy_map=dummy_map,
        ordinal_k=ordinal_k,
        binary_columns=tuple(binary_columns),
        qt_columns=tuple(qt_columns),
        qt=qt,
    )


def apply_genai(canonical: pd.DataFrame, enc: GenaiEncoding) -> pd.DataFrame:
    """Encode a canonical frame into the generative models' [0, 1] space."""
    pieces: dict[str, np.ndarray] = {}

    for col, names in enc.dummy_map.items():
        codes = pd.to_numeric(canonical[col], errors="coerce").to_numpy("float64")
        missing = np.isnan(codes)
        # A code outside the declared range means the codebook and the matrix
        # have drifted apart; that must not be silently one-hot'ed to all zeros.
        present = codes[~missing]
        if present.size and (present.min() < 0 or present.max() > len(names) - 1):
            raise ValueError(
                f"{col}: codes span [{present.min()}, {present.max()}] but the "
                f"codebook declares {len(names)} levels"
            )
        block = np.zeros((len(codes), len(names)), dtype="float64")
        rows = np.flatnonzero(~missing)
        block[rows, codes[rows].astype(int)] = 1.0
        block[missing, :] = np.nan
        for j, name in enumerate(names):
            pieces[name] = block[:, j]

    for col, k in enc.ordinal_k.items():
        codes = pd.to_numeric(canonical[col], errors="coerce").to_numpy("float64")
        pieces[col] = codes / float(k - 1)

    for col in enc.binary_columns:
        pieces[col] = pd.to_numeric(canonical[col], errors="coerce").to_numpy("float64")

    if enc.qt_columns:
        block = enc.qt.transform(canonical[list(enc.qt_columns)].astype("float64"))
        for j, col in enumerate(enc.qt_columns):
            pieces[col] = block[:, j]

    return pd.DataFrame(
        {name: pieces[name] for name in enc.columns}, index=canonical.index,
    )


def invert_genai(encoded: pd.DataFrame, enc: GenaiEncoding) -> pd.DataFrame:
    """Map a generative model's output back into the canonical space.

    The inverse of :func:`apply_genai` on valid inputs, and a projection to the
    nearest valid value otherwise -- an imputer returns a real number where a
    one-hot block or an ordinal code belongs, and the argmax / rounding here is
    what turns it back into a level.
    """
    out: dict[str, np.ndarray] = {}

    for col, names in enc.dummy_map.items():
        block = encoded[names].to_numpy("float64")
        all_missing = np.isnan(block).all(axis=1)
        filled = np.where(np.isnan(block), -np.inf, block)
        codes = filled.argmax(axis=1).astype("float64")
        codes[all_missing] = np.nan
        out[col] = codes

    for col, k in enc.ordinal_k.items():
        values = encoded[col].to_numpy("float64") * float(k - 1)
        out[col] = np.clip(np.round(values), 0, k - 1)

    for col in enc.binary_columns:
        out[col] = encoded[col].to_numpy("float64")

    if enc.qt_columns:
        block = enc.qt.inverse_transform(encoded[list(enc.qt_columns)].astype("float64"))
        for j, col in enumerate(enc.qt_columns):
            out[col] = block[:, j]

    return pd.DataFrame(
        {col: out[col] for col in enc.source_columns}, index=encoded.index,
    )
