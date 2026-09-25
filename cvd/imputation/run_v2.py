"""Impute one v2 unit from its frozen, pre-encoded inputs.

Paper section: Methods 4.3, Imputation of Masked Data.

A parallel of :func:`cvd.imputation.registry.impute`, deliberately NOT a change
to it: ``results/`` is frozen and must keep regenerating byte-identically, and
registry.impute is what regenerates it.

The one structural difference is where the encoding happens. registry.impute
encodes on the fly, inside the call, so nothing about the matrix an imputer
actually saw could be inspected -- or even reconstructed afterwards. Here
``cvd.stages.inputs`` has already written that matrix to disk and Check stop 3
has already gated it, so this function only loads, runs and inverts.

Everything is returned in the **canonical space** (continuous and count
z-scored, everything else on its integer codebook code), which is the space
both families are scored in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from cvd.conf import Config, active
from cvd.prep import encode_v2
from cvd.prep.postprocess import snap_binary, snap_multiclass

from .ensemble import ensemble

logger = logging.getLogger(__name__)

__all__ = ["PreparedUnit", "impute_v2", "col_types_3way"]


@dataclass
class PreparedUnit:
    """One unit's frozen inputs, as written by :mod:`cvd.stages.inputs`."""

    tag: str
    cohort: str
    canonical_masked: pd.DataFrame     # includes the id column
    canonical_clean: pd.DataFrame      # includes the id column
    genai_masked: pd.DataFrame         # features only, every value in [0, 1]
    encoding: encode_v2.GenaiEncoding
    kinds: Mapping[str, str]
    id_column: str

    @property
    def features(self) -> list[str]:
        return [c for c in self.canonical_masked.columns if c != self.id_column]

    def of_kind(self, *wanted: str) -> list[str]:
        return [c for c in self.features if self.kinds.get(c) in wanted]


def classic_groups(unit: PreparedUnit) -> dict[str, list[str]]:
    """What each classic imputer should treat as what.

    NOT the same grouping as :func:`col_types_3way`, and the difference is the
    whole point of having a four-way taxonomy:

    * ``bridge.R`` turns everything in ``binary_cols`` or ``multiclass_cols``
      into an **unordered** R factor (``bridge.R:43``). Passing ordinals there
      throws away exactly the order that Phase 3 exists to preserve -- the
      generative-family defect, relocated. Ordinals go to R as numeric instead,
      where PMM draws from observed values and so can only ever return a valid
      level.
    * ``impute_mean_mode`` mode-fills whatever is in ``binary_cols`` and
      mean-fills everything else. v1 put nominal columns in ``multiclass``, so a
      15-level nominal was filled with its *mean code* (~7) and snapped to
      level 7 regardless of frequency -- arbitrary, since the codes are
      unordered, and not what a method called "Mean/Mode" should do. Unordered
      columns are mode-filled here; ordered ones keep the mean.
    """
    return {
        "mode_fill": unit.of_kind("binary", "nominal"),
        "factor": unit.of_kind("nominal"),
        "numeric": unit.of_kind("ordinal", "count", "continuous"),
        "snap_levels": unit.of_kind("nominal", "ordinal", "count"),
    }


def col_types_3way(unit: PreparedUnit) -> dict[str, list[str]]:
    """Collapse the five declared kinds onto the three the scorers take.

    Counts join the continuous group (RMSE on an integer count is meaningful;
    accuracy on one with 8 levels is not), ordinals join the categorical group
    but keep their order in the codes, so Phase 5 can split them back out using
    ``codebook["kinds"]`` -- the per-column metrics are preserved.
    """
    return {
        "continuous": unit.of_kind("continuous", "count"),
        "binary": unit.of_kind("binary"),
        "multiclass": unit.of_kind("nominal", "ordinal"),
    }


def _snap_to_canonical(
    frame: pd.DataFrame, unit: PreparedUnit, *, snap_numeric_codes: bool,
) -> pd.DataFrame:
    """Project a raw imputer output onto values the codebook allows."""
    out = snap_binary(frame, unit.of_kind("binary"))
    # Nominal and ordinal codes arrive exact from invert_genai (argmax and
    # rounding respectively), so only the classic family needs them snapped.
    targets = unit.of_kind("count")
    if snap_numeric_codes:
        targets = unit.of_kind("nominal", "ordinal", "count")
    return snap_multiclass(out, targets, unit.canonical_clean)


def impute_v2(
    unit: PreparedUnit,
    method: str,
    *,
    k: int,
    params: Mapping,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Run one method on one prepared unit. Returns a canonical-space frame."""
    cfg = cfg or active()
    families = cfg["imputation.families"].as_dict()
    family = next(f for f, members in families.items() if method in members)
    params = dict(params)

    if family == "reference":
        # The baseline arm is the masked matrix: XGBoost handles the NaNs.
        return unit.canonical_masked.copy()

    ct = col_types_3way(unit)
    logger.info("[%s/%s] family=%s K=%d", unit.tag, method, family, k)

    if family == "classic":
        out = _run_classic(unit, method, params, k, cfg, ct)
    else:
        out = _run_genai(unit, method, params, k, cfg)

    out.insert(0, unit.id_column, unit.canonical_masked[unit.id_column].to_numpy())
    return out


def _run_classic(
    unit: PreparedUnit,
    method: str,
    params: dict,
    k: int,
    cfg: Config,
    ct: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    from .classic import impute_knn, impute_mean_mode
    from .mice import impute_mice
    from .missranger import impute_missranger

    X = unit.canonical_masked[unit.features].copy()
    clean = unit.canonical_clean[unit.features]
    groups = classic_groups(unit)

    if method == "mean":
        # Unordered columns get the mode, ordered ones the mean; see
        # classic_groups for why that split is not what v1 did.
        run_one = lambda seed: impute_mean_mode(                # noqa: E731
            X, groups["mode_fill"], groups["snap_levels"], clean,
        )
    elif method == "knn":
        run_one = lambda seed: impute_knn(                      # noqa: E731
            X, list(ct["binary"]), groups["snap_levels"], clean, **params,
        )
    elif method == "mice":
        # Only NOMINAL columns become R factors. Ordinals and counts stay
        # numeric, so PMM draws an observed -- hence always valid -- level and
        # the order survives.
        run_one = lambda seed: impute_mice(                     # noqa: E731
            X, list(ct["binary"]), groups["factor"], clean,
            max_iter=params["max_iter"], random_state=seed,
            use_miceforest=params.get("use_miceforest", False),
        )
    elif method == "missranger":
        run_one = lambda seed: impute_missranger(               # noqa: E731
            X, list(ct["binary"]), groups["factor"], clean,
            n_trees=params["n_trees"], max_iter=params["max_iter"],
            pmm_k=params["pmm_k"], random_state=seed,
        )
    else:
        raise KeyError(method)

    out = ensemble(
        run_one, k=k, method=method,
        mode=cfg["imputation.combine"].as_dict()["classic"], col_types=ct,
    )
    return _snap_to_canonical(out, unit, snap_numeric_codes=True)


def _run_genai(
    unit: PreparedUnit, method: str, params: dict, k: int, cfg: Config,
) -> pd.DataFrame:
    from .genai import impute_hyperimpute, impute_remasker, set_experiment

    set_experiment("v2")
    X = unit.genai_masked
    if method == "remasker":
        raw = impute_remasker(X, params={**params, "n_ensemble": k})
    else:
        raw = impute_hyperimpute(X, method, params={**params, "n_ensemble": k})

    # Back to the canonical space BEFORE anything is snapped or scored. The
    # ensemble has already been averaged in the model's own output space, so a
    # dummy block carries soft class scores here and the argmax inside
    # invert_genai is a probability-level vote, not a post-hoc majority vote.
    canonical = encode_v2.invert_genai(raw, unit.encoding)
    return _snap_to_canonical(canonical, unit, snap_numeric_codes=False)
