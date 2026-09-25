"""Target lists, method ordering and display constants for the prediction path.

These were module-level constants in a 3,760-line script. The *values* now live
in ``conf/`` -- this module re-exports them in the shapes the plotting and
statistics code already expects, so there is one source of truth without
rewriting 1,600 lines of matplotlib.

Paper sections: Methods 4.1 (target definitions), 4.5 (prediction).
"""
from __future__ import annotations

import json
import logging
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from cvd.conf import active, all_targets, analysed_targets, target_labels

_cfg = active()

# Repository root; the frozen result trees hang off it.
BASE_DIR = Path(str(_cfg["paths.root"]))

# ── holdout intensities ───────────────────────────────────────────────────
# As bare-integer strings, which is how they appear in the frozen filenames
# and in the `intensity` column of every stored result CSV.
INTENSITIES = tuple(
    str(int(round(a * 100))) for a in _cfg["missingness.alphas"]
)

# ── split protocol (conf/prediction.yaml) ─────────────────────────────────
DEFAULT_SPLIT_SEEDS = tuple(_cfg["prediction.split.seeds"])
BINARY_NEGATIVE_TO_POSITIVE_RATIO = float(
    _cfg["prediction.binary.downsample.max_negative_to_positive_ratio"]
)

# ── method identity (conf/imputation/_base.yaml) ──────────────────────────
# Display labels, not registry keys, because every stored result CSV and every
# figure legend uses the labels.
_REGISTRY = _cfg["imputation.registry"].as_dict()
_ORDER = list(_cfg["imputation.order"])
METHOD_ORDER = tuple(_REGISTRY[k]["label"] for k in _ORDER)
METHOD_COLORS = {_REGISTRY[k]["label"]: _REGISTRY[k]["color"] for k in _ORDER}
# The Baseline is a reference arm, not an imputer, so it is excluded from the
# arms the omnibus and pairwise tests compare.
_REFERENCE = set(_cfg["imputation.families.reference"])
STAT_TEST_METHODS = tuple(
    _REGISTRY[k]["label"] for k in _ORDER if k not in _REFERENCE
)

# ── statistics (conf/stats.yaml) ──────────────────────────────────────────
STAT_TEST_ALPHA = float(_cfg["stats.alpha"])

# ── figure styling (conf/reporting.yaml) ──────────────────────────────────
COMPACT_STAR_MARKER_SIZE = float(_cfg["reporting.defaults.star_marker_size"])
COMPACT_LEGEND_STAR_MARKER_SIZE = float(
    _cfg["reporting.defaults.legend_star_marker_size"]
)
COMPACT_V4_FONT_SCALE = float(_cfg["reporting.defaults.compact_font_scale"])

# ── endpoints (conf/variables.yaml) ───────────────────────────────────────
# TWO DISTINCT SETS, and keeping them distinct matters:
#
#   BINARY_TARGETS / CONTINUOUS_TARGETS
#       every declared endpoint. These drive the GRID -- all of them must be
#       fitted, because the supplement reports `unrespstat` (Note 1) and
#       `perfustm` / `xclamptm` (Figures S5, S6). Narrowing these would leave
#       supplementary figures with no data behind them.
#
#   MAIN_BINARY_TARGETS / MAIN_CONTINUOUS_TARGETS
#       the endpoints the MAIN TEXT analyses, i.e. those flagged
#       `analysed: true`. Figures 4-9 and every main-text pooled test use
#       these.
#
# Conflating the two is precisely what let Tables S9/S10 keep pooling four
# binary endpoints (144 blocks) after the main text had moved to three (108) --
# so the tables still contained what Reviewer 1 had objected to, and
# contradicted the main text.
BINARY_TARGETS = tuple(all_targets(_cfg, "binary"))
CONTINUOUS_TARGETS = tuple(all_targets(_cfg, "continuous"))
MAIN_BINARY_TARGETS = tuple(analysed_targets(_cfg, "binary"))
MAIN_CONTINUOUS_TARGETS = tuple(analysed_targets(_cfg, "continuous"))
# Backwards-compatible aliases for the supplementary-only readers.
ALL_BINARY_TARGETS = BINARY_TARGETS
ALL_CONTINUOUS_TARGETS = CONTINUOUS_TARGETS

TARGET_LABELS = target_labels(_cfg)

# Objective families for the continuous endpoints.
_OBJ = _cfg["prediction.single_fit.regression_objectives"].as_dict()
RMSLE_TARGETS = frozenset(
    t for t, o in _OBJ.items() if o == "reg:squaredlogerror"
)
MPHE_TARGETS = frozenset(
    t for t, o in _OBJ.items() if o == "reg:pseudohubererror"
)

# ── leakage control (conf/variables.yaml -> Supplementary Table S6) ───────
_EXCLUDED = _cfg["variables.excluded_from_predictors"].as_dict()
LEAKAGE_CONTINUOUS_TARGETS = CONTINUOUS_TARGETS
LEAKAGE_BINARY_TARGETS = tuple(
    t for t in _EXCLUDED["endpoints"]
    if t not in CONTINUOUS_TARGETS
)
LEAKAGE_MULTICLASS_TARGETS = tuple(_EXCLUDED["discharge_medication"])
# NOTE: `mortality_anytype` sits in the endpoints group here, where the
# pre-refactor code had it in the "additional outcomes" list. Only the UNION
# (OUTCOME_COLUMNS) is ever consumed and it is byte-identical, so the grouping
# change has no effect; it is grouped this way because it is a modelled binary
# endpoint, not an incidental outcome field.
ADDITIONAL_OUTCOME_COLUMNS = tuple(_EXCLUDED["other_outcomes"])
OUTCOME_COLUMNS = tuple(dict.fromkeys(
    LEAKAGE_CONTINUOUS_TARGETS
    + LEAKAGE_BINARY_TARGETS
    + LEAKAGE_MULTICLASS_TARGETS
    + ADDITIONAL_OUTCOME_COLUMNS
))

# Cohort-specific predictor exclusions: the single target-specific entry in
# Supplementary Table S6. In Cohort2, `caortreint` missingness almost perfectly
# identifies non-complication rows.
LEAKAGE_FEATURES_BY_COHORT_TARGET = {
    (cohort, target): tuple(fields)
    for cohort in _cfg["cohorts"].keys()
    for target, fields in (
        (_cfg.get(f"cohorts.{cohort}.leakage_exclusions") or {})
        and _cfg[f"cohorts.{cohort}.leakage_exclusions"].as_dict().items()
        or []
    )
}








STAT_OMNIBUS_COLUMNS = (
    "cohort",
    "outcome_type",
    "target",
    "target_label",
    "metric",
    "direction",
    "alpha",
    "methods_tested",
    "n_methods_expected",
    "n_methods_available",
    "missing_methods",
    "block_columns",
    "n_blocks_total",
    "n_complete_blocks",
    "best_median_method",
    "best_median_value",
    "friedman_chi2",
    "friedman_p",
    "omnibus_reject",
    "status",
)

STAT_PAIRWISE_COLUMNS = (
    "cohort",
    "outcome_type",
    "target",
    "target_label",
    "metric",
    "direction",
    "method_a",
    "method_b",
    "n_complete_blocks",
    "median_method_a",
    "median_method_b",
    "median_diff_a_minus_b",
    "better_method",
    "worse_method",
    "wilcoxon_statistic",
    "p_value",
    "holm_p",
    "holm_reject",
    "alpha",
    "n_nonzero_differences",
)







LEAKAGE_MULTICLASS_TARGETS = ("dcasa", "dcadp", "dccoum", "dcothanticoag")










# ── ordering helpers ───────────────────────────────────────────────────────
# Sort keys over METHOD_ORDER / INTENSITIES. They live here, next to the
# orderings they read, because both the figures and the statistics modules
# need them -- keeping them in the orchestrator made those two import it and
# created a cycle.


def _method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def _intensity_sort_key(intensity: str) -> int:
    return INTENSITIES.index(str(intensity)) if str(intensity) in INTENSITIES else 99
