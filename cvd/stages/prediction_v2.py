"""Stage 05 (v2) -- downstream prediction from an imputed matrix.

Paper section: Methods 4.5, Post-Operative Adverse Outcome Prediction.

Runs the published protocol unchanged -- repeated randomized 60/20/20 splits at
the encounter level, three repeats with seeds 42/314/2718, NO cross-validation
anywhere -- against the v2 imputed matrices instead of the v1 ones, reusing
``cvd.prediction.modeling``'s fitters so the model side is identical to what the
paper describes.

Two things are simpler here than in v1 and both are consequences of the
rebuild:

* **Leakage control is structural, and the v1 exclusion list must NOT be
  reapplied.** Every held-out endpoint was moved into its own targets file at
  harmonisation, so the predictor matrix cannot contain one. Deriving the
  exclusion from that file is therefore both correct and self-maintaining.

  Applying ``variables.excluded_from_predictors`` instead would be actively
  wrong for v2: it still lists ``perfustm`` and ``xclamptm``, which were
  *endpoints* in v1 but are deliberately *predictors* here -- the decision was
  to hold out the post-operative fields and keep the intra-operative ones. A
  blanket reapplication silently dropped both, which is how a stale list
  quietly changes the experiment.
* **The baseline arm needs no special case.** It is the canonical masked matrix,
  which is already on disk as a prepared input; XGBoost handles its NaNs
  natively.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cvd.conf import Config
from cvd.prediction.modeling import _fit_binary, _fit_continuous

logger = logging.getLogger(__name__)

TAG_RE = re.compile(r"^(?P<cohort>Cohort1|Cohort2)_(?P<design>[A-Z]+)_(?P<alpha>\d+)_(?P<method>.+)$")


def parse_stem(stem: str) -> dict | None:
    m = TAG_RE.match(stem)
    return m.groupdict() if m else None


def feature_meta(kinds: dict[str, str], columns) -> dict[str, list[str]]:
    """The three-way grouping `_prepare_model_matrix` expects.

    Declared, not inferred. v1 re-derived this from the data with cardinality
    heuristics every time, which is how a column could be treated as one type by
    the imputer and another by the predictor.
    """
    cols = [c for c in columns if c in kinds]
    return {
        "binary": [c for c in cols if kinds[c] == "binary"],
        # Ordinals stay ONE ordered column, grouped with the numerics rather
        # than one-hot encoded. A tree can then split an ordered scale at a
        # threshold in a single cut ("status >= Urgent"), which one-hot makes
        # impossible -- it forces a separate split per level and throws the
        # order away. The z-scoring these pick up is irrelevant to a tree,
        # which splits on order statistics, so nothing else changes.
        "continuous": [c for c in cols if kinds[c] in ("continuous", "count", "ordinal")],
        "multiclass": [c for c in cols if kinds[c] == "nominal"],
    }


def held_out_columns(cfg: Config, cohort: str, id_column: str) -> set[str]:
    """Everything set aside at harmonisation, read from the targets file itself.

    The file IS the definition of what was held out, so this cannot drift from
    the harmonisation the way a hand-maintained list does.
    """
    targets = pd.read_csv(str(cfg[f"harmonise.target_outputs.{cohort}"]), nrows=0)
    return {c for c in targets.columns if c != id_column}


def load_unit(cfg: Config, cohort: str, path: Path, id_column: str):
    """Predictor matrix and the target frame, joined on the identifier."""
    X = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    targets = pd.read_csv(str(cfg[f"harmonise.target_outputs.{cohort}"]))
    leaked = held_out_columns(cfg, cohort, id_column) & set(X.columns)
    if leaked:
        # Should be impossible: a held-out column in the predictor matrix means
        # harmonisation regressed. Raise rather than drop -- silently repairing
        # it would hide the regression and change the experiment.
        raise ValueError(
            f"{path.name}: {len(leaked)} held-out endpoint(s) present as "
            f"predictors: {sorted(leaked)[:5]}"
        )
    merged = X.merge(targets, on=id_column, how="left", validate="one_to_one")
    return X, merged


def predict_unit(
    cfg: Config,
    X: pd.DataFrame,
    merged: pd.DataFrame,
    kinds: dict[str, str],
    target: str,
    is_binary: bool,
    id_column: str,
    n_jobs: int = 4,
) -> dict:
    features = [c for c in X.columns if c != id_column]
    meta = feature_meta(kinds, features)
    y_all = pd.to_numeric(merged[target], errors="coerce")
    keep = y_all.notna().to_numpy()
    n_kept = int(keep.sum())
    if n_kept < int(cfg["prediction.min_rows"]):
        return {"error": f"only {n_kept} rows with an observed target"}

    Xf = X.loc[keep, features].reset_index(drop=True)
    y = y_all[keep].reset_index(drop=True)

    runs = []
    for seed in [int(s) for s in cfg["prediction.split.seeds"]]:
        if is_binary:
            m = _fit_binary(
                Xf, y, meta, n_jobs, split_seed=seed,
                negative_to_positive_ratio=float(
                    cfg["prediction.binary.downsample.max_negative_to_positive_ratio"]
                ),
            )
        else:
            m = _fit_continuous(Xf, y, meta, n_jobs, split_seed=seed, target=target)
        runs.append({"seed": seed, **{k: v for k, v in m.items() if not k.startswith("_")}})
    return {"n_rows": n_kept, "n_features": len(features), "runs": runs}
