"""Reading and preparing the frozen analysis matrices.

Paper sections: Methods 4.1 (variable taxonomy) and 4.3 (encoding).

Covers loading a cohort's clean and imputed matrices, classifying columns,
inverting one-hot blocks where a stored matrix is still encoded, building the
Baseline and Mean/Mode frames from the masked matrix, and extracting a target
series.

``_target_series`` holds the one constructed endpoint: additional ICU hours is
set to zero where ICU readmission is 0, on the reading that no readmission
means no additional ICU time. In Cohort2 that rule supplies 3,687 of 3,875
values, so results for that endpoint must be read with its composition in mind.
The field it is built from is in the uniformly excluded set and is never a
predictor, so the rule introduces no leakage.
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

from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from scipy.stats import friedmanchisquare, wilcoxon
from xgboost import XGBClassifier, XGBRegressor

from cvd.prediction.constants import OUTCOME_COLUMNS
from cvd.prediction.cohort_paths import (CohortConfig, _CLEAN_CACHE,
                                         _FEATURE_META_CACHE, _METHOD_CACHE)
logger = logging.getLogger(__name__)
def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_columns(df)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"or time": "ortime"})
    if "recordid" in df.columns:
        df["recordid"] = df["recordid"].astype("string").str.strip()
    return df

def _as_binary(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    nonmissing = s.dropna()
    rounded_vals = set(nonmissing.round().unique().tolist())
    if rounded_vals and rounded_vals.issubset({1, 2}) and 2 in rounded_vals:
        s = s.round().replace({1: 0, 2: 1})
    else:
        s = s.round().clip(0, 1)
    return s

def _is_int_like(series: pd.Series) -> bool:
    vals = pd.to_numeric(series.dropna(), errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return False
    return bool(np.all(np.abs(vals - np.round(vals)) < 1e-9))

def _feature_meta(config: CohortConfig, intensity: str) -> dict[str, list[str]]:
    key = (config.name, intensity)
    if key in _FEATURE_META_CACHE:
        return _FEATURE_META_CACHE[key]

    masked = _drop_outcomes(_read_csv(config.data_dir / config.masked_template.format(intensity=intensity)))
    clean = load_clean(config)
    feature_cols = [c for c in masked.columns if c != "recordid"]
    binary_cols: list[str] = []
    continuous_cols: list[str] = []
    multiclass_cols: list[str] = []

    for col in feature_cols:
        s = masked[col]
        n_unique = s.nunique(dropna=True)
        unique_rt = n_unique / max(1, int(s.notna().sum()))
        if n_unique == 2:
            binary_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            if (_is_int_like(s) and n_unique <= 10) or unique_rt <= 0.005:
                multiclass_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            multiclass_cols.append(col)

    levels = {
        c: np.sort(pd.to_numeric(clean[c], errors="coerce").dropna().unique()).tolist()
        for c in multiclass_cols
        if c in clean.columns and pd.to_numeric(clean[c], errors="coerce").dropna().nunique() > 0
    }
    meta = {
        "binary": binary_cols,
        "continuous": continuous_cols,
        "multiclass": multiclass_cols,
        "levels": levels,
    }
    _FEATURE_META_CACHE[key] = meta
    return meta

def load_clean(config: CohortConfig) -> pd.DataFrame:
    if config.name in _CLEAN_CACHE:
        return _CLEAN_CACHE[config.name].copy()

    clean = _read_csv(config.clean_path)
    if {"mortality_inhospital", "mt30stat"}.issubset(clean.columns):
        mortality = _as_binary(clean["mortality_inhospital"])
        mt30 = _as_binary(clean["mt30stat"])
        both_missing = mortality.isna() & mt30.isna()
        clean["mortality_anytype"] = ((mortality == 1) | (mt30 == 1)).astype(float)
        clean.loc[both_missing, "mortality_anytype"] = np.nan
    _CLEAN_CACHE[config.name] = clean
    return clean.copy()

def _outcome_like_columns(df: pd.DataFrame) -> list[str]:
    outcomes = set(OUTCOME_COLUMNS)
    return [
        c
        for c in df.columns
        if c != "recordid" and (c in outcomes or any(c.startswith(f"{outcome}_") for outcome in outcomes))
    ]

def _drop_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=_outcome_like_columns(df), errors="ignore")

def _coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "recordid" not in out.columns:
        raise ValueError("Expected recordid column in feature frame")
    rid = out["recordid"].astype("string").str.strip()
    out = out.drop(columns=["recordid"])
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out.insert(0, "recordid", rid)
    return out

def _snap_multiclass(df: pd.DataFrame, meta: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    levels = meta.get("levels", {})
    for col in meta["multiclass"]:
        if col not in out.columns or col not in levels:
            continue
        cats = np.asarray(levels[col], dtype=float)
        values = pd.to_numeric(out[col], errors="coerce").to_numpy()
        missing = np.isnan(values)
        if cats.size == 0:
            continue
        idx = np.abs(values[:, None] - cats[None, :]).argmin(axis=1)
        snapped = cats[idx]
        snapped[missing] = np.nan
        out[col] = snapped
    return out

def _invert_onehot_multiclass(df: pd.DataFrame, meta: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    levels = meta.get("levels", {})
    for col in meta["multiclass"]:
        prefix = f"{col}_"
        dummies = [c for c in out.columns if c.startswith(prefix)]
        if not dummies:
            continue

        dummy_values = out[dummies].apply(pd.to_numeric, errors="coerce")
        best_dummy = dummy_values.idxmax(axis=1)
        restored = pd.to_numeric(best_dummy.str[len(prefix) :], errors="coerce")

        if col in levels and len(levels[col]) > 0:
            cats = np.asarray(levels[col], dtype=float)
            values = restored.to_numpy(dtype=float)
            missing = np.isnan(values)
            idx = np.abs(values[:, None] - cats[None, :]).argmin(axis=1)
            snapped = cats[idx]
            snapped[missing] = np.nan
            restored = pd.Series(snapped, index=out.index)

        out[col] = restored
        out = out.drop(columns=dummies)
    return out

def _prep_masked_like(df: pd.DataFrame, meta: dict[str, list[str]], *, fill: bool) -> pd.DataFrame:
    out = _drop_outcomes(_coerce_feature_frame(df))
    for col in meta["binary"]:
        if col in out.columns:
            out[col] = _as_binary(out[col])
    out = _snap_multiclass(out, meta)

    if not fill:
        return out

    for col in out.columns:
        if col == "recordid":
            continue
        s = out[col]
        if col in meta["binary"] or col in meta["multiclass"]:
            mode = s.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else 0
        else:
            fill_value = s.mean(skipna=True)
            if pd.isna(fill_value):
                fill_value = 0
        out[col] = s.fillna(fill_value)
    return out

def _prepare_model_matrix(
    X: pd.DataFrame,
    meta: dict[str, list[str]],
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a numeric XGBoost matrix with train-fold feature normalization."""
    x = X.copy().reset_index(drop=True)
    x = x.replace([np.inf, -np.inf], np.nan)

    known_binary = [c for c in meta["binary"] if c in x.columns]
    known_continuous = [c for c in meta["continuous"] if c in x.columns]
    known_multiclass = [c for c in meta["multiclass"] if c in x.columns]
    known = set(known_binary) | set(known_continuous) | set(known_multiclass)
    unknown_cols = [c for c in x.columns if c not in known]

    inferred_binary: list[str] = []
    inferred_continuous: list[str] = []
    for col in unknown_cols:
        s = pd.to_numeric(x[col], errors="coerce")
        vals = set(s.dropna().unique().tolist())
        if vals and vals.issubset({0, 1, 2}):
            inferred_binary.append(col)
        else:
            inferred_continuous.append(col)

    pieces: list[pd.DataFrame] = []

    binary_cols = known_binary + inferred_binary
    if binary_cols:
        binary = pd.DataFrame(index=x.index)
        for col in binary_cols:
            binary[col] = _as_binary(x[col]).astype(float)
        pieces.append(binary)

    continuous_cols = known_continuous + inferred_continuous
    standardized_cols: list[str] = []
    dropped_continuous: list[str] = []
    if continuous_cols:
        continuous = pd.DataFrame(index=x.index)
        for col in continuous_cols:
            s = pd.to_numeric(x[col], errors="coerce").astype(float)
            train_s = s.iloc[train_idx]
            if train_s.notna().sum() == 0:
                dropped_continuous.append(col)
                continue
            mean = float(train_s.mean(skipna=True))
            std = float(train_s.std(skipna=True, ddof=0))
            if not np.isfinite(std) or std <= 1e-12:
                std = 1.0
            continuous[col] = (s - mean) / std
            standardized_cols.append(col)
        if not continuous.empty:
            pieces.append(continuous)

    ohe_cols_count = 0
    levels = meta.get("levels", {})
    for col in known_multiclass:
        raw = pd.to_numeric(x[col], errors="coerce")
        categories = levels.get(col)
        if categories:
            cat = pd.Categorical(raw, categories=categories)
        else:
            train_levels = np.sort(raw.iloc[train_idx].dropna().unique()).tolist()
            cat = pd.Categorical(raw, categories=train_levels)
        dummies = pd.get_dummies(cat, prefix=col, dummy_na=True, dtype=float)
        dummies.index = x.index
        ohe_cols_count += len(dummies.columns)
        pieces.append(dummies)

    if not pieces:
        raise ValueError("No usable feature columns after preprocessing.")

    matrix = pd.concat(pieces, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()].astype("float32")
    train_nonmissing = matrix.iloc[train_idx].notna().sum(axis=0)
    keep_cols = train_nonmissing[train_nonmissing > 0].index.tolist()
    matrix = matrix[keep_cols]
    return matrix, {
        "feature_preprocessing": "binary_0_1__continuous_train_zscore__multiclass_onehot",
        "binary_encoding": "0/1",
        "continuous_scaling": "train_fold_zscore",
        "multiclass_encoding": "one_hot_with_missing_indicator",
        "n_model_features": int(matrix.shape[1]),
        "n_binary_features": int(len(binary_cols)),
        "n_standardized_continuous_features": int(len(standardized_cols)),
        "n_onehot_features": int(ohe_cols_count),
        "n_dropped_empty_continuous_features": int(len(dropped_continuous)),
        "missing_cells_after_preprocessing": int(matrix.isna().sum().sum()),
    }

def _make_knn(config: CohortConfig, intensity: str, meta: dict[str, list[str]]) -> pd.DataFrame:
    masked = _read_csv(config.data_dir / config.masked_template.format(intensity=intensity))
    base = _prep_masked_like(masked, meta, fill=False)
    recordid = base["recordid"]
    x = base.drop(columns=["recordid"])
    imputer = KNNImputer(n_neighbors=3, weights="uniform", keep_empty_features=True)
    values = imputer.fit_transform(x)
    out = pd.DataFrame(values, columns=x.columns, index=x.index)
    for col in meta["binary"]:
        if col in out.columns:
            out[col] = out[col].round().clip(0, 1)
    out = _snap_multiclass(out.assign(recordid=recordid), meta)
    cols = ["recordid"] + [c for c in out.columns if c != "recordid"]
    return out[cols]

def load_method_frame(config: CohortConfig, intensity: str, method: str) -> pd.DataFrame:
    key = (config.name, intensity, method)
    if key in _METHOD_CACHE:
        return _METHOD_CACHE[key].copy()

    meta = _feature_meta(config, intensity)
    if method == "Baseline":
        masked = _read_csv(config.data_dir / config.masked_template.format(intensity=intensity))
        frame = _prep_masked_like(masked, meta, fill=False)
    elif method == "Mean/Mode":
        masked = _read_csv(config.data_dir / config.masked_template.format(intensity=intensity))
        frame = _prep_masked_like(masked, meta, fill=True)
    elif method == "KNN":
        frame = _make_knn(config, intensity, meta)
    else:
        raw = _drop_outcomes(_read_csv(_method_path(config, method, intensity)))
        raw = _invert_onehot_multiclass(raw, meta)
        frame = _coerce_feature_frame(raw)
        frame = _drop_outcomes(frame)
        for col in meta["binary"]:
            if col in frame.columns:
                frame[col] = _as_binary(frame[col])
        frame = _snap_multiclass(frame, meta)

    _METHOD_CACHE[key] = frame
    return frame.copy()

def _target_series(clean: pd.DataFrame, target: str, target_type: str) -> pd.DataFrame:
    if target not in clean.columns:
        return pd.DataFrame(columns=["recordid", target])
    y = clean[["recordid", target]].copy()
    if target_type == "binary":
        y[target] = _as_binary(y[target])
    else:
        y[target] = pd.to_numeric(y[target], errors="coerce")
        if target == "icuadhrs" and "icureadm" in clean.columns:
            icu_readmit = _as_binary(clean["icureadm"])
            fill_zero = y[target].isna() & icu_readmit.eq(0)
            y.loc[fill_zero, target] = 0.0
    return y