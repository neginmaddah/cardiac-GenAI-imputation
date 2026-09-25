"""
Preprocessing utilities for imputation.

Handles:
    - StandardScaler on continuous columns (fit on masked, transform both)
    - Binary recoding (sorted unique values → 0/1)
    - Multiclass integer encoding (sorted unique values → 1..K)
    - One-hot encoding that preserves NaN structure
    - QuantileTransformer for GenAI methods
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler


# ── Scaling ────────────────────────────────────────────────────────────

def scale_continuous(
    df: pd.DataFrame,
    cont_cols: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize continuous columns (zero mean, unit variance).

    If *scaler* is ``None`` a new one is fitted on *df*; otherwise the
    provided scaler is used for transform only (important: fit on masked
    data, transform clean data with the same scaler).

    Returns (df_copy, scaler).
    """
    df = df.copy()
    cols = [c for c in cont_cols if c in df.columns]
    if not cols:
        return df, scaler or StandardScaler()

    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


# ── Binary recoding ───────────────────────────────────────────────────

def recode_binary(
    df: pd.DataFrame,
    binary_cols: List[str],
) -> pd.DataFrame:
    """
    Recode binary columns to 0/1 and mark as ``category`` dtype.

    Original values ``{1, 2}`` are mapped to ``{0, 1}``.
    If the column already contains only 0/1, it is left unchanged.
    """
    df = df.copy()
    for col in binary_cols:
        if col not in df.columns:
            continue
        uniq = sorted(df[col].dropna().unique())
        if len(uniq) == 2 and set(uniq) == {1, 2}:
            df[col] = df[col].replace({1: 0, 2: 1})
        df[col] = df[col].astype("category")
    return df


# ── Multiclass integer encoding ──────────────────────────────────────

def encode_multiclass(
    df: pd.DataFrame,
    multiclass_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Encode multiclass columns as integers 1..K (sorted unique values).

    Returns (df_copy, category_maps) where *category_maps* maps column
    name → sorted array of original unique values.
    """
    df = df.copy()
    cat_maps: Dict[str, np.ndarray] = {}
    for col in multiclass_cols:
        if col not in df.columns:
            continue
        cats = np.sort(df[col].dropna().unique())
        cat_maps[col] = cats
        mapping = {v: i + 1 for i, v in enumerate(cats)}
        df[col] = df[col].map(mapping)
    return df, cat_maps


# ── One-hot encoding with NaN preservation ────────────────────────────

def one_hot_with_nans(
    df: pd.DataFrame,
    multiclass_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Create one-hot dummies for multiclass columns, preserving NaN rows.

    For a column with categories [A, B, C], creates three 0/1 columns.
    Rows where the original column is NaN get NaN in all dummy columns.

    Returns (encoded_df, dummy_map) where *dummy_map* maps each original
    column name to its list of dummy column names.
    """
    dummy_map: Dict[str, List[str]] = {}
    parts = []

    for col in multiclass_cols:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col).astype(float)
        # Restore NaNs where original was missing
        mask = df[col].isna()
        dummies.loc[mask] = np.nan
        dummy_map[col] = list(dummies.columns)
        parts.append(dummies)

    # Columns NOT being one-hot encoded
    other_cols = [c for c in df.columns if c not in multiclass_cols]
    result = df[other_cols].copy()
    for p in parts:
        result = pd.concat([result, p], axis=1)

    return result, dummy_map


# ── QuantileTransformer (for GenAI methods) ───────────────────────────

def quantile_transform(
    df: pd.DataFrame,
    cont_cols: List[str],
    qt: Optional[QuantileTransformer] = None,
    n_quantiles: Optional[int] = None,
) -> Tuple[pd.DataFrame, QuantileTransformer]:
    """
    Apply QuantileTransformer (uniform output) to continuous columns.

    Returns (df_copy, qt_fitted).
    """
    df = df.copy()
    cols = [c for c in cont_cols if c in df.columns]
    if not cols:
        return df, qt or QuantileTransformer()

    if qt is None:
        nq = n_quantiles or min(1000, len(df))
        qt = QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=nq,
            random_state=42,
        )
        df[cols] = qt.fit_transform(df[cols])
    else:
        df[cols] = qt.transform(df[cols])
    return df, qt


# ── Prepare features for imputation ──────────────────────────────────

def prepare_for_imputation(
    masked_df: pd.DataFrame,
    binary_cols: List[str],
    id_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cast columns to proper dtypes and drop the ID column.

    Binary columns → ``category``, everything else → ``float``.
    """
    X = masked_df.drop(columns=[id_column] if id_column else []).copy()
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    for col in X.columns:
        if col not in binary_cols:
            X[col] = X[col].astype(float)
    return X
