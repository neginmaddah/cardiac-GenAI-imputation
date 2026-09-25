"""
Post-processing of imputed data.

After imputation, raw outputs need:
    - One-hot columns inverted back to original multiclass values
    - Binary columns rounded and clipped to {0, 1}
    - Multiclass columns snapped to nearest valid category level
    - (Optional) QuantileTransformer inversion for GenAI methods
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def invert_one_hot(
    df: pd.DataFrame,
    dummy_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Convert one-hot dummy columns back to a single multiclass column.

    For each group of dummies, the column with the highest value is
    selected.  The original multiclass column name is restored.

    Parameters
    ----------
    df : DataFrame
        Imputed DataFrame containing the dummy columns.
    dummy_map : dict
        Mapping from original column name → list of dummy column names
        (as returned by :func:`preprocessing.one_hot_with_nans`).

    Returns
    -------
    DataFrame with dummy columns replaced by the original multiclass columns.
    """
    df = df.copy()
    for orig_col, dummy_cols in dummy_map.items():
        present = [c for c in dummy_cols if c in df.columns]
        if not present:
            continue
        # argmax across the dummy columns → index of highest value
        sub = df[present].astype(float)
        idx = sub.values.argmax(axis=1)
        # Map back to category label (strip prefix from dummy column names)
        prefix = orig_col + "_"
        labels = [c.replace(prefix, "", 1) for c in present]
        df[orig_col] = [labels[i] for i in idx]
        # Try to convert to numeric if all labels are numeric
        try:
            df[orig_col] = pd.to_numeric(df[orig_col])
        except (ValueError, TypeError):
            pass
        df = df.drop(columns=present)

    return df


def snap_binary(
    df: pd.DataFrame,
    binary_cols: List[str],
) -> pd.DataFrame:
    """Round binary columns to {0, 1} and cast to ``category``."""
    df = df.copy()
    for col in binary_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .round()
                .clip(0, 1)
                .astype(int)
                .astype("category")
            )
    return df


def snap_multiclass(
    df: pd.DataFrame,
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Snap multiclass columns to the nearest valid category value.

    Valid categories are taken from *clean_df*.
    """
    df = df.copy()
    for col in multiclass_cols:
        if col not in df.columns or col not in clean_df.columns:
            continue
        cats = np.sort(clean_df[col].dropna().unique())
        if len(cats) == 0:
            continue
        preds = pd.to_numeric(df[col], errors="coerce").to_numpy()
        best = np.abs(preds[:, None] - cats[None, :]).argmin(axis=1)
        df[col] = pd.Categorical.from_codes(best, categories=cats)
    return df


def inverse_quantile_transform(
    df: pd.DataFrame,
    cont_cols: List[str],
    qt: QuantileTransformer,
) -> pd.DataFrame:
    """Invert the QuantileTransformer on continuous columns."""
    df = df.copy()
    cols = [c for c in cont_cols if c in df.columns]
    if cols:
        # Generative models can emit small excursions outside the fitted
        # uniform quantile range; sklearn expects values in [0, 1].
        df[cols] = df[cols].clip(0.0, 1.0)
        df[cols] = qt.inverse_transform(df[cols])
    return df


def postprocess_imputed(
    df: pd.DataFrame,
    binary_cols: List[str],
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
    dummy_map: Optional[Dict[str, List[str]]] = None,
    qt: Optional[QuantileTransformer] = None,
    cont_cols: Optional[List[str]] = None,
    invert_qt: bool = False,
) -> pd.DataFrame:
    """
    Full post-processing pipeline for an imputed DataFrame.

    Steps (in order):
        1. Invert one-hot encoding (if dummy_map provided)
        2. Snap binary columns to {0, 1}
        3. Snap multiclass columns to nearest valid category
        4. Invert QuantileTransformer on continuous columns (if requested)

    Parameters
    ----------
    df : DataFrame
        Raw imputation output.
    binary_cols, multiclass_cols : list of str
        Column type lists.
    clean_df : DataFrame
        Clean reference for valid categories.
    dummy_map : dict, optional
        One-hot → original column mapping (for GenAI methods).
    qt : QuantileTransformer, optional
        Fitted transformer for inversion.
    cont_cols : list of str, optional
        Continuous column names (required if inverting QT).
    invert_qt : bool
        Whether to invert the QuantileTransformer.
    """
    result = df.copy()

    if dummy_map:
        result = invert_one_hot(result, dummy_map)

    result = snap_binary(result, binary_cols)
    result = snap_multiclass(result, multiclass_cols, clean_df)

    if invert_qt and qt is not None and cont_cols:
        result = inverse_quantile_transform(result, cont_cols, qt)

    return result
