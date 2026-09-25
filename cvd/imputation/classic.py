"""Deterministic imputers: Mean/Mode and KNN.

Both are deterministic, so ``conf/imputation/_base.yaml`` lists them under
``deterministic`` and they are never ensembled (K is forced to 1).

Category snapping is shared with every other imputer via
:mod:`cvd.prep.postprocess` -- that block used to be copy-pasted eleven times
across this package.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from cvd.conf import active
from cvd.prep.postprocess import snap_binary, snap_multiclass

# conf/imputation/knn.yaml
_KNN = active()["imputation.methods.knn.params"]
KNN_NEIGHBORS: int = _KNN["n_neighbors"]
KNN_WEIGHTS: str = _KNN["weights"]


def impute_mean_mode(
    X: pd.DataFrame,
    binary_cols: List[str],
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mean imputation for continuous columns, mode for binary/categorical.

    After filling, multiclass columns are snapped to the nearest valid
    category value from *clean_df*.

    Parameters
    ----------
    X : DataFrame
        Feature matrix with NaNs. Binary cols should be ``category`` dtype,
        others ``float``.
    binary_cols, multiclass_cols : list of str
        Column name lists from :func:`classify_columns`.
    clean_df : DataFrame
        Original clean dataset (without outcome vars) used to look up
        valid category values.
    """
    df = X.copy()

    for col in df.columns:
        if col in binary_cols:
            m = df[col].mode(dropna=True)
            fill = m.iloc[0] if not m.empty else np.nan
        else:
            fill = df[col].mean()
        df[col] = df[col].fillna(fill)

    return snap_multiclass(df, multiclass_cols, clean_df)


def impute_knn(
    X: pd.DataFrame,
    binary_cols: List[str],
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
    n_neighbors: int = KNN_NEIGHBORS,
    weights: str = KNN_WEIGHTS,
) -> pd.DataFrame:
    """
    KNN imputation with post-processing for categorical columns.

    Binary columns are rounded and clipped to {0, 1}.
    Multiclass columns are snapped to the nearest valid category.

    Parameters
    ----------
    X : DataFrame
        Feature matrix with NaNs.
    binary_cols, multiclass_cols : list of str
        Column name lists.
    clean_df : DataFrame
        Clean reference for valid category values.
    n_neighbors : int
        Number of neighbours (default 3).
    weights : str
        Weight function (default ``"uniform"``).
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    raw = imputer.fit_transform(X)
    df = pd.DataFrame(raw, columns=X.columns, index=X.index)

    df = snap_binary(df, binary_cols)
    return snap_multiclass(df, multiclass_cols, clean_df)
