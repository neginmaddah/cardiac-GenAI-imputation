"""
Automatic column-type classification.

Classifies DataFrame columns into three groups:
    - binary:     exactly 2 unique non-null values
    - multiclass: categorical with >2 levels (integer-like with few uniques,
                  low unique-ratio numerics, or non-numeric columns)
    - continuous: high-cardinality numeric columns

The thresholds (cat_max, cat_frac, eps) match the original analysis scripts.
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from cvd.conf import active

# Thresholds come from conf/variables.yaml (`variables.typing`). Resolved once
# at import so the existing signatures keep working; the YAML is the only place
# these numbers are written down.
_T = active()["variables.typing"]
CAT_MAX: int = _T["cat_max"]
CAT_FRAC: float = _T["cat_frac"]
EPS: float = _T["eps"]


def is_int_like(series: pd.Series, eps: float = EPS) -> bool:
    """Check whether all non-null values in *series* are integer-valued."""
    vals = series.dropna().values
    if len(vals) == 0:
        return False
    return bool(np.all(np.abs(vals - np.round(vals)) < eps))


def classify_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[Set[str]] = None,
    cat_max: int = CAT_MAX,
    cat_frac: float = CAT_FRAC,
    eps: float = EPS,
) -> Dict[str, List[str]]:
    """
    Classify columns into binary, multiclass, and continuous.

    Parameters
    ----------
    df : DataFrame
        The dataset whose columns are to be classified.
    exclude_cols : set of str, optional
        Column names to skip (e.g. record ID, outcome variables).
    cat_max : int
        Maximum unique values for an integer-like column to be categorical.
    cat_frac : float
        Maximum unique-ratio (n_unique / n_non_null) for a column to be
        categorical when it has more than *cat_max* unique values.
    eps : float
        Tolerance for the integer-like check.

    Returns
    -------
    dict with keys ``"binary"``, ``"multiclass"``, ``"continuous"``,
    each mapping to a sorted list of column names.
    """
    exclude = exclude_cols or set()
    binary_cols: List[str] = []
    multiclass_cols: List[str] = []
    cont_cols: List[str] = []

    for col in df.columns:
        if col in exclude:
            continue

        s = df[col]
        n_unique = s.nunique(dropna=True)

        if n_unique <= 1:
            # constant or all-null — skip
            continue

        if n_unique == 2:
            binary_cols.append(col)
            continue

        # Numeric columns
        if pd.api.types.is_numeric_dtype(s):
            n_non_null = s.notna().sum()
            unique_ratio = n_unique / n_non_null if n_non_null > 0 else 0

            if is_int_like(s, eps) and n_unique <= cat_max:
                multiclass_cols.append(col)
            elif unique_ratio <= cat_frac:
                multiclass_cols.append(col)
            else:
                cont_cols.append(col)
        else:
            # Non-numeric → treat as multiclass
            multiclass_cols.append(col)

    return {
        "binary": sorted(binary_cols),
        "multiclass": sorted(multiclass_cols),
        "continuous": sorted(cont_cols),
    }
