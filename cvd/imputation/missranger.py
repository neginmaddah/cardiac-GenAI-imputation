"""MissRanger -- iterative random-forest imputation with predictive mean matching.

Paper section: Methods - "Imputation of Masked Data".

A separate random forest is trained for each variable containing missing
values and predicts its blanks in an iterative round-robin loop; predictive
mean matching with k nearest neighbours preserves plausible numerical values.

The reference backend is the original R ``missRanger`` package, reached through
:mod:`cvd.imputation.r_bridge` and ``bridge.R``, so PMM and factor handling
match the analysis scripts exactly. Parameters come from
``conf/imputation/missranger.yaml``.

Category snapping is shared via :mod:`cvd.prep.postprocess`.
"""

from typing import List

import pandas as pd

from cvd.conf import active
from cvd.prep.postprocess import snap_binary, snap_multiclass

from .r_bridge import run_r_imputer

# conf/imputation/missranger.yaml
_MR = active()["imputation.methods.missranger.params"]
MISSRANGER_N_TREES: int = _MR["n_trees"]
MISSRANGER_MAX_ITER: int = _MR["max_iter"]
MISSRANGER_PMM_K: int = _MR["pmm_k"]
MISSRANGER_RANDOM_STATE: int = _MR["random_state"]

__all__ = ["impute_missranger"]


def impute_missranger(
    X: pd.DataFrame,
    binary_cols: List[str],
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
    n_trees: int = MISSRANGER_N_TREES,
    max_iter: int = MISSRANGER_MAX_ITER,
    pmm_k: int = MISSRANGER_PMM_K,
    random_state: int = MISSRANGER_RANDOM_STATE,
) -> pd.DataFrame:
    """Impute *X* with R ``missRanger``.

    Parameters
    ----------
    X : DataFrame
        Feature matrix with NaNs, numeric throughout.
    binary_cols, multiclass_cols : list of str
        Column lists from :func:`cvd.prep.typing.classify_columns`.
    clean_df : DataFrame
        Clean reference supplying the valid category values.
    n_trees : int
        Trees per forest.
    max_iter : int
        Round-robin iterations.
    pmm_k : int
        Neighbours for predictive mean matching.
    random_state : int
        Seed.
    """
    df = run_r_imputer(
        X,
        method="missranger",
        binary_cols=binary_cols,
        multiclass_cols=multiclass_cols,
        clean_df=clean_df,
        seed=random_state,
        max_iter=max_iter,
        n_trees=n_trees,
        pmm_k=pmm_k,
    )
    df = snap_binary(df, binary_cols)
    return snap_multiclass(df, multiclass_cols, clean_df)
