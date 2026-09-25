"""MICE -- multiple imputation by chained equations.

Paper section: Methods - "Imputation of Masked Data".

The reference backend is the original R ``mice`` package, reached through
:mod:`cvd.imputation.r_bridge` and ``bridge.R``. Per-variable elementary
methods are assigned automatically by type and cardinality (PMM for continuous,
logistic regression for balanced binary factors, CART for imbalanced ones,
Bayesian polytomous regression for low-cardinality multiclass and CART above
that); the thresholds live in ``conf/imputation/mice.yaml``.

**Disclosed protocol deviation.** R ``mice`` cannot be installed in the
cluster environment used for the injected MAR/EV grid: the CRAN ``nloptr``
build requires NLopt and there is no root access. Those runs therefore used the
``miceforest`` (LightGBM) backend. This is a real difference from the
uniform-holdout runs and is recorded in ``conf/imputation/mice.yaml``.

Category snapping is shared via :mod:`cvd.prep.postprocess`.
"""

from typing import List

import numpy as np
import pandas as pd

from cvd.conf import active
from cvd.prep.postprocess import snap_binary, snap_multiclass

from .r_bridge import run_r_imputer

# conf/imputation/mice.yaml
_MICE = active()["imputation.methods.mice"]
MICE_MAX_ITER: int = _MICE["params"]["max_iter"]
MICE_RANDOM_STATE: int = _MICE["params"]["random_state"]
MICE_NUM_DATASETS: int = _MICE["miceforest"]["num_datasets"]
MICEFOREST_MIN_DATA_IN_LEAF: int = _MICE["miceforest"]["min_data_in_leaf"]
MICEFOREST_LOGODDS_EPS: float = _MICE["miceforest"]["logodds_clip_eps"]
DEFAULT_BACKEND: str = _MICE["backend"]

__all__ = ["impute_mice"]


def impute_mice(
    X: pd.DataFrame,
    binary_cols: List[str],
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
    max_iter: int = MICE_MAX_ITER,
    random_state: int = MICE_RANDOM_STATE,
    use_miceforest: bool = False,
) -> pd.DataFrame:
    """Impute *X* with MICE.

    Parameters
    ----------
    X : DataFrame
        Feature matrix with NaNs. Numeric throughout: binary as 0/1, multiclass
        as integer level codes, continuous as float.
    binary_cols, multiclass_cols : list of str
        Column lists from :func:`cvd.prep.typing.classify_columns`.
    clean_df : DataFrame
        Clean reference supplying the valid category values.
    max_iter : int
        Chained-equation rounds.
    random_state : int
        Seed.
    use_miceforest : bool
        Use the LightGBM backend instead of R. See the module docstring: this
        is the disclosed deviation, not a tuning knob.
    """
    backend = _mice_miceforest if use_miceforest else _mice_r
    df = backend(
        X, binary_cols, multiclass_cols, clean_df,
        max_iter=max_iter, random_state=random_state,
    )
    df = snap_binary(df, binary_cols)
    return snap_multiclass(df, multiclass_cols, clean_df)


def _mice_r(
    X: pd.DataFrame,
    binary_cols: List[str],
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
    *,
    max_iter: int,
    random_state: int,
) -> pd.DataFrame:
    """The reference R ``mice`` backend."""
    return run_r_imputer(
        X,
        method="mice",
        binary_cols=binary_cols,
        multiclass_cols=multiclass_cols,
        clean_df=clean_df,
        seed=random_state,
        max_iter=max_iter,
    )


def _mice_miceforest(
    X: pd.DataFrame,
    binary_cols: List[str],
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
    *,
    max_iter: int,
    random_state: int,
) -> pd.DataFrame:
    """The ``miceforest`` (LightGBM) backend."""
    try:
        import miceforest as mf
        import miceforest.imputation_kernel as _mf_kernel_module
    except ImportError as exc:
        raise ImportError(
            "miceforest is not installed. Install with: pip install miceforest\n"
            "The R backend (use_miceforest=False) needs Rscript on PATH or "
            "CVD_RSCRIPT set."
        ) from exc

    # Several columns here have categories below 0.2% frequency (a rare
    # multiclass level with <10 rows in the smaller cohort), which lets LightGBM
    # output an exact 0.0/1.0 probability for some leaf. miceforest's
    # mean-matching then takes log-odds of that -- probability / (1 -
    # probability) -- and crashes with a division by zero, surfacing as "data
    # must be finite" in the downstream KDTree. Raising `min_data_in_leaf`
    # alone (miceforest's own suggested fix) was NOT sufficient on this data
    # even at 30, so `logodds` itself is patched to clip away from the exact
    # 0/1 boundary -- a standard numerical-stability fix for logit transforms.
    #
    # `imputation_kernel.py` does `from .utils import logodds`, binding its own
    # local reference, so `miceforest.utils.logodds` must be left alone and
    # this module's copy patched directly for the change to take effect.
    def _safe_logodds(probability):
        eps = MICEFOREST_LOGODDS_EPS
        p = np.clip(probability, eps, 1 - eps)
        return np.log(p / (1 - p))

    _mf_kernel_module.logodds = _safe_logodds

    # miceforest 6.0.5 renamed the `datasets` kwarg to `num_datasets`.
    kernel = mf.ImputationKernel(
        X, num_datasets=MICE_NUM_DATASETS, random_state=random_state,
    )
    kernel.mice(max_iter, min_data_in_leaf=MICEFOREST_MIN_DATA_IN_LEAF)
    return kernel.complete_data(dataset=0)
