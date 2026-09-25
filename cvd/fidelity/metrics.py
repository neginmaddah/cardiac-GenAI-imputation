"""
Imputation quality evaluation.

Compares imputed values against ground truth **only** at positions where
missingness was artificially injected (missing_status == 2).

Metrics:
    - Continuous columns: RMSE (root mean squared error)
    - Categorical columns: Accuracy, Macro-F1, Balanced Accuracy
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_imputation(
    imputed_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    missing_status: pd.DataFrame,
    cont_cols: List[str],
    binary_cols: List[str],
    multiclass_cols: List[str],
    injected_code: int = 2,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate imputed values against ground truth on injected positions.

    Parameters
    ----------
    imputed_df : DataFrame
        Imputed dataset.
    clean_df : DataFrame
        Ground-truth dataset (same scaling as imputed_df).
    missing_status : DataFrame
        Missing-status codes (0=complete, 1=original missing, 2=injected).
    cont_cols, binary_cols, multiclass_cols : list of str
        Column type lists.
    injected_code : int
        Status code for injected missingness (default 2).

    Returns
    -------
    dict with keys:
        ``"rmse"``          → {col: float} for continuous columns
        ``"accuracy"``      → {col: float} for categorical columns
        ``"f1_macro"``      → {col: float} for categorical columns
        ``"balanced_acc"``  → {col: float} for categorical columns
    """
    fake_mask = (missing_status == injected_code)

    rmse: Dict[str, float] = {}
    accuracy: Dict[str, float] = {}
    f1_macro: Dict[str, float] = {}
    balanced_acc: Dict[str, float] = {}

    # ── Continuous: RMSE ──
    for col in cont_cols:
        if col not in fake_mask.columns:
            continue
        mask = fake_mask[col]
        if not mask.any():
            continue
        y_true = clean_df.loc[mask, col].to_numpy().astype(float)
        y_pred = imputed_df.loc[mask, col].to_numpy().astype(float)
        # Drop pairs where either is NaN
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() == 0:
            continue
        rmse[col] = _rmse(y_true[valid], y_pred[valid])

    # ── Categorical: accuracy, F1, balanced accuracy ──
    for col in binary_cols + multiclass_cols:
        if col not in fake_mask.columns:
            continue
        mask = fake_mask[col]
        if not mask.any():
            continue
        y_true = clean_df.loc[mask, col].to_numpy()
        y_pred = imputed_df.loc[mask, col].to_numpy()
        # Ensure numeric for comparison
        try:
            y_true = y_true.astype(float).astype(int)
            y_pred = y_pred.astype(float).astype(int)
        except (ValueError, TypeError):
            continue
        # Drop pairs where either is NaN-like
        valid = ~(pd.isna(y_true) | pd.isna(y_pred))
        if valid.sum() == 0:
            continue
        yt, yp = y_true[valid], y_pred[valid]
        accuracy[col] = float(accuracy_score(yt, yp))
        f1_macro[col] = float(f1_score(yt, yp, average="macro", zero_division=0))
        balanced_acc[col] = float(balanced_accuracy_score(yt, yp))

    return {
        "rmse": rmse,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "balanced_acc": balanced_acc,
    }


def evaluate_multiple_methods(
    imputed_datasets: Dict[str, pd.DataFrame],
    clean_df: pd.DataFrame,
    missing_status: pd.DataFrame,
    cont_cols: List[str],
    binary_cols: List[str],
    multiclass_cols: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate several imputation methods at once.

    Parameters
    ----------
    imputed_datasets : dict
        Mapping method_name → imputed DataFrame.
    clean_df, missing_status : DataFrame
        Ground truth and status codes.
    cont_cols, binary_cols, multiclass_cols : list of str
        Column type lists.

    Returns
    -------
    dict mapping method_name → metrics dict (as returned by
    :func:`evaluate_imputation`).
    """
    results = {}
    for name, imp_df in imputed_datasets.items():
        results[name] = evaluate_imputation(
            imp_df, clean_df, missing_status,
            cont_cols, binary_cols, multiclass_cols,
        )
    return results


def results_to_dataframe(
    all_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
) -> pd.DataFrame:
    """
    Flatten nested evaluation results into a tidy DataFrame.

    Parameters
    ----------
    all_results : dict
        Structure: ``{simulation_label: {method_name: metrics_dict}}``,
        where metrics_dict is the output of :func:`evaluate_imputation`.

    Returns
    -------
    DataFrame with columns:
        simulation, method, metric, and one column per evaluated variable.
    """
    rows = []
    for sim_label, methods in all_results.items():
        for method_name, metrics_dict in methods.items():
            for metric_name, col_values in metrics_dict.items():
                row = {
                    "simulation": sim_label,
                    "method": method_name,
                    "metric": metric_name,
                }
                row.update(col_values)
                rows.append(row)
    return pd.DataFrame(rows)


# ── Human-readable method name mapping ────────────────────────────────

METHOD_DISPLAY_NAMES = {
    "mean": "Mean/Mode",
    "knn": "k-Nearest-Neighbor",
    "mice": "MICE",
    "missranger": "MissForest (ensemble tree-based)",
    "miwae": "MIWAE (VAE-based)",
    "gain": "GAIN (GAN-based)",
    "miracle": "MIRACLE (diffusion/energy-based)",
    "remasker": "ReMasker (transformer-based)",
}
