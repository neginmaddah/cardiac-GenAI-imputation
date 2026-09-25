"""Fitting the XGBoost models.

Paper section: Methods 4.5.

Splitting is three independently randomized 60/20/20 train/validation/test
partitions at the patient-encounter level. **No cross-validation is used
anywhere in this study.**

Binary endpoints get inverse-prevalence class weighting, majority-class
downsampling in the training fold only, and a threshold chosen on the
validation fold. Continuous endpoints get mean-consistent log-link objectives.

One rule matters more than any hyperparameter here: **model selection uses the
same metric that is reported.** An earlier version selected on validation RMSLE
while reporting R2; because a squared-log-error objective estimates a
conditional geometric rather than arithmetic mean, that combination
systematically understated R2 on the duration endpoints. See the provenance notes.
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

from cvd.prep.frames import _prepare_model_matrix
from cvd.prediction.constants import RMSLE_TARGETS
logger = logging.getLogger(__name__)
def stratification_bins(
    y: pd.Series, *, n_bins: int = 10, min_count: int = 2,
) -> pd.Series | None:
    """Bin labels to stratify a split on, or None when stratification is unsafe.

    A binary target uses its two classes. A continuous one is cut into `n_bins`
    quantile bins, so train / validation / test each carry the same share of
    every part of the distribution -- previously continuous targets were split
    with no stratification at all, which on a skewed outcome lets a fold end up
    with a materially different mean than the others.

    Degenerate cases are handled rather than raised on: repeated quantile edges
    collapse (`duplicates="drop"`), any bin below `min_count` is merged into its
    neighbour, and if fewer than two bins survive the caller falls back to an
    unstratified split.
    """
    s = pd.Series(y).reset_index(drop=True)
    if s.nunique(dropna=True) <= 1:
        return None
    if s.nunique(dropna=True) == 2:
        labels = s
    else:
        try:
            labels = pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
        except (ValueError, IndexError):
            return None
        labels = pd.Series(labels).astype("float")
        # Merge undersized bins upward so every surviving bin can appear in all
        # three folds; an orphan bin is what makes train_test_split raise.
        counts = labels.value_counts().sort_index()
        for b in list(counts.index):
            if counts.get(b, 0) < min_count:
                nxt = [x for x in counts.index if x > b]
                labels = labels.replace(b, nxt[0] if nxt else labels.max())
                counts = labels.value_counts().sort_index()
    vc = labels.value_counts()
    if len(vc) < 2 or vc.min() < min_count:
        return None
    return labels


def _split_indices(
    y: pd.Series,
    split_seed: int,
    *,
    n_bins: int = 10,
    min_count: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    strat = stratification_bins(y, n_bins=n_bins, min_count=min_count)
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=0.20,
        random_state=split_seed,
        stratify=None if strat is None else strat.to_numpy(),
    )
    strat2 = None
    if strat is not None:
        s2 = strat.iloc[train_val_idx]
        if s2.value_counts().min() >= min_count and s2.nunique() >= 2:
            strat2 = s2.to_numpy()
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=split_seed + 1,
        stratify=strat2,
    )
    return train_idx, val_idx, test_idx

def _safe_auc(metric_fn, y_true: pd.Series, score: np.ndarray) -> float:
    try:
        if pd.Series(y_true).nunique(dropna=True) < 2:
            return float("nan")
        return float(metric_fn(y_true, score))
    except ValueError:
        return float("nan")

def _best_threshold(y_true: pd.Series, score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising validation balanced accuracy; ties keep the lowest.

    This used to carry an AU-PRC tie-breaker, which was inert and has been
    removed: average precision is a function of ``score``, which does not depend
    on ``threshold``, so it took the same value at every point of the grid and
    the tie-break clause could only fire on the first iteration -- where the
    strict comparison already held. The rule the code has always applied is the
    one written here, and removing the clause changes no result.
    """
    best_threshold = 0.5
    best_bal = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (score >= threshold).astype(int)
        bal = balanced_accuracy_score(y_true, pred)
        if bal > best_bal:
            best_bal = float(bal)
            best_threshold = float(threshold)
    return best_threshold, best_bal

def _binary_param_grid(scale_pos_weight: float, n_jobs: int) -> list[dict[str, object]]:
    return [
        {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": scale_pos_weight,
            "max_delta_step": 1,
            "n_jobs": n_jobs,
        },
        {
            "n_estimators": 350,
            "learning_rate": 0.02,
            "max_depth": 4,
            "min_child_weight": 2,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "scale_pos_weight": scale_pos_weight,
            "max_delta_step": 1,
            "n_jobs": n_jobs,
        },
        {
            "n_estimators": 180,
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_child_weight": 1,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "scale_pos_weight": max(1.0, math.sqrt(scale_pos_weight)),
            "max_delta_step": 0,
            "n_jobs": n_jobs,
        },
    ]

def _regressor_param_grid(n_jobs: int) -> list[dict[str, object]]:
    return [
        {
            "objective": "reg:squarederror",
            "n_estimators": 250,
            "learning_rate": 0.04,
            "max_depth": 3,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "n_jobs": n_jobs,
        },
        {
            "objective": "reg:squarederror",
            "n_estimators": 350,
            "learning_rate": 0.025,
            "max_depth": 4,
            "min_child_weight": 2,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "n_jobs": n_jobs,
        },
        {
            "objective": "reg:squarederror",
            "n_estimators": 180,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_child_weight": 1,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "n_jobs": n_jobs,
        },
        {
            "objective": "reg:pseudohubererror",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "n_jobs": n_jobs,
        },
    ]

def _base_xgb_params(random_state: int = 42) -> dict[str, object]:
    return {
        "tree_method": "hist",
        "random_state": random_state,
        "verbosity": 0,
        "missing": np.nan,
    }

def _compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.maximum(np.asarray(y_true, dtype=float), 0.0)
    b = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float(np.sqrt(np.mean((np.log1p(a) - np.log1p(b)) ** 2)))

def _compute_mphe(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(delta**2 * (np.sqrt(1.0 + (diff / delta) ** 2) - 1.0)))

def _rmsle_param_grid(n_jobs: int) -> list[dict[str, object]]:
    return [
        {
            "objective": "reg:squaredlogerror",
            "n_estimators": 250,
            "learning_rate": 0.04,
            "max_depth": 3,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "n_jobs": n_jobs,
        },
        {
            "objective": "reg:squaredlogerror",
            "n_estimators": 350,
            "learning_rate": 0.025,
            "max_depth": 4,
            "min_child_weight": 2,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "n_jobs": n_jobs,
        },
        {
            "objective": "reg:squaredlogerror",
            "n_estimators": 180,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_child_weight": 1,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "n_jobs": n_jobs,
        },
    ]

def _downsample_binary_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    max_negative_to_positive_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    y_train = y_train.astype(int)
    pos_idx = y_train.index[y_train.eq(1)].to_numpy()
    neg_idx = y_train.index[y_train.eq(0)].to_numpy()
    n_pos = int(len(pos_idx))
    n_neg = int(len(neg_idx))

    if n_pos == 0 or n_neg == 0 or max_negative_to_positive_ratio <= 0:
        return X_train, y_train, {
            "binary_downsampling": "none",
            "binary_downsample_neg_pos_ratio": float(max_negative_to_positive_ratio),
            "n_train_before_downsampling": int(len(y_train)),
            "n_train_positive_before_downsampling": n_pos,
            "n_train_negative_before_downsampling": n_neg,
            "n_train_positive": n_pos,
            "n_train_negative": n_neg,
        }

    max_neg = max(n_pos, int(math.ceil(n_pos * max_negative_to_positive_ratio)))
    if n_neg <= max_neg:
        return X_train, y_train, {
            "binary_downsampling": "not_needed",
            "binary_downsample_neg_pos_ratio": float(max_negative_to_positive_ratio),
            "n_train_before_downsampling": int(len(y_train)),
            "n_train_positive_before_downsampling": n_pos,
            "n_train_negative_before_downsampling": n_neg,
            "n_train_positive": n_pos,
            "n_train_negative": n_neg,
        }

    rng = np.random.default_rng(random_state)
    keep_neg = rng.choice(neg_idx, size=max_neg, replace=False)
    keep_idx = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep_idx)
    y_down = y_train.loc[keep_idx]
    return X_train.loc[keep_idx], y_down, {
        "binary_downsampling": "train_fold_majority_0_downsampled",
        "binary_downsample_neg_pos_ratio": float(max_negative_to_positive_ratio),
        "n_train_before_downsampling": int(len(y_train)),
        "n_train_positive_before_downsampling": n_pos,
        "n_train_negative_before_downsampling": n_neg,
        "n_train_positive": int(y_down.eq(1).sum()),
        "n_train_negative": int(y_down.eq(0).sum()),
    }

def _fit_binary(
    X: pd.DataFrame,
    y: pd.Series,
    meta: dict[str, list[str]],
    n_jobs: int,
    split_seed: int,
    negative_to_positive_ratio: float,
    selection_metric: str = "balanced_accuracy",
) -> dict[str, object]:
    if y.nunique(dropna=True) < 2:
        return {"error": "target has fewer than two classes"}

    train_idx, val_idx, test_idx = _split_indices(y, split_seed)
    X_model, prep_info = _prepare_model_matrix(X, meta, train_idx)
    X_train, X_val, X_test = X_model.iloc[train_idx], X_model.iloc[val_idx], X_model.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx].astype(int), y.iloc[val_idx].astype(int), y.iloc[test_idx].astype(int)

    X_train, y_train, downsample_info = _downsample_binary_training_data(
        X_train,
        y_train,
        max_negative_to_positive_ratio=negative_to_positive_ratio,
        random_state=split_seed + 1000,
    )
    counts = y_train.value_counts()
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    best: dict[str, object] | None = None
    for params in _binary_param_grid(scale_pos_weight, n_jobs):
        model = XGBClassifier(
            **_base_xgb_params(split_seed),
            **params,
            eval_metric="aucpr",
            objective="binary:logistic",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_score = model.predict_proba(X_val)[:, 1]
        threshold, val_bal = _best_threshold(y_val, val_score)
        val_ap = _safe_auc(average_precision_score, y_val, val_score)
        candidate = {
            "model": model,
            "threshold": threshold,
            "val_balanced_accuracy": val_bal,
            "val_au_prc": val_ap,
            "params": params,
        }
        if best is None:
            best = candidate
        elif selection_metric == "au_prc":
            better_ap = candidate["val_au_prc"] > best["val_au_prc"]
            better_bal = math.isclose(candidate["val_au_prc"], best["val_au_prc"]) and (
                candidate["val_balanced_accuracy"] > best["val_balanced_accuracy"]
            )
            if better_ap or better_bal:
                best = candidate
        else:
            better_bal = candidate["val_balanced_accuracy"] > best["val_balanced_accuracy"]
            better_ap = math.isclose(candidate["val_balanced_accuracy"], best["val_balanced_accuracy"]) and (
                candidate["val_au_prc"] > best["val_au_prc"]
            )
            if better_bal or better_ap:
                best = candidate

    assert best is not None
    model = best["model"]
    test_score = model.predict_proba(X_test)[:, 1]
    test_pred = (test_score >= float(best["threshold"])).astype(int)
    y_test_array = y_test.to_numpy(dtype=int)
    bal_acc = float(balanced_accuracy_score(y_test, test_pred))
    au_prc = _safe_auc(average_precision_score, y_test, test_score)
    test_positive_rate = float(y_test.mean())
    true_positive = int(((y_test_array == 1) & (test_pred == 1)).sum())
    false_negative = int(((y_test_array == 1) & (test_pred == 0)).sum())
    true_negative = int(((y_test_array == 0) & (test_pred == 0)).sum())
    false_positive = int(((y_test_array == 0) & (test_pred == 1)).sum())
    n_test_positive = true_positive + false_negative
    n_test_negative = true_negative + false_positive
    return {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "roc_auc": _safe_auc(roc_auc_score, y_test, test_score),
        "balanced_accuracy": bal_acc,
        "au_prc": au_prc,
        "au_prc_lift": float(au_prc / test_positive_rate) if test_positive_rate > 0 and not math.isnan(au_prc) else float("nan"),
        "threshold": float(best["threshold"]),
        "val_balanced_accuracy": float(best["val_balanced_accuracy"]),
        "val_au_prc": float(best["val_au_prc"]),
        "selection_metric": selection_metric,
        "scale_pos_weight": float(scale_pos_weight),
        "below_075_balanced_accuracy": bool(bal_acc < 0.75),
        "n_train": int(len(y_train)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_test_positive": n_test_positive,
        "n_test_negative": n_test_negative,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "sensitivity": float(true_positive / n_test_positive) if n_test_positive > 0 else float("nan"),
        "specificity": float(true_negative / n_test_negative) if n_test_negative > 0 else float("nan"),
        "false_negative_rate": float(false_negative / n_test_positive) if n_test_positive > 0 else float("nan"),
        "positive_rate": float(y.mean()),
        "test_positive_rate": test_positive_rate,
        "best_params": json.dumps(best["params"], sort_keys=True),
        **downsample_info,
        **prep_info,
    }

def _fit_continuous(
    X: pd.DataFrame,
    y: pd.Series,
    meta: dict[str, list[str]],
    n_jobs: int,
    split_seed: int,
    target: str = "",
    force_mphe: bool = False,
) -> dict[str, object]:
    use_rmsle = (target in RMSLE_TARGETS) and not force_mphe
    train_idx, val_idx, test_idx = _split_indices(y, split_seed)
    X_model, prep_info = _prepare_model_matrix(X, meta, train_idx)
    X_train, X_val, X_test = X_model.iloc[train_idx], X_model.iloc[val_idx], X_model.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx].astype(float), y.iloc[val_idx].astype(float), y.iloc[test_idx].astype(float)

    target_mean = float(y_train.mean(skipna=True))
    target_std = float(y_train.std(skipna=True, ddof=0))
    if not np.isfinite(target_std) or target_std <= 1e-12:
        target_std = 1.0
    y_val_z = (y_val - target_mean) / target_std
    y_test_z = (y_test - target_mean) / target_std

    zero_fraction = float((y_train <= 0).mean()) if y_train.min(skipna=True) >= 0 else 0.0
    positive_count = int((y_train > 0).sum())

    if use_rmsle:
        # Non-negative, heavily right-skewed targets: reg:squaredlogerror, report RMSLE
        y_train_nn = y_train.clip(lower=0)
        y_val_nn = y_val.clip(lower=0)

        best: dict[str, object] | None = None
        for params in _rmsle_param_grid(n_jobs):
            model_params = params.copy()
            objective = str(model_params.pop("objective"))
            model = XGBRegressor(
                **_base_xgb_params(split_seed),
                **model_params,
                eval_metric="rmsle",
                objective=objective,
            )
            model.fit(X_train, y_train_nn, eval_set=[(X_val, y_val_nn)], verbose=False)
            val_pred = np.maximum(model.predict(X_val), 0.0)
            val_rmsle = _compute_rmsle(y_val_nn.to_numpy(), val_pred)
            candidate = {
                "model": model,
                "model_family": "single_regressor",
                "target_transform": "none_rmsle",
                "val_rmsle": val_rmsle,
                "val_rmse_original_units": float(root_mean_squared_error(y_val, val_pred)),
                "params": {**params, "objective": objective},
            }
            if best is None or val_rmsle < best["val_rmsle"]:
                best = candidate

        if zero_fraction >= 0.40 and positive_count >= 20 and y_val.nunique(dropna=True) > 1:
            y_train_pos_flag = (y_train > 0).astype(int)
            y_val_pos_flag = (y_val > 0).astype(int)
            pos = int(y_train_pos_flag.sum())
            neg = int((y_train_pos_flag == 0).sum())
            scale_pos_weight = neg / pos if pos > 0 else 1.0
            classifier = XGBClassifier(
                **_base_xgb_params(split_seed),
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=1,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                max_delta_step=1,
                n_jobs=n_jobs,
                eval_metric="aucpr",
                objective="binary:logistic",
            )
            classifier.fit(X_train, y_train_pos_flag, eval_set=[(X_val, y_val_pos_flag)], verbose=False)
            val_positive_prob = classifier.predict_proba(X_val)[:, 1]

            X_train_pos = X_train.loc[y_train > 0]
            y_train_amount = y_train.loc[y_train > 0].clip(lower=0)

            for params in _rmsle_param_grid(n_jobs):
                model_params = params.copy()
                objective = str(model_params.pop("objective"))
                amount_model = XGBRegressor(
                    **_base_xgb_params(split_seed),
                    **model_params,
                    eval_metric="rmsle",
                    objective=objective,
                )
                amount_model.fit(X_train_pos, y_train_amount, verbose=False)
                val_amount = np.maximum(amount_model.predict(X_val), 0.0)
                val_pred = np.maximum(val_positive_prob * val_amount, 0.0)
                val_rmsle = _compute_rmsle(y_val_nn.to_numpy(), val_pred)
                candidate = {
                    "classifier": classifier,
                    "model": amount_model,
                    "model_family": "zero_inflated_hurdle",
                    "target_transform": "none_rmsle",
                    "val_rmsle": val_rmsle,
                    "val_rmse_original_units": float(root_mean_squared_error(y_val, val_pred)),
                    "params": {**params, "objective": objective, "positive_scale_pos_weight": scale_pos_weight},
                }
                if best is None or val_rmsle < best["val_rmsle"]:
                    best = candidate

        assert best is not None
        if best["model_family"] == "zero_inflated_hurdle":
            probability = best["classifier"].predict_proba(X_test)[:, 1]
            amount = np.maximum(best["model"].predict(X_test), 0.0)
            pred = np.maximum(probability * amount, 0.0)
        else:
            pred = np.maximum(best["model"].predict(X_test), 0.0)

        pred_z = (pred - target_mean) / target_std
        return {
            "rmsle": _compute_rmsle(y_test.to_numpy(), pred),
            "mphe": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "rmse_z": float(root_mean_squared_error(y_test_z, pred_z)),
            "mae_z": float(mean_absolute_error(y_test_z, pred_z)),
            "rmse_original_units": float(root_mean_squared_error(y_test, pred)),
            "mae_original_units": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "val_rmsle": float(best["val_rmsle"]),
            "val_rmse": float("nan"),
            "val_rmse_z": float("nan"),
            "val_rmse_original_units": float(best["val_rmse_original_units"]),
            "target_transform": str(best["target_transform"]),
            "continuous_target_scaling": "none",
            "model_family": str(best["model_family"]),
            "target_train_mean": target_mean,
            "target_train_std": target_std,
            "target_zero_fraction": zero_fraction,
            "target_group": "rmsle",
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "best_params": json.dumps(best["params"], sort_keys=True),
            **prep_info,
        }

    # MLPE path: z-score target, pseudohubererror + squarederror, report the
    # log-aware pseudo-Huber error stored in the legacy ``mphe`` column.
    target_transforms = ["zscore"]
    if y_train.min(skipna=True) >= 0 and y_train.skew(skipna=True) > 1.0:
        target_transforms.append("log1p_zscore")

    best: dict[str, object] | None = None
    for transform in target_transforms:
        if transform == "log1p_zscore":
            log_train = np.log1p(y_train)
            log_val = np.log1p(y_val)
            log_mean = float(log_train.mean(skipna=True))
            log_std = float(log_train.std(skipna=True, ddof=0))
            if not np.isfinite(log_std) or log_std <= 1e-12:
                log_std = 1.0
            train_target = (log_train - log_mean) / log_std
            eval_target = (log_val - log_mean) / log_std
        else:
            log_mean = float("nan")
            log_std = float("nan")
            train_target = (y_train - target_mean) / target_std
            eval_target = y_val_z

        for params in _regressor_param_grid(n_jobs):
            model_params = params.copy()
            objective = str(model_params.pop("objective"))
            model = XGBRegressor(
                **_base_xgb_params(split_seed),
                **model_params,
                eval_metric="rmse",
                objective=objective,
            )
            model.fit(X_train, train_target, eval_set=[(X_val, eval_target)], verbose=False)
            val_pred = model.predict(X_val)
            if transform == "log1p_zscore":
                val_pred_original = np.expm1((val_pred * log_std) + log_mean)
            else:
                val_pred_original = (val_pred * target_std) + target_mean
            if y_train.min(skipna=True) >= 0:
                val_pred_original = np.maximum(val_pred_original, 0)
            val_pred_z = (val_pred_original - target_mean) / target_std
            val_rmse_z = float(root_mean_squared_error(y_val_z, val_pred_z))
            candidate = {
                "model": model,
                "model_family": "single_regressor",
                "target_transform": transform,
                "log_mean": log_mean,
                "log_std": log_std,
                "val_rmse": val_rmse_z,
                "val_rmse_original_units": float(root_mean_squared_error(y_val, val_pred_original)),
                "params": {**params, "objective": objective},
            }
            if best is None or val_rmse_z < best["val_rmse"]:
                best = candidate

    if zero_fraction >= 0.40 and positive_count >= 20 and y_val.nunique(dropna=True) > 1:
        y_train_positive = (y_train > 0).astype(int)
        y_val_positive = (y_val > 0).astype(int)
        pos = int(y_train_positive.sum())
        neg = int((y_train_positive == 0).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        classifier = XGBClassifier(
            **_base_xgb_params(split_seed),
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=1,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            max_delta_step=1,
            n_jobs=n_jobs,
            eval_metric="aucpr",
            objective="binary:logistic",
        )
        classifier.fit(X_train, y_train_positive, eval_set=[(X_val, y_val_positive)], verbose=False)
        val_positive_probability = classifier.predict_proba(X_val)[:, 1]

        X_train_positive = X_train.loc[y_train > 0]
        y_train_amount = y_train.loc[y_train > 0]
        amount_log = np.log1p(y_train_amount)
        amount_log_mean = float(amount_log.mean(skipna=True))
        amount_log_std = float(amount_log.std(skipna=True, ddof=0))
        if not np.isfinite(amount_log_std) or amount_log_std <= 1e-12:
            amount_log_std = 1.0
        amount_target = (amount_log - amount_log_mean) / amount_log_std

        for params in _regressor_param_grid(n_jobs):
            model_params = params.copy()
            objective = str(model_params.pop("objective"))
            amount_model = XGBRegressor(
                **_base_xgb_params(split_seed),
                **model_params,
                eval_metric="rmse",
                objective=objective,
            )
            amount_model.fit(X_train_positive, amount_target, verbose=False)
            val_amount = np.expm1((amount_model.predict(X_val) * amount_log_std) + amount_log_mean)
            val_amount = np.maximum(val_amount, 0)
            val_pred_original = val_positive_probability * val_amount
            val_pred_z = (val_pred_original - target_mean) / target_std
            val_rmse_z = float(root_mean_squared_error(y_val_z, val_pred_z))
            candidate = {
                "classifier": classifier,
                "model": amount_model,
                "model_family": "zero_inflated_hurdle",
                "target_transform": "positive_log1p_zscore",
                "log_mean": amount_log_mean,
                "log_std": amount_log_std,
                "val_rmse": val_rmse_z,
                "val_rmse_original_units": float(root_mean_squared_error(y_val, val_pred_original)),
                "params": {**params, "objective": objective, "positive_scale_pos_weight": scale_pos_weight},
            }
            if best is None or val_rmse_z < best["val_rmse"]:
                best = candidate

    assert best is not None
    if best["model_family"] == "zero_inflated_hurdle":
        probability = best["classifier"].predict_proba(X_test)[:, 1]
        amount = np.expm1((best["model"].predict(X_test) * best["log_std"]) + best["log_mean"])
        pred = probability * np.maximum(amount, 0)
    else:
        pred_model_scale = best["model"].predict(X_test)
        if best["target_transform"] == "log1p_zscore":
            pred = np.expm1((pred_model_scale * best["log_std"]) + best["log_mean"])
        else:
            pred = (pred_model_scale * target_std) + target_mean
    if y_train.min(skipna=True) >= 0:
        pred = np.maximum(pred, 0)
    pred_z = (pred - target_mean) / target_std
    return {
        "rmsle": float("nan"),
        "mphe": _compute_mphe(y_test_z.to_numpy(), pred_z),
        "rmse": float(root_mean_squared_error(y_test_z, pred_z)),
        "mae": float(mean_absolute_error(y_test_z, pred_z)),
        "rmse_z": float(root_mean_squared_error(y_test_z, pred_z)),
        "mae_z": float(mean_absolute_error(y_test_z, pred_z)),
        "rmse_original_units": float(root_mean_squared_error(y_test, pred)),
        "mae_original_units": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "val_rmsle": float("nan"),
        "val_rmse": float(best["val_rmse"]),
        "val_rmse_z": float(best["val_rmse"]),
        "val_rmse_original_units": float(best["val_rmse_original_units"]),
        "target_transform": str(best["target_transform"]),
        "continuous_target_scaling": "train_fold_zscore",
        "model_family": str(best["model_family"]),
        "target_train_mean": target_mean,
        "target_train_std": target_std,
        "target_zero_fraction": zero_fraction,
        "target_group": "mphe",
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_params": json.dumps(best["params"], sort_keys=True),
        **prep_info,
    }