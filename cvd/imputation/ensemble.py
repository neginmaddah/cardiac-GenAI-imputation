"""Combining the K seeded members of a stochastic imputer.

Paper section: Methods - "Imputation of Masked Data", the paragraph beginning
"Two configuration settings differ between the two benchmark families".

There are two combination modes, and which one applies is a property of what
the imputer *returns*, not a tuning choice:

``raw_mean``
    For the four generative methods, which emit numeric matrices where
    multiclass variables are still one-hot blocks. Averaging in the model's own
    output space turns each dummy block into soft class scores, which
    :func:`cvd.prep.postprocess.postprocess_imputed` then argmaxes. That is a
    probability-level vote, not a post-hoc majority vote.

``mixed_vote``
    For MICE and MissRanger, which return frames whose categoricals are already
    snapped to labels and so cannot be averaged the same way: mean for
    continuous columns, majority vote for binary and multiclass. Ties resolve to
    the first member, which keeps the result deterministic given the seeds.

Two things worth keeping in mind:

* Ensembling is applied to **every** stochastic imputer, not to ReMasker alone.
  Ranking an ensembled method against single-draw baselines would not be a
  like-for-like comparison, and the gain is method-dependent: at K=2 the
  accuracy penalty is ~0.8% for GAIN and 2.4% for ReMasker but 8.2% for MICE
  and 12.8% for MissRanger.
* Ensembling **shrinks the imputed distribution**. Methods discloses this.

Before this refactor these two modes lived in different files
(``genai._ensemble`` and ``run_pipeline._ensemble_mixed``) with no shared
contract, so which one an imputer got depended on which call path reached it.
"""

from __future__ import annotations

import logging
from typing import Callable, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["combine_raw_mean", "combine_mixed_vote", "ensemble"]

RunOne = Callable[[int], pd.DataFrame]


def combine_raw_mean(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Element-wise mean in the model's own numeric output space."""
    if len(frames) == 1:
        return frames[0]
    total = frames[0].to_numpy(dtype=float).copy()
    for frame in frames[1:]:
        total += frame.to_numpy(dtype=float)
    total /= len(frames)
    return pd.DataFrame(total, columns=frames[0].columns, index=frames[0].index)


def combine_mixed_vote(
    frames: Sequence[pd.DataFrame],
    col_types: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Mean for continuous columns, majority vote for categorical ones."""
    if len(frames) == 1:
        return frames[0]

    categorical = set(col_types["binary"]) | set(col_types["multiclass"])
    continuous = set(col_types["continuous"])
    out = frames[0].copy()
    for col in out.columns:
        values = [f[col] for f in frames]
        if col in categorical:
            stacked = pd.concat(values, axis=1)
            # mode(axis=1) returns candidates in sorted order; taking column 0
            # resolves ties to the lowest label, deterministically.
            out[col] = (
                stacked.mode(axis=1, dropna=False).iloc[:, 0].astype(out[col].dtype)
            )
        elif col in continuous:
            out[col] = pd.concat(values, axis=1).mean(axis=1)
        # Columns in neither group (e.g. a re-attached identifier) pass through
        # from the first member unchanged.
    return out


def ensemble(
    run_one: RunOne,
    *,
    k: int,
    method: str,
    mode: str,
    col_types: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Run ``run_one(seed)`` for ``k`` seeds and combine the members.

    Seeds are ``0 .. k-1``, so a run is reproducible from ``k`` alone.

    Parameters
    ----------
    run_one
        Called once per member with the member's seed.
    k
        Number of members. ``k <= 1`` short-circuits to a single call, which is
        what deterministic imputers always get.
    method
        Name, for logging.
    mode
        ``"raw_mean"`` or ``"mixed_vote"``; see the module docstring.
    col_types
        Required for ``mixed_vote``.
    """
    if k <= 1:
        return run_one(0)

    if mode == "mixed_vote" and col_types is None:
        raise ValueError("mixed_vote combination requires col_types")
    if mode not in ("raw_mean", "mixed_vote"):
        raise ValueError(f"unknown combination mode {mode!r}")

    frames = []
    for i, seed in enumerate(range(k), start=1):
        logger.info("[%s] ensemble member %d/%d (seed=%d)", method, i, k, seed)
        frames.append(run_one(seed))

    combined = (
        combine_raw_mean(frames)
        if mode == "raw_mean"
        else combine_mixed_vote(frames, col_types)
    )
    logger.info("[%s] combined %d members via %s", method, k, mode)
    return combined
