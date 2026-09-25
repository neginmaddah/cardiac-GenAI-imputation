#!/usr/bin/env python
"""Canonical prediction ranking table: the four omnibus families, one place.

Every consumer -- Figures 5/6/8/9, the supplementary ranking section and the
Results prose -- reads this file rather than recomputing, so the figure caption
and the table can never disagree.

One family is one (cohort, endpoint type). Each endpoint type is compared on a
SINGLE metric: balanced accuracy for the binary endpoints, standardized RMSE for
the continuous ones. The second metric each type used to carry is not a second
result. On the continuous side $R^2$ is a strictly decreasing function of
standardized RMSE within a block, so it returns the identical rank matrix and
the identical Friedman statistic -- reporting both invited the two columns to be
read as corroboration when they are one test printed twice. On the binary side
AU-PRC is a genuinely different ordering, but ranking on two metrics at once
leaves no defined answer when they disagree, which they do. Both remaining
metrics are reported descriptively (Supplementary Tables S3, S12); neither is
ranked or tested.

Blocks are design x intensity x endpoint x split, matching what the boxes pool.

The omnibus gate is applied to the HOLM-CORRECTED p-value, not the raw one.
This is why the pass structure below is explicit rather than a `compare_methods`
call: that helper decides `reject` from the uncorrected p and so cannot see the
family it belongs to. Holm has to run across all four omnibus tests before any
pairwise family may be opened, so the omnibus pass completes first and the
pairwise pass runs only for the survivors. Pairwise Wilcoxon inside a family
then carries its own Holm correction over that family's 36 pairs.

    python pipeline/06b_rank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvd import conf  # noqa: E402
from cvd.stats.tests import (  # noqa: E402
    friedman_test, holm_bonferroni, paired_matrix, pairwise_wilcoxon,
)

BLOCKS = ["design", "intensity", "target", "split_seed"]
# (endpoint type, metric, direction, label, expected complete blocks)
FAMILIES = [
    ("binary", "balanced_accuracy", "higher_is_better", "Balanced accuracy", 36),
    ("continuous", "rmse_z", "lower_is_better", "Standardized RMSE", 24),
]
ALPHA = 0.05


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = conf.load()
    tables = Path(str(cfg["masking.output_root"])) / "prediction_tables"

    # ── pass 1: every omnibus, before any pairwise family is opened ─────────
    held = []
    for fam, metric, direction, label, expected in FAMILIES:
        df = pd.read_csv(tables / f"{fam}_prediction_results.csv", low_memory=False)
        df = df[df.status == "ok"]
        for cohort in ("Cohort1", "Cohort2"):
            sub = df[df.cohort == cohort]
            matrix, n_total = paired_matrix(
                sub, block_keys=BLOCKS, arm_key="method", value_key=metric,
            )
            if len(matrix) != expected:
                raise AssertionError(
                    f"expected {expected} complete blocks for {cohort} {fam} "
                    f"{metric!r}, got {len(matrix)} (of {n_total}). A changed "
                    f"arm or target list is the usual cause."
                )
            res = friedman_test(
                matrix, metric=metric, direction=direction,
                n_blocks_total=n_total, alpha=ALPHA,
            )
            order = pd.Series(res.mean_ranks).sort_values()
            held.append({
                "row": {
                    "endpoint_type": fam, "cohort": cohort, "metric": metric,
                    "metric_label": label, "n_endpoints": sub.target.nunique(),
                    "n_arms": len(res.arms), "n_complete_blocks": res.n_complete_blocks,
                    "friedman_chi2": res.chi2, "friedman_p": res.p_value,
                    "rank1": order.index[0], "rank1_mean_rank": order.iloc[0],
                    "rank2": order.index[1], "rank2_mean_rank": order.iloc[1],
                    "rank_last": order.index[-1], "rank_last_mean_rank": order.iloc[-1],
                    "rank_spread": order.iloc[-1] - order.iloc[0],
                    "mean_ranks": ";".join(f"{k}={v:.4g}" for k, v in order.items()),
                },
                "matrix": matrix, "direction": direction,
            })

    out = pd.DataFrame([h["row"] for h in held])
    out["friedman_p_holm"] = holm_bonferroni(out.friedman_p.tolist())
    out["omnibus_significant"] = out.friedman_p_holm < ALPHA

    # ── pass 2: pairwise, only where the CORRECTED omnibus rejects ──────────
    tested, significant, pair_rows = [], [], []
    for h, (_, r) in zip(held, out.iterrows()):
        if not bool(r.omnibus_significant):
            tested.append(0)
            significant.append(0)
            continue
        pw = pairwise_wilcoxon(h["matrix"], direction=h["direction"], alpha=ALPHA)
        tested.append(len(pw))
        significant.append(int(pw.reject_holm.sum()) if len(pw) else 0)
        keep = pw[pw.reject_holm].copy()
        if len(keep):
            keep.insert(0, "cohort", r.cohort)
            keep.insert(1, "metric", r.metric)
            keep.insert(2, "endpoint_type", r.endpoint_type)
            pair_rows.append(keep)
    out["n_pairwise_tested"] = tested
    out["n_pairwise_significant"] = significant

    pairs = (pd.concat(pair_rows, ignore_index=True) if pair_rows
             else pd.DataFrame(columns=["cohort", "metric", "endpoint_type"]))
    return out, pairs


def main() -> int:
    cfg = conf.load()
    dest = Path(str(cfg["masking.output_root"])) / "prediction_tables"
    out, pairs = build()
    out.to_csv(dest / "prediction_rankings.csv", index=False)
    pairs.to_csv(dest / "prediction_pairwise_significant.csv", index=False)
    cols = ["cohort", "endpoint_type", "metric", "n_complete_blocks", "friedman_chi2",
            "friedman_p", "friedman_p_holm", "omnibus_significant", "rank1",
            "rank1_mean_rank", "rank_last", "rank_last_mean_rank",
            "n_pairwise_tested", "n_pairwise_significant"]
    pd.set_option("display.width", 250)
    print(out[cols].to_string(index=False))
    print(f"\n-> {dest/'prediction_rankings.csv'}")
    print(f"-> {dest/'prediction_pairwise_significant.csv'}  ({len(pairs)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
