#!/usr/bin/env python
"""Re-render Figures 2 and 3 -- the per-cohort fidelity panels -- from v2.

Figure 2 is Cohort1, Figure 3 is Cohort2: categorical balanced accuracy and
continuous RMSE per imputer, with a within-cohort Friedman ranking row under
each panel.

The default pools the final two mechanisms and two intensities and saves the
long-form values, descriptive summaries, and pooled/within-cohort rankings.
Single-mechanism diagnostic outputs receive suffixes, leaving those canonical
tables and figures intact. Existing plotting helpers supply the artwork.

Scores must represent original variables after decoding, with matching
feature/metric sets for every method within a unit. The number of evaluated
features varies across units because some columns have no held-out cells.

    python pipeline/07a_fidelity_figures.py                  # final MCAR/MNAR grid
    python pipeline/07a_fidelity_figures.py --designs mnar
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvd import conf  # noqa: E402
from cvd.reporting.figures.fidelity import (  # noqa: E402
    plot_cohort_fidelity_figure,
    run_overall_fidelity_friedman_tests,
    summarize_fidelity,
)
from cvd.reporting.figures.selection import selected  # noqa: E402

METRIC_TO_TYPE = {"balanced_acc": ("Categorical", "Balanced accuracy"),
                  "rmse": ("Continuous", "RMSE")}


def build_values(cfg, designs: set[str]) -> pd.DataFrame:
    labels = {m: str(cfg[f"imputation.registry.{m}.label"])
              for m in cfg["imputation.order"] if m != "baseline"}
    rows = []
    # Scored-variable count per (unit, method). It varies BETWEEN units -- a 5%
    # unit has fewer columns with any held-out cell than a 20% one -- so the
    # invariant is that every method scored the same set WITHIN a unit.
    per_unit: dict[tuple, dict[str, set[tuple[str, str]]]] = {}
    cohorts = set(cfg["experiment.v2.cohorts"])
    intensities = {int(round(float(a) * 100)) for a in cfg["experiment.v2.alphas"]}
    for path in sorted(Path(str(cfg["results_v2.fidelity_dir"])).glob("*.json")):
        b = json.loads(path.read_text())
        design = b["design"].lower()
        if b["method"] not in labels or design not in designs or b["cohort"] not in cohorts:
            continue
        pct = int(round(b["alpha"] * 100))
        if pct not in intensities:
            continue
        expected_k = 1 if b["method"] in cfg["imputation.deterministic"] else 2
        if b.get("n_ensemble") != expected_k:
            raise AssertionError(f"{path.name}: expected ensemble size {expected_k}")
        sim = f"{b['design'].upper()}{pct}"
        scored = set()
        for metric_key, (var_type, metric_label) in METRIC_TO_TYPE.items():
            for feature, value in b["metrics"].get(metric_key, {}).items():
                if value is None or pd.isna(value):
                    continue
                if not math.isfinite(float(value)):
                    raise AssertionError(f"{path.name}: non-finite {metric_key}/{feature}")
                scored.add((metric_key, feature))
                rows.append({
                    "cohort": b["cohort"], "variable_type": var_type,
                    "design": design, "intensity": pct,
                    "metric": metric_label, "simulation": sim,
                    "simulation_label": f"{b['design'].upper()} {pct}%",
                    "method": labels[b["method"]], "feature": feature,
                    "value": float(value), "source_file": path.name,
                })
        by_method = per_unit.setdefault((b["cohort"], design, pct), {})
        if b["method"] in by_method:
            raise AssertionError(f"{path.name}: duplicate unit/method")
        by_method[b["method"]] = scored
        # A one-hot dummy reaching this point would mean post-processing did not
        # invert the encoding, and the generative arms would be scored on a
        # different -- and much wider -- feature set than the classic ones.
        bad = [f for _, f in scored if "__" in f]
        if bad:
            raise AssertionError(f"{path.name}: one-hot columns in the metrics: {bad[:5]}")
    expected_units = set(product(cohorts, designs, intensities))
    if set(per_unit) != expected_units:
        raise AssertionError(f"Missing fidelity units: {sorted(expected_units - set(per_unit))}")
    for unit, by_method in sorted(per_unit.items()):
        if set(by_method) != set(labels):
            raise AssertionError(f"{unit}: missing methods {sorted(set(labels) - set(by_method))}")
        scored_sets = {frozenset(scored) for scored in by_method.values()}
        if len(scored_sets) != 1:
            raise AssertionError(
                f"{unit}: arms scored different feature/metric sets"
            )
    spread = sorted({len(scored) for d in per_unit.values() for scored in d.values()})
    print(f"  {len(per_unit)} units; within each, every arm scored the same "
          f"variables (counts across units: {spread[0]}-{spread[-1]}, "
          f"higher intensities touch more columns)")
    return pd.DataFrame(rows)


def write_ranking_table(values: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Save the within-cohort tests plus every method's median and IQR.

    There is deliberately no pooled two-cohort row. Fidelity is ranked inside a
    cohort and the two registries are then compared by whether their rankings
    agree; merging their blocks into one Friedman test answers a question the
    study does not ask, and a pooled mean rank sitting in this file is an
    invitation to quote it.
    """
    frames = []
    for cohorts in (("Cohort1",), ("Cohort2",)):
        frame = run_overall_fidelity_friedman_tests(values, cohorts=cohorts)
        if not frame["status"].eq("ok").all():
            raise AssertionError(f"Invalid Friedman result for {cohorts}")
        frame.insert(0, "figure", {"Cohort1": "Figure 2", "Cohort2": "Figure 3"}[cohorts[0]])
        frame.insert(1, "designs", ";".join(sorted(values["design"].unique())))
        frame.insert(2, "intensities", ";".join(map(str, sorted(values["intensity"].unique()))))
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table


def main(argv=None) -> int:
    cfg = conf.load()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--designs", nargs="+", choices=["mcar", "mnar"],
                   default=list(cfg["experiment.v2.designs"]),
                   help="mechanisms to pool (default: the final MCAR/MNAR grid)")
    p.add_argument("--out", default=str(cfg["paths.paper.figures_v2_dir"]))
    a = p.parse_args(argv)

    designs = {d.lower() for d in a.designs}
    values = build_values(cfg, designs)
    if values.empty:
        raise SystemExit(f"no fidelity results for designs {sorted(designs)}")
    print(f"  pooling {sorted(values['simulation'].unique())} "
          f"over {values['method'].nunique()} imputers")

    summary = summarize_fidelity(values)
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = designs == set(cfg["experiment.v2.designs"])
    suffix = "" if canonical else "_" + "_".join(sorted(designs)).upper()
    tables_dir = Path(str(cfg["results_v2.fidelity_dir"])).parent / "fidelity_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(tables_dir / f"fidelity_long{suffix}.csv", index=False)
    summary.to_csv(tables_dir / f"fidelity_summary{suffix}.csv", index=False)
    rankings = write_ranking_table(values, out_dir / f"fidelity_rankings{suffix}.csv")

    for fig_no, cohort in ((2, "Cohort1"), (3, "Cohort2")):
        if f"Figure_{fig_no}" not in selected({"Figure_2", "Figure_3"}):
            continue
        ranking = rankings.loc[rankings["cohorts"] == cohort]
        dest = out_dir / f"Figure_{fig_no}{suffix}.png"
        plot_cohort_fidelity_figure(summary, cohort, dest, ranking_df=ranking)
        print(f"\nFigure {fig_no} -- {cohort} -> {dest}")
        for _, r in ranking.iterrows():
            print(f"    {r['variable_type']:12s} chi2={r['friedman_chi2']:8.2f} "
                  f"p={r['friedman_p']:.3g}  blocks={r['n_complete_blocks']}")
            print(f"      best median: {r['best_median_method']} "
                  f"({r['best_median_value']:.3f})   "
                  f"best mean rank: {r['best_mean_rank_method']} "
                  f"({r['best_mean_rank']:.2f})")
        sub = summary.loc[summary["cohort"] == cohort]
        for vt in ("Categorical", "Continuous"):
            s = sub.loc[sub["variable_type"] == vt].sort_values(
                "median", ascending=(vt == "Continuous"))
            best = ", ".join(f"{r['method']} {r['median']:.3f}"
                             for _, r in s.head(3).iterrows())
            print(f"      best {vt.lower()}: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
