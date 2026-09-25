#!/usr/bin/env python
"""Generate two synthetic cohorts so the benchmark runs without real data.

This exists so the pipeline can be exercised end to end by someone who has not
yet wired up their own registry exports: to check the environment, to see the
shape of every artifact, and to read the figures and tables that come out.

It writes the artifacts of stage 01 rather than raw exports, so it starts the
pipeline at stage 02:

    data/clean_Cohort1.csv       predictors, with realistic missingness
    data/clean_Cohort2.csv
    data/targets_Cohort1.csv     record id + the endpoint columns
    data/targets_Cohort2.csv
    data/harmonise_report/codebook.json

**The numbers mean nothing.** Columns are drawn independently apart from a weak
signal injected into each endpoint, so every imputer is reconstructing noise and
every ranking you get from this data is noise too. It tells you the pipeline
runs. It tells you nothing about which imputer is better.

    python examples/make_synthetic_cohorts.py
    python examples/make_synthetic_cohorts.py --n1 2000 --n2 800

To run on real data instead, delete these files, point
``conf/harmonise.yaml:sources`` at your exports and start from stage 01.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvd import conf                        # noqa: E402

# Shapes chosen to look like a peri-operative registry: mostly binary and
# nominal fields, a minority continuous, and missingness concentrated in a
# subset of columns rather than spread evenly.
N_BINARY, N_NOMINAL, N_ORDINAL, N_CONTINUOUS, N_COUNT = 63, 81, 10, 22, 2
NOMINAL_LEVELS, ORDINAL_LEVELS = 4, 5


def _columns() -> tuple[list[str], dict[str, str], dict[str, list]]:
    kinds: dict[str, str] = {}
    levels: dict[str, list] = {}
    for i in range(N_BINARY):
        kinds[f"bin_{i:02d}"] = "binary"; levels[f"bin_{i:02d}"] = [0, 1]
    for i in range(N_NOMINAL):
        kinds[f"nom_{i:02d}"] = "nominal"
        levels[f"nom_{i:02d}"] = list(range(NOMINAL_LEVELS))
    for i in range(N_ORDINAL):
        kinds[f"ord_{i:02d}"] = "ordinal"
        levels[f"ord_{i:02d}"] = list(range(ORDINAL_LEVELS))
    for i in range(N_CONTINUOUS):
        kinds[f"cont_{i:02d}"] = "continuous"; levels[f"cont_{i:02d}"] = []
    for i in range(N_COUNT):
        kinds[f"cnt_{i:02d}"] = "count"; levels[f"cnt_{i:02d}"] = []
    return list(kinds), kinds, levels


def _cohort(n: int, seed: int, cols, kinds, binary_targets, continuous_targets):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(index=range(n))
    for c in cols:
        k = kinds[c]
        if k == "binary":
            frame[c] = rng.binomial(1, rng.uniform(0.05, 0.5), n)
        elif k == "nominal":
            frame[c] = rng.integers(0, NOMINAL_LEVELS, n)
        elif k == "ordinal":
            frame[c] = rng.integers(0, ORDINAL_LEVELS, n)
        elif k == "count":
            frame[c] = rng.poisson(1.5, n)
        else:
            frame[c] = rng.normal(0, 1, n)

    # A weak, genuine signal, so the downstream models are not fitting pure
    # noise and the prediction stage produces something other than chance.
    signal = (frame["cont_00"] * 0.6 + frame["cont_01"] * 0.4
              + frame["bin_00"] * 0.8 + rng.normal(0, 1.0, n))

    targets = pd.DataFrame(index=frame.index)
    for i, t in enumerate(binary_targets):
        p = 1 / (1 + np.exp(-(signal + rng.normal(0, 0.5, n) - 1.0 - 0.3 * i)))
        targets[t] = rng.binomial(1, p)
    for i, t in enumerate(continuous_targets):
        # right-skewed, like a duration
        targets[t] = np.exp(0.5 * signal + rng.normal(0, 0.6, n) + 2.5 + i)

    # Missingness: two thirds of columns complete, the rest missing at rates
    # from a few per cent to a third. The benchmark's holdout budget is
    # proportional to this, so a cohort with no native missingness would
    # produce no held-out cells and nothing to score.
    for c in cols:
        if rng.random() < 0.62:
            continue
        frac = rng.uniform(0.02, 0.33)
        frame.loc[rng.random(n) < frac, c] = np.nan

    # A little missingness in the outcomes too, which the pipeline drops
    # per-endpoint rather than imputing.
    for t in targets.columns:
        targets.loc[rng.random(n) < 0.02, t] = np.nan
    return frame, targets


def main(argv: list[str] | None = None) -> int:
    cfg = conf.active()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n1", type=int, default=3000, help="records in Cohort 1")
    ap.add_argument("--n2", type=int, default=1200, help="records in Cohort 2")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args(argv)

    id_col = str(cfg["harmonise.id_column"])
    # `harmonise.targets` is the ANALYSED set -- what stage 05 iterates.
    # `variables.targets` is a superset that also names endpoints held out of
    # the analysis, which still have to be absent from the predictor matrix.
    binary_targets = [str(t) for t in cfg["harmonise.targets.categorical"]]
    continuous_targets = [str(t) for t in cfg["harmonise.targets.continuous"]]
    cols, kinds, levels = _columns()

    report = Path(str(cfg["harmonise.report_dir"]))
    report.mkdir(parents=True, exist_ok=True)
    (report / "codebook.json").write_text(
        json.dumps({"kinds": kinds, "levels": levels}, indent=1))

    # Stage 01 writes a second codebook for the endpoints, which the endpoint
    # figures read to label the classes. Same shape as the predictor one.
    target_kinds = {t: "binary" for t in binary_targets}
    target_levels = {t: ["No", "Yes"] for t in binary_targets}
    target_kinds.update({t: "continuous" for t in continuous_targets})
    target_levels.update({t: [] for t in continuous_targets})
    (report / "targets_codebook.json").write_text(
        json.dumps({"kinds": target_kinds, "levels": target_levels}, indent=1))

    for idx, (cohort, n) in enumerate(
            zip(cfg["harmonise.outputs"], (args.n1, args.n2))):
        frame, targets = _cohort(n, args.seed + idx, cols, kinds,
                                 binary_targets, continuous_targets)
        frame.insert(0, id_col, [f"{cohort}-{i:06d}" for i in range(n)])
        targets.insert(0, id_col, frame[id_col].to_numpy())

        out = Path(str(cfg[f"harmonise.outputs.{cohort}"]))
        tgt = Path(str(cfg[f"harmonise.target_outputs.{cohort}"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        tgt.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        targets.to_csv(tgt, index=False)
        print(f"{cohort}: {frame.shape[0]} records x {frame.shape[1] - 1} predictors "
              f"-> {out}")
        print(f"{cohort}: {targets.shape[1] - 1} endpoints -> {tgt}")

    print(f"\ncodebook -> {report / 'codebook.json'}")
    print("\nSynthetic data. Rankings computed from it are noise; it is here to "
          "prove the pipeline runs.\nNext: python pipeline/02_build_masks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
