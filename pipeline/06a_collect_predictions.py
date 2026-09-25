#!/usr/bin/env python
"""Flatten results_v2/prediction/*.json into the flat CSVs the generators read.

Every figure and table generator downstream expects one row per
(cohort, target, intensity, method, split) in the schema the v1 pipeline wrote.
The v2 runner emits one JSON per cell with the three splits nested under
`runs[]`, so this explodes them and adds the identity columns.

`design` is a NEW column: v1 had one mechanism per output directory, v2 carries
MCAR and MNAR in one table, so anything that pools must group on it explicitly
rather than assume a single design.

    python pipeline/06a_collect_predictions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvd import conf  # noqa: E402

ID_COLS = ["cohort", "design", "target", "target_type", "intensity",
           "method", "method_key", "split_repeat", "split_seed", "status"]


def main() -> int:
    cfg = conf.load()
    labels = {m: str(cfg[f"imputation.registry.{m}.label"])
              for m in cfg["imputation.order"]}
    binary = {str(t) for t in cfg["harmonise.targets.categorical"]}
    continuous = {str(t) for t in cfg["harmonise.targets.continuous"]}
    # Filter to the declared endpoint set. Retired endpoints still have result
    # JSONs on disk; without this they would be swept in and silently typed as
    # continuous, because target_type below is a two-way split on `binary`.
    analysed = binary | continuous

    rows: list[dict] = []
    for path in sorted(Path(str(cfg["results_v2.prediction_dir"])).glob("*.json")):
        b = json.loads(path.read_text())
        if "runs" not in b:
            continue
        if b.get("target") not in analysed:
            continue
        for i, r in enumerate(b["runs"]):
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            row.update({
                "cohort": b["cohort"],
                "design": str(b["design"]).upper(),
                "target": b["target"],
                "target_type": "binary" if b["target"] in binary else "continuous",
                "intensity": int(round(float(b["alpha"]) * 100)),
                "method": labels.get(b["method"], b["method"]),
                "method_key": b["method"],
                "split_repeat": i + 1,
                "split_seed": r.get("seed"),
                "status": "ok" if "error" not in r else "error",
                "n_rows": b.get("n_rows"),
                "n_features": b.get("n_features"),
            })
            rows.append(row)

    if not rows:
        raise SystemExit("no prediction cells found")
    df = pd.DataFrame(rows)
    out_dir = Path(str(cfg["masking.output_root"])) / "prediction_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in ("binary", "continuous"):
        sub = df.loc[df["target_type"] == kind].copy()
        lead = [c for c in ID_COLS if c in sub.columns]
        sub = sub[lead + [c for c in sub.columns if c not in lead]]
        dest = out_dir / f"{kind}_prediction_results.csv"
        sub.to_csv(dest, index=False)
        dup = sub.duplicated(
            ["cohort", "design", "target", "intensity", "method", "split_seed"]
        ).sum()
        if dup:
            raise AssertionError(f"{dest.name}: {dup} duplicated cells")
        print(f"  {dest.name:36s} {len(sub):5d} rows  "
              f"{sub['target'].nunique()} targets x {sub['method'].nunique()} arms "
              f"x {sub['design'].nunique()} designs x {sub['intensity'].nunique()} "
              f"intensities x {sub['split_seed'].nunique()} splits")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
