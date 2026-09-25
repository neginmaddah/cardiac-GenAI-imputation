#!/usr/bin/env python
"""Ingest MICE members produced in RStudio into the v2 grid.

`bridge.R` emits ONE member per seed. The cluster path then combines them,
snaps categoricals to valid levels and scores at the held-out cells -- all
Python side. This does exactly those steps, using the same functions, so a
laptop-produced unit is indistinguishable from a cluster-produced one.

Combination is `mixed_vote` (conf/imputation/_base.yaml): mean for continuous,
majority vote for categorical, ties to the first member so the result is
deterministic given the seeds. That is NOT the same as the generative family's
`raw_mean`, and using the wrong one would quietly bias the categoricals.

    python pipeline/03c_ingest_r_mice.py --src results_v2/inputs/imputed_Rstudio
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvd import conf  # noqa: E402
from cvd.cluster.worker_v2 import fidelity_path, imputed_path, load_prepared  # noqa: E402
from cvd.conf import ensemble_size, method_params  # noqa: E402
from cvd.imputation.ensemble import combine_mixed_vote  # noqa: E402
from cvd.imputation.run_v2 import _snap_to_canonical, col_types_3way  # noqa: E402
from cvd.stages.fidelity import score_unit  # noqa: E402

DESIGN_KEY = {"MCAR": "mcar", "MAR": "mar", "MNAR": "mnar"}


def main(argv=None) -> int:
    cfg = conf.load()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="results_v2/inputs/imputed_Rstudio")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    src = Path(a.src)
    members: dict[str, dict[int, Path]] = collections.defaultdict(dict)
    for f in sorted(src.glob("*_mice_seed*.csv")):
        tag, seed = f.stem.rsplit("_mice_seed", 1)
        members[tag][int(seed)] = f

    k = ensemble_size(cfg, "v2", "mice")
    wanted = set(range(k))
    done = incomplete = 0
    for tag, by_seed in sorted(members.items()):
        if set(by_seed) < wanted:
            print(f"  SKIP {tag}: has seeds {sorted(by_seed)}, needs {sorted(wanted)}")
            incomplete += 1
            continue
        out_path = imputed_path(cfg, tag, "mice")
        if out_path.exists() and not a.force:
            print(f"  have {tag}")
            continue

        cohort, design, pct = tag.split("_")
        unit = load_prepared(cfg, cohort, DESIGN_KEY[design], int(pct) / 100)
        features = unit.features
        frames = []
        for seed in sorted(wanted):
            df = pd.read_csv(by_seed[seed])
            if list(df.columns) != features:
                raise ValueError(f"{by_seed[seed].name}: columns differ from the schema")
            if len(df) != len(unit.canonical_clean):
                raise ValueError(f"{by_seed[seed].name}: {len(df)} rows, expected "
                                 f"{len(unit.canonical_clean)}")
            frames.append(df)

        ct = col_types_3way(unit)
        combined = combine_mixed_vote(frames, ct)
        combined = _snap_to_canonical(combined, unit, snap_numeric_codes=True)
        combined.insert(0, unit.id_column,
                        unit.canonical_masked[unit.id_column].to_numpy())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)

        status = pd.read_parquet(
            Path(str(cfg["masking.status_dir"]))
            / str(cfg["masking.filenames"]["status"]).format(tag=tag)
        )
        metrics = score_unit(cfg, combined, unit.canonical_clean, status, ct)
        dest = fidelity_path(cfg, tag, "mice")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            "cohort": cohort, "design": DESIGN_KEY[design], "alpha": int(pct) / 100,
            "method": "mice", "experiment": "v2", "n_ensemble": k,
            "params": method_params(cfg, "v2", "mice"),
            "seconds": None, "imputed_externally": "RStudio",
            "kinds": {c: unit.kinds[c] for c in features},
            "metrics": metrics,
        }, indent=2))
        done += 1
        print(f"  ingested {tag}  ({len(frames)} members combined)")
    print(f"ingest: {done} units, {incomplete} incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
