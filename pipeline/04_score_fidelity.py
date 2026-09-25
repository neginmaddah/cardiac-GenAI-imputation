#!/usr/bin/env python
"""Score an imputed matrix that already exists, without re-imputing it.

`cvd.cluster.worker_v2` skips a unit only when BOTH its matrix and its fidelity
JSON are present, so a matrix arriving on its own -- imputed on a laptop and
copied back -- would otherwise be recomputed from scratch. This scores whatever
is on disk and leaves everything else alone.

    python pipeline/04_score_fidelity.py                 # every unscored matrix
    python pipeline/04_score_fidelity.py --methods remasker
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvd import conf  # noqa: E402
from cvd.cluster.worker_v2 import fidelity_path, load_prepared  # noqa: E402
from cvd.conf import ensemble_size, method_params  # noqa: E402
from cvd.imputation.run_v2 import col_types_3way  # noqa: E402
from cvd.stages.fidelity import score_unit  # noqa: E402
from cvd.stages.prediction_v2 import parse_stem  # noqa: E402

DESIGN_TO_KEY = {"MCAR": "mcar", "MAR": "mar", "MNAR": "mnar"}


def main(argv=None) -> int:
    cfg = conf.load()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="*")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    scored = skipped = 0
    for path in sorted(Path(str(cfg["results_v2.imputed_dir"])).glob("*.csv")):
        info = parse_stem(path.stem)
        if not info or (a.methods and info["method"] not in a.methods):
            continue
        tag = f"{info['cohort']}_{info['design']}_{info['alpha']}"
        dest = fidelity_path(cfg, tag, info["method"])
        if dest.exists() and not a.force:
            skipped += 1
            continue

        alpha = int(info["alpha"]) / 100
        unit = load_prepared(cfg, info["cohort"], DESIGN_TO_KEY[info["design"]], alpha)
        result = pd.read_csv(path)
        if list(result.columns) != list(unit.canonical_clean.columns):
            raise ValueError(f"{path.name}: columns differ from the ground truth")
        # Row order carries the join: the matrices are positional against the
        # status mask, so a reordered file would score the wrong cells.
        if not (result[unit.id_column].astype(str).to_numpy()
                == unit.canonical_clean[unit.id_column].astype(str).to_numpy()).all():
            raise ValueError(f"{path.name}: identifier order differs from the ground truth")

        status = pd.read_parquet(
            Path(str(cfg["masking.status_dir"]))
            / str(cfg["masking.filenames"]["status"]).format(tag=tag)
        )
        metrics = score_unit(cfg, result, unit.canonical_clean, status, col_types_3way(unit))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            "cohort": info["cohort"], "design": DESIGN_TO_KEY[info["design"]],
            "alpha": alpha, "method": info["method"], "experiment": "v2",
            "n_ensemble": ensemble_size(cfg, "v2", info["method"]),
            "params": method_params(cfg, "v2", info["method"]),
            "seconds": None,           # imputed elsewhere; runtime is not ours
            "scored_from_existing_matrix": True,
            "kinds": {c: unit.kinds[c] for c in unit.features},
            "metrics": metrics,
        }, indent=2))
        scored += 1
        print(f"  scored {tag}_{info['method']}")
    print(f"scoring: {scored} written, {skipped} already had fidelity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
