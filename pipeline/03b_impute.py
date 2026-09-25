#!/usr/bin/env python
"""Stage 03b -- run the imputation grid.

Walks every (cohort, mechanism, intensity, method) unit declared in
``conf/experiment/v2.yaml`` and imputes it, scoring reconstruction fidelity at
the held-out cells as it goes. Both outputs are written per unit, so the grid
is resumable: a unit whose imputed matrix and fidelity record both exist is
skipped unless ``--force``.

The grid is embarrassingly parallel over units. Run this command several times
concurrently and each process takes units the others have not claimed; the
claim is an atomic file create, so no two processes duplicate a unit. There is
nothing scheduler-specific about it -- several shells, several containers or
several nodes all work the same way.

The four generative imputers want a GPU and are the great majority of the cost;
the four classical ones and the baseline arm are CPU-only and quick. Use
``--methods`` to split the grid along that line:

    python pipeline/03b_impute.py --methods baseline mean knn mice missranger
    python pipeline/03b_impute.py --methods miwae gain remasker miracle

MICE and missRanger call R through ``cvd/imputation/bridge.R``. If R is not
available, run the grid without them and the rest of the benchmark still works;
they simply do not appear in the rankings.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvd import conf                                  # noqa: E402
from cvd.imputation import unit_runner                # noqa: E402
from cvd.stages.masking import tag_for                # noqa: E402
from cvd.util.lease import Claim                      # noqa: E402

logger = logging.getLogger("impute")


def main(argv: list[str] | None = None) -> int:
    cfg = conf.active()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohorts", nargs="+", default=list(cfg["experiment.v2.cohorts"]))
    ap.add_argument("--designs", nargs="+", default=list(cfg["experiment.v2.designs"]))
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[float(a) for a in cfg["experiment.v2.alphas"]])
    ap.add_argument("--methods", nargs="+", default=list(cfg["experiment.v2.methods"]))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lease-hours", type=float, default=12.0,
                    help="how long before another process may steal a claim "
                         "whose owner died; make it longer than your slowest unit")
    args = ap.parse_args(argv)

    logging.basicConfig(level=cfg["log_level"],
                        format="%(asctime)s %(levelname)-7s %(message)s")

    claims = Path(str(cfg["results_v2.claims_dir"])) if cfg.get(
        "results_v2.claims_dir") else Path(str(cfg["masking.output_root"])) / "claims"

    done = skipped = failed = 0
    for cohort in args.cohorts:
        for design in args.designs:
            for alpha in args.alphas:
                tag = tag_for(cohort, design, alpha)
                for method in args.methods:
                    lease = Claim(claims / f"impute_{tag}_{method}.claim",
                                  args.lease_hours * 3600)
                    with lease:
                        if not lease.held:
                            skipped += 1
                            continue
                        call = ["--cohort", cohort, "--design", design,
                                "--alpha", str(alpha), "--method", method]
                        if args.force:
                            call.append("--force")
                        try:
                            rc = unit_runner.main(call)
                        except Exception:            # one bad unit, not the grid
                            logger.exception("[%s_%s] failed", tag, method)
                            failed += 1
                            continue
                        done += 1 if rc == 0 else 0
                        failed += 1 if rc != 0 else 0

    logger.info("imputation: %d units run, %d claimed elsewhere, %d failed",
                done, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
