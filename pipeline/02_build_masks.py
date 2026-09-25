#!/usr/bin/env python
"""Stage 02 -- draw the frozen holdout masks.

For each (cohort, mechanism) this holds out cells that were originally
OBSERVED, so there is a known ground truth to score reconstruction against. The
per-column budget is ``floor(N * alpha * m0)`` where ``m0`` is the column's own
native missing fraction, so injected missingness is proportional to how often a
field is genuinely absent and a column with no native missingness receives
none.

Intensities are nested by construction. One Gumbel-perturbed ordering is drawn
per column and each intensity takes a prefix of it, so the 5% held-out set is a
strict subset of the 20% set and intensity is the only thing that differs
between them. Drawing them independently would confound intensity with the draw.

Masks are written once and frozen. Every imputer sees exactly the same missing
cells, which is what makes the comparison paired.

    python pipeline/02_build_masks.py [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvd import conf                        # noqa: E402
from cvd.stages import masking              # noqa: E402


def main(argv: list[str] | None = None) -> int:
    cfg = conf.active()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="redraw masks that already exist (changes every "
                         "downstream result; do not do this mid-run)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=cfg["log_level"],
                        format="%(asctime)s %(levelname)-7s %(message)s")
    masking.run(cfg, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
