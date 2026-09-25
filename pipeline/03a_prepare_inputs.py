#!/usr/bin/env python
"""Stage 03a -- freeze one prepared input per masked matrix, per encoding family.

Each masked matrix is written out twice, because the two families of imputer
consume different spaces:

* **canonical** -- continuous and count columns z-scored, everything else on its
  integer codebook code. What the classical imputers consume, and the space in
  which fidelity is scored.
* **generative** -- nominal one-hot, ordinals rescaled to the unit interval,
  continuous and count quantile-transformed -- plus the fitted transformers, so
  the mapping back is inverted rather than reconstructed from memory at scoring
  time.

Every transformer is fitted on that unit's masked matrix and no other data.
That is what stops the 5% unit from seeing statistics that exist only because
the 20% unit was built on top of it.

Doing this as its own stage, with md5s in ``manifest.json``, means the encoding
is inspectable BEFORE any GPU time is spent. Folding it inside the imputer call
hides exactly the kind of mistake that is most expensive to find late.

    python pipeline/03a_prepare_inputs.py [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvd import conf                        # noqa: E402
from cvd.stages import inputs               # noqa: E402


def main(argv: list[str] | None = None) -> int:
    cfg = conf.active()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=cfg["log_level"],
                        format="%(asctime)s %(levelname)-7s %(message)s")
    inputs.run(cfg, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
