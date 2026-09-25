#!/usr/bin/env python
"""Stage 01 -- build one analysis matrix per cohort from the raw exports.

Reads the two registry exports named in ``conf/harmonise.yaml`` and writes, for
each cohort, a cleaned predictor matrix and a separate targets file. Every
rename, recode, structural fill and exclusion is declared in
``conf/harmonise.yaml``, ``conf/label_aliases.yaml`` and
``conf/variable_types.yaml``; nothing is inferred from the data except the
category domains, which are taken as the union across cohorts so that a code
means the same label at both sites.

Separating the targets out here is what makes leakage structural rather than a
matter of discipline: a predictor matrix cannot contain an endpoint, because
the endpoints are not in the file.

Also writes, into ``harmonise.report_dir``:

``codebook.json``      every column's kind and its ordered level list
``selection.json``     which columns survived, and the missingness cutoff used
``missingness_histogram.csv``  per-column missing fraction in each cohort
``fills.json``         per-rule counts of structurally filled cells

Read ``selection.json`` before trusting a run on new data: it records how many
predictors the two cohorts actually shared.

    python pipeline/01_harmonise.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvd import conf                        # noqa: E402
from cvd.stages import harmonise            # noqa: E402


def main() -> int:
    cfg = conf.active()
    logging.basicConfig(level=cfg["log_level"],
                        format="%(asctime)s %(levelname)-7s %(message)s")
    harmonise.run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
