"""Stage 04 -- Adult Cardiac Surgery Data Fidelity Check.

Paper sections: Methods 4.4; Results 2.1 (categorical) and 2.2 (continuous).

Scores each imputed matrix against the held-out ground truth **only at cells
with status 2**. That restriction is what isolates imputation error from
uncertainty in the source data, and it is why the mask has to be frozen.

Metrics follow ``conf/fidelity.yaml``: balanced accuracy (plus macro-F1 and
accuracy) for categorical variables, RMSE for continuous ones. Continuous
columns are z-scored, so mean-imputation has an expected RMSE of 1.0 and any
value below 1 is an improvement on that naive baseline.

The reporting statistic is the MEDIAN, not the mean: per-feature error
distributions are strongly right-skewed because a small number of clinically
extreme rows contribute disproportionately large errors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from cvd.conf import Config, ensemble_size
from cvd.fidelity.metrics import evaluate_imputation

logger = logging.getLogger(__name__)


def score_unit(cfg: Config, imputed, clean_scaled, status, col_types) -> dict:
    return evaluate_imputation(
        imputed, clean_scaled, status,
        list(col_types["continuous"]),
        list(col_types["binary"]),
        list(col_types["multiclass"]),
        injected_code=int(cfg["fidelity.score_on_status"]),
    )
