"""One unit-method of the v2 grid, as its own process.

Paper section: Methods 4.3-4.4.

Run as a subprocess by :mod:`cvd.cluster.grid`, deliberately: an imputer that
segfaults, OOMs or corrupts its CUDA context takes only its own unit down.

Loads the frozen inputs that Check stop 3 has already verified, imputes,
scores at the held-out cells, and writes both artifacts. Scoring happens in
this process rather than a later stage because both matrices are already in
memory here, and because a unit that is imputed but unscored looks "done" to a
resuming grid.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from cvd import conf
from cvd.conf import Config, ensemble_size, method_params
from cvd.prep import encode_v2
from cvd.imputation.run_v2 import PreparedUnit, col_types_3way, impute_v2
from cvd.stages.inputs import _paths as input_paths
from cvd.stages.masking import tag_for

logger = logging.getLogger(__name__)

EXPERIMENT = "v2"


def _genai_epochs() -> list:
    """Epochs each ensemble member actually used; empty unless MIWAE ran."""
    from cvd.imputation import genai

    return list(genai.LAST_RUN_EPOCHS)


def imputed_path(cfg: Config, tag: str, method: str) -> Path:
    return Path(str(cfg["results_v2.imputed_dir"])) / str(
        cfg["results_v2.filenames"]["imputed"]
    ).format(tag=tag, method=method)


def fidelity_path(cfg: Config, tag: str, method: str) -> Path:
    return Path(str(cfg["results_v2.fidelity_dir"])) / str(
        cfg["results_v2.filenames"]["fidelity"]
    ).format(tag=tag, method=method)


def load_prepared(cfg: Config, cohort: str, design: str, alpha: float) -> PreparedUnit:
    tag = tag_for(cohort, design, alpha)
    paths = input_paths(cfg, tag)
    meta = json.loads(paths["genai_meta"].read_text())
    book = json.loads(
        (Path(str(cfg["harmonise.report_dir"])) / "codebook.json").read_text()
    )
    encoding = encode_v2.GenaiEncoding(
        source_columns=tuple(meta["source_columns"]),
        columns=tuple(meta["columns"]),
        dummy_map={k: list(v) for k, v in meta["dummy_map"].items()},
        ordinal_k=dict(meta["ordinal_k"]),
        binary_columns=tuple(meta["binary_columns"]),
        qt_columns=tuple(meta["qt_columns"]),
        qt=joblib.load(paths["genai_qt"]),
    )
    return PreparedUnit(
        tag=tag,
        cohort=cohort,
        canonical_masked=pd.read_parquet(paths["scaled_masked"]),
        canonical_clean=pd.read_parquet(paths["scaled_clean"]),
        genai_masked=pd.read_parquet(paths["genai_masked"]),
        encoding=encoding,
        kinds=book["kinds"],
        id_column=str(cfg["harmonise.id_column"]),
    )


def main(argv: list[str] | None = None) -> int:
    cfg = conf.active()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--method", required=True, choices=list(cfg["imputation.order"]))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=cfg["log_level"], format="%(asctime)s %(levelname)-7s %(message)s",
    )

    tag = tag_for(args.cohort, args.design, args.alpha)
    out = imputed_path(cfg, tag, args.method)
    scored = fidelity_path(cfg, tag, args.method)
    if out.exists() and scored.exists() and not args.force:
        logger.info("[%s_%s] already complete", tag, args.method)
        return 0

    unit = load_prepared(cfg, args.cohort, args.design, args.alpha)
    k = ensemble_size(cfg, EXPERIMENT, args.method)
    params = method_params(cfg, EXPERIMENT, args.method)

    started = time.monotonic()
    result = impute_v2(unit, args.method, k=k, params=params, cfg=cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    if args.method != "baseline":
        from cvd.stages.fidelity import score_unit

        status = pd.read_parquet(
            Path(str(cfg["masking.status_dir"]))
            / str(cfg["masking.filenames"]["status"]).format(tag=tag)
        )
        metrics = score_unit(
            cfg, result, unit.canonical_clean, status, col_types_3way(unit),
        )
        scored.parent.mkdir(parents=True, exist_ok=True)
        scored.write_text(json.dumps({
            "cohort": args.cohort,
            "design": args.design,
            "alpha": args.alpha,
            "method": args.method,
            "experiment": EXPERIMENT,
            "n_ensemble": k,
            "params": params,
            "seconds": round(time.monotonic() - started, 1),
            # None per member when the member ran its full epoch budget.
            "epochs_used": list(getattr(_genai_epochs(), "__iter__", list)())
            if args.method in list(cfg["imputation.families.genai"]) else None,
            # The five-way taxonomy, so Phase 5 can split ordinal out of the
            # categorical pool and count out of the continuous pool without
            # re-deriving either from the data.
            "kinds": {c: unit.kinds[c] for c in unit.features},
            "metrics": metrics,
        }, indent=2))

    logger.info("[%s_%s] done in %.0fs", tag, args.method, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
