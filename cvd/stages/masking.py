"""Stage 02 (v2) -- write the 18 frozen masked matrices.

Paper section: Methods 4.2, Missing-Data Simulation.

Per cohort: 3 mechanisms x 3 intensities = 9 masked matrices and 9 status
matrices, plus a manifest recording the per-unit seed, the per-column budget and
the md5 of every file. The manifest is what makes "frozen" auditable --
``conf/missingness.yaml``'s v1 ``frozen_mask: true`` flag was read by no code at
all, and freezing was enforced only by the accident of a file already existing.

Nesting is produced inside :func:`cvd.missingness.nested.inject_nested`, which
draws each column once per ``(cohort, design)`` and takes prefixes for the three
intensities. This stage therefore calls it **once per cohort-design**, not once
per unit.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cvd import conf
from cvd.conf import Config
from cvd.missingness import nested

logger = logging.getLogger(__name__)


def _as_list(node: Any) -> list:
    return node.as_list() if hasattr(node, "as_list") else list(node)


def _as_dict(node: Any) -> dict:
    return node.as_dict() if hasattr(node, "as_dict") else dict(node)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def tag_for(cohort: str, design: str, alpha: float) -> str:
    return f"{cohort}_{design.upper()}_{int(round(alpha * 100))}"


def run(cfg: Config | None = None, *, force: bool = False) -> None:
    cfg = cfg or conf.active()
    if not bool(cfg["masking.enabled"]):
        logger.info("masking disabled; nothing to do")
        return

    id_column = str(cfg["harmonise.id_column"])
    designs = [str(d) for d in _as_list(cfg["masking.designs"])]
    alphas = [float(a) for a in _as_list(cfg["masking.alphas"])]
    base_seed = int(cfg["masking.seed"])

    masked_dir = Path(str(cfg["masking.masked_dir"]))
    status_dir = Path(str(cfg["masking.status_dir"]))
    masked_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    names = _as_dict(cfg["masking.filenames"])

    codebook = json.loads(
        (Path(str(cfg["harmonise.report_dir"])) / "codebook.json").read_text()
    )
    kinds: dict[str, str] = codebook["kinds"]

    manifest: dict[str, Any] = {
        "base_seed": base_seed,
        "designs": designs,
        "alphas": alphas,
        "units": {},
    }

    for cohort in _as_dict(cfg["harmonise.outputs"]):
        clean_path = Path(str(cfg[f"harmonise.outputs.{cohort}"]))
        clean = pd.read_csv(clean_path)
        logger.info("[%s] clean matrix %s", cohort, clean.shape)

        for design in designs:
            seed = nested.unit_seed(base_seed, cohort, design)
            targets = {
                a: (
                    masked_dir / str(names["masked"]).format(tag=tag_for(cohort, design, a)),
                    status_dir / str(names["status"]).format(tag=tag_for(cohort, design, a)),
                )
                for a in alphas
            }
            if not force and all(m.exists() and s.exists() for m, s in targets.values()):
                logger.info("[%s/%s] all intensities present; skipping", cohort, design)
                continue

            result = nested.inject_nested(
                clean,
                design=design,
                alphas=alphas,
                seed=seed,
                kinds=kinds,
                id_column=id_column,
                params=_as_dict(cfg[f"masking.params.{design}"]),
            )

            for a in alphas:
                masked_path, status_path = targets[a]
                result.masked[a].to_parquet(masked_path, index=False)
                result.status[a].to_parquet(status_path, index=False)
                tag = tag_for(cohort, design, a)
                held = int(
                    (result.status[a] == nested.STATUS_HELD_OUT).to_numpy().sum()
                )
                manifest["units"][tag] = {
                    "cohort": cohort,
                    "design": design,
                    "alpha": a,
                    "seed": seed,
                    "held_out_cells": held,
                    "shape": list(result.masked[a].shape),
                    "masked_md5": _md5(masked_path),
                    "status_md5": _md5(status_path),
                }
                logger.info("[%s] wrote %s (%d held out)", cohort, tag, held)

            manifest.setdefault("budgets", {})[f"{cohort}_{design}"] = {
                col: {str(k): v for k, v in counts.items()}
                for col, counts in result.budget.items()
            }

    Path(str(cfg["masking.manifest"])).write_text(json.dumps(manifest, indent=1))
    logger.info("manifest written to %s", cfg["masking.manifest"])


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(conf.load())
