"""Stage 03a (v2) -- freeze one prepared input per masked matrix and family.

Paper section: Methods 4.3, Imputation of Masked Data.

For each of the 18 masked matrices this writes:

* the **canonical** space -- continuous and count z-scored, everything else on
  its integer codebook code -- as both the masked matrix and the ground truth.
  This is what the four classic imputers consume and the space in which
  fidelity is scored, for both families.
* the **generative** space -- one-hot nominal, ordinals rescaled to [0, 1],
  continuous and count quantile-transformed -- plus the fitted transformers, so
  the mapping is invertible rather than reconstructed from memory at scoring
  time.

Every transformer is fitted on that unit's masked matrix and no other data.
That is what stops the 5% unit from seeing statistics that only exist because
the 10% and 20% units were built on top of it (the re-run plan), and Check
stop 3 asserts the constants actually differ across the three intensities.

The outputs are frozen with md5s in ``manifest.json``. They are the artifact the
GPU gate inspects, and the reason this exists as a stage at all: in v1 the same
encoding happened inside ``registry.impute()``, invisible until after the GPU
hours had been spent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from cvd import conf
from cvd.conf import Config
from cvd.prep import encode_v2
from cvd.stages.masking import tag_for

logger = logging.getLogger(__name__)


def _as_list(node: Any) -> list:
    return node.as_list() if hasattr(node, "as_list") else list(node)


def _as_dict(node: Any) -> dict:
    return node.as_dict() if hasattr(node, "as_dict") else dict(node)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _paths(cfg: Config, tag: str) -> dict[str, Path]:
    root = Path(str(cfg["inputs.inputs_dir"]))
    names = _as_dict(cfg["inputs.filenames"])
    return {key: root / str(tpl).format(tag=tag) for key, tpl in names.items()}


def prepare_unit(
    cfg: Config,
    masked: pd.DataFrame,
    clean: pd.DataFrame,
    kinds: dict[str, str],
    levels: dict[str, list],
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, encode_v2.CanonicalScaler, pd.DataFrame, encode_v2.GenaiEncoding]:
    """Everything derived from one masked matrix, in one place."""
    features = [c for c in masked.columns if c != id_column]

    scaler = encode_v2.fit_canonical(
        masked[features],
        kinds,
        standardize_kinds=[str(k) for k in _as_list(cfg["inputs.canonical.standardize"])],
        ddof=int(cfg["inputs.canonical.ddof"]),
        min_scale=float(cfg["inputs.canonical.min_scale"]),
    )
    canon_masked = encode_v2.apply_canonical(masked[features], scaler)
    canon_clean = encode_v2.apply_canonical(clean[features], scaler)

    encoding = encode_v2.fit_genai(
        canon_masked,
        kinds,
        levels,
        feature_columns=features,
        one_hot_template=str(cfg["inputs.one_hot_template"]),
        quantile_kwargs=_as_dict(cfg["inputs.quantile_transform"]),
    )
    genai_masked = encode_v2.apply_genai(canon_masked, encoding)

    canon_masked.insert(0, id_column, masked[id_column].to_numpy())
    canon_clean.insert(0, id_column, clean[id_column].to_numpy())
    return canon_masked, canon_clean, scaler, genai_masked, encoding


def run(cfg: Config | None = None, *, force: bool = False) -> None:
    cfg = cfg or conf.active()
    if not bool(cfg["inputs.enabled"]):
        logger.info("input preparation disabled; nothing to do")
        return

    id_column = str(cfg["harmonise.id_column"])
    designs = [str(d) for d in _as_list(cfg["masking.designs"])]
    alphas = [float(a) for a in _as_list(cfg["masking.alphas"])]
    masked_dir = Path(str(cfg["masking.masked_dir"]))
    mask_names = _as_dict(cfg["masking.filenames"])

    out_dir = Path(str(cfg["inputs.inputs_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    book = json.loads(
        (Path(str(cfg["harmonise.report_dir"])) / "codebook.json").read_text()
    )
    kinds: dict[str, str] = book["kinds"]
    levels: dict[str, list] = book["levels"]

    manifest: dict[str, Any] = {"units": {}}

    for cohort in _as_dict(cfg["harmonise.outputs"]):
        clean = pd.read_csv(Path(str(cfg[f"harmonise.outputs.{cohort}"])))
        for design in designs:
            for alpha in alphas:
                tag = tag_for(cohort, design, alpha)
                paths = _paths(cfg, tag)
                if not force and all(p.exists() for p in paths.values()):
                    logger.info("[%s] already prepared; skipping", tag)
                    manifest["units"][tag] = _describe(tag, cohort, design, alpha, paths)
                    continue

                masked = pd.read_parquet(
                    masked_dir / str(mask_names["masked"]).format(tag=tag)
                )
                if list(masked.columns) != list(clean.columns):
                    raise ValueError(f"{tag}: masked and clean schemas differ")

                canon_masked, canon_clean, scaler, genai_masked, encoding = prepare_unit(
                    cfg, masked, clean, kinds, levels, id_column,
                )

                canon_masked.to_parquet(paths["scaled_masked"], index=False)
                canon_clean.to_parquet(paths["scaled_clean"], index=False)
                paths["scalers"].write_text(json.dumps(scaler.to_json(), indent=1))
                genai_masked.to_parquet(paths["genai_masked"], index=False)
                joblib.dump(encoding.qt, paths["genai_qt"])
                paths["genai_meta"].write_text(json.dumps(encoding.to_json(), indent=1))

                entry = _describe(tag, cohort, design, alpha, paths)
                entry["canonical_width"] = int(canon_masked.shape[1])
                entry["genai_width"] = int(genai_masked.shape[1])
                entry["standardized_columns"] = len(scaler.columns)
                entry["degenerate_columns"] = list(scaler.degenerate)
                manifest["units"][tag] = entry
                logger.info(
                    "[%s] canonical %s, genai %s",
                    tag, canon_masked.shape, genai_masked.shape,
                )

    Path(str(cfg["inputs.manifest"])).write_text(json.dumps(manifest, indent=1))
    logger.info("manifest written to %s", cfg["inputs.manifest"])


def _describe(tag, cohort, design, alpha, paths) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "design": design,
        "alpha": alpha,
        "md5": {key: _md5(path) for key, path in sorted(paths.items())},
    }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(conf.load())
