"""Stage 00 -- build the analysis matrices from the raw registry exports.

Paper section: Methods 4.1, "After cleaning and harmonization".

This is the only producer of ``data/clean_{cohort}.csv`` and
``data/targets_{cohort}.csv``. It replaces the two notebooks whose output was
audited on 2026-09-10 and retired to the retired inputs (not distributed) -- see that directory's
``MANIFEST.md`` for the defects, and :mod:`cvd.prep.harmonise` for the rules
that prevent each one from recurring.

The stage is deliberately config-driven: every rename, recode, fill and
exclusion lives in ``conf/harmonise.yaml``, ``conf/label_aliases.yaml`` and
``conf/variable_types.yaml``. Nothing is inferred from the data except the
category domains, which are taken as the union across cohorts so that a code
means the same label in both.

Outputs, all written under ``harmonise.report_dir`` alongside the matrices:

``codebook.json``
    Every column's kind and its ordered level list. This is the artifact that
    makes the encoding auditable -- v1 had no equivalent, which is why nobody
    noticed Cohort1 and Cohort2 had encoded the same variable in opposite directions.
``selection.json``
    Which columns were kept, the realised missingness cutoff, and the five
    columns that just missed it.
``missingness_histogram.csv``
    Per-column missing fraction in each cohort, post-fill. The input to the
    Check-stop-1a cutoff decision.
``fills.json``
    Per-rule, per-column counts of structurally filled cells.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvd import conf
from cvd.conf import Config
from cvd.prep import harmonise as H

logger = logging.getLogger(__name__)


def _as_list(node: Any) -> list:
    return node.as_list() if hasattr(node, "as_list") else list(node)


def _as_dict(node: Any) -> dict:
    return node.as_dict() if hasattr(node, "as_dict") else dict(node)


def _derive_mortality_anytype(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """``mortality_anytype`` = died in hospital OR dead at 30 days.

    Defined here rather than in the prediction code so that the endpoint has one
    definition. v1 built it at
    ``cvd/prediction/continuous_v2.py:265`` from already-encoded integers, which
    meant it silently depended on each cohort's encoding.
    """
    out = df.copy()
    positive = {"mortality_inhospital": "Yes", "mt30stat": "Dead"}
    parts, missing = [], []
    for col in _as_list(spec["any_of"]):
        if col not in out.columns:
            raise KeyError(f"mortality_anytype needs {col!r}, which is absent")
        s = out[col].astype("string")
        parts.append((s == positive[col]).fillna(False).to_numpy(dtype=bool))
        missing.append(s.isna().to_numpy(dtype=bool))
    any_positive = np.logical_or.reduce(parts)
    all_missing = np.logical_and.reduce(missing)
    out["mortality_anytype"] = np.where(
        all_missing, None, np.where(any_positive, "Yes", "No")
    )
    return out


def _apply_row_filters(
    df: pd.DataFrame, filters: list[dict[str, Any]], *, cohort: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    report: dict[str, int] = {}
    out = df
    for f in filters:
        col = str(f["column"])
        if col not in out.columns:
            raise KeyError(f"[{cohort}] row filter {f['name']!r} needs absent column {col!r}")
        values = {str(v) for v in _as_list(f["exclude_values"])}
        s = out[col].astype("string")
        drop = s.isin(values)
        if not bool(f.get("keep_null", True)):
            drop = drop | s.isna()
        n = int(drop.fillna(False).sum())
        out = out.loc[~drop.fillna(False)].reset_index(drop=True)
        report[str(f["name"])] = n
        logger.info("[%s] row filter %s: dropped %d record(s)", cohort, f["name"], n)
    return out, report


def _prepare(cfg: Config, cohort: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Everything up to, but not including, column selection."""
    report: dict[str, Any] = {}
    src = Path(str(cfg[f"harmonise.sources.{cohort}"]))
    df = pd.read_csv(src, low_memory=False)
    report["raw_shape"] = list(df.shape)

    df = H.normalise_column_names(df)
    df = H.apply_column_aliases(
        df, _as_dict(cfg[f"harmonise.column_aliases.{cohort}"]), cohort=cohort
    )
    df, report["missing_tokens"] = H.apply_missing_tokens(
        df, _as_list(cfg["harmonise.missing_tokens"])
    )
    df, report["label_aliases"] = H.apply_label_aliases(
        df, [_as_dict(g) for g in cfg["label_aliases.groups"]]
    )
    df, report["row_filters"] = _apply_row_filters(
        df, [_as_dict(f) for f in cfg["harmonise.row_filters"]], cohort=cohort
    )
    df, report["structural_na"] = H.apply_structural_na(
        df, [_as_dict(r) for r in cfg["harmonise.structural_na"]]
    )
    ladder = _as_dict(cfg["harmonise.graft_ladder"])
    df, report["graft_ladder"] = H.apply_graft_ladder(
        df,
        families=_as_list(ladder["families"]),
        indices=_as_list(ladder["indices"]),
        fill=str(ladder.get("fill", H.NOT_APPLICABLE)),
    )
    df, report["implausible"] = H.apply_implausible(
        df,
        _as_dict(cfg["variable_types.range_check.implausible"]),
        _as_dict(cfg["variable_types.range_check.sentinels"]),
    )
    df = _derive_mortality_anytype(
        df, _as_dict(cfg["harmonise.targets.derived.mortality_anytype"])
    )
    report["prepared_shape"] = list(df.shape)
    return df, report


def run(cfg: Config | None = None) -> None:
    cfg = cfg or conf.active()
    if not bool(cfg["harmonise.enabled"]):
        logger.info("harmonise disabled; nothing to do")
        return

    id_column = str(cfg["harmonise.id_column"])
    cohorts = list(_as_dict(cfg["harmonise.sources"]))
    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {}

    for name in cohorts:
        frames[name], reports[name] = _prepare(cfg, name)
        logger.info("[%s] prepared %s", name, frames[name].shape)

    # --- selection -------------------------------------------------------
    held_out = _as_list(cfg["harmonise.targets.also_exclude_from_predictors"])
    held_out = sorted(set(held_out) | {"mortality_anytype"})
    sel = cfg["harmonise.selection"]
    columns, info = H.select_columns(
        frames,
        max_missing_fraction=float(sel["max_missing_fraction"]),
        always_keep=_as_list(sel["always_keep"]),
        always_drop=_as_list(sel["always_drop"]),
        held_out=held_out,
    )
    logger.info(
        "selected %d predictors (cutoff %.4f) from %d shared candidates",
        info["n_predictors"],
        info["realised_cutoff"],
        info["n_shared_candidates"],
    )

    # --- encoding --------------------------------------------------------
    types_cfg = _as_dict(cfg["variable_types"])
    codebook = H.build_codebook(frames, columns, types_cfg, id_column=id_column)

    present_targets = [
        c for c in held_out if all(c in f.columns for f in frames.values())
    ]
    target_book = H.build_codebook(
        frames, [id_column, *present_targets], types_cfg, id_column=id_column
    )

    # --- write -----------------------------------------------------------
    report_dir = Path(str(cfg["harmonise.report_dir"]))
    report_dir.mkdir(parents=True, exist_ok=True)

    for name, df in frames.items():
        matrix = H.encode(
            df[[c for c in columns if c in df.columns]], codebook, id_column=id_column
        )
        out_path = Path(str(cfg[f"harmonise.outputs.{name}"]))
        matrix.to_csv(out_path, index=False)
        logger.info("[%s] wrote %s %s", name, out_path.name, matrix.shape)

        targets = H.encode(
            df[[id_column, *present_targets]], target_book, id_column=id_column
        )
        tgt_path = Path(str(cfg[f"harmonise.target_outputs.{name}"]))
        targets.to_csv(tgt_path, index=False)
        logger.info("[%s] wrote %s %s", name, tgt_path.name, targets.shape)
        reports[name]["matrix_shape"] = list(matrix.shape)
        reports[name]["targets_shape"] = list(targets.shape)

    H.missingness_table(frames).to_csv(
        report_dir / "missingness_histogram.csv", index=False
    )
    (report_dir / "codebook.json").write_text(
        json.dumps({"kinds": codebook.kinds, "levels": codebook.levels}, indent=1)
    )
    (report_dir / "targets_codebook.json").write_text(
        json.dumps({"kinds": target_book.kinds, "levels": target_book.levels}, indent=1)
    )
    (report_dir / "selection.json").write_text(
        json.dumps(
            {
                "columns": columns,
                "held_out": present_targets,
                "analysed_categorical": _as_list(cfg["harmonise.targets.categorical"]),
                "analysed_continuous": _as_list(cfg["harmonise.targets.continuous"]),
                **info,
            },
            indent=1,
        )
    )
    (report_dir / "fills.json").write_text(json.dumps(reports, indent=1, default=str))
    logger.info("harmonise report written to %s", report_dir)


if __name__ == "__main__":  # pragma: no cover - convenience for a manual run
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(conf.load())
