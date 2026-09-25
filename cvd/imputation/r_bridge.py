"""Utilities for delegating imputation to the original R implementations."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DEFAULT_CLUSTER_RSCRIPT = "/shared/EL9/explorer/R/4.4.1/bin/Rscript"


def _resolve_rscript() -> str:
    env_path = os.environ.get("CVD_RSCRIPT")
    if env_path:
        return env_path
    path_rscript = shutil.which("Rscript")
    if path_rscript:
        return path_rscript
    if Path(DEFAULT_CLUSTER_RSCRIPT).exists():
        return DEFAULT_CLUSTER_RSCRIPT
    raise RuntimeError(
        "Rscript was not found. Load R first, for example `module load R/4.4.1`, "
        "or set CVD_RSCRIPT=/path/to/Rscript."
    )


def _write_column_file(path: Path, cols: Iterable[str]) -> None:
    path.write_text("\n".join(cols) + ("\n" if cols else ""))


def _write_levels_file(
    path: Path,
    clean_df: pd.DataFrame,
    categorical_cols: List[str],
) -> None:
    rows = []
    for col in categorical_cols:
        if col not in clean_df.columns:
            continue
        vals = pd.Series(clean_df[col]).dropna().drop_duplicates().sort_values()
        for order, val in enumerate(vals.astype(str), start=1):
            rows.append({"column": col, "level": val, "order": order})
    pd.DataFrame(rows, columns=["column", "level", "order"]).to_csv(path, index=False)


def run_r_imputer(
    X: pd.DataFrame,
    *,
    method: str,
    binary_cols: List[str],
    multiclass_cols: List[str],
    clean_df: pd.DataFrame,
    seed: int,
    max_iter: int,
    n_trees: int = 100,
    pmm_k: int = 3,
) -> pd.DataFrame:
    """Run the repository R bridge and return the imputed DataFrame."""
    bridge = Path(__file__).with_name("bridge.R")
    rscript = _resolve_rscript()

    X_to_r = X.copy()
    for col in X_to_r.columns:
        if col not in set(binary_cols) | set(multiclass_cols):
            X_to_r[col] = pd.to_numeric(X_to_r[col], errors="coerce")

    with tempfile.TemporaryDirectory(prefix=f"cvd_{method}_") as tmp:
        tmpdir = Path(tmp)
        input_csv = tmpdir / "input.csv"
        output_csv = tmpdir / "output.csv"
        binary_file = tmpdir / "binary_cols.txt"
        multiclass_file = tmpdir / "multiclass_cols.txt"
        levels_file = tmpdir / "levels.csv"

        X_to_r.to_csv(input_csv, index=False)
        _write_column_file(binary_file, [c for c in binary_cols if c in X.columns])
        _write_column_file(
            multiclass_file, [c for c in multiclass_cols if c in X.columns],
        )
        _write_levels_file(
            levels_file,
            clean_df,
            [c for c in binary_cols + multiclass_cols if c in X.columns],
        )

        cmd = [
            rscript,
            str(bridge),
            method,
            str(input_csv),
            str(output_csv),
            str(binary_file),
            str(multiclass_file),
            str(levels_file),
            str(seed),
            str(max_iter),
            str(n_trees),
            str(pmm_k),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"R {method} imputation failed with exit code {proc.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_csv.exists():
            raise RuntimeError(
                f"R {method} imputation completed but did not create {output_csv}."
            )
        out = pd.read_csv(output_csv)

    out = out.reindex(columns=X.columns)
    out.index = X.index
    return out
