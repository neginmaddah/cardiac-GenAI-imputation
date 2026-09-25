#!/usr/bin/env python
"""Re-render Figures 4 and 7 -- the endpoint distributions -- from the v2 data.

WHY THIS EXISTS
---------------
Figures 4 and 7 are the only two main-text figures that depend on nothing but
the endpoint values: Figure 4 is the class balance of the three binary
outcomes, Figure 7 the z-scored histograms of the three post-operative
durations. Everything else in the paper needs the masking -> imputation ->
prediction grid first.

That makes them the one thing that can be inspected *before* committing GPU
hours to the re-run, which is exactly what they are for here: a pre-flight
check that the rebuilt targets look right.

The plotting itself is NOT reimplemented -- this calls the same
`plot_binary_target_distribution` / `plot_continuous_target_distribution` the
real generators call (`cvd/reporting/figures/prediction.py`), and composes the
(a)/(b) panels the same way the published Figure 4 does. So a difference you
see against the published figure is a difference in the data, not in the
drawing.

Output goes to the configured figures directory.

    python pipeline/07c_endpoint_figures.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvd import conf  # noqa: E402
from cvd.reporting.figures.panels import _panel_label, _save_300dpi  # noqa: E402
from cvd.reporting.figures.prediction import (  # noqa: E402
    plot_binary_target_distribution,
    plot_continuous_target_distribution,
)

COHORTS = ("Cohort1", "Cohort2")


def _decode_targets(cfg: conf.Config, cohort: str) -> pd.DataFrame:
    """Load the v2 targets and put binaries back on the 0/1 scale the plots want.

    The targets file stores every categorical as its codebook index. For the
    three analysed binaries the codebook is ``[negative, positive]``, so the
    stored code already IS 0/1 -- but assert it rather than assume, because the
    whole point of the rebuild was that a silent coding mismatch is exactly how
    v1 went wrong.
    """
    path = Path(str(cfg[f"harmonise.target_outputs.{cohort}"]))
    df = pd.read_csv(path)
    import json

    book = json.loads(
        (Path(str(cfg["harmonise.report_dir"])) / "targets_codebook.json").read_text()
    )
    for target in cfg["harmonise.targets.categorical"]:
        levels = book["levels"][str(target)]
        if levels != ["No", "Yes"]:
            raise AssertionError(
                f"{cohort}/{target}: expected codebook ['No', 'Yes'], got {levels}"
            )
        observed = set(df[str(target)].dropna().unique())
        if not observed <= {0.0, 1.0}:
            raise AssertionError(f"{cohort}/{target}: codes {observed} are not 0/1")
    return df


def _stack(parts: list[Image.Image], labels: list[str], out: Path) -> None:
    """Stack panels vertically with a labelled band above each."""
    w = max(p.width for p in parts)
    parts = [
        p
        if p.width == w
        else p.resize((w, int(p.height * w / p.width)), Image.LANCZOS)
        for p in parts
    ]
    fs = max(48, int(w * 0.026))
    band = int(fs * 1.55)
    pad = 30
    combined = Image.new(
        "RGB", (w, sum(p.height for p in parts) + pad + 2 * band), "white"
    )
    combined.paste(parts[0], (0, band))
    combined.paste(parts[1], (0, band + parts[0].height + pad + band))
    from PIL import ImageDraw

    d = ImageDraw.Draw(combined)
    _panel_label(d, labels[0], int(fs * 0.4), int(fs * 0.15), fs)
    _panel_label(
        d, labels[1], int(fs * 0.4), band + parts[0].height + pad + int(fs * 0.15), fs
    )
    _save_300dpi(combined, out)


def main() -> int:
    cfg = conf.load()
    out_dir = Path(str(cfg["paths.paper.figures_v2_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = [str(t) for t in cfg["harmonise.targets.categorical"]]
    continuous = [str(t) for t in cfg["harmonise.targets.continuous"]]

    frames = {c: _decode_targets(cfg, c) for c in COHORTS}

    print("=" * 74)
    print("FIGURE 4 -- binary endpoint class distributions")
    print("=" * 74)
    fig4_parts = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for cohort in COHORTS:
            clean = frames[cohort]
            plot_binary_target_distribution(clean, tmp_path, cohort, targets=binary)
            src = tmp_path / "binary_clinical_target_distribution_v4.png"
            dst = tmp_path / f"fig4_{cohort}.png"
            src.rename(dst)
            fig4_parts.append(Image.open(dst).copy())
            for t in binary:
                y = clean[t].dropna().astype(int)
                print(
                    f"    {cohort:6s} {t:20s} n={len(y):6d}  "
                    f"positive={100 * y.mean():6.2f}%   missing={clean[t].isna().sum()}"
                )
        _stack(
            fig4_parts,
            ["(a) Cohort1 cohort", "(b) Cohort2 cohort"],
            out_dir / "Figure_4.png",
        )
    print(f"  -> {out_dir / 'Figure_4.png'}")

    print()
    print("=" * 74)
    print("FIGURE 7 -- continuous endpoint distributions (z-scored)")
    print("=" * 74)
    fig7_parts = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for cohort in COHORTS:
            clean = frames[cohort]
            plot_continuous_target_distribution(
                clean, tmp_path, cohort, targets=continuous
            )
            src = tmp_path / "continuous_clinical_target_distribution_standardized_v4.png"
            dst = tmp_path / f"fig7_{cohort}.png"
            src.rename(dst)
            fig7_parts.append(Image.open(dst).copy())
            for t in continuous:
                v = pd.to_numeric(clean[t], errors="coerce")
                print(
                    f"    {cohort:6s} {t:20s} n={int(v.notna().sum()):6d}  "
                    f"missing={int(v.isna().sum()):4d}  median={v.median():8.2f}  "
                    f"max={v.max():9.2f}  zeros={int((v == 0).sum()):5d}"
                )
        _stack(
            fig7_parts,
            ["(a) Cohort1 cohort", "(b) Cohort2 cohort"],
            out_dir / "Figure_7.png",
        )
    print(f"  -> {out_dir / 'Figure_7.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
