"""Text and number formatting shared by the figure captions and the tables.

Kept separate from the plotting code so that :mod:`cvd.stats.prediction_tests`
can format its caption strings without importing matplotlib -- which is what
previously forced a statistics module to depend on a figures module.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def _format_plot_value(value: float, decimals: int, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    sign = "+" if signed else ""
    return f"{value:{sign}.{decimals}f}"
