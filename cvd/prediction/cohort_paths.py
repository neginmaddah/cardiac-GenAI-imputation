"""Where one cohort's matrices live, plus the frame caches keyed on it.

Extracted from a module whose other half resolved filenames belonging to a
superseded benchmark family, and which therefore could not be shipped. Only the
declaration and the caches are needed: the loaders in :mod:`cvd.prep.frames`
and the plotting code populate and read them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CohortConfig:
    """Where one cohort's matrices live."""
    name: str
    data_dir: Path
    #: Absolute path to the cohort's clean analysis matrix.
    clean_path: Path
    clean_file: str
    output_dir: Path
    masked_template: str
    method_templates: dict[str, str]
    #: {method_label: {intensity: filename}} -- the irregular cases.
    template_overrides: dict[str, dict[str, str]]
    #: Method labels whose stored matrix is still one-hot encoded.
    one_hot_encoded: frozenset[str]


_CLEAN_CACHE: dict[str, pd.DataFrame] = {}

# Keyed by (cohort, intensity, method). NOTE: this key cannot distinguish one
# holdout DESIGN from another at the same intensity, so a consumer that
# switches design between calls must clear the cache.
_METHOD_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}

_FEATURE_META_CACHE: dict[tuple[str, str], dict[str, list[str]]] = {}
