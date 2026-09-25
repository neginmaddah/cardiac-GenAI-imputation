"""Per-figure selection for the script-shaped figure generators.

Each generator renders every figure it owns in one top-level pass, which is how
the published figures were made. Stage 07 sometimes needs a strict subset of
one generator's outputs -- most importantly when a generator owns both a
``protected: true`` figure whose bytes are frozen and an ordinary figure that
must be re-rendered. Figures 2, 3 and 7 are all owned by
``cvd.reporting.figures.journal``, so without this the whole module was skipped
and Figure 7 stayed content-stale.

The stage sets ``CVD_FIGURE_KEYS`` to a comma-separated list of registry keys;
a generator's ``owned_figures()`` intersects with it. An unset or empty
variable means "everything you own", so running a generator by hand is
unchanged.
"""

from __future__ import annotations

import os

ENV_VAR = "CVD_FIGURE_KEYS"


def selected(owned: set[str]) -> set[str]:
    """Narrow *owned* to the keys stage 07 asked for, if it asked."""
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return owned
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return owned & wanted
