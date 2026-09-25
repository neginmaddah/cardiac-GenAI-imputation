"""Cheap checks that the distribution is internally consistent.

These do not need data. They catch the failures that make a repository
unusable for someone who just cloned it: a module that does not import, a
config key a module reads but no file defines, an imputer declared in one place
and missing from another.

    pytest -q
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _modules() -> list[str]:
    import cvd
    out = []
    for m in pkgutil.walk_packages(cvd.__path__, prefix="cvd."):
        out.append(m.name)
    return sorted(out)


@pytest.mark.parametrize("name", _modules())
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_config_loads() -> None:
    from cvd import conf
    assert conf.active()["seed"]


def test_every_method_is_declared_everywhere() -> None:
    """A method must appear in the registry, the order and exactly one family.

    The registry drives encoding, the order drives figure layout and the family
    drives which runner is called. A method present in two of the three and
    missing from the third fails somewhere downstream, usually after the
    expensive part.
    """
    from cvd import conf
    cfg = conf.active()
    registry = set(cfg["imputation.registry"].as_dict())
    order = set(cfg["imputation.order"])
    families = cfg["imputation.families"].as_dict()
    claimed = [m for members in families.values() for m in members]

    assert order == registry, f"order/registry differ: {order ^ registry}"
    assert sorted(claimed) == sorted(registry), "a method is in no family, or two"


def test_benchmark_grid_is_a_subset_of_the_masks() -> None:
    """You cannot run an arm whose mask was never drawn."""
    from cvd import conf
    cfg = conf.active()
    assert set(cfg["experiment.v2.designs"]) <= set(cfg["masking.designs"])
    assert set(cfg["experiment.v2.alphas"]) <= set(cfg["masking.alphas"])
