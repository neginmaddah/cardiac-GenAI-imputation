"""Configuration loading.

One YAML tree under ``conf/`` is the single source of truth for every
parameter in this project. Nothing in ``cvd/`` may hardcode a target name, a
hyperparameter, a path or a seed.

Composition rule
----------------
Each entry in ``conf/config.yaml``'s ``defaults`` list names a file relative to
``conf/``. Where it lands in the merged namespace is determined by its
location:

===========================  ==================================
``conf/paths.yaml``          ``paths``
``conf/imputation/_base``    ``imputation``
``conf/imputation/gain``     ``imputation.methods.gain``
``conf/cohorts/cohort1``         ``cohorts.Cohort1``   (keyed by its own ``name``)
``the superseded injected family`` ``experiment.injected``
``the named runs (not distributed)``           ``run_defs.smoke``
``conf/config.yaml``         merged at the root
===========================  ==================================

Interpolation
-------------
``${a.b.c}`` is resolved against the merged root, so every reference is
unambiguous regardless of which file it appears in. ``${oc.env:VAR,default}``
reads an environment variable. Resolution is iterative so a reference may
point at another reference; a cycle raises rather than looping.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
REPO_ROOT = Path(__file__).resolve().parents[1]

_REF = re.compile(r"\$\{([^}]+)\}")
_MAX_PASSES = 20


class ConfigError(RuntimeError):
    """Raised for a malformed or unresolvable configuration."""


# ── plain-dict helpers ─────────────────────────────────────────────────────

def _deep_merge(base: dict, incoming: dict) -> dict:
    """Recursively merge ``incoming`` into ``base``, returning ``base``."""
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_path(root: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = root
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot set {dotted!r}: {part!r} is not a mapping")
    node[parts[-1]] = value


def _get_path(root: dict, dotted: str) -> Any:
    node: Any = root
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise KeyError(dotted)
    return node


# ── interpolation ──────────────────────────────────────────────────────────

def _resolve_env(expr: str) -> str:
    """``oc.env:VAR,default`` -> the environment value or the default."""
    body = expr[len("oc.env:"):]
    name, _, default = body.partition(",")
    value = os.environ.get(name.strip())
    if value is not None:
        return value
    if not _:
        raise ConfigError(f"env var {name.strip()!r} is unset and has no default")
    return default.strip()


def _resolve_scalar(text: str, root: dict) -> Any:
    """Substitute every ``${...}`` in ``text``. Returns a non-str when the
    string is exactly one reference to a non-str value."""
    matches = list(_REF.finditer(text))
    if not matches:
        return text

    # A lone reference keeps the referent's type (int, list, dict, ...).
    if len(matches) == 1 and matches[0].group(0) == text.strip():
        expr = matches[0].group(1).strip()
        if expr.startswith("oc.env:"):
            return _resolve_env(expr)
        return _get_path(root, expr)

    def sub(match: re.Match) -> str:
        expr = match.group(1).strip()
        if expr.startswith("oc.env:"):
            return str(_resolve_env(expr))
        return str(_get_path(root, expr))

    return _REF.sub(sub, text)


def _walk_resolve(node: Any, root: dict) -> tuple[Any, int]:
    """One resolution pass. Returns ``(new_node, n_unresolved_remaining)``."""
    if isinstance(node, dict):
        pending = 0
        out = {}
        for key, value in node.items():
            out[key], sub_pending = _walk_resolve(value, root)
            pending += sub_pending
        return out, pending
    if isinstance(node, list):
        pending = 0
        out = []
        for value in node:
            resolved, sub_pending = _walk_resolve(value, root)
            out.append(resolved)
            pending += sub_pending
        return out, pending
    if isinstance(node, str) and _REF.search(node):
        try:
            resolved = _resolve_scalar(node, root)
        except KeyError:
            return node, 1          # referent not resolved yet; retry next pass
        still = 1 if isinstance(resolved, str) and _REF.search(resolved) else 0
        return resolved, still
    return node, 0


def resolve(cfg: dict) -> dict:
    """Resolve every interpolation in ``cfg`` to a fixed point."""
    current = cfg
    previous_pending = None
    for _ in range(_MAX_PASSES):
        current, pending = _walk_resolve(current, current)
        if pending == 0:
            return current
        if pending == previous_pending:
            break
        previous_pending = pending
    unresolved = sorted(_unresolved_keys(current))
    raise ConfigError(
        "unresolvable or circular interpolation at: " + ", ".join(unresolved[:10])
    )


def _unresolved_keys(node: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _unresolved_keys(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _unresolved_keys(value, f"{prefix}[{i}]")
    elif isinstance(node, str) and _REF.search(node):
        yield f"{prefix} = {node!r}"


# ── loading ────────────────────────────────────────────────────────────────

def _read(rel: str) -> dict:
    path = CONF_DIR / f"{rel}.yaml"
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return loaded


def _namespace_for(rel: str, payload: dict) -> tuple[str | None, dict]:
    """Where does ``conf/<rel>.yaml`` land in the merged namespace?"""
    if "/" not in rel:
        return (None if rel == "config" else rel), payload

    group, _, stem = rel.partition("/")
    if group == "imputation":
        if stem == "_base":
            return "imputation", payload
        return f"imputation.methods.{stem}", payload
    if group == "cohorts":
        # keyed by the cohort's own declared name, so `cohorts.Cohort1` not `cohorts.cohort1`
        return f"cohorts.{payload.get('name', stem)}", payload
    if group == "experiment":
        return f"experiment.{payload.get('name', stem)}", payload
    if group == "run":
        return f"run_defs.{payload.get('name', stem)}", payload
    return f"{group}.{stem}", payload


def load(
    overrides: Iterable[str] = (),
    *,
    run: str | None = None,
    extra_defaults: Iterable[str] = (),
) -> "Config":
    """Compose, override and resolve the configuration.

    Parameters
    ----------
    overrides
        ``dotted.key=value`` strings, as passed by ``--set``. Values are parsed
        as YAML, so ``a.b=[1,2]`` and ``a.b=null`` behave as written.
    run
        Name of a ``the named runs (not distributed)`` entry to activate.
    extra_defaults
        Additional config files to merge, after the ``defaults`` list.
    """
    root_cfg = _read("config")
    defaults = list(root_cfg.pop("defaults", []))

    # Named runs are auto-discovered rather than listed, so adding
    # conf/run/<name>.yaml is enough to make `--run <name>` work.
    defaults += sorted(
        f"run/{path.stem}" for path in (CONF_DIR / "run").glob("*.yaml")
    )

    merged: dict = {}
    for rel in [*defaults, *extra_defaults]:
        payload = _read(rel)
        target, body = _namespace_for(rel, payload)
        if target is None:
            _deep_merge(merged, body)
        else:
            existing = {}
            try:
                existing = _get_path(merged, target)
            except KeyError:
                pass
            if isinstance(existing, dict):
                body = _deep_merge(dict(existing), body)
            _set_path(merged, target, body)
    _deep_merge(merged, root_cfg)

    # `paths.root` defaults to the repo root rather than the process cwd, so a
    # run from any directory resolves the same artifacts.
    if os.environ.get("CVD_ROOT") is None:
        _set_path(merged, "paths.root", str(REPO_ROOT))

    if run is not None:
        merged["run"] = run

    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set expects dotted.key=value, got {item!r}")
        key, _, raw = item.partition("=")
        _set_path(merged, key.strip(), yaml.safe_load(raw))

    resolved = resolve(merged)
    _validate(resolved)
    return Config(resolved)


# ── validation ─────────────────────────────────────────────────────────────

def _validate(cfg: dict) -> None:
    """Fail fast on the mistakes that silently corrupt results."""
    problems: list[str] = []

    # Named runs (conf/run/*.yaml) belonged to the DAG driver; the stages are
    # invoked directly here, so their absence is not an error.

    registry = _get_path(cfg, "imputation.registry")
    order = _get_path(cfg, "imputation.order")
    if set(order) != set(registry):
        problems.append(
            f"imputation.order {sorted(set(order) ^ set(registry))} does not match registry"
        )

    families = _get_path(cfg, "imputation.families")
    claimed = [m for members in families.values() for m in members]
    if sorted(claimed) != sorted(registry):
        problems.append("every method must appear in exactly one imputation family")

    for name, method in _get_path(cfg, "imputation.methods").items():
        if name not in registry:
            problems.append(f"conf/imputation/{name}.yaml has no registry entry")

    # An imputer with no encoding rule would silently receive raw columns.
    encoding = _get_path(cfg, "imputation.encoding")
    missing_encoding = set(registry) - set(encoding)
    if missing_encoding:
        problems.append(f"no encoding rule for: {sorted(missing_encoding)}")

    # The continuous metric trap: `rmse` is populated for only 2 of 5 targets.
    if _get_path(cfg, "prediction.continuous.primary_metric") != "rmse_z":
        problems.append(
            "prediction.continuous.primary_metric must be 'rmse_z'; plain 'rmse' is "
            "NaN for the three log-transformed targets and silently analyses 2 of 5"
        )

    # Exactly one generator per output path.
    seen: dict[str, str] = {}
    for group in ("figures", "supp_figures", "supp_tables"):
        for key, spec in _get_path(cfg, f"reporting.{group}").items():
            out = spec.get("output")
            if out is None:
                continue
            if out in seen:
                problems.append(
                    f"reporting.{group}.{key} and {seen[out]} both claim output {out}"
                )
            seen[out] = f"reporting.{group}.{key}"

    # Every design must declare a legacy alias, or frozen filenames cannot be read.
    for name, design in _get_path(cfg, "missingness.designs").items():
        if "legacy_alias" not in design:
            problems.append(f"missingness.designs.{name} has no legacy_alias")

    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))


# ── access ─────────────────────────────────────────────────────────────────

class Config:
    """Read-only dotted/attribute access over the resolved mapping."""

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return _wrap(self._data[name])
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, key: str) -> Any:
        return _wrap(_get_path(self._data, key))

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def path(self, key: str) -> Path:
        """A config value as a ``Path``."""
        return Path(str(self[key]))

    def as_dict(self) -> dict:
        return self._data

    def __contains__(self, key: str) -> bool:
        try:
            _get_path(self._data, key)
            return True
        except KeyError:
            return False

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return ((k, _wrap(v)) for k, v in self._data.items())

    def __repr__(self) -> str:
        return f"Config({sorted(self._data)})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    return value


# ── derived lookups used across stages ─────────────────────────────────────

def analysed_targets(cfg: Config, kind: str) -> list[str]:
    """Targets with ``analysed: true`` for ``kind`` in {'binary','continuous'}."""
    spec = cfg[f"variables.targets.{kind}"].as_dict()
    return [name for name, meta in spec.items() if meta.get("analysed")]


def all_targets(cfg: Config, kind: str) -> list[str]:
    """Every declared target for ``kind``, analysed or not."""
    return list(cfg[f"variables.targets.{kind}"].as_dict())


def target_labels(cfg: Config) -> dict[str, str]:
    """Display label for every endpoint, including retired ones.

    Retired endpoints are included so that a frozen artifact keyed on one still
    renders with its clinical name rather than its raw column name.
    """
    labels = {}
    for kind in ("binary", "continuous", "legacy"):
        section = cfg.get(f"variables.targets.{kind}")
        if section is None:
            continue
        for name, meta in section.as_dict().items():
            labels[name] = meta["label"]
    return labels


def excluded_predictors(cfg: Config, cohort: str | None = None,
                        target: str | None = None) -> list[str]:
    """Fields removed from the predictor matrix (Supplementary Table S6)."""
    groups = cfg["variables.excluded_from_predictors"].as_dict()
    excluded = {name for members in groups.values() for name in members}
    if cohort and target:
        per_cohort = cfg.get(f"cohorts.{cohort}.leakage_exclusions")
        if per_cohort is not None:
            excluded.update(per_cohort.as_dict().get(target, []))
    return sorted(excluded)


def ensemble_size(cfg: Config, experiment: str, method: str) -> int:
    """K for ``method`` under benchmark family ``experiment``.

    Deterministic methods are forced to 1: ensembling them is pure waste and
    would misrepresent them as stochastic.
    """
    if method in cfg["imputation.deterministic"]:
        return 1
    spec = cfg[f"experiment.{experiment}.ensemble"]
    by_method = spec.get("by_method")
    if by_method is not None and method in by_method:
        return int(by_method[method])
    return int(spec["default"])


def method_params(cfg: Config, experiment: str, method: str) -> dict:
    """Hyperparameters for ``method``, with the benchmark family's overrides
    applied on top of the method's own defaults."""
    base = cfg.get(f"imputation.methods.{method}.params")
    params = dict(base.as_dict()) if base is not None else {}
    overrides = cfg.get(f"experiment.{experiment}.overrides")
    if overrides is not None:
        params.update(overrides.as_dict().get(method, {}))
    return params


def design_alias(cfg: Config, design: str) -> str:
    """Paper-facing design name -> the token used in frozen result filenames.

    ``uniform -> MCAR``, ``mar -> MAR``, ``ev -> MNAR``. The manuscript retired
    the MNAR label but the stored files predate the rename and are not renamed.
    """
    return cfg[f"missingness.designs.{design}.legacy_alias"]


def design_from_alias(cfg: Config, alias: str) -> str:
    """The inverse of :func:`design_alias`."""
    for name, spec in cfg["missingness.designs"].as_dict().items():
        if spec["legacy_alias"].upper() == alias.upper():
            return name
    raise KeyError(f"no design with legacy alias {alias!r}")


def normalise_method(cfg: Config, label: str) -> str:
    """Map any spelling found in the frozen result files to a registry key."""
    aliases = cfg["imputation.legacy_labels"].as_dict()
    if label in aliases:
        return aliases[label]
    lowered = str(label).strip().lower()
    if lowered in cfg["imputation.registry"]:
        return lowered
    raise KeyError(f"unknown imputation method label {label!r}")


# ── lazily-loaded active configuration ─────────────────────────────────────
# Library modules take explicit parameters; `None` means "use the configured
# default". They resolve it through here rather than importing constants, so
# there is exactly one place a value can come from.

_ACTIVE: Config | None = None


def active() -> Config:
    """The process-wide configuration, loaded on first use."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load()
    return _ACTIVE


def set_active(cfg: Config) -> None:
    """Install a configuration (used by the pipeline runner and by tests)."""
    global _ACTIVE
    _ACTIVE = cfg
