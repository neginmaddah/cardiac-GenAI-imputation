"""
Generative-AI imputation methods: MIWAE, GAIN, MIRACLE, ReMasker.

MIWAE, GAIN, and MIRACLE are accessed through the ``hyperimpute`` library.
ReMasker is fetched from upstream by ``scripts/get_remasker.sh`` (see
``docs/remasker.md``).

All methods expect a numeric-only DataFrame (one-hot encoded multiclass,
float binary, and continuous columns).  Post-processing (inverting one-hot,
snapping to categories) is handled in :mod:`cvd.prep.postprocess`.

Hyperparameters come from ``conf/imputation/{miwae,gain,miracle,remasker}.yaml``
and the *benchmark-family* overrides in ``conf/experiment/*.yaml``. The two
families differ deliberately -- ReMasker's batch size and MIRACLE's step cap --
and the reasons are recorded in those files.
"""

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cvd.conf import active, ensemble_size, method_params

# Which benchmark family's overrides apply. Callers set it per run; there
# is only one family in this distribution, so that is the default.
_EXPERIMENT = "v2"


def set_experiment(name: str) -> None:
    """Select the benchmark family whose overrides and K apply."""
    global _EXPERIMENT
    _EXPERIMENT = name


# Epochs consumed by each ensemble member of the most recent run. Reset by
# impute_hyperimpute, read by cvd.cluster.worker_v2 into the fidelity record so
# the epoch budget actually used is visible without grepping a log.
LAST_RUN_EPOCHS: list[int | None] = []


def _params(method: str) -> dict:
    """Configured hyperparameters for ``method`` under the active family."""
    return method_params(active(), _EXPERIMENT, method)


def _n_ensemble(method: str) -> int:
    return ensemble_size(active(), _EXPERIMENT, method)

logger = logging.getLogger(__name__)

# Set MIWAE_LOG_MEM=1 to log CUDA memory every 100 epochs during MIWAE
# training -- used to verify the no_grad fix in _make_miwae_fit actually
# keeps memory bounded on large cohorts.
_MIWAE_LOG_MEM = os.environ.get("MIWAE_LOG_MEM", "") not in ("", "0", "false", "False")



def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs.

    Needed because several of these imputers do not seed themselves. ReMasker is
    the worst case: ``ReMasker.__init__`` never reads ``args.seed`` and nothing in
    the package calls ``torch.manual_seed``, so ``remasker.yaml``'s ``seed`` has no
    effect and every run is an independent draw. Seeding here, immediately before
    the model is constructed, is what actually makes a run reproducible.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _ensemble(run_one: Any, n: int, method: str, X: pd.DataFrame) -> pd.DataFrame:
    """Average ``n`` independently-seeded runs of ``run_one(seed) -> DataFrame``.

    Averaging happens on the raw model output, before any post-processing, so
    continuous columns average in their transformed space and one-hot dummy
    blocks average into soft class scores for ``postprocess_imputed`` to argmax.

    Rationale and measurements: see ``the superseded benchmark families (not distributed)``.
    """
    if n <= 1:
        _seed_everything(0)
        return run_one(0)

    total = None
    for i, seed in enumerate(range(n), start=1):
        logger.info("[%s] ensemble member %d/%d (seed=%d)", method, i, n, seed)
        _seed_everything(seed)
        arr = run_one(seed).to_numpy(dtype=float)
        total = arr if total is None else total + arr
    logger.info("[%s] averaged %d ensemble members", method, n)
    return pd.DataFrame(total / n, columns=X.columns, index=X.index)


def _make_miwae_fit(plugin_cls: type) -> Any:
    """
    Build a memory-safe replacement for ``MIWAEPlugin._fit``.

    Fixes an unbounded GPU-memory growth bug in hyperimpute's MIWAE: every
    100 epochs ``_fit`` re-evaluates the loss AND runs a full imputation
    pass over the *entire* (unbatched) dataset for logging purposes only,
    with no ``torch.no_grad()`` -- so autograd retains a full-dataset
    computation graph (up to ~n/batch_size times larger than a normal
    training-batch graph) on every one of those ~50 passes across a
    5000-epoch run.  On Cohort1 (10159 rows) this exhausts even a 140GB H200
    ~34 minutes in (confirmed: fails with ~136GB allocated and only
    ~200MB reserved-but-unallocated, i.e. genuine exhaustion rather than
    fragmentation -- ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments`` alone
    does not help).

    Also adds OPT-IN early stopping, read from
    ``experiment.<family>.early_stopping.miwae``. It is absent from the v1
    families, so ``the superseded uniform-holdout family`` and ``injected`` take exactly the code path
    they always did and their frozen artifacts still regenerate bit-identically.
    Under ``v2`` it stops when the epoch-mean training loss has not improved for
    ``patience`` epochs. That is a CONVERGENCE criterion on the IWELBO, not
    validation-based model selection -- there is no held-out split here, and
    calling it regularisation would be dressing it up. It is honest as "stop
    when the objective has plateaued" and nothing more.

    Wrapping just that diagnostic block in ``torch.no_grad()`` removes the
    leak while leaving results **bit-identical**: the block is never
    backpropagated through, and the ``xhat`` it computes is a pure local
    that is never stored on ``self`` and never read after the loop (
    ``_transform`` recomputes the imputation from scratch).  Keeping the
    ``_miwae_impute`` call rather than deleting it as dead code preserves
    the torch RNG stream, so Cohort1 runs the same numerical path the already
    -completed Cohort2 units did.

    .. note::
       This must be bound to the **instance** returned by
       ``Imputers().get("miwae")``, not to the class reached via a normal
       ``import``.  hyperimpute's ``PluginLoader._load_single_plugin``
       (``plugins/core/base_plugin.py``) loads plugins with
       ``importlib.util.spec_from_file_location`` + ``exec_module``, which
       re-executes ``plugin_miwae.py`` into a *fresh, unregistered module
       object* on every call -- so its ``MIWAEPlugin`` is a different class
       object than the importable one, and patching the importable one has
       no effect whatsoever (this is why an earlier class-level patch here
       silently never ran).  Module globals are therefore read off
       ``plugin_cls`` itself rather than from an imported module.
    """
    g = plugin_cls._fit.__globals__
    torch = g["torch"]
    DEVICE, td, nn, optim = g["DEVICE"], g["td"], g["nn"], g["optim"]
    weights_init, log = g["weights_init"], g["log"]

    stop_cfg = active().get(f"experiment.{_EXPERIMENT}.early_stopping.miwae")
    stop_cfg = stop_cfg.as_dict() if stop_cfg is not None else None

    def _fit(self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        X = torch.from_numpy(np.asarray(X)).float().to(DEVICE)
        mask = np.isfinite(X.cpu()).bool().to(DEVICE)

        xhat_0 = torch.clone(X)
        xhat_0[np.isnan(X.cpu()).bool()] = 0

        n = X.shape[0]
        p = X.shape[1]

        self.p_z = td.Independent(
            td.Normal(
                loc=torch.zeros(self.latent_size).to(DEVICE),
                scale=torch.ones(self.latent_size).to(DEVICE),
            ),
            1,
        )
        self.latent_sizeecoder = nn.Sequential(
            torch.nn.Linear(self.latent_size, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, 3 * p),
        ).to(DEVICE)
        self.encoder = nn.Sequential(
            torch.nn.Linear(p, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, 2 * self.latent_size),
        ).to(DEVICE)

        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.latent_sizeecoder.parameters()),
            lr=1e-3,
        )

        xhat = torch.clone(xhat_0)
        self.encoder.apply(weights_init)
        self.latent_sizeecoder.apply(weights_init)

        bs = min(self.batch_size, n)

        # Early stopping on the epoch-mean MINIBATCH loss, which is already
        # computed by the training step. The full-dataset bound below is the
        # better signal but is evaluated unbatched, so checking it every epoch
        # would cost more than the epochs it saves.
        patience = int(stop_cfg["patience"]) if stop_cfg else 0
        min_delta = float(stop_cfg.get("min_delta", 0.0)) if stop_cfg else 0.0
        # RELATIVE threshold, and it is the one that does the work. With an
        # absolute min_delta of 0 any improvement over the best-so-far resets
        # the counter, and the epoch-mean minibatch loss is stochastic in both
        # the batch permutation and the K importance samples -- so once training
        # has plateaued a lucky epoch still sets a new best every so often
        # (P(new minimum at epoch n) ~ 1/n for a stationary series), the counter
        # keeps resetting, and the run goes the full 5,000 epochs having learnt
        # nothing after the first few hundred. A threshold proportional to the
        # loss scale ignores noise-level "improvements" without needing to know
        # that scale in advance.
        rel_delta = float(stop_cfg.get("min_rel_delta", 0.0)) if stop_cfg else 0.0
        best, stale = float("inf"), 0
        self.stopped_epoch_ = None

        for ep in range(1, self.n_epochs):
            perm = np.random.permutation(n)
            batches_data = np.array_split(xhat_0[perm, ], int(n / bs))
            batches_mask = np.array_split(mask[perm, ], int(n / bs))
            running = 0.0
            for it in range(len(batches_data)):
                optimizer.zero_grad()
                self.encoder.zero_grad()
                self.latent_sizeecoder.zero_grad()
                b_data = batches_data[it]
                b_mask = batches_mask[it].float()
                loss = self._miwae_loss(iota_x=b_data, mask=b_mask)
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
            if patience:
                epoch_loss = running / max(len(batches_data), 1)
                needed = max(min_delta, rel_delta * abs(best)) if np.isfinite(best) else 0.0
                if epoch_loss < best - needed:
                    best, stale = epoch_loss, 0
                else:
                    stale += 1
                    if stale >= patience:
                        self.stopped_epoch_ = ep
                        logger.info(
                            "[miwae] early stop at epoch %d/%d "
                            "(no improvement > %.3g for %d epochs; best %.5f)",
                            ep, self.n_epochs, needed, patience, best,
                        )
                        break
            if ep % 100 == 1:
                # PATCHED: no_grad -- diagnostic only, unbatched over the
                # full dataset, which is what causes the OOM without it.
                with torch.no_grad():
                    log.debug(f"Epoch {ep}")
                    log.debug(
                        "MIWAE likelihood bound  %g"
                        % (
                            -np.log(self.K)
                            - self._miwae_loss(iota_x=xhat_0, mask=mask).cpu().data.numpy()
                        )
                    )
                    xhat[~mask] = self._miwae_impute(iota_x=xhat_0, mask=mask, L=10)[~mask]
                if _MIWAE_LOG_MEM:
                    logger.info(
                        "[miwae] epoch %d: cuda allocated=%.2fGB reserved=%.2fGB",
                        ep,
                        torch.cuda.memory_allocated() / 2**30,
                        torch.cuda.memory_reserved() / 2**30,
                    )

        return self

    return _fit


def impute_hyperimpute(
    X: pd.DataFrame,
    method: str,
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Impute using a hyperimpute plugin (MIWAE, GAIN, or MIRACLE).

    Parameters
    ----------
    X : DataFrame
        Numeric feature matrix with NaNs (one-hot encoded multiclass).
    method : str
        One of ``"miwae"``, ``"gain"``, ``"miracle"``.
    params : dict, optional
        Override default hyperparameters.  If None, uses defaults from
        ``conf/experiment/*.yaml``.

    Returns
    -------
    DataFrame with imputed values (same columns as input).
    """
    try:
        from hyperimpute.plugins.imputers import Imputers
    except ImportError:
        raise ImportError(
            "hyperimpute is not installed. Install with: pip install hyperimpute"
        )

    defaults = {
        "miwae": _params("miwae"),
        "gain": _params("gain"),
        "miracle": _params("miracle"),
    }
    if method not in defaults:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(defaults)}")

    cfg = {**defaults[method], **(params or {})}
    n_ens = int(cfg.pop("n_ensemble", _n_ensemble(method)))

    def _run_one(seed: int) -> pd.DataFrame:
        # random_state must be set unconditionally, not only when the method's
        # defaults already carry one. Every hyperimpute plugin calls
        # enable_reproducible_results(random_state) in __init__ (e.g.
        # plugin_gain.py:342, default random_state=0), which re-seeds the global
        # torch/numpy RNGs and would otherwise *undo* _seed_everything(seed) and
        # make all K ensemble members bit-identical -- a silent no-op that costs
        # K x the compute. the GAIN config in particular has no random_state key.
        run_cfg = dict(cfg, random_state=seed)
        imp = Imputers().get(method, **run_cfg)

        if method == "miwae":
            # Must patch the instance, not the imported class -- see
            # _make_miwae_fit's note on hyperimpute's exec_module plugin loader.
            imp._fit = types.MethodType(_make_miwae_fit(type(imp)), imp)
            logger.info("Applied MIWAE no_grad memory fix to %s", type(imp))

        # hyperimpute expects numpy-like input
        raw = imp.fit_transform(X.copy())
        LAST_RUN_EPOCHS.append(getattr(imp, "stopped_epoch_", None))
        out = raw.to_numpy() if hasattr(raw, "to_numpy") else np.asarray(raw)
        return pd.DataFrame(out, columns=X.columns, index=X.index)

    LAST_RUN_EPOCHS.clear()
    return _ensemble(_run_one, n_ens, method, X)



def _install_fast_transform(imputer: Any, chunk: int = 512) -> None:
    """Replace ReMasker.transform's one-row-at-a-time loop with grouped batching.

    Upstream ``transform`` runs ``for i in range(no)`` at batch size 1 and
    accumulates through a growing ``torch.cat`` -- one GPU launch per row plus
    quadratic copying (10,159 launches on Cohort1).

    It cannot be naively batched. At eval, ``random_masking`` sets
    ``len_keep = int(min(sum(mask)))`` over the *batch*, so a mixed batch would
    truncate every row down to its worst-observed member and silently discard
    observed values. Grouping rows by observed-count sidesteps that: within one
    group ``len_keep`` equals each row's own observed count, and
    ``noise[m < eps] = 1`` sorts missing features last, so each row keeps exactly
    its own observed set. Token order does not matter because ``pos_embed`` is
    added before masking and ``ids_restore`` unshuffles afterwards.

    Verified equivalent to the upstream path to 2.98e-08 (float32 rounding) over
    400 rows spanning 65 distinct observed-counts -- see
    ``tests/test_fast_transform.py``.
    """
    import torch as _torch

    ri = sys.modules.get(type(imputer).__module__)
    if ri is None or not hasattr(ri, "eps"):
        logger.warning("Could not locate ReMasker module; leaving transform() as-is")
        return
    _eps, _dev = ri.eps, ri.device

    def _transform(self: Any, X_raw: Any) -> Any:
        X = X_raw.clone()
        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]
        no, dim = X.shape
        X = X.cpu()
        for i in range(dim):
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + _eps)

        M = 1 - (1 * (np.isnan(X)))
        X = np.nan_to_num(X)
        X = _torch.from_numpy(X).to(_dev)
        M = M.to(_dev)

        self.model.eval()
        imputed = _torch.zeros((no, dim), device=_dev)
        counts = M.sum(dim=1)
        with _torch.no_grad():
            for c in _torch.unique(counts):
                idx = (counts == c).nonzero(as_tuple=True)[0]
                for s0 in range(0, idx.numel(), chunk):
                    sel = idx[s0:s0 + chunk]
                    _, pred, _, _ = self.model(X[sel].unsqueeze(dim=1), M[sel])
                    imputed[sel] = pred.squeeze(dim=2)

        for i in range(dim):
            imputed[:, i] = imputed[:, i] * (max_val[i] - min_val[i] + _eps) + min_val[i]

        M = M.cpu()
        imputed = imputed.detach().cpu()
        return M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed

    imputer.transform = types.MethodType(_transform, imputer)


def impute_remasker(
    X: pd.DataFrame,
    params: Optional[Dict] = None,
    remasker_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Impute using the ReMasker transformer-based model.

    Parameters
    ----------
    X : DataFrame
        Numeric feature matrix with NaNs (one-hot encoded multiclass).
    params : dict, optional
        Override default ReMasker hyperparameters.
    remasker_dir : str, optional
        Path to the ReMasker checkout.  If None, uses
        ``paths.external.remasker`` (``external/remasker``, populated by
        ``scripts/get_remasker.sh``).

    Returns
    -------
    DataFrame with imputed values.
    """
    # ReMasker (external/remasker, fetched by scripts/get_remasker.sh) uses FLAT
    # internal imports -- `from utils import ...`, not `from .utils import ...`
    # -- so it is directory-scoped rather than package-scoped and must be
    # imported with its own directory on sys.path.
    rdir = remasker_dir or str(active()["paths.external.remasker"])
    if rdir not in sys.path:
        sys.path.insert(0, rdir)
    try:
        from remasker_impute import ReMasker
        from utils import get_args_parser
    except ImportError as exc:
        raise ImportError(
            f"Cannot import ReMasker from {rdir}. Run scripts/get_remasker.sh "
            "(see docs/remasker.md)."
        ) from exc

    import torch

    cfg = {**_params("remasker"), **(params or {})}
    n_ens = int(cfg.pop("n_ensemble", _n_ensemble("remasker")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: ReMasker.__init__ builds its own args via
    # get_args_parser().parse_args() (strict) using the *real* process
    # sys.argv, and copies fields onto individual instance attributes at
    # construction time. It never reads back a `.args` attribute set
    # afterward. That means (a) invoking this from any script that has its
    # own CLI flags previously crashed with "unrecognized arguments", and
    # (b) even when it didn't crash, hyperparameter overrides assigned via
    # `imputer.args = args` after construction were silently discarded.
    # `mask_strategy`/`drop_rate` in remasker.yaml have no corresponding
    # parser flag in this ReMasker implementation (only `mask_ratio` exists,
    # left at its default) and are intentionally not forwarded.
    argv_overrides = [
        "--batch_size", str(cfg["batch_size"]),
        "--max_epochs", str(cfg["max_epochs"]),
        "--embed_dim", str(cfg["embed_dim"]),
        "--depth", str(cfg["encoder_depth"]),
        "--decoder_depth", str(cfg["decoder_depth"]),
        "--seed", str(cfg["seed"]),
        "--device", device,
    ]
    def _run_one(seed: int) -> pd.DataFrame:
        # _ensemble() has already called _seed_everything(seed); ReMasker itself
        # ignores --seed entirely (see the note in conf/imputation/remasker.yaml), so the
        # global torch RNG state set there is the only thing that makes this draw
        # reproducible.
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0] if old_argv else "remasker", *argv_overrides]
            imputer = ReMasker()
        finally:
            sys.argv = old_argv

        _install_fast_transform(imputer)
        raw = imputer.fit_transform(X.copy())
        out = raw.to_numpy() if hasattr(raw, "to_numpy") else np.asarray(raw)
        return pd.DataFrame(out, columns=X.columns, index=X.index)

    return _ensemble(_run_one, n_ens, "remasker", X)


# ── Convenience: run all GenAI methods ────────────────────────────────

# Display names come from the one registry in conf/imputation/_base.yaml,
# which replaced six divergent copies of this mapping.
GENAI_METHODS = {
    name: active()[f"imputation.registry.{name}.long"]
    for name in active()["imputation.families.genai"]
}


def impute_all_genai(
    X: pd.DataFrame,
    methods: Optional[list] = None,
    params_override: Optional[Dict[str, Dict]] = None,
    remasker_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple GenAI imputation methods.

    Parameters
    ----------
    X : DataFrame
        Numeric feature matrix with NaNs.
    methods : list of str, optional
        Which methods to run.  Default: all four.
    params_override : dict of dict, optional
        Per-method parameter overrides, e.g.
        ``{"miwae": {"n_epochs": 2000}}``.
    remasker_dir : str, optional
        Path to remasker package.

    Returns
    -------
    dict mapping method name → imputed DataFrame.
    """
    methods = methods or list(GENAI_METHODS)
    overrides = params_override or {}
    results: Dict[str, pd.DataFrame] = {}

    for m in methods:
        logger.info("Running GenAI imputation: %s", GENAI_METHODS.get(m, m))
        if m == "remasker":
            results[m] = impute_remasker(
                X, params=overrides.get(m), remasker_dir=remasker_dir,
            )
        else:
            results[m] = impute_hyperimpute(X, method=m, params=overrides.get(m))

    return results
