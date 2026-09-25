#!/usr/bin/env python
"""The imputation configuration table for Methods, generated from conf/.

Methods used to carry every hyperparameter inline, which made the prose hard to
read and easy to let drift. The table below is built from the same YAML the
runners load, so a configuration change reaches the manuscript by regenerating
this file rather than by someone remembering to edit a sentence.

Rows follow the figure ordering. A method occupies one row unless it treats
variable types differently, in which case it occupies one row per treatment and
the shared settings sit on a final "all variables" row.

    python pipeline/08c_imputation_config_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvd import conf  # noqa: E402

OUT = Path(str(conf.active()["paths.paper.tables_dir"]))

# (label, [(variable scope, configuration), ...]). The configuration strings
# read their numbers out of conf below, so nothing here is a literal.
def _sci(x: float) -> str:
    """Scientific notation the way the manuscript writes it."""
    mant, exp = f"{float(x):.0e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def rows(c):
    from cvd.conf import method_params
    P = {m: method_params(c, "v2", m)
         for m in ("mean", "knn", "mice", "missranger", "miwae", "gain",
                   "remasker", "miracle")}
    p = lambda m, k: P[m][k]                                  # noqa: E731
    mq = c["imputation.methods.mice.quickpred"].as_dict()
    ms = c["imputation.methods.mice.stabilisers"].as_dict()
    es = c[f"experiment.v2.early_stopping.miwae"].as_dict()
    ens_k = int(c["experiment.v2.ensemble.default"])
    det = r"$K_{\mathrm{ens}}=1$ (deterministic)"
    sto = rf"$K_{{\mathrm{{ens}}}}={ens_k}$ (seeds 0, 1)"
    return [
        ("Mean/Mode", [
            ("Binary, nominal", "Modal observed category"),
            ("Ordinal, continuous", "Arithmetic mean of observed values"),
        ], det),
        ("KNN", [
            ("All", rf"$k={p('knn','n_neighbors')}$ neighbours, {p('knn','weights')} "
                    r"weights, in the encoded feature space; categorical "
                    r"predictions projected onto valid codes"),
        ], det),
        ("MICE", [
            ("Binary, nominal", r"Classification trees (\texttt{cart}), passed to R "
                                r"as unordered factors"),
            ("Ordinal, continuous", r"Predictive mean matching, passed to R "
                                           r"as numeric so draws are observed levels"),
            ("All", rf"$m=1$ completed dataset per member, "
                    rf"\texttt{{maxit}}={p('mice','max_iter')}; predictor matrix from "
                    rf"\texttt{{quickpred}} (\texttt{{mincor}}={mq['mincor']:.2f}, "
                    rf"\texttt{{minpuc}}={mq['minpuc']:.2f}); "
                    rf"\texttt{{ridge}}={ms['ridge']:g}, \texttt{{eps}}={ms['eps']:g}, "
                    rf"\texttt{{maxcor}}={ms['maxcor']:g}"),
        ], sto),
        ("MissRanger", [
            ("All", rf"{p('missranger','n_trees')} trees per forest, "
                    rf"{p('missranger','max_iter')} iterations, predictive mean "
                    rf"matching with {p('missranger','pmm_k')} neighbours"),
        ], sto),
        ("MIWAE", [
            ("All", rf"{p('miwae','n_hidden')} hidden units, "
                    rf"{p('miwae','latent_size')} latent dimensions, "
                    rf"$K_{{\mathrm{{IW}}}}={p('miwae','K')}$ importance samples, "
                    rf"batch {p('miwae','batch_size')}; IWELBO minimised for up to "
                    rf"{int(p('miwae','n_epochs')):,} epochs, early stopping at "
                    rf"patience {es['patience']} and relative threshold "
                    + _sci(es["min_rel_delta"])),
        ], sto),
        ("GAIN", [
            ("All", rf"Adversarial objective with a hint mechanism; up to "
                    rf"{int(p('gain','n_epochs')):,} epochs, batch "
                    rf"{p('gain','batch_size')}, hint rate {p('gain','hint_rate')}, "
                    rf"reconstruction weight {p('gain','loss_alpha')}"),
        ], sto),
        ("ReMasker", [
            ("All", rf"{p('remasker','encoder_depth')} encoder and "
                    rf"{p('remasker','decoder_depth')} decoder blocks, "
                    rf"{p('remasker','embed_dim')}-dimensional embeddings; up to "
                    rf"{p('remasker','max_epochs')} epochs, batch "
                    rf"{p('remasker','batch_size')}"),
        ], sto),
        ("MIRACLE", [
            ("All", rf"Median initialisation, learning rate $10^{{-3}}$, batch "
                    rf"{p('miracle','batch_size')}, hidden width "
                    rf"{p('miracle','n_hidden')}, $\beta={p('miracle','reg_beta')}$, "
                    rf"$\lambda={p('miracle','reg_lambda')}$, moving-average window "
                    rf"{p('miracle','window')}, up to "
                    rf"{p('miracle','max_steps')} optimisation steps"),
        ], sto),
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c = conf.load()
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\caption{\revisedthree{\textbf{Imputation configuration.} Every setting below is "
        r"identical for the two cohorts: each imputer is fitted independently "
        r"within each cohort--mechanism--intensity unit, but from the same "
        r"specification, so a cohort difference in the results cannot come from "
        r"a difference in configuration. Rows follow the ordering used in "
        r"Figures~\ref{fig_fidelity_Cohort1} and~\ref{fig_fidelity_Cohort2}. A method "
        r"spans several rows only where it treats variable types differently; "
        r"``All'' marks settings that apply to every variable. $K_{\mathrm{ens}}$ "
        r"is the number of ensemble members: stochastic imputers use two, seeded "
        r"0 and 1, combined in the model's own output space before decoding: the "
        r"neural imputers average throughout, while MICE and MissRanger average "
        r"continuous predictions and \emph{vote} on categorical ones, ties to the "
        r"lowest code. The "
        r"two deterministic imputers return an identical matrix on a repeated "
        r"run, so combining would be a no-op and they are run once.}}",
        r"\label{tab:imputation_config}",
        r"\begin{tabular}{@{}l p{0.175\linewidth} p{0.425\linewidth} "
        r"p{0.175\linewidth}@{}}",
        r"\hline",
        r"Method & Variables & Model configuration & Ensemble \\",
        r"\hline",
    ]
    for label, blocks, ens in rows(c):
        n = len(blocks)
        for i, (scope, detail) in enumerate(blocks):
            # Only the METHOD and ENSEMBLE cells span; both are single lines, so
            # \multirow can centre them without overflowing the block.
            method_cell = (rf"\multirow{{{n}}}{{*}}{{{label}}}" if i == 0 and n > 1
                           else ("" if i else label))
            ens_cell = (rf"\multirow{{{n}}}{{*}}{{{ens}}}" if i == 0 and n > 1
                        else ("" if i else ens))
            lines.append(f"{method_cell} & {scope} & {detail} & {ens_cell} \\\\")
            # A rule across the two middle columns separates one variable-type
            # treatment from the next without cutting the spanning cells.
            if i < n - 1:
                lines.append(r"\cline{2-3}")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    dest = OUT / "table_imputation_config.tex"
    dest.write_text("\n".join(lines))
    print(f"  -> {dest.relative_to(ROOT)}  ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
