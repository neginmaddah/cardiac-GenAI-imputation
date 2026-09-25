#!/usr/bin/env python
"""Supplementary ranking tables, fidelity and prediction, straight from v2.

Two supplementary sections cite these: the fidelity ranking section reads the
table written by the fidelity figures, the prediction ranking section reads the
canonical table written by pipeline/06b_rank.py. Nothing is
recomputed here, so a supplementary cell and a figure caption cannot disagree.

Bold marks a rank position that the omnibus supports. In the prediction tables
that means the omnibus rejected after Holm across the four reported families;
where it did not, no cell in that column is bold.

Each endpoint type is ranked on ONE metric -- balanced accuracy for the binary
endpoints, standardized RMSE for the continuous ones. AU-PRC and $R^2$ are
reported as descriptive medians (the per-endpoint tables at the end of this
module) and are never ranked, tested or used to break a tie.

    python pipeline/08a_ranking_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvd import conf  # noqa: E402

OUT = Path(str(conf.active()["paths.paper.tables_dir"]))
ARMS = ["Baseline", "Mean/Mode", "KNN", "MICE", "MissRanger", "MIWAE", "GAIN",
        "ReMasker", "MIRACLE"]


def parse_ranks(s: str) -> dict[str, float]:
    out = {}
    for part in str(s).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def fmt_p(p: float) -> str:
    # Three decimals, but never an apparent exact zero. The cutover is 1e-3 and
    # not 1e-4 because three-decimal rounding prints anything below 5e-4 as
    # "0.000" -- which is how a Holm-adjusted p of 3.5e-4 came to be displayed
    # as zero in the per-endpoint table: a value that rejects, shown as though
    # it were unmeasurably small.
    p = float(p)
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return rf"${mant}\times10^{{{int(exp)}}}$"
    return f"${p:.3f}$"


def esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


# ── fidelity ───────────────────────────────────────────────────────────────

def fidelity_tables() -> None:
    src = (Path(str(conf.active()["paths.paper.diagnostics_dir"]))
           / "fidelity_rankings.csv")
    d = pd.read_csv(src)
    # Fidelity is ranked WITHIN a cohort only. Pooling the two registries into a
    # single Friedman test would answer a question the study does not ask: the
    # cohorts are compared by whether their rankings agree, not by merging their
    # blocks.
    d = d[~d.scope.str.startswith("All")].reset_index(drop=True)
    d["scope_label"] = d.cohorts
    d["type_label"] = d.variable_type

    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\setlength{\tabcolsep}{3pt}", r"\small",
        r"\caption{\textbf{Imputation fidelity rankings.} Friedman omnibus over "
        r"per-variable reconstruction accuracy, blocked by cohort, simulation "
        r"(mechanism $\times$ intensity) and variable. Categorical fidelity is "
        r"balanced accuracy (higher is better), continuous fidelity is "
        r"standardized RMSE (lower is better). Mean rank 1 is best of eight "
        r"imputers; the no-imputation baseline does not appear because fidelity "
        r"is only defined where a value was imputed.}",
        r"\label{tab:fidelity_rankings}",
        r"\begin{tabular}{llrrrlll}", r"\hline",
        r"Scope & Data type & Blocks & $\chi^2$ & $p$ & Rank 1 & Rank 2 & Rank 8 \\",
        r"\hline",
    ]
    for _, r in d.iterrows():
        ranks = parse_ranks(r.method_mean_ranks)
        order = sorted(ranks, key=ranks.get)
        cells = [f"{m} ({ranks[m]:.2f})" for m in (order[0], order[1], order[-1])]
        lines.append(
            f"{r.scope_label} & {r.type_label} & {int(r.n_complete_blocks)} & "
            f"{r.friedman_chi2:.1f} & {fmt_p(r.friedman_p)} & "
            + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_fidelity_rankings.tex").write_text("\n".join(lines))

    # full mean-rank matrix
    cols = list(d.index)
    head = [f"{d.loc[i,'scope_label']}" for i in cols]
    sub = [f"{d.loc[i,'type_label'][:4]}." for i in cols]
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{\textbf{Imputation fidelity mean ranks, every imputer.} "
        r"Friedman mean rank within each scope, 1 = best of eight. Cat.\ is "
        r"categorical balanced accuracy, Cont.\ is continuous "
        r"standardized RMSE. The best two in each column are bold. Every column "
        r"rejects its omnibus test (Table~\ref{tab:fidelity_rankings}).}",
        r"\label{tab:fidelity_mean_ranks}",
        r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline",
        "Imputer & " + " & ".join(head) + r" \\",
        " & " + " & ".join(sub) + r" \\", r"\hline",
    ]
    mats = {i: parse_ranks(d.loc[i, "method_mean_ranks"]) for i in cols}
    tops = {i: sorted(mats[i], key=mats[i].get)[:2] for i in cols}
    for arm in ARMS:
        if arm == "Baseline":
            continue
        cells = []
        for i in cols:
            v = mats[i].get(arm)
            txt = "--" if v is None else f"{v:.2f}"
            cells.append(rf"\textbf{{{txt}}}" if arm in tops[i] else txt)
        lines.append(f"{esc(arm)} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_fidelity_mean_ranks.tex").write_text("\n".join(lines))


# ── prediction ─────────────────────────────────────────────────────────────

def prediction_tables(cfg) -> None:
    tdir = Path(str(cfg["masking.output_root"])) / "prediction_tables"
    d = pd.read_csv(tdir / "prediction_rankings.csv")
    pairs = pd.read_csv(tdir / "prediction_pairwise_significant.csv")
    d["type_label"] = d.endpoint_type.map({"binary": "Binary", "continuous": "Continuous"})

    lines = [
        r"\begin{table}[H]", r"\centering",
        # Ten columns do not fit the text block at \small with default padding:
        # the row overran the right margin by 108pt. Halving \tabcolsep and
        # dropping one font step recovers it without shortening any value.
        r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
        r"\caption{\textbf{Downstream prediction rankings.} Friedman omnibus over "
        r"the four reported families, blocked by mechanism $\times$ intensity "
        r"$\times$ endpoint $\times$ split. Nine arms: eight imputers and the "
        r"no-imputation baseline. Each endpoint type is compared on one metric: "
        r"balanced accuracy for the binary endpoints, standardized RMSE for the "
        r"continuous ones. $p_{\text{Holm}}$ is Holm--Bonferroni across these "
        r"four omnibus tests. Pairs is the number of the 36 all-pairs Wilcoxon "
        r"signed-rank comparisons that survive Holm within the family; pairwise "
        r"testing is run only where the omnibus rejects \emph{after} that "
        r"correction, so a dash means no pair was tested. Rank 1 and rank 2 are the two "
        r"best \emph{descriptive} mean ranks in each family. A rejecting omnibus "
        r"establishes that the nine arms are not all alike; it does not establish "
        r"that these two differ from each other, which only a surviving pairwise "
        r"comparison can support. ``Bal.\ accuracy'' is balanced "
        r"accuracy and ``Std.\ RMSE'' is standardized RMSE.}",
        r"\label{tab:prediction_rankings}",
        r"\begin{tabular}{lllrrrrllr}", r"\hline",
        r"Cohort & Endpoints & Metric & Blocks & $\chi^2$ & $p$ & "
        r"$p_{\text{Holm}}$ & Rank 1 & Rank 2 & Pairs \\",
        r"\hline",
    ]
    ABBREV = {"Balanced accuracy": r"Bal.\ accuracy",
              "Standardized RMSE": r"Std.\ RMSE"}
    for _, r in d.iterrows():
        sig = bool(r.omnibus_significant)
        r1 = f"{r.rank1} ({r.rank1_mean_rank:.2f})"
        r2 = f"{r.rank2} ({r.rank2_mean_rank:.2f})"
        if sig:
            r1, r2 = rf"\textbf{{{r1}}}", rf"\textbf{{{r2}}}"
        pr = "--" if not int(r.n_pairwise_tested) else (
            f"{int(r.n_pairwise_significant)}/{int(r.n_pairwise_tested)}")
        metric_tex = ("$R^2$" if r.metric == "r2"
                      else ABBREV.get(r.metric_label, esc(r.metric_label)))
        lines.append(
            f"{r.cohort} & {r.type_label} & {metric_tex} & "
            f"{int(r.n_complete_blocks)} & {r.friedman_chi2:.1f} & "
            f"{fmt_p(r.friedman_p)} & {fmt_p(r.friedman_p_holm)} & {r1} & {r2} & {pr}"
            + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_prediction_rankings.tex").write_text("\n".join(lines))

    # full mean-rank matrix, one column per family
    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\caption{\textbf{Downstream prediction mean ranks, every arm.} Friedman "
        r"mean rank within each family, 1 = best of nine. BA is balanced accuracy "
        r"(the binary comparison metric) and RMSE is standardized RMSE (the "
        r"continuous one). The best two are bold only in the one family whose omnibus "
        r"rejects after Holm (Table~\ref{tab:prediction_rankings}); in the other three "
        r"no arm is significantly best and the ranking is shown for completeness only.}",
        r"\label{tab:prediction_mean_ranks}",
        r"\begin{tabular}{l" + "r" * len(d) + "}", r"\hline",
        "Arm & " + " & ".join(f"{r.cohort}" for _, r in d.iterrows()) + r" \\",
        " & " + " & ".join(
            {"balanced_accuracy": "BA", "rmse_z": "RMSE"}[r.metric]
            for _, r in d.iterrows()) + r" \\",
        r"\hline",
    ]
    mats, tops = [], []
    for _, r in d.iterrows():
        m = parse_ranks(r.mean_ranks)
        mats.append(m)
        tops.append(sorted(m, key=m.get)[:2] if bool(r.omnibus_significant) else [])
    for arm in ARMS:
        cells = []
        for m, top in zip(mats, tops):
            v = m.get(arm)
            txt = "--" if v is None else f"{v:.2f}"
            cells.append(rf"\textbf{{{txt}}}" if arm in top else txt)
        lines.append(f"{esc(arm)} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_prediction_mean_ranks.tex").write_text("\n".join(lines))

    # surviving pairwise comparisons
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{\textbf{Surviving pairwise comparisons.} "
        r"All-pairs Wilcoxon signed-rank within a family, "
        r"36 pairs per family, Holm applied within the family. Inference uses the "
        r"signed ranks of the individual paired within-block differences; the "
        r"median of those differences is used only to label which arm the "
        r"comparison favours. Blocks favouring the winner counts "
        r"how many of the 24 blocks the winner led. Cohort1 standardized RMSE is the "
        r"only family whose omnibus rejects after correction and therefore the "
        r"only family in which pairwise testing was run at all; no comparison "
        r"exists to report for either cohort's binary family or for Cohort2. "
        r"Three of the four are an imputer beating the no-imputation Baseline "
        r"and the fourth is one imputer beating another, which is why the paper "
        r"reads this family as evidence that imputing helps rather than that any "
        r"particular imputer does.}",
        r"\label{tab:prediction_pairwise}",
        r"\begin{tabular}{lllrr}", r"\hline",
        r"Cohort & Metric & Comparison & Blocks favouring winner & "
        r"$p_{\text{Holm}}$ \\", r"\hline",
    ]
    if len(pairs):
        met = {"rmse_z": "Standardized RMSE",
               "balanced_accuracy": "Balanced accuracy"}
        pairs = pairs.sort_values("p_holm")
        for _, r in pairs.iterrows():
            n = int(r.n_blocks_favouring_a)
            tot = int(r.n_complete_blocks)
            won = n if r.better_arm == r.arm_a else tot - n
            lines.append(
                f"{r.cohort} & {met.get(r.metric, r.metric)} & "
                f"{esc(r.better_arm)} $>$ {esc(r.worse_arm)} & {won} of {tot} & "
                f"{fmt_p(r.p_holm)}" + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_prediction_pairwise.tex").write_text("\n".join(lines))


ALPHA = 0.05


def _per_target_verdict(out) -> str:
    """One sentence stating which per-endpoint omnibus tests survive Holm.

    Derived, never asserted. This was a fixed string reading "No endpoint-level
    test survives correction", which stopped being true when Cohort1 `ppef` came in
    at a Holm-adjusted 3.5e-4 and nothing in the pipeline noticed.
    """
    surv = out[out.p_holm < ALPHA]
    tail = (r" Twelve blocks rarely resolve nine arms, which is why the paper "
            r"reports the pooled families of "
            r"Table~\ref{tab:prediction_rankings} rather than endpoint-level "
            r"winners.")
    if surv.empty:
        return (r"\emph{No endpoint-level test survives correction.}" + tail)
    named = ", ".join(rf"{r.cohort} \texttt{{{esc(r.target)}}}"
                      for _, r in surv.iterrows())
    verb = "is the only endpoint" if len(surv) == 1 else "are the only endpoints"
    return (rf"\emph{{{named} {verb} whose omnibus survives correction}}; none of "
            r"the other tests in this table does." + tail)


def _no_bold_reason(fam: str) -> str:
    """Why nothing is bolded, stated truthfully for this endpoint type.

    The binary and continuous tables cannot share one sentence any more: no
    binary endpoint's omnibus survives correction, but Cohort1 `ppef` does.
    """
    here = sorted(t for c, t in _SURVIVORS
                  if (t in ("icuinhrs", "ppef")) == (fam == "continuous"))
    if not here:
        return (r"No value is bolded: no per-endpoint omnibus test in this family "
                r"survives correction (Table~\ref{tab:prediction_per_target}), so "
                r"a best value per column would assert a difference the tests do "
                r"not support.")
    named = ", ".join(rf"\texttt{{{esc(t)}}}" for t in here)
    return (rf"No value is bolded. The omnibus does survive correction for {named} "
            r"(Table~\ref{tab:prediction_per_target}), but an omnibus rejection "
            r"says only that the nine arms are not all alike within that endpoint; "
            r"it does not identify a leading arm, and no endpoint-level pairwise "
            r"testing was run. Bolding a best value per column would assert a "
            r"difference no test here supports.")


def per_target_survivors(cfg) -> set:
    """(cohort, endpoint) pairs whose per-endpoint omnibus survives Holm."""
    return _SURVIVORS


_SURVIVORS: set = set()


def per_target_table(cfg) -> None:
    """Per-endpoint omnibus, the detail the main text points at."""
    from cvd.stats.tests import compare_methods, holm_bonferroni
    tdir = Path(str(cfg["masking.output_root"])) / "prediction_tables"
    blocks = ["design", "intensity", "split_seed"]
    # The ranking metrics only, one per endpoint type. AU-PRC and R^2 appear in
    # the descriptive per-endpoint median tables, not in any omnibus.
    spec = [("binary", "balanced_accuracy", "higher_is_better", "Balanced accuracy"),
            ("continuous", "rmse_z", "lower_is_better", "Standardized RMSE")]
    labels = dict(cfg["variables.labels"].as_dict()) if cfg.get("variables.labels") else {}
    rows = []
    for fam, metric, direction, mlabel in spec:
        df = pd.read_csv(tdir / f"{fam}_prediction_results.csv", low_memory=False)
        df = df[df.status == "ok"]
        for cohort in ("Cohort1", "Cohort2"):
            for target in sorted(df.target.unique()):
                sub = df[(df.cohort == cohort) & (df.target == target)]
                res, _ = compare_methods(sub, block_keys=blocks, arm_key="method",
                                         value_key=metric, metric=metric,
                                         direction=direction)
                order = pd.Series(res.mean_ranks).sort_values()
                rows.append({"cohort": cohort, "target": target, "metric": mlabel,
                             "blocks": res.n_complete_blocks, "chi2": res.chi2,
                             "p": res.p_value, "best": order.index[0],
                             "best_rank": order.iloc[0],
                             "spread": order.iloc[-1] - order.iloc[0]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_bonferroni(out.p.tolist())
    # The verdict sentence is DERIVED, never asserted. It was previously a fixed
    # string saying no endpoint-level test survives, which stopped being true
    # without anything noticing.
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{\textbf{Per-endpoint prediction omnibus tests.} Friedman over "
        r"the nine arms within a single endpoint, blocked by mechanism $\times$ "
        r"intensity $\times$ split ($12$ blocks). $p_{\text{Holm}}$ is "
        r"Holm--Bonferroni across all " + str(len(out)) + r" tests in this table. "
        + _per_target_verdict(out) +
        r" No best-performing arm is named per endpoint: an omnibus rejection says "
        r"the nine arms are not all alike within that endpoint, not which arm leads, "
        r"and no endpoint-level pairwise testing was run.}",
        r"\label{tab:prediction_per_target}",
        r"\begin{tabular}{lllrrrr}", r"\hline",
        r"Cohort & Endpoint & Metric & Blocks & $\chi^2$ & $p$ & "
        r"$p_{\text{Holm}}$ \\", r"\hline",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"{r.cohort} & \\texttt{{{esc(r.target)}}} & {r.metric} & "
            f"{int(r.blocks)} & {r.chi2:.1f} & {fmt_p(r.p)} & {fmt_p(r.p_holm)}"
            + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_prediction_per_target.tex").write_text("\n".join(lines))
    global _SURVIVORS
    _SURVIVORS = {(r.cohort, r.target) for _, r in out.iterrows()
                  if r.p_holm < ALPHA}


def per_target_performance_tables(cfg) -> None:
    """Per-endpoint prediction PERFORMANCE, as distinct from per-endpoint tests.

    The main text pools endpoints inside a family, and `per_target_table` gives
    the omnibus run within each endpoint; neither shows what a method actually
    scored on a given outcome. These two tables do, as the median over the
    twelve cells (two mechanisms x two intensities x three splits) that every
    (cohort, endpoint, method) combination contributes.

    Nothing is bolded. No per-endpoint omnibus survives correction, so marking a
    best value per column would assert a difference the tests do not support.
    """
    from cvd.prediction.constants import METHOD_ORDER, TARGET_LABELS
    tdir = Path(str(cfg["masking.output_root"])) / "prediction_tables"

    SPEC = [
        ("binary", "table_per_target_binary.tex", "tab:per_target_binary",
         [("balanced_accuracy", "Balanced accuracy (higher is better)", "{:.3f}"),
          ("au_prc", "AU-PRC (higher is better)", "{:.3f}")],
         "binary endpoints"),
        ("continuous", "table_per_target_continuous.tex", "tab:per_target_continuous",
         [("rmse_z", "Standardized RMSE (lower is better)", "{:.4f}"),
          ("r2", "$R^2$ (higher is better)", "{:.4f}")],
         "post-operative continuous endpoints"),
    ]

    for fam, fname, label, metrics, what in SPEC:
        df = pd.read_csv(tdir / f"{fam}_prediction_results.csv", low_memory=False)
        df = df[df.status == "ok"]
        targets = sorted(df.target.unique())
        cols = [(c, t) for c in ("Cohort1", "Cohort2") for t in targets]
        n_blocks = int(df.groupby(["cohort", "target", "method"]).size().median())

        head_cohort = " & ".join(
            rf"\multicolumn{{{len(targets)}}}{{c}}{{{c}}}" for c in ("Cohort1", "Cohort2"))
        head_target = " & ".join(rf"\texttt{{{esc(t)}}}" for _, t in cols)
        lines = [
            r"\begin{table}[H]", r"\centering",
            r"\setlength{\tabcolsep}{4pt}", r"\small",
            rf"\caption{{\textbf{{Per-endpoint prediction performance, {what}.}} "
            rf"Median over the {n_blocks} cells each (cohort, endpoint, method) "
            r"combination contributes: two mechanisms $\times$ two intensities "
            r"$\times$ three splits. The no-imputation Baseline is included as the "
            r"ninth arm. Values are absolute, not differences from the Baseline as "
            r"in the main-text figures, so a column may be read straight down to "
            r"compare methods on that endpoint. Reading \emph{across} columns "
            r"compares endpoints of different difficulty, and for AU-PRC also of "
            r"different positive-class prevalence, which is what sets its chance "
            r"level. " + _no_bold_reason(fam) + r"}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline",
            rf"Method & {head_cohort} \\",
            rf" & {head_target} \\", r"\hline",
        ]
        for metric, mlabel, fmt in metrics:
            piv = (df.pivot_table(index="method", columns=["cohort", "target"],
                                  values=metric, aggfunc="median"))
            lines.append(rf"\multicolumn{{{len(cols)+1}}}{{l}}{{\emph{{{mlabel}}}}} \\")
            for m in METHOD_ORDER:
                if m not in piv.index:
                    continue
                vals = " & ".join(fmt.format(piv.loc[m, c]) for c in cols)
                lines.append(f"{esc(m)} & {vals}" + r" \\")
            lines.append(r"\hline")
        lines += [r"\end{tabular}", r"\end{table}", ""]
        (OUT / fname).write_text("\n".join(lines))


def error_profile_table(cfg) -> None:
    """False negatives with the rest of the error profile beside them.

    Reporting a false-negative reduction on its own is not interpretable -- any
    method can buy one with false positives -- so specificity, PPV and the false
    positives are on the same row.
    """
    tdir = Path(str(cfg["masking.output_root"])) / "prediction_tables"
    d = pd.read_csv(tdir / "binary_prediction_results.csv", low_memory=False)
    d = d[d.status == "ok"]
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{\textbf{Binary error profile, pooled over the three binary "
        r"endpoints.} Counts are summed over endpoints, mechanisms, intensities "
        r"and splits; sensitivity, specificity and positive predictive value are "
        r"computed from those sums. $\Delta$FN and $\Delta$FP are the change in "
        r"false negatives and false positives relative to the no-imputation "
        r"Baseline, as a percentage of the Baseline count. The two percentages "
        r"have different denominators and are not on a common scale, so a "
        r"false-negative reduction bought with a larger false-positive increase "
        r"cannot be called better or worse without a loss function or clinical "
        r"utility criterion weighting the two errors. This study defines none; "
        r"both columns are shown so the trade-off is visible rather than "
        r"adjudicated.}",
        r"\label{tab:error_profile}",
        r"\begin{tabular}{llrrrrrrr}", r"\hline",
        r"Cohort & Arm & FN & FP & $\Delta$FN & $\Delta$FP & Sens. & Spec. & PPV \\",
        r"\hline",
    ]
    for cohort in ("Cohort1", "Cohort2"):
        sub = d[d.cohort == cohort]
        agg = sub.groupby("method")[["true_positive", "false_negative",
                                     "true_negative", "false_positive"]].sum()
        base = agg.loc["Baseline"]
        for arm in ARMS:
            if arm not in agg.index:
                continue
            r = agg.loc[arm]
            sens = r.true_positive / max(r.true_positive + r.false_negative, 1)
            spec = r.true_negative / max(r.true_negative + r.false_positive, 1)
            ppv = r.true_positive / max(r.true_positive + r.false_positive, 1)
            dfn = 100 * (r.false_negative - base.false_negative) / base.false_negative
            dfp = 100 * (r.false_positive - base.false_positive) / base.false_positive
            dfn_s = "--" if arm == "Baseline" else f"{dfn:+.1f}\\%"
            dfp_s = "--" if arm == "Baseline" else f"{dfp:+.1f}\\%"
            lines.append(
                f"{cohort} & {esc(arm)} & {int(r.false_negative)} & "
                f"{int(r.false_positive)} & {dfn_s} & {dfp_s} & {sens:.3f} & "
                f"{spec:.3f} & {ppv:.3f}" + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_error_profile.tex").write_text("\n".join(lines))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = conf.load()
    fidelity_tables()
    prediction_tables(cfg)
    per_target_table(cfg)
    per_target_performance_tables(cfg)
    error_profile_table(cfg)
    for f in sorted(OUT.glob("*.tex")):
        print(f"  -> {f.relative_to(ROOT)}  ({len(f.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
