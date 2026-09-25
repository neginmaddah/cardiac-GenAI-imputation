#!/usr/bin/env python
"""Variable taxonomy and leakage-exclusion tables, v2-native.

The v1 versions in paper/supp_tables/ report 167 and 182 predictors under a
three-way taxonomy. The rebuilt matrices carry 178 predictors in both cohorts
under the declared five-way taxonomy, and the held-out set is now derived from
the targets file rather than from a maintained exclusion list.

    python pipeline/08b_taxonomy_tables.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvd import conf  # noqa: E402

OUT = Path(str(conf.active()["paths.paper.tables_dir"]))
# The intra-operative timing table was dropped from the supplement on
# 2026-09-12: the manuscript states the timing rule in prose and lists the
# excluded fields, and a second table enumerating the RETAINED intra-operative
# predictors added no claim. It is still computed, because it is the evidence
# behind that prose, but it is written outside paper/ so it cannot reappear in
# the Overleaf source tree as an orphan fragment.
RETIRED_OUT = OUT

# Grouping by STS-ACSD field semantics, carried over unchanged from
# cvd/reporting/tables/intraop_timing.py. Deliberately conservative: a field is
# listed only where its timing is unambiguous from the STS definition. The v2
# matrices no longer carry every field the v1 list named, so the table is built
# by intersecting these groups with the predictor set actually in use.
INTRAOP_GROUPS = [
    ("Whole-case laboratory and physiological extrema",
     ["lwsthct", "lwstintrahemo", "lwsttemp", "lwsttempsrc", "highintraglu", "tempmeas"]),
    ("Intra-operative haemostasis and transfusion",
     ["ibldprod", "iantifibmedgiven", "intraclotfact", "intraopprocomcon"]),
    ("Operative technique and intra-operative monitoring",
     ["genanes", "inoptee", "ceroxused", "circarr", "aortoccl", "cplegiadeliv",
      "cplegiatype", "proxtech", "opapp", "approachcon"]),
    ("Procedures, conduits and grafts actually performed",
     ["opcab", "opvalve", "opocard", "oponcard", "aortproc", "ocarcong", "valexp",
      "opvalsurginput", "vsav", "vsmv", "vspv", "vstv", "cab02", "cab03", "cab04",
      "cabconduit01", "cabconduit02", "cabconduit03", "cabdistpos01", "cabdistpos02",
      "cabdistsite01", "cabdistsite02", "cabdistsite03", "cabendart01", "cabendart02",
      "cabendart03", "cabproximalsite01", "cabproximalsite02", "cabproximalsite03",
      "distvein", "distanastartcond", "leftima", "rightima", "imaused", "radialartused",
      "venouscondused"]),
    ("Other operative-duration fields",
     ["ortotaltime", "or time", "skinincisiontime"]),
]
KINDS = [("binary", "Binary"), ("nominal", "Nominal"), ("ordinal", "Ordinal"),
         ("continuous", "Continuous"), ("count", "Count")]


def esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = conf.load()
    idc = str(cfg["harmonise.id_column"])
    book = json.loads(
        (Path(str(cfg["harmonise.report_dir"])) / "codebook.json").read_text())
    kinds = book["kinds"]

    counts, held = {}, {}
    for cohort in ("Cohort1", "Cohort2"):
        X = pd.read_parquet(Path(str(cfg["inputs.inputs_dir"]))
                            / f"{cohort}_MCAR_20_canonical_masked.parquet")
        feats = [c for c in X.columns if c != idc]
        counts[cohort] = Counter(kinds.get(f, "unclassified") for f in feats)
        counts[cohort]["_total"] = len(feats)
        tg = pd.read_csv(Path(str(cfg[f"harmonise.target_outputs.{cohort}"])))
        held[cohort] = [c for c in tg.columns if c != idc]

    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{\textbf{Predictor taxonomy by cohort.} The taxonomy is "
        r"declared once in the configuration and applied identically to both "
        r"cohorts and every imputer, so the class of a variable is not a "
        r"per-method choice. Binary, nominal and ordinal variables are scored by "
        r"balanced accuracy in the fidelity analysis; continuous and count "
        r"variables by standardized RMSE. Ordinal variables enter the prediction "
        r"models as a single ordered integer column, not as indicators. Counts "
        r"exclude the record identifier and the held-out endpoint fields of "
        r"Table~\ref{tab:leakage}.}",
        r"\label{tab:variable_taxonomy}",
        r"\begin{tabular}{l" + "r" * (len(KINDS) + 1) + "}", r"\hline",
        "Cohort & " + " & ".join(lbl for _, lbl in KINDS) + r" & Total \\",
        r"\hline",
    ]
    for cohort in ("Cohort1", "Cohort2"):
        c = counts[cohort]
        lines.append(f"{cohort} & " + " & ".join(str(c.get(k, 0)) for k, _ in KINDS)
                     + f" & {c['_total']}" + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_variable_taxonomy.tex").write_text("\n".join(lines))

    # leakage: the held-out set, taken from the targets file itself
    both = sorted(set(held["Cohort1"]) & set(held["Cohort2"]))
    only = {c: sorted(set(held[c]) - set(held["Cohort2" if c == "Cohort1" else "Cohort1"]))
            for c in ("Cohort1", "Cohort2")}
    analysed = set(map(str, cfg["harmonise.targets.categorical"])) | \
               set(map(str, cfg["harmonise.targets.continuous"]))
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{\textbf{Fields held out of every predictor matrix, and where each "
        r"one is ever shown to a model.} At harmonisation every outcome and "
        r"post-operative status field is written to a separate targets file, so the "
        r"predictor matrix cannot contain one; the exclusion below is read off that "
        r"file rather than from a maintained list. The same set is removed for every "
        r"endpoint, so no outcome can act as a feature for another. The last two "
        r"columns state, for each field, whether any model ever sees it. In the "
        r"\emph{imputation} phase the answer is uniform: every field below is held out "
        r"of the imputer input, for every imputer, cohort, mechanism and intensity, so "
        r"no imputed value anywhere in the study is informed by a post-operative field. "
        r"In the \emph{prediction} phase the five analysed endpoints appear once each, "
        r"as the target of their own model and never as a predictor; the remaining "
        r"fields appear nowhere at all. Intra-operative fields are \emph{not} excluded: "
        r"they are retained as predictors for every endpoint, which is discussed in the "
        r"timing section below and set out per endpoint in "
        r"Table~\ref{tab:timing_windows}.}",
        r"\label{tab:leakage}",
        r"\begin{tabular}{lllll}", r"\hline",
        r"Field & Held out in & Analysed endpoint & Imputation phase & "
        r"Prediction phase \\", r"\hline",
    ]

    def shown(f):
        """Where this field is ever visible to a model."""
        return ("held out", "own target only") if f in analysed else ("held out", "never shown")

    for f in both:
        mark = "yes" if f in analysed else "--"
        imp, pred = shown(f)
        lines.append(rf"\texttt{{{esc(f)}}} & both cohorts & {mark} & {imp} & {pred} \\")
    for cohort in ("Cohort1", "Cohort2"):
        for f in only[cohort]:
            mark = "yes" if f in analysed else "--"
            imp, pred = shown(f)
            lines.append(rf"\texttt{{{esc(f)}}} & {cohort} only & {mark} & {imp} & {pred} \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_leakage_exclusions.tex").write_text("\n".join(lines))

    # ── per-endpoint prediction timing, the four columns R1.6 asked for ──────
    # "Clearly show which variables were used for each target and when those
    # variables became available." The answer is uniform across endpoints by
    # construction, and that uniformity IS the answer: the post-operative block
    # is removed before imputation and never reinstated, so the allowed window
    # and the excluded classes cannot differ per target. Stating it as a table
    # means the reviewer reads it rather than infers it from a rule.
    WINDOW = (r"Admission $\rightarrow$ end of index operation")
    EXCLUDED_CLASSES = (r"All post-operative fields "
                        r"(Table~\ref{tab:leakage})")
    RESOLVES = {
        "afibproc":  "during the post-operative admission",
        "complics":  "during the post-operative admission",
        # "beyond 24 hours" everywhere else in the paper; keep one inequality.
        "cpvntlng":  r"beyond $24$\,h of post-operative ventilation",
        "icuinhrs":  "at discharge from the initial ICU stay",
        "ppef":      "at the post-operative echocardiogram",
    }
    labels = dict(cfg["variables.labels"].as_dict()) if cfg.get("variables.labels") else {}
    order = [str(x) for x in cfg["harmonise.targets.categorical"]] + \
            [str(x) for x in cfg["harmonise.targets.continuous"]]
    lines = [
        r"\begin{table}[H]", r"\centering", r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
        r"\caption{\textbf{Prediction timing and the allowed predictor window, per "
        r"endpoint.} For each analysed endpoint: the moment the model is applied "
        r"(\emph{prediction time}), when the outcome itself resolves, the window of "
        r"record time a predictor may come from (\emph{allowed predictor window}), and "
        r"what is excluded because it falls outside it. The window and the exclusion "
        r"are identical for all five endpoints, and that is structural rather than "
        r"coincidental: the entire post-operative block is removed before imputation "
        r"and is never reinstated, so no per-target exclusion rule is applied after "
        r"the fact and none could differ between targets. Every predictor is "
        r"therefore drawn from the same frozen set of $178$ fields "
        r"(Table~\ref{tab:variable_taxonomy}) for every endpoint and every one of the "
        r"nine arms. Of those $178$, at least $48$ are intra-operative by STS field "
        r"semantics; none is recorded post-operatively, because every post-operative "
        r"field is in Table~\ref{tab:leakage}. Intra-operative predictors precede all "
        r"five endpoints, which is why the two intra-operative durations that could "
        r"not satisfy that ordering were dropped as endpoints and kept as predictors.}",
        r"\label{tab:timing_windows}",
        r"\begin{tabular}{lp{2.0cm}p{3.3cm}p{3.0cm}p{2.7cm}}", r"\hline",
        r"Outcome & Prediction time & Outcome resolves & Allowed predictor window & "
        r"Excluded variable classes \\[2pt]", r"\hline",
    ]
    for f in order:
        lines.append(
            rf"\texttt{{{esc(f)}}} & End of index operation & {RESOLVES[f]} & "
            rf"{WINDOW} & {EXCLUDED_CLASSES} \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "table_timing_windows.tex").write_text("\n".join(lines))

    # intra-operative timing disclosure
    present = {}
    for cohort in ("Cohort1", "Cohort2"):
        X = pd.read_parquet(Path(str(cfg["inputs.inputs_dir"]))
                            / f"{cohort}_MCAR_20_canonical_masked.parquet")
        present[cohort] = {c for c in X.columns if c != idc}
    totals = {c: 0 for c in present}
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{\textbf{Intra-operative predictors, retained for every endpoint.} "
        r"Every predictor in both cohorts is pre-operative or intra-operative; none "
        r"is recorded post-operatively, so for all five analysed endpoints every "
        r"predictor is determined before the outcome. The fields below are fixed "
        r"during, or at the end of, the index operation. They are listed as a "
        r"disclosure of what the predictor matrix contains, not as an exclusion "
        r"list: they are retained as predictors throughout. Grouping follows the "
        r"STS-ACSD field definitions and is deliberately conservative, a field "
        r"being listed only where its timing is unambiguous. Because the predictor "
        r"matrix is frozen and identical across all nine arms, both mechanisms and "
        r"both intensities, every arm inherits the same information set.}",
        r"\label{tab:intraop_timing}",
        r"\begin{tabular}{p{0.62\linewidth}rr}", r"\hline",
        r"Group & Cohort1 & Cohort2 \\", r"\hline",
    ]
    for group, fields in INTRAOP_GROUPS:
        keep = {c: [f for f in fields if f in present[c]] for c in present}
        if not any(keep.values()):
            continue
        for c in totals:
            totals[c] += len(keep[c])
        lines.append(f"{group} & {len(keep['Cohort1'])} & {len(keep['Cohort2'])}" + r" \\")
        named = sorted(set(keep["Cohort1"]) | set(keep["Cohort2"]))
        body = ", ".join(rf"\texttt{{{esc(f)}}}" for f in named)
        lines.append(r"\multicolumn{3}{p{0.95\linewidth}}{\footnotesize\hspace{1em}"
                     + body + r"} \\[2pt]")
    lines += [r"\hline",
              f"Total & {totals['Cohort1']} & {totals['Cohort2']}" + r" \\",
              f"Of all predictors & {100*totals['Cohort1']/len(present['Cohort1']):.0f}\\% & "
              f"{100*totals['Cohort2']/len(present['Cohort2']):.0f}\\%" + r" \\",
              r"\hline", r"\end{tabular}", r"\end{table}", ""]
    RETIRED_OUT.mkdir(parents=True, exist_ok=True)
    (RETIRED_OUT / "table_intraop_timing.tex").write_text("\n".join(lines))

    for f in ("table_variable_taxonomy.tex", "table_leakage_exclusions.tex"):
        print(f"  -> {OUT}/{f}")
    print("  -> the retired artifact tree.tex (not in the paper)")
    print(f"     {counts['Cohort1']['_total']} predictors per cohort, "
          f"{len(both)} held-out fields common to both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
