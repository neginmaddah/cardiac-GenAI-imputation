# Classical vs. Generative AI Imputation of Missing Data in Cardiac Surgery Health Records (Nezami Lab)

Analysis code for:

> **Generative AI Imputation for Data Fidelity and Outcome Prediction in
> Cardiac Surgery**
>
> Negin Maddah\*, Amin Ramezani†, Qingchu Jin, Jakob Wollborn, Akinobu Itoh,
> Jaime B. Rabb, Felistas Mazhude, Robert S. Kramer, Douglas B. Sawyer,
> Raimond L. Winslow, Farhad R. Nezami‡
>
> \* Corresponding author: Negin Maddah (maddah.n@northeastern.edu)
> † Co-first author: Amin Ramezani, with Negin Maddah
> ‡ Senior author: Farhad R. Nezami. This work was carried out in the
>   Nezami Lab.
>
> Preprint (Research Square, 2026):
> [doi:10.21203/rs.3.rs-8303629/v1](https://doi.org/10.21203/rs.3.rs-8303629/v1)
> · [PDF](https://assets-eu.researchsquare.com/files/rs-8303629/v1_covered_20c126c0-8402-4d09-ae8f-5ef3dca5fef3.pdf)


A reproducible benchmark that asks two questions about missing data in adult
cardiac-surgery registries, and keeps them separate:

1. **Reconstruction fidelity.** How closely does an imputer recover values we
   deliberately hid, at cells where the truth is known?
2. **Downstream utility.** Does the choice of imputer change how well a model
   predicts post-operative outcomes?

Eight imputers are compared, four classical (mean/mode, k-NN, MICE,
missRanger) and four generative (MIWAE, GAIN, MIRACLE, ReMasker), against a
ninth arm that does not impute at all and leaves the missing values to
XGBoost's own sparsity-aware splitting.

The benchmark is run independently on **two cohorts**, referred to throughout
as `Cohort1` and `Cohort2`. Nothing about either site is encoded in this
repository, and no patient data is included. The point of two cohorts is to ask
whether a ranking obtained at one site reproduces at another.

## What this repository is, and is not

It is the analysis pipeline: masking, imputation, scoring, prediction, the
statistical tests, and the figures and tables. Run it against your own registry
export and you get your own version of every artifact.

It is **not** a trained model and contains no data. Nothing here transfers a
fitted imputer or predictor between sites; every arm is re-fitted from scratch
within each cohort. That is a deliberate property of the design, not a
limitation of the packaging. The question is whether a *protocol* reproduces,
not whether a *model* transports.

## Data you need

Two tabular exports, one per cohort, at the encounter level. The variable names
used in development were standard adult cardiac-surgery registry fields of the
kind most cardiac-surgery departments already collect, so a site with a
comparable export will mostly be renaming columns rather than deriving them.

Nothing about the schema is hard-coded. What a column *is* (binary, nominal,
ordinal, continuous or count) is declared in `conf/variable_types.yaml`, label
harmonisation between the two cohorts in `conf/label_aliases.yaml`, and the
endpoints and their exclusions in `conf/variables.yaml`. Stage 01 reads those
and writes a codebook so the encoding is auditable rather than implicit.

Three requirements are real, and a run will be meaningless without them:

* **The two cohorts must share predictors.** The analysis is restricted to the
  intersection, and stage 01 reports how large it turned out to be. Check
  `selection.json` before trusting anything downstream.
* **A field must mean the same thing at both sites.** A column carrying the
  same name but a different quantity has to be dropped at harmonisation, not
  reconciled later. This is easy to miss and expensive to find late.
* **There must be native missingness.** The holdout budget is proportional to
  each column's own missing rate, so a complete column is never scored and a
  complete dataset produces nothing to measure.

### Trying it without data

```bash
python examples/make_synthetic_cohorts.py
```

Writes two synthetic cohorts and a codebook, so stages 02–08 run end to end.
The rankings it produces are noise, because the columns are independent, but it
proves the environment works and shows the shape of every output.

## Install

```bash
pip install -e .                 # core: classical imputers, XGBoost, reporting
pip install -e ".[genai]"        # adds the four generative imputers (wants a GPU)
scripts/get_remasker.sh          # fetches ReMasker (needed for the GenAI arms)
```

ReMasker ([alps-lab/remasker](https://github.com/alps-lab/remasker)) is not
included in this repository because upstream publishes it without a licence.
`scripts/get_remasker.sh` clones it into `external/remasker` at the exact
commit the published results used and applies a one-line NumPy compatibility
fix; see `docs/remasker.md`. Please cite Du et al. (ICLR 2024) if you use it.

MICE and missRanger call R through `cvd/imputation/bridge.R` and need R with
`mice` and `missRanger` installed. Without R, run the grid without those two
arms; the rest of the benchmark is unaffected and they simply do not appear in
the rankings.

## Running it

```bash
bash pipeline/run_all.sh                 # everything, in order
```

or stage by stage:

| Stage | Command | Produces |
|---|---|---|
| 01 | `01_harmonise.py` | one clean predictor matrix and one targets file per cohort, plus a codebook |
| 02 | `02_build_masks.py` | the frozen holdout masks and per-cell status matrices |
| 03a | `03a_prepare_inputs.py` | per-unit encodings, one per imputer family, with md5s |
| 03b | `03b_impute.py` | an imputed matrix and a fidelity record per unit |
| 03c | `03c_ingest_r_mice.py` | optional: collect MICE draws produced out of process |
| 04 | `04_score_fidelity.py` | fidelity scores pooled into per-cohort tables |
| 05 | `05_predict.py` | XGBoost results for every (arm, endpoint, split) |
| 06a/b | `06a_collect_predictions.py`, `06b_rank.py` | the prediction table, then the Friedman/Wilcoxon rankings |
| 07a–c | `07a_fidelity_figures.py`, `07b_prediction_figures.py`, `07c_endpoint_figures.py` | the figures |
| 08a–c | `08a_ranking_tables.py`, `08b_taxonomy_tables.py`, `08c_imputation_config_table.py` | the tables, as LaTeX |

Stages are resumable: a unit whose outputs already exist is skipped unless you
pass `--force`.

Two stages need the **complete** set of eight imputers and will stop rather
than quietly rank a partial field: `07a_fidelity_figures.py`, which names the
arms it could not find, and `08a_ranking_tables.py`, which reads what 07a
writes. Everything else degrades gracefully and reports on whichever arms ran,
so you can get a prediction ranking out of the classical imputers alone while
the generative half is still queued. Stage 05 takes `--targets` if you want to fit one endpoint
rather than all five, which is the quickest way to get something out the far
end while you are still checking the wiring.

Stage 03b is the expensive one and is embarrassingly parallel over units. Start
the same command several times and each process claims units the others have
not taken; the claim is an atomic file create, so nothing is done twice. Split
it by method to keep the CPU and GPU halves apart:

```bash
python pipeline/03b_impute.py --methods baseline mean knn mice missranger   # CPU
python pipeline/03b_impute.py --methods miwae gain remasker miracle         # GPU
```

## Design decisions worth knowing before you trust a result

**Masks are frozen and shared.** Every imputer sees exactly the same missing
cells, which is what makes the comparison paired and the tests legitimate.

**Intensities are nested.** One Gumbel-perturbed ordering is drawn per column
and each intensity takes a prefix of it, so the 5% held-out set is a strict
subset of the 20% set. Drawing them independently would confound intensity with
the draw.

**Fidelity is scored only at held-out cells.** Cells that were missing in the
source have no ground truth and are never scored. This is what separates
imputation error from uncertainty in the data.

**Encoding is fitted per unit.** Every transformer is fitted on that unit's
masked matrix alone, so a lower-intensity unit cannot see statistics that exist
only because a higher-intensity one was built on top of it.

**Imputation is transductive with respect to prediction.** Each imputer sees
the whole masked predictor matrix, including rows later assigned to validation
and test. No imputer ever sees an outcome. Absolute performance here is
therefore not an estimate of what a model would achieve on genuinely unseen
patients; the comparison between arms is what the design supports.

**The no-imputation arm is differently informed, not simply worse.** It resolves
missingness inside the model using the training outcomes, where the imputers use
no outcomes and all predictor rows. Both are legitimate pipelines; they are not
informationally symmetric, and a contrast between them is not the same kind of
claim as a contrast between two imputers.

**The ensemble budget is small and is not neutral across methods.** Stochastic
imputers are averaged over two draws. Multiple-imputation methods keep reducing
variance with more draws while neural ones plateau early, so a small budget
favours the neural arms. `conf/experiment/v2.yaml` records the measured sizes.
Read a ranking as conditional on that budget, or run a sensitivity check.

**Rankings are never pooled across cohorts.** Each cohort is tested on its own
blocks. The two are compared by asking whether their rankings agree.

## Configuration

Everything is in `conf/`, composed from `conf/config.yaml`. No path, seed,
target name or hyperparameter appears anywhere else.

| File | Contents |
|---|---|
| `paths.yaml` | every filesystem location |
| `cohorts/cohort1.yaml`, `cohorts/cohort2.yaml` | per-cohort name, label, site-specific exclusions |
| `variables.yaml`, `variable_types.yaml` | endpoints, exclusions, column typing |
| `harmonise.yaml`, `label_aliases.yaml` | source files and the rules that make the two cohorts comparable |
| `masking.yaml`, `missingness.yaml` | holdout mechanisms, intensities, budget rule, seed |
| `imputation/_base.yaml` + one file per method | the method registry and every hyperparameter |
| `experiment/v2.yaml` | the benchmark family: grid, ensemble budget, per-method overrides |
| `fidelity.yaml`, `prediction.yaml`, `stats.yaml` | metrics, split protocol, tests |
| `reporting.yaml` | figure and table registry, one generator per artifact |

To point the pipeline at your own data, edit `conf/harmonise.yaml:sources` and
`conf/paths.yaml`. To change what a column is, edit `conf/variable_types.yaml`.
Neither requires touching code.

## Layout

```
conf/            all configuration
cvd/             the library
  prep/          harmonisation, typing, encoding
  missingness/   the nested holdout draw
  imputation/    the eight imputers, the ensemble, the per-unit runner
  fidelity/      reconstruction metrics
  prediction/    XGBoost fitting and the split protocol
  stats/         Friedman, Wilcoxon, Holm
  reporting/     figures and tables
pipeline/        numbered entry points, one per stage
scripts/         setup helpers (get_remasker.sh)
docs/            notes on third-party code (remasker.md)
external/        third-party code fetched by scripts/; git-ignored
examples/        synthetic-data generator
tests/           import and configuration-consistency checks (pytest -q)
outputs/         everything a run produces; git-ignored, created on first run
```

`outputs/figures` and `outputs/tables` are where the figures and LaTeX table
fragments land. Intermediate artifacts (masks, encodings, imputed matrices,
per-unit fidelity and prediction records) go under the directory named by
`masking.output_root` in `conf/paths.yaml`.

## Reproducibility

Seeds are declared in configuration, not in code: the masking seed in
`conf/masking.yaml`, split seeds in `conf/prediction.yaml`. Masks are derived
per cohort and mechanism from a base seed, and frozen on first write.

Given the same inputs and configuration, the figures and tables regenerate
byte-identically. If yours do not, something is reading state from outside
`conf/` and that is a bug worth reporting.

## Citation and licence

The code in this repository is released under the MIT License (see
`LICENSE`). If you use it, please cite the accompanying paper:

> Maddah N, Ramezani A, Jin Q, Wollborn J, Itoh A, Rabb JB, Mazhude F,
> Kramer RS, Sawyer DB, Winslow RL, Nezami FR. Generative AI Imputation for
> Data Fidelity and Outcome Prediction in Cardiac Surgery. *Research Square* (preprint), 2026. https://doi.org/10.21203/rs.3.rs-8303629/v1

ReMasker is
third-party code that is not distributed here and is not covered by this
licence; see `docs/remasker.md`.
