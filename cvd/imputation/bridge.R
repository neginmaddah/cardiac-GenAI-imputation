suppressPackageStartupMessages({
  args <- commandArgs(trailingOnly = TRUE)
})

if (length(args) < 10) {
  stop(
    "Usage: Rscript r_impute_bridge.R METHOD INPUT_CSV OUTPUT_CSV ",
    "BINARY_COLS_TXT MULTICLASS_COLS_TXT LEVELS_CSV SEED MAXIT NUM_TREES PMM_K"
  )
}

method_name <- args[[1]]
input_csv <- args[[2]]
output_csv <- args[[3]]
binary_cols_file <- args[[4]]
multiclass_cols_file <- args[[5]]
levels_csv <- args[[6]]
seed <- as.integer(args[[7]])
maxit <- as.integer(args[[8]])
num_trees <- as.integer(args[[9]])
pmm_k <- as.integer(args[[10]])

read_cols <- function(path) {
  if (!file.exists(path) || file.info(path)$size == 0) return(character())
  cols <- readLines(path, warn = FALSE)
  cols[nzchar(cols)]
}

require_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      "Required R package '", pkg, "' is not installed. ",
      "Install it in the R library used by Rscript before running this pipeline."
    )
  }
}

set.seed(seed)

df <- read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)
binary_cols <- intersect(read_cols(binary_cols_file), names(df))
multiclass_cols <- intersect(read_cols(multiclass_cols_file), names(df))
factor_cols <- unique(c(binary_cols, multiclass_cols))
levels_df <- read.csv(levels_csv, check.names = FALSE, stringsAsFactors = FALSE)

levels_for <- function(col) {
  vals <- levels_df$level[levels_df$column == col]
  vals <- vals[!is.na(vals)]
  if (length(vals) == 0) vals <- sort(unique(na.omit(df[[col]])))
  as.character(vals)
}

for (col in names(df)) {
  if (col %in% factor_cols) {
    df[[col]] <- factor(df[[col]], levels = levels_for(col))
  } else {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  }
}

if (method_name == "mice") {
  require_package("mice")

  orig_names <- names(df)
  safe_names <- make.names(orig_names, unique = TRUE)
  names(df) <- safe_names
  binary_safe <- safe_names[match(binary_cols, orig_names, nomatch = 0)]
  multiclass_safe <- safe_names[match(multiclass_cols, orig_names, nomatch = 0)]
  factor_safe <- unique(c(binary_safe, multiclass_safe))

  pred <- tryCatch(
    # mincor 0.5, not 0.3. With 178 columns a 0.3 threshold admits a large
    # predictor set for every target, and `cart` -- which the factors now use --
    # costs a tree per target per iteration, so predictor count drives the whole
    # runtime. quickpred exists precisely because mice's default of "everything
    # predicts everything" does not scale past ~25 variables; this is tightening
    # a knob the method is designed around, not cutting a corner.
    mice::quickpred(as.data.frame(df), mincor = 0.5, minpuc = 0.25),
    error = function(e) mice::make.predictorMatrix(df)
  )
  diag(pred) <- 0
  method <- mice::make.method(df)

  for (v in names(df)) {
    x <- df[[v]]
    if (all(!is.na(x))) {
      method[v] <- ""
    } else if (is.numeric(x) && !(v %in% factor_safe)) {
      method[v] <- "pmm"
    } else if (is.factor(x)) {
      # CART for every factor, not logreg/polyreg.
      #
      # Both of those fit a design matrix and break when it is rank deficient:
      # glm.fit silently drops aliased columns, so the coefficient vector comes
      # back shorter than the covariance Cholesky and mice dies inside
      # `beta + rv %*% rnorm(ncol(rv))` with
      #     dims [product 17] do not match the length of object [18]
      # Measured on Cohort2/MCAR/5% in the v2 matrix, which makes this certain
      # rather than unlucky: the structural "NotApplicable" fill means whole
      # families are perfectly collinear by construction -- cab03 is
      # NotApplicable exactly when cab02 is, for all six cab* families -- so
      # some subset of any large predictor set is always aliased.
      #
      # cart is a standard mice elementary method, was already used here for
      # rare-class binaries and >10-level factors, handles factors natively,
      # returns observed levels only, and does not care about collinearity.
      # Using it uniformly also removes a silent inconsistency: two different
      # model families were being applied to columns of the same type depending
      # on their class balance.
      method[v] <- if (nlevels(x) < 2) "" else "cart"
    } else {
      method[v] <- ""
    }
  }

  imp <- mice::mice(
    df,
    m = 1,
    maxit = maxit,
    method = method,
    predictorMatrix = pred,
    printFlag = FALSE,
    seed = seed,
    # Passed through to mice.impute.pmm -> .norm.draw, which Choleskys X'X and
    # fails outright on a singular one:
    #     the leading minor of order 7 is not positive
    # mice's default ridge of 1e-5 is not enough here. The cause is the same
    # structural collinearity that forced cart on the factors -- the
    # "NotApplicable" fill ties whole cab* families together exactly -- and it
    # reaches the numeric path too because quickpred selects those columns as
    # predictors for the continuous targets. 1e-3 is mice's documented remedy
    # for a computationally singular system and keeps PMM as PMM, which matters:
    # dropping to cart for numeric as well would turn this arm into tree-based
    # multiple imputation and collapse the contrast with MissRanger.
    # 1e-2. Swept on Cohort1_MCAR_5, which failed at mice's documented 1e-3 with
    #     chol.default(sym(p$v)): the leading minor of order 4 is not positive
    # 1e-3 completed 5 of 8 units and 1e-2 completes all of them. Note 1e-1
    # fails again, so this is not monotone -- .norm.draw Choleskys the INVERSE
    # of (xtx + ridge*diag(xtx)) and the draw is RNG-dependent, so the usable
    # band is narrow rather than "more is safer".
    #
    # Raising it means re-running every MICE unit, not just the three that
    # failed: an arm has to share one configuration or its units are not
    # comparable with each other.
    ridge = as.numeric(Sys.getenv("CVD_MICE_RIDGE", unset = "1e-2")),
    # Passed through mice(...) to mice:::remove.lindep, which is the knob
    # DESIGNED for this problem: it drops predictors that are linearly
    # dependent on the rest before each univariate model is fitted. The
    # defaults (eps 1e-4, maxcor 0.99) leave enough near-collinearity in a
    # 178-column matrix that .norm.draw still hits a singular Cholesky on some
    # draws -- and the failure is RNG-dependent, so a unit can succeed at
    # seed 0 and fail at seed 1. Tightening both is what makes the arm run
    # deterministically rather than most of the time.
    eps = as.numeric(Sys.getenv("CVD_MICE_EPS", unset = "1e-3")),
    maxcor = as.numeric(Sys.getenv("CVD_MICE_MAXCOR", unset = "0.95"))
  )
  out <- mice::complete(imp, 1)
  names(out) <- orig_names
} else if (method_name == "missranger") {
  require_package("missRanger")
  n_threads <- as.integer(Sys.getenv("CVD_NUM_THREADS", unset = "4"))
  out <- missRanger::missRanger(
    df,
    pmm.k = pmm_k,
    num.trees = num_trees,
    num.threads = n_threads,
    verbose = 1
  )
} else {
  stop("Unknown R imputation method: ", method_name)
}

out <- out[, names(read.csv(input_csv, check.names = FALSE, nrows = 1)), drop = FALSE]
# ---------------------------------------------------------------------------
# Residual-NA guard. An imputer MUST return a complete matrix; a column that
# comes back with NAs still in it silently shrinks the fidelity denominator
# downstream (the scorer drops NaN pairs) or, as happened here, crashes
# snap_binary's astype(int) with a pandas cast error that says nothing about
# the cause.
#
# The real case: Cohort2's `race_white` has ONE observed level and 143 missing
# cells. mice cannot fit a model for a constant outcome, so it is assigned
# method "" and left untouched. The only value it can legitimately take is the
# one level that was observed, so fill it and SAY SO on stderr rather than
# passing the NAs on.
for (v in names(out)) {
  na_idx <- is.na(out[[v]])
  if (!any(na_idx)) next
  obs <- out[[v]][!na_idx]
  if (length(obs) == 0L) {
    stop("column '", v, "' is entirely missing after imputation")
  }
  fill <- if (is.numeric(obs)) {
    stats::median(obs)
  } else {
    tab <- sort(table(obs), decreasing = TRUE)
    names(tab)[1]
  }
  out[[v]][na_idx] <- fill
  message(sprintf(
    "[bridge] %s: %d cell(s) left unimputed by %s (%d distinct observed value(s)); filled with %s",
    v, sum(na_idx), method_name, length(unique(obs)), as.character(fill)
  ))
}

write.csv(out, output_csv, row.names = FALSE, na = "")
