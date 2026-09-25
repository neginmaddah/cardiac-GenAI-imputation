# ReMasker (external, not redistributed)

Upstream: <https://github.com/alps-lab/remasker>
Pinned commit: `b2eca0eed92a03cedca9a19b3c42f6d7da7231bb`
Reference: Du et al., "ReMasker: Imputing Tabular Missing Values with Masked
Autoencoding", ICLR 2024.

Upstream publishes ReMasker without a licence, so its source is not included in
this repository. Fetch it with:

```bash
scripts/get_remasker.sh
```

This clones upstream into `external/remasker` (git-ignored), checks out the
pinned commit, and applies the one local modification below.
`cvd/imputation/genai.py` imports it from there via `paths.external.remasker`
in `conf/paths.yaml`.

## Local modification

Exactly one, in `utils.py`:

```diff
@@ -56,7 +56,7 @@ def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
     """
 
     assert embed_dim % 2 == 0
-    omega = np.arange(embed_dim // 2, dtype=np.float)
+    omega = np.arange(embed_dim // 2, dtype=float)
     omega /= embed_dim / 2.
     omega = 1. / 10000**omega  # (D/2,)
 
```

`np.float` was removed in NumPy 1.24 and this project pins 1.26.4, so the
unpatched file raises `AttributeError` on every model construction:
`utils.get_1d_sincos_pos_embed` is called from `model_mae.py` lines 69 and 72.

## What this project actually uses

Only `remasker_impute.ReMasker` (`__init__`, `fit`, `transform`,
`fit_transform`) and `utils.get_args_parser`. Everything else upstream is
demo/paper code with no importer here.

## Two upstream behaviours worked around in `cvd/imputation/genai.py`

1. `ReMasker.__init__` parses the real `sys.argv` and never reads
   `args.seed`; nothing in the package calls `torch.manual_seed`. Seeding is
   therefore applied by `cvd.imputation.genai.seed_everything()` immediately
   before each `ReMasker` is constructed.
2. `transform()` is O(n^2), imputing one row at a time. It is replaced at
   runtime by an observed-count-grouped batched implementation
   (`install_fast_transform`), verified to reproduce upstream exactly.
