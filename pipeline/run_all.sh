#!/usr/bin/env bash
# The whole benchmark, in order, on one machine.
#
# Stage 03b is the expensive one: the four generative imputers want a GPU and
# dominate the wall time. Everything else is CPU-only and comparatively quick.
# The two 03b lines below are separate so the CPU half can run while the GPU
# half queues, and so a missing GPU costs you four arms rather than the run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python pipeline/01_harmonise.py
python pipeline/02_build_masks.py
python pipeline/03a_prepare_inputs.py

python pipeline/03b_impute.py --methods baseline mean knn mice missranger
python pipeline/03b_impute.py --methods miwae gain remasker miracle

python pipeline/04_score_fidelity.py
python pipeline/05_predict.py
python pipeline/06a_collect_predictions.py
python pipeline/06b_rank.py

python pipeline/07a_fidelity_figures.py
python pipeline/07b_prediction_figures.py
python pipeline/07c_endpoint_figures.py
python pipeline/07d_schematic.py
python pipeline/08a_ranking_tables.py
python pipeline/08b_taxonomy_tables.py
python pipeline/08c_imputation_config_table.py

echo
echo "done. figures and tables are under the output directory named in conf/paths.yaml"
