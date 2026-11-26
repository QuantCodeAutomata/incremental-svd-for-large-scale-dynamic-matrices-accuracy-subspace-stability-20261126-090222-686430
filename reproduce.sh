#!/usr/bin/env bash
# Runs the test suite and every experiment of the reduced (default)
# configuration described in README.md, writing results into results/.
set -e

python -m pytest -m "not market_data"

python -m exp.exp1_rank1_streaming
python -m exp.exp2_truncation_rank
python -m exp.exp3_structural_growth
python -m exp.exp4_mixed_streaming
python -m exp.exp5_bound_and_refresh
python -m exp.exp6_runtime_scaling
python -m exp.exp7_etf_factor_model

echo "Done. See results/RESULTS.md for the comparison against the paper."
