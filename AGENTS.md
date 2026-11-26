# Repository guide

This repository reproduces seven experiments from Staykov, S. (2026),
"Incremental SVD for Large-Scale Dynamic Matrices: Accuracy, Subspace
Stability, Refresh Strategies, and Financial Factor-Based Risk Models"
(arXiv:2605.24514): an online, Brand-style rank-1/row-append/column-append
incremental truncated SVD engine, its a posteriori Frobenius error bound and
refresh policies (periodic and bound-triggered), and its use as an online
factor model for ETF-return covariance and portfolio-risk estimation.

## Layout

```
src/     isvd_core.py (IncrementalSVD engine), isvd_metrics.py (frob_ratio,
         angle_to_opt, EVR), isvd_policies.py (periodic/bound-triggered
         refresh), streaming.py (update-stream generators for the
         experiments), finance.py (causal de-meaning, online covariance and
         portfolio-risk tracking), config.py (every experiment parameter,
         with its paper section or "not specified in the paper")
exp/     one script per experiment (exp1_rank1_streaming.py ... 
         exp7_etf_factor_model.py), run as `python -m exp.<name>`
data/    data_loader.py (ETF prices via the QuantX Dataloader REST API),
         generate_synthetic.py (low-rank matrix and update-stream generators)
tests/   pytest unit tests for src/ components and smoke tests per experiment
results/ aggregate metrics (.csv/.json) and static .png plots per experiment,
         plus RESULTS.md with the comparison against the paper
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY, needed only for exp_7
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit and synthetic-data tests, no network
python -m pytest                          # full suite; the market_data test needs API keys
```

## Experiments

```bash
python -m exp.exp1_rank1_streaming
python -m exp.exp2_truncation_rank
python -m exp.exp3_structural_growth
python -m exp.exp4_mixed_streaming
python -m exp.exp5_bound_and_refresh
python -m exp.exp6_runtime_scaling
python -m exp.exp7_etf_factor_model          # a reduced ETF universe, downloaded on each run
python -m exp.exp7_etf_factor_model --full   # the paper's full 60-ETF-scale configuration
```

`reproduce.sh` at the repository root runs the unit tests and every
experiment in this order.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  UPPER_CASE variable with a comment citing its paper section, or stating
  that the paper does not specify it. No parameters are hardcoded in `exp/`
  or `src/` modules.
- Market data is loaded only through `data/data_loader.py` (QuantX Dataloader
  REST API); synthetic streams are generated only through
  `data/generate_synthetic.py`. No other module fetches or generates data.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from an uncommitted
  `.env` file (see `.env.example`) and are never written to source, logs, or
  results.
- Running the experiment commands overwrites the corresponding files in
  `results/`. The numbers and verdicts in `results/RESULTS.md` must be
  checked against the newly written files and updated by hand after a rerun;
  they are not regenerated automatically.
- Raw or market-derived time series, caches, and `.html` plots are never
  committed; `data/cache/` is gitignored and only aggregate tables
  (`.csv`/`.json`, indexed by experiment/policy/parameter, never by date or
  step) and static `.png` figures go into `results/`.
