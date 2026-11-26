# Incremental SVD for Large-Scale Dynamic Matrices

Reproduction of selected experiments from Staykov, S. (2026), "Incremental
SVD for Large-Scale Dynamic Matrices: Accuracy, Subspace Stability, Refresh
Strategies, and Financial Factor-Based Risk Models" (arXiv:2605.24514).

The paper maintains a rank-k truncated SVD of a dense matrix under rank-1
entry updates, row/column appends and mixed streams, tracks the drift of the
incremental factors against a full-SVD oracle, derives a computable a
posteriori Frobenius error bound (Proposition 1), and applies the engine as
an online factor model for asset-return covariance and portfolio-risk
estimation (Sec. 5).

## Repository layout

```
src/     incremental SVD engine, metrics, refresh policies, finance pipeline, config
exp/     one script per experiment, run as `python -m exp.<name>`
data/    synthetic-data generators and the QuantX Dataloader ETF price loader
tests/   pytest unit and integration tests
results/ aggregate metrics (.csv/.json) and static plots (.png) per experiment
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY for exp_7
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network calls
python -m pytest                          # full suite, requires API keys for exp_7 tests
```

## Experiments

```bash
python -m exp.exp1_rank1_streaming
python -m exp.exp2_truncation_rank
python -m exp.exp3_structural_growth
python -m exp.exp4_mixed_streaming
python -m exp.exp5_bound_and_refresh
python -m exp.exp6_runtime_scaling
python -m exp.exp7_etf_factor_model          # downloads ETF prices through the QuantX Dataloader
python -m exp.exp7_etf_factor_model --full   # paper's full 60-ETF universe and rank/refresh grid
```

Each script writes its summary tables and figures directly to `results/` and
prints a short summary to stdout.

`reproduce.sh` runs the full test suite followed by every experiment in the
reduced (default) configuration.

See `results/RESULTS.md` for the numbers obtained, their comparison with the
paper, and every deviation from the paper's methodology.
