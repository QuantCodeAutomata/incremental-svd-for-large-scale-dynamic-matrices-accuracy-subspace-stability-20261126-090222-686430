"""Experiment 6: runtime scaling of per-update time with matrix size and
truncation rank (Staykov, 2026, Sec. 4.6, cf. Sec. 4.2).

Times 300 rank-1 entry updates (median over 5 independent runs, each on a
fresh matrix with its own update stream) for a small 50x40 matrix at
k in {5, 8, 12} and a larger 500x400 matrix at k in {5, 10, 20, 40}, both of
true rank 5, and compares the incremental per-update time against a full thin
SVD of a matrix of the same size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from data.generate_synthetic import low_rank_matrix, make_rng, rank1_entry_stream
from src import config
from src.streaming import time_entry_updates, time_full_svd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _time_ranks(shape: tuple, ranks: tuple) -> List[dict]:
    m, n = shape
    rows = []
    for k in ranks:
        rng = make_rng(config.EXP6_SEED + k)
        A0, _, _ = low_rank_matrix(rng, m, n, config.EXP6_TRUE_RANK)
        events = rank1_entry_stream(
            rng, m, n, config.EXP6_N_UPDATES, config.RANK1_DELTA_STD
        )
        updates = [(i, j, delta) for (_, i, j, delta) in events]
        update_time = time_entry_updates(
            A0, k, updates, n_runs=config.EXP6_TIMING_RUNS
        )
        rows.append({"shape": f"{m}x{n}", "m": m, "n": n, "k": k, "mean_update_time_s": update_time})
    return rows


def run() -> pd.DataFrame:
    rows = _time_ranks(config.EXP6_SMALL_SHAPE, config.EXP6_SMALL_RANKS)
    rows += _time_ranks(config.EXP6_LARGE_SHAPE, config.EXP6_LARGE_RANKS)
    return pd.DataFrame(rows)


def time_full_svd_baselines() -> Dict[str, float]:
    baselines = {}
    for shape in (config.EXP6_SMALL_SHAPE, config.EXP6_LARGE_SHAPE):
        m, n = shape
        rng = make_rng(config.EXP6_SEED)
        A0, _, _ = low_rank_matrix(rng, m, n, config.EXP6_TRUE_RANK)
        baselines[f"{m}x{n}"] = time_full_svd(A0, n_runs=config.EXP6_TIMING_RUNS)
    return baselines


def summarize(df: pd.DataFrame, full_svd_times: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["full_svd_time_s"] = df["shape"].map(full_svd_times)
    df["speedup_full_svd_over_update"] = df["full_svd_time_s"] / df["mean_update_time_s"]
    return df


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = run()
    full_svd_times = time_full_svd_baselines()
    summary = summarize(df, full_svd_times)
    summary.to_csv(RESULTS_DIR / "exp6_summary.csv", index=False)
    with open(RESULTS_DIR / "exp6_summary.json", "w") as f:
        json.dump(
            {"timings": summary.to_dict(orient="records"), "full_svd_time_s": full_svd_times},
            f,
            indent=2,
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
