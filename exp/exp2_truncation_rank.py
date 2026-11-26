"""Experiment 2: effect of the truncation rank k on the error ratio and
subspace tracking (Staykov, 2026, Sec. 4.2).

Streams 6,000 random rank-1 entry updates into the same 50x40, true-rank-5
synthetic matrix of Sec. 4.1, under a fixed periodic refresh every 2,000
steps, for three truncation ranks k in {5, 8, 12} (one at the true rank, two
above it). All three runs replay the identical update stream, so differences
in the logged Frobenius error ratio E_inc(t)/E_opt(t), the absolute
incremental and optimal errors, and the principal angle to the optimal
rank-k subspace are attributable only to the truncation rank. The experiment
also times the per-update cost of the rank-1 entry update as a function of
k, matching the runtime-measurement protocol shared with Sec. 4.6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import low_rank_matrix, make_rng, rank1_entry_stream
from src import config
from src.streaming import run_stream, time_entry_updates

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> pd.DataFrame:
    rng = make_rng(config.EXP2_SEED)
    A0, _, _ = low_rank_matrix(
        rng, config.RANK1_ROWS, config.RANK1_COLS, config.RANK1_TRUE_RANK
    )
    events = rank1_entry_stream(
        rng,
        config.RANK1_ROWS,
        config.RANK1_COLS,
        config.EXP2_N_STEPS,
        config.RANK1_DELTA_STD,
    )

    frames = []
    for k in config.EXP2_RANKS:
        df = run_stream(
            A0,
            k,
            events,
            refresh_period=config.EXP2_REFRESH_EVERY,
            log_every=config.EXP2_LOG_EVERY,
        )
        df["k"] = k
        frames.append(df)
        print(
            f"[exp2] k={k:3d} final frob_ratio={df['frob_ratio'].iloc[-1]:.4f} "
            f"final angle={df['angle_to_opt'].iloc[-1]:.4f} n_refreshes={df['n_refreshes'].iloc[-1]}"
        )

    all_df = pd.concat(frames, ignore_index=True)
    return all_df


def time_update_per_rank() -> Dict[int, float]:
    rng = make_rng(config.EXP2_SEED)
    A0, _, _ = low_rank_matrix(
        rng, config.RANK1_ROWS, config.RANK1_COLS, config.RANK1_TRUE_RANK
    )
    timings: Dict[int, float] = {}
    for k in config.EXP2_RANKS:
        timing_rng = make_rng(config.EXP2_SEED + 1000 + k)
        events = rank1_entry_stream(
            timing_rng,
            config.RANK1_ROWS,
            config.RANK1_COLS,
            config.EXP2_TIMING_STEPS,
            config.RANK1_DELTA_STD,
        )
        updates = [(i, j, delta) for (_, i, j, delta) in events]
        timings[k] = time_entry_updates(
            A0, k, updates, n_runs=config.EXP2_TIMING_RUNS
        )
    return timings


def summarize(all_df: pd.DataFrame) -> pd.DataFrame:
    tail = all_df[all_df["step"] >= 100]
    rows = []
    for k, df in tail.groupby("k"):
        mean_e_inc = float(df["E_inc"].mean())
        mean_e_opt = float(df["E_opt"].mean())
        rows.append(
            {
                "k": int(k),
                "mean_frob_ratio": float(df["frob_ratio"].mean()),
                "mean_E_inc": mean_e_inc,
                "mean_E_opt": mean_e_opt,
                "mean_gap": mean_e_inc - mean_e_opt,
                "mean_angle_to_opt": float(df["angle_to_opt"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def plot(all_df: pd.DataFrame) -> None:
    tail = all_df[all_df["step"] >= 100]
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, df in tail.groupby("k"):
        ax.plot(df["step"], df["frob_ratio"], label=f"k={k}", linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step t")
    ax.set_ylabel("Error ratio E_inc(t) / E_opt(t)")
    ax.set_title(
        "Effect of truncation rank k on the Frobenius error ratio (Sec. 4.2)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_truncation_rank.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    all_df = run()
    summary = summarize(all_df)

    timing = time_update_per_rank()
    summary["mean_update_time_s"] = summary["k"].map(timing)

    summary.to_csv(RESULTS_DIR / "exp2_summary.csv", index=False)
    plot(all_df)
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
