"""Experiment 1: rank-1 streaming, tracking and drift under periodic refresh
(Staykov, 2026, Sec. 4.1).

Streams 10,000 random rank-1 entry updates into a 50x40, true-rank-5 matrix
under three refresh policies (no refresh, every 1000 steps, every 5000
steps), logging the Frobenius error ratio and the principal angle to the
optimal rank-k subspace against a full-SVD oracle at every logged step.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import low_rank_matrix, make_rng, rank1_entry_stream
from src import config
from src.streaming import run_stream

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> pd.DataFrame:
    rng = make_rng(config.EXP1_SEED)
    A0, _, _ = low_rank_matrix(
        rng, config.RANK1_ROWS, config.RANK1_COLS, config.RANK1_TRUE_RANK
    )
    events = rank1_entry_stream(
        rng,
        config.RANK1_ROWS,
        config.RANK1_COLS,
        config.EXP1_N_STEPS,
        config.RANK1_DELTA_STD,
    )

    policy_labels = {None: "no_refresh", 1000: "refresh_1000", 5000: "refresh_5000"}
    frames = []
    for period in config.EXP1_REFRESH_PERIODS:
        label = policy_labels[period]
        df = run_stream(
            A0,
            config.EXP1_TRUNCATION_RANK,
            events,
            refresh_period=period,
            log_every=config.EXP1_LOG_EVERY,
        )
        df["policy"] = label
        frames.append(df)
        print(
            f"[exp1] policy={label:14s} final frob_ratio={df['frob_ratio'].iloc[-1]:.4f} "
            f"final angle={df['angle_to_opt'].iloc[-1]:.4f} n_refreshes={df['n_refreshes'].iloc[-1]}"
        )

    all_df = pd.concat(frames, ignore_index=True)
    return all_df


def summarize(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, df in all_df.groupby("policy"):
        rows.append(
            {
                "policy": policy,
                "final_frob_ratio": float(df["frob_ratio"].iloc[-1]),
                "mean_frob_ratio": float(df["frob_ratio"].mean()),
                "max_frob_ratio": float(df["frob_ratio"].max()),
                "final_angle_to_opt": float(df["angle_to_opt"].iloc[-1]),
                "mean_angle_to_opt": float(df["angle_to_opt"].mean()),
                "max_angle_to_opt": float(df["angle_to_opt"].max()),
                "n_refreshes": int(df["n_refreshes"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)


def plot(all_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for policy, df in all_df.groupby("policy"):
        axes[0].plot(df["step"], df["frob_ratio"], label=policy, linewidth=1.2)
        axes[1].plot(df["step"], df["angle_to_opt"], label=policy, linewidth=1.2)
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Error ratio E_inc(t) / E_opt(t)")
    axes[0].set_title("Rank-1 streaming: Frobenius error ratio vs. step (Sec. 4.1)")
    axes[0].legend()
    axes[1].set_ylabel("Principal angle to optimal subspace (rad)")
    axes[1].set_xlabel("Step t")
    axes[1].set_title("Principal angle to the optimal rank-5 subspace vs. step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_rank1_streaming.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    all_df = run()
    summary = summarize(all_df)
    summary.to_csv(RESULTS_DIR / "exp1_summary.csv", index=False)
    plot(all_df)
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
