"""Experiment 4: mixed streaming with heterogeneous update types, rank-1
entry perturbations interleaved with periodic row appends (Staykov, 2026,
Sec. 4.4, "Mixed Streaming: Heterogeneous Update Types with Rank-1 and Row
Appends").

A 50x40, true-rank-5 matrix is streamed for 2,000 steps, where at every step
either a random rank-1 entry update is applied, or, once every
`row_period = 50` steps, a new row sampled from the rank-5 model (plus
noise) is appended, so the row dimension of the matrix grows over the
stream. The incremental SVD is tracked at a fixed truncation rank under two
refresh policies, no periodic refresh and a periodic refresh every 200
steps, and the Frobenius error ratio and principal angle to the optimal
rank-k subspace are logged against a full-SVD oracle at every logged step,
to show that the incremental scheme accommodates both update types within a
single stream and that periodic refresh keeps the error ratio close to 1
even as the matrix grows.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import low_rank_matrix, make_rng, mixed_rank1_and_row_stream
from src import config
from src.streaming import run_stream

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> pd.DataFrame:
    rng = make_rng(config.EXP4_SEED)
    A0, U_true, V_true = low_rank_matrix(
        rng, config.EXP4_ROWS, config.EXP4_COLS, config.EXP4_TRUE_RANK
    )
    events = mixed_rank1_and_row_stream(
        rng,
        config.EXP4_ROWS,
        config.EXP4_COLS,
        V_true,
        config.EXP4_TRUE_RANK,
        config.EXP4_N_STEPS,
        config.EXP4_ROW_PERIOD,
        config.EXP4_DELTA_STD,
        config.EXP4_ROW_NOISE_STD,
    )

    policy_labels = {None: "no_refresh", config.EXP4_REFRESH_EVERY: f"refresh_{config.EXP4_REFRESH_EVERY}"}
    frames = []
    for policy_period in (None, config.EXP4_REFRESH_EVERY):
        label = policy_labels[policy_period]
        df = run_stream(
            A0,
            config.EXP4_TRUNCATION_RANK,
            events,
            refresh_period=policy_period,
            log_every=config.EXP4_LOG_EVERY,
        )
        df["policy"] = label
        frames.append(df)
        print(
            f"[exp4] policy={label:14s} final frob_ratio={df['frob_ratio'].iloc[-1]:.4f} "
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
                "n_refreshes": int(df["n_refreshes"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)


def plot(all_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy, df in all_df.groupby("policy"):
        ax.plot(df["step"], df["frob_ratio"], label=policy, linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step t")
    ax.set_ylabel("Error ratio E_inc(t) / E_opt(t)")
    ax.set_title(
        "Mixed streaming: rank-1 updates and row appends, Frobenius error ratio vs. step (Sec. 4.4)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_mixed_streaming.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    all_df = run()
    summary = summarize(all_df)
    summary.to_csv(RESULTS_DIR / "exp4_summary.csv", index=False)
    plot(all_df)
    with open(RESULTS_DIR / "exp4_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
