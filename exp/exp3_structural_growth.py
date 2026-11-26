"""Experiment 3: structural growth, incremental SVD under row and column
appends (Staykov, 2026, Sec. 4.3).

A 50x40 matrix is sampled from a rank-6 generating model with additive
Gaussian noise (true factors U_true, V_true). Two independent streams of
200 structural-growth events are then applied to a rank-4 incremental SVD
(below the true rank 6, as in the paper's methodology):

* a row-append stream, where each new row is drawn from the same rank-6
  model (u_new @ V_true.T + noise) and appended to the bottom of the
  matrix, growing its row dimension;
* a column-append stream, where each new column is drawn analogously
  (U_true @ v_new + noise) and appended to the matrix, growing its column
  dimension.

For each stream, two refresh policies are compared: no periodic refresh,
and a periodic refresh every 50 appends. At every logged step the Frobenius
error ratio E_inc(t) / E_opt(t) against a full-SVD oracle and the principal
angle to the optimal rank-k subspace are recorded, to assess how well the
incremental SVD tracks the optimal low-rank subspace as the matrix grows
structurally, and how periodic refresh affects that tracking accuracy
(Sec. 4.3).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import (
    col_append_stream,
    low_rank_matrix_noisy,
    make_rng,
    row_append_stream,
)
from src import config
from src.streaming import run_stream

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _policy_label(period: "int | None") -> str:
    return "no_refresh" if period is None else f"refresh_{period}"


def run_row_streaming() -> pd.DataFrame:
    """Run the row-append stream under both refresh policies (Sec. 4.3)."""
    rng = make_rng(config.EXP3_SEED)
    A0, _U_true, V_true = low_rank_matrix_noisy(
        rng,
        config.EXP3_INIT_ROWS,
        config.EXP3_INIT_COLS,
        config.EXP3_TRUE_RANK,
        config.EXP3_NOISE_STD,
    )
    events = row_append_stream(
        rng, V_true, config.EXP3_TRUE_RANK, config.EXP3_N_APPENDS, config.EXP3_NOISE_STD
    )

    frames = []
    for period in (None, config.EXP3_REFRESH_EVERY):
        label = _policy_label(period)
        df = run_stream(
            A0,
            config.EXP3_WORKING_RANK,
            events,
            refresh_period=period,
            log_every=config.EXP3_LOG_EVERY,
        )
        df["policy"] = label
        df["stream"] = "row"
        frames.append(df)
        print(
            f"[exp3] stream=row policy={label:14s} final frob_ratio={df['frob_ratio'].iloc[-1]:.4f} "
            f"final angle={df['angle_to_opt'].iloc[-1]:.4f} n_refreshes={df['n_refreshes'].iloc[-1]}"
        )

    return pd.concat(frames, ignore_index=True)


def run_col_streaming() -> pd.DataFrame:
    """Run the column-append stream under both refresh policies (Sec. 4.3)."""
    rng2 = make_rng(config.EXP3_SEED + 1)
    A0c, U_truec, _V_truec = low_rank_matrix_noisy(
        rng2,
        config.EXP3_INIT_ROWS,
        config.EXP3_INIT_COLS,
        config.EXP3_TRUE_RANK,
        config.EXP3_NOISE_STD,
    )
    events_c = col_append_stream(
        rng2, U_truec, config.EXP3_TRUE_RANK, config.EXP3_N_APPENDS, config.EXP3_NOISE_STD
    )

    frames = []
    for period in (None, config.EXP3_REFRESH_EVERY):
        label = _policy_label(period)
        df = run_stream(
            A0c,
            config.EXP3_WORKING_RANK,
            events_c,
            refresh_period=period,
            log_every=config.EXP3_LOG_EVERY,
        )
        df["policy"] = label
        df["stream"] = "col"
        frames.append(df)
        print(
            f"[exp3] stream=col policy={label:14s} final frob_ratio={df['frob_ratio'].iloc[-1]:.4f} "
            f"final angle={df['angle_to_opt'].iloc[-1]:.4f} n_refreshes={df['n_refreshes'].iloc[-1]}"
        )

    return pd.concat(frames, ignore_index=True)


def summarize(all_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(stream, policy) aggregate summary of tracking accuracy."""
    rows = []
    for (stream, policy), df in all_df.groupby(["stream", "policy"]):
        rows.append(
            {
                "stream": stream,
                "policy": policy,
                "final_frob_ratio": float(df["frob_ratio"].iloc[-1]),
                "mean_frob_ratio": float(df["frob_ratio"].mean()),
                "max_frob_ratio": float(df["frob_ratio"].max()),
                "final_angle_to_opt": float(df["angle_to_opt"].iloc[-1]),
                "n_appends": config.EXP3_N_APPENDS,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["stream", "policy"])
        .reset_index(drop=True)
    )


def plot(all_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    stream_titles = {
        "row": "Row appends: Frobenius error ratio vs. step (Sec. 4.3)",
        "col": "Column appends: Frobenius error ratio vs. step (Sec. 4.3)",
    }
    for ax, stream in zip(axes, ("row", "col")):
        stream_df = all_df[all_df["stream"] == stream]
        for policy, df in stream_df.groupby("policy"):
            ax.plot(df["step"], df["frob_ratio"], label=policy, linewidth=1.2)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Step t")
        ax.set_title(stream_titles[stream])
        ax.legend()
    axes[0].set_ylabel("Error ratio E_inc(t) / E_opt(t)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_structural_growth.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    row_df = run_row_streaming()
    col_df = run_col_streaming()
    all_df = pd.concat([row_df, col_df], ignore_index=True)

    summary = summarize(all_df)
    summary.to_csv(RESULTS_DIR / "exp3_summary.csv", index=False)
    plot(all_df)
    with open(RESULTS_DIR / "exp3_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
