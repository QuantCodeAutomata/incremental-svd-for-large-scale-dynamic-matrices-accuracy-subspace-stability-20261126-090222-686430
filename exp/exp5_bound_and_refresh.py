"""Experiment 5: a posteriori error bound and bound-triggered refresh policy
(Staykov, 2026, Sec. 4.5).

For five random seeds, streams 10,000 rank-1 entry updates into a 50x40,
true-rank-5 matrix (the stream of seed 0 is identical to Sec. 4.1) under six
policies: no refresh, periodic refresh every 1000 steps, and bound-triggered
refresh (eq. 16) at tau in {0.4, 0.2, 0.1, 0.05}. At every logged step the
maintained a posteriori bound beta_t (Proposition 1) is checked against the
true incremental error E_inc(t), and its tightness beta_t/E_inc(t) is
tabulated together with the Frobenius error ratio, the principal angle to
the optimal subspace, and the number of full SVD recomputations per policy.
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

_POLICIES = [
    ("no_refresh", None, None),
    ("refresh_1000", config.EXP5_PERIODIC_REFRESH, None),
] + [(f"bound_tau_{tau}", None, tau) for tau in config.EXP5_BOUND_TAUS]


def run() -> pd.DataFrame:
    frames = []
    for seed in config.EXP5_SEEDS:
        rng = make_rng(seed)
        A0, _, _ = low_rank_matrix(
            rng, config.RANK1_ROWS, config.RANK1_COLS, config.RANK1_TRUE_RANK
        )
        events = rank1_entry_stream(
            rng,
            config.RANK1_ROWS,
            config.RANK1_COLS,
            config.EXP5_N_STEPS,
            config.RANK1_DELTA_STD,
        )
        for label, refresh_period, bound_tau in _POLICIES:
            df = run_stream(
                A0,
                config.EXP5_TRUNCATION_RANK,
                events,
                refresh_period=refresh_period,
                bound_tau=bound_tau,
                log_every=config.EXP5_LOG_EVERY,
                tol=config.EXP5_TOL,
            )
            df["policy"] = label
            df["seed"] = seed
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df


def verify_proposition_1c(all_df: pd.DataFrame) -> dict:
    """Checks beta_t >= E_inc(t) at every logged step (Proposition 1, bound
    holds up to roundoff) across every seed and policy."""
    gap = all_df["E_inc"] - all_df["beta"]
    return {
        "max_bound_violation": float(gap.max()),
        "bound_holds_up_to_roundoff": bool(gap.max() < 1e-8),
    }


def summarize(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, df in all_df.groupby("policy"):
        half_mask = df["step"] > (config.EXP5_N_STEPS / 2)
        tightness = df["beta_over_E_inc"].replace([np.inf, -np.inf], np.nan).dropna()
        n_svds_per_seed = df.groupby("seed")["n_refreshes"].max() + 1  # +1 for the initial SVD
        rows.append(
            {
                "policy": policy,
                "mean_frob_ratio": float(df["frob_ratio"].mean()),
                "max_frob_ratio": float(df["frob_ratio"].max()),
                "max_frob_ratio_second_half": float(df.loc[half_mask, "frob_ratio"].max())
                if half_mask.any()
                else float("nan"),
                "mean_angle_to_opt": float(df["angle_to_opt"].mean()),
                "max_angle_to_opt": float(df["angle_to_opt"].max()),
                "median_beta_over_E_inc": float(tightness.median()) if len(tightness) else float("nan"),
                "mean_n_full_svds": float(n_svds_per_seed.mean()),
            }
        )
    order = [label for label, _, _ in _POLICIES]
    out = pd.DataFrame(rows).set_index("policy").loc[order].reset_index()
    return out


def plot_bound_seed0(all_df: pd.DataFrame) -> None:
    """E_inc(t), beta_t and E_opt(t) on a log scale, no-refresh policy, seed 0
    (paper Sec. 4.5)."""
    df = all_df[(all_df["policy"] == "no_refresh") & (all_df["seed"] == 0)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(df["step"], df["E_inc"], label="E_inc(t)", linewidth=1.2)
    ax.semilogy(df["step"], df["beta"], label="beta_t (a posteriori bound)", linewidth=1.2)
    ax.semilogy(df["step"], df["E_opt"], label="E_opt(t)", linewidth=1.2)
    ax.set_xlabel("Step t")
    ax.set_ylabel("Frobenius error (log scale)")
    ax.set_title("A posteriori bound vs. true error, no refresh, seed 0 (Sec. 4.5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_bound_seed0.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame) -> None:
    """Trade-off between the number of full SVD recomputations and the mean
    error ratio, across policies (paper Sec. 4.5)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary["mean_n_full_svds"], summary["mean_frob_ratio"])
    for _, row in summary.iterrows():
        ax.annotate(row["policy"], (row["mean_n_full_svds"], row["mean_frob_ratio"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean number of full SVD recomputations")
    ax.set_ylabel("Mean error ratio E_inc(t)/E_opt(t)")
    ax.set_title("Tolerance tau vs. refresh budget trade-off (Sec. 4.5)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_tradeoff.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    all_df = run()
    bound_check = verify_proposition_1c(all_df)
    summary = summarize(all_df)

    summary.to_csv(RESULTS_DIR / "exp5_summary.csv", index=False)
    plot_bound_seed0(all_df)
    plot_tradeoff(summary)
    with open(RESULTS_DIR / "exp5_summary.json", "w") as f:
        json.dump(
            {"policies": summary.to_dict(orient="records"), "bound_check": bound_check},
            f,
            indent=2,
        )
    print(summary.to_string(index=False))
    print(bound_check)


if __name__ == "__main__":
    main()
