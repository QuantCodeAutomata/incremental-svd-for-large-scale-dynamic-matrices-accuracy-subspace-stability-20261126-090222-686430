"""Experiment 7: online factor model and covariance estimation via
incremental SVD on ETF returns (Staykov, 2026, Sec. 5).

Streams daily causally de-meaned log-returns of a panel of liquid US-listed
ETFs into a row-append `IncrementalSVD`, evaluating the drift of the
incremental factor subspace, the relative error of the induced low-rank
covariance estimate, and the relative error of portfolio risk, against a
full-SVD oracle recomputed on the same causally centered return history.
Reuses `src.finance.run_finance_stream`, which carries the row-append engine,
the refresh policies and the covariance/risk formulas shared with Sec. 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.finance import load_price_data_and_returns, run_finance_stream

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_data():
    R, dates, tickers, meta = load_price_data_and_returns(
        config.EXP7_TICKERS, config.EXP7_START_DATE, config.EXP7_END_DATE
    )
    print(
        f"[exp7] loaded {R.shape[0]} trading days x {R.shape[1]} tickers, "
        f"{dates[0].date()} to {dates[-1].date()}"
    )
    non_massive = meta.get("non_massive_source", {})
    if non_massive:
        print(f"[exp7] tickers with non-Massive data_source: {non_massive}")
    failed = meta.get("failed", {})
    if failed:
        print(f"[exp7] tickers that failed to load: {failed}")
    return R, dates, tickers, meta


def run_primary_timeseries(R):
    """Primary rank k=5 under no_refresh, refresh_100, refresh_20 (Sec. 5)."""
    frames = []
    summaries = {}
    for refresh in config.EXP7_TIMESERIES_REFRESH:
        label = "no_refresh" if refresh is None else f"refresh_{refresh}"
        df, summary = run_finance_stream(
            R,
            config.EXP7_PRIMARY_RANK,
            config.EXP7_INITIAL_WINDOW,
            refresh_period=refresh,
            eval_every=config.EXP7_EVAL_EVERY,
            n_random_portfolios=config.EXP7_N_RANDOM_PORTFOLIOS,
            portfolio_seed=config.EXP7_PORTFOLIO_SEED,
        )
        df["policy"] = label
        frames.append(df)
        summaries[label] = summary
        print(
            f"[exp7] primary k={config.EXP7_PRIMARY_RANK} policy={label} "
            f"final angle={df['angle_factor'].iloc[-1]:.4f} "
            f"final cov_err={df['cov_rel_error'].iloc[-1]:.2e} "
            f"final risk_err_eq={df['risk_rel_err_eq'].iloc[-1]:.2e}"
        )
    return pd.concat(frames, ignore_index=True), summaries


def run_rank_refresh_grid(R):
    """Final-day angle, covariance error, portfolio risk error and mean
    update time over k in {3,5,8,12,20} x refresh in {None,30,50,100}
    (Sec. 5.3)."""
    rows = []
    for k in config.EXP7_GRID_RANKS:
        for refresh in config.EXP7_GRID_REFRESH:
            df, summary = run_finance_stream(
                R,
                k,
                config.EXP7_INITIAL_WINDOW,
                refresh_period=refresh,
                eval_every=config.EXP7_EVAL_EVERY,
                n_random_portfolios=config.EXP7_N_RANDOM_PORTFOLIOS,
                portfolio_seed=config.EXP7_PORTFOLIO_SEED,
            )
            last = df.iloc[-1]
            rows.append(
                {
                    "k": k,
                    "refresh_every": "none" if refresh is None else refresh,
                    "final_angle": float(last["angle_factor"]),
                    "final_cov_rel_error": float(last["cov_rel_error"]),
                    "final_risk_rel_err_eq": float(last["risk_rel_err_eq"]),
                    "final_risk_rel_err_rand_mean": float(last["risk_rel_err_rand_mean"]),
                    "mean_update_time_s": summary["mean_update_time"],
                    "n_refreshes": summary["n_refreshes"],
                }
            )
            print(f"[exp7] grid k={k} refresh={refresh} final_angle={last['angle_factor']:.4f}")
    return pd.DataFrame(rows)


def run_bound_triggered(R):
    """Bound-triggered refresh (eq. 16) for k in {5,8,12}, tau in
    {0.5,0.6,0.7} (Sec. 5.3)."""
    rows = []
    for k in config.EXP7_BOUND_RANKS:
        for tau in config.EXP7_BOUND_TAUS:
            df, summary = run_finance_stream(
                R,
                k,
                config.EXP7_INITIAL_WINDOW,
                bound_tau=tau,
                eval_every=config.EXP7_EVAL_EVERY,
                n_random_portfolios=config.EXP7_N_RANDOM_PORTFOLIOS,
                portfolio_seed=config.EXP7_PORTFOLIO_SEED,
            )
            last = df.iloc[-1]
            rows.append(
                {
                    "k": k,
                    "tau": tau,
                    "final_angle": float(last["angle_factor"]),
                    "final_cov_rel_error": float(last["cov_rel_error"]),
                    "final_risk_rel_err_eq": float(last["risk_rel_err_eq"]),
                    "mean_update_time_s": summary["mean_update_time"],
                    "n_full_svds": summary["n_refreshes"] + 1,
                }
            )
    return pd.DataFrame(rows)


def summarize_primary(primary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics per refresh policy (final value, median, max over
    the streamed days) -- never a per-day series -- for the primary k=5 run
    (paper Sec. 5.1-5.2)."""
    rows = []
    for policy, df in primary_df.groupby("policy"):
        rows.append(
            {
                "policy": policy,
                "final_angle_factor": float(df["angle_factor"].iloc[-1]),
                "median_angle_factor": float(df["angle_factor"].median()),
                "max_angle_factor": float(df["angle_factor"].max()),
                "final_cov_rel_error": float(df["cov_rel_error"].iloc[-1]),
                "median_cov_rel_error": float(df["cov_rel_error"].median()),
                "max_cov_rel_error": float(df["cov_rel_error"].max()),
                "final_risk_rel_err_eq": float(df["risk_rel_err_eq"].iloc[-1]),
                "max_risk_rel_err_eq": float(df["risk_rel_err_eq"].max()),
                "final_risk_rel_err_rand_mean": float(df["risk_rel_err_rand_mean"].iloc[-1]),
            }
        )
    order = [
        "no_refresh" if r is None else f"refresh_{r}" for r in config.EXP7_TIMESERIES_REFRESH
    ]
    out = pd.DataFrame(rows).set_index("policy").loc[order].reset_index()
    return out


def plot_primary(primary_df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, df in primary_df.groupby("policy"):
        axes[0].plot(df["t"], df["angle_factor"], label=label, linewidth=1.0)
        axes[1].semilogy(df["t"], df["cov_rel_error"].clip(lower=1e-6), label=label, linewidth=1.0)
    axes[0].set_xlabel("Trading day index")
    axes[0].set_ylabel("Principal angle to oracle factor subspace (rad)")
    axes[0].set_title("Factor-subspace drift (Sec. 5.1)")
    axes[0].legend()
    axes[1].set_xlabel("Trading day index")
    axes[1].set_ylabel("Relative covariance error (log scale)")
    axes[1].set_title("Low-rank covariance error (Sec. 5.1)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp7_primary_timeseries.png", dpi=150)
    plt.close(fig)


def plot_runtime(grid_df) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for refresh, df in grid_df.groupby("refresh_every"):
        df = df.sort_values("k")
        ax.plot(df["k"], df["mean_update_time_s"] * 1e3, marker="o", label=f"refresh={refresh}")
    ax.set_xlabel("Truncation rank k")
    ax.set_ylabel("Mean update time (ms)")
    ax.set_title("Per-update runtime vs. k and refresh cadence (Sec. 5.3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp7_runtime_vs_k.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    R, dates, tickers, meta = load_data()

    primary_df, primary_summaries = run_primary_timeseries(R)
    grid_df = run_rank_refresh_grid(R)
    bound_df = run_bound_triggered(R)

    primary_agg = summarize_primary(primary_df)
    primary_agg.to_csv(RESULTS_DIR / "exp7_primary_summary.csv", index=False)
    grid_df.to_csv(RESULTS_DIR / "exp7_rank_refresh_grid.csv", index=False)
    bound_df.to_csv(RESULTS_DIR / "exp7_bound_triggered.csv", index=False)
    plot_primary(primary_df)
    plot_runtime(grid_df)

    with open(RESULTS_DIR / "exp7_summary.json", "w") as f:
        json.dump(
            {
                "n_trading_days": int(R.shape[0]),
                "n_tickers": int(R.shape[1]),
                "date_range": [str(dates[0].date()), str(dates[-1].date())],
                "non_massive_source": meta.get("non_massive_source", {}),
                "failed_tickers": meta.get("failed", {}),
                "primary_summaries": primary_summaries,
                "bound_triggered": bound_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(grid_df.to_string(index=False))
    print(bound_df.to_string(index=False))


if __name__ == "__main__":
    main()
