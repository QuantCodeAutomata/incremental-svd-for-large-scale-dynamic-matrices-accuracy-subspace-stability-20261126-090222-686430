"""Online factor model and covariance/risk evaluation on ETF returns
(Sec. 5 of Staykov, 2026).

Reuses `IncrementalSVD` (row appends) and the refresh policies of
`src.isvd_policies`; the covariance and portfolio-risk formulas below are a
custom implementation of eqs. (18)-(22) of the paper (Sec. 5.1-5.2). Context7
was queried for `pyportfolioopt` covariance estimators (`CovarianceShrinkage`,
sample covariance); none of them expose an online, rank-truncated covariance
estimator built from an incrementally maintained SVD, so the low-rank
covariance below is implemented directly from the paper's definitions.
"""

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data.data_loader import load_price_panel
from src.isvd_core import IncrementalSVD
from src.isvd_metrics import optimal_rank_k, principal_angle
from src.isvd_policies import should_refresh_bound, should_refresh_periodic


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_{t,i} = log(P_{t,i} / P_{t-1,i}) (paper Sec. 5, data_requirements)."""
    log_prices = np.log(prices)
    returns = log_prices.diff().dropna(how="any")
    return returns


def causal_demean(returns: np.ndarray) -> np.ndarray:
    """Expanding-mean causal de-meaning (paper Sec. 5, Option A):
    mu_t = mu_{t-1} + (1/t)(r_t - mu_{t-1}), R_t = r_t - mu_t. Implemented via
    the equivalent closed form mu_t = (1/t) sum_{s<=t} r_s to avoid an
    explicit Python loop; the result is identical to the online recurrence."""
    T = returns.shape[0]
    counts = np.arange(1, T + 1).reshape(-1, 1)
    running_mean = np.cumsum(returns, axis=0) / counts
    return returns - running_mean


def load_price_data_and_returns(
    tickers: List[str], start_date: str, end_date: str, use_cache: bool = True
) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """Causally de-meaned daily log-return matrix R (T x N) for `tickers`
    over [start_date, end_date], after dropping days with any missing price
    (paper Sec. 5, data_requirements). No per-asset volatility standardization
    is applied, per the paper's explicit instruction to work in return units.
    """
    prices, meta = load_price_panel(tickers, start_date, end_date, use_cache=use_cache)
    returns = compute_log_returns(prices)
    R = causal_demean(returns.to_numpy())
    return R, returns.index, list(returns.columns), meta


def low_rank_covariance(V_k: np.ndarray, S_k: np.ndarray, T_eff: int) -> np.ndarray:
    """Sigma_hat_k(t) = (1/(t-1)) V_k diag(sigma_1^2,...,sigma_k^2) V_k^T,
    the rank-k factor covariance of the causally centered returns (paper Sec.
    5.1, eq. for Sigma_full,k(t) / Sigma_inc,k(t))."""
    if T_eff <= 1:
        raise ValueError("T_eff must be at least 2 to form a covariance estimate")
    return (V_k * (S_k ** 2)) @ V_k.T / (T_eff - 1)


def covariance_rel_error(Sigma_inc: np.ndarray, Sigma_opt: np.ndarray) -> float:
    """err_Sigma(t) = ||Sigma_inc,k(t) - Sigma_opt,k(t)||_F / ||Sigma_opt,k(t)||_F."""
    denom = float(np.linalg.norm(Sigma_opt, ord="fro"))
    if denom <= 1e-14:
        return 0.0
    return float(np.linalg.norm(Sigma_inc - Sigma_opt, ord="fro") / denom)


def portfolio_risk(w: np.ndarray, Sigma: np.ndarray) -> float:
    """sigma(t) = sqrt(w^T Sigma w), clipped at 0 for numerical robustness."""
    val = float(w @ Sigma @ w)
    return float(np.sqrt(max(val, 0.0)))


def risk_rel_error(sigma_inc: float, sigma_opt: float) -> float:
    """err_risk(t) = |sigma_inc(t) - sigma_opt(t)| / sigma_opt(t)."""
    if sigma_opt <= 1e-14:
        return 0.0
    return float(abs(sigma_inc - sigma_opt) / sigma_opt)


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """Long-only equal-weight portfolio w = 1/N (paper Sec. 5.2)."""
    return np.full(n_assets, 1.0 / n_assets)


def random_long_short_portfolios(
    rng: np.random.Generator, n_assets: int, n_portfolios: int
) -> List[np.ndarray]:
    """`n_portfolios` random long-short portfolios with iid standard-normal
    entries scaled to unit norm (paper Sec. 5.2)."""
    portfolios = []
    for _ in range(n_portfolios):
        w = rng.standard_normal(n_assets)
        w = w / np.linalg.norm(w)
        portfolios.append(w)
    return portfolios


def _evaluate_finance_step(
    inc: IncrementalSVD,
    k: int,
    w_eq: np.ndarray,
    w_rand_list: List[np.ndarray],
) -> dict:
    T_eff = inc.A.shape[0]
    U_opt, S_opt, Vt_opt = optimal_rank_k(inc.A, k)
    V_opt = Vt_opt.T
    angle = principal_angle(inc.U[:, :k], U_opt)

    Sigma_opt = low_rank_covariance(V_opt, S_opt, T_eff)
    Sigma_inc = low_rank_covariance(inc.V[:, :k], inc.S[:k], T_eff)
    cov_err = covariance_rel_error(Sigma_inc, Sigma_opt)

    sigma_opt_eq = portfolio_risk(w_eq, Sigma_opt)
    sigma_inc_eq = portfolio_risk(w_eq, Sigma_inc)
    risk_err_eq = risk_rel_error(sigma_inc_eq, sigma_opt_eq)

    rand_errs = []
    for w in w_rand_list:
        s_opt = portfolio_risk(w, Sigma_opt)
        s_inc = portfolio_risk(w, Sigma_inc)
        rand_errs.append(risk_rel_error(s_inc, s_opt))

    return {
        "angle_factor": angle,
        "cov_rel_error": cov_err,
        "risk_rel_err_eq": risk_err_eq,
        "risk_rel_err_rand_mean": float(np.mean(rand_errs)) if rand_errs else np.nan,
        "risk_rel_err_rand_median": float(np.median(rand_errs)) if rand_errs else np.nan,
        "n_rows": T_eff,
        "beta_over_frob_norm": inc.beta / inc.frob_norm() if inc.frob_norm() > 1e-12 else np.nan,
    }


def run_finance_stream(
    R: np.ndarray,
    rank_k: int,
    T0: int,
    refresh_period: Optional[int] = None,
    bound_tau: Optional[float] = None,
    eval_every: int = 5,
    n_random_portfolios: int = 10,
    portfolio_seed: int = 0,
    tol: float = 1e-10,
) -> Tuple[pd.DataFrame, dict]:
    """Stream the rows of `R` (days T0..T-1) into an `IncrementalSVD` of rank
    `rank_k` initialized on `R[:T0]`, under at most one refresh policy, and
    evaluate factor-subspace angle, relative covariance error and relative
    portfolio risk error every `eval_every` streamed rows (paper Sec. 5.1-5.2).

    Returns (metrics_df, summary) with summary = {mean_update_time,
    total_update_time, n_stream_events, n_refreshes}.
    """
    if refresh_period is not None and bound_tau is not None:
        raise ValueError("choose at most one of refresh_period, bound_tau")
    T, N = R.shape
    rng = np.random.default_rng(portfolio_seed)
    w_eq = equal_weight_portfolio(N)
    w_rand_list = random_long_short_portfolios(rng, N, n_random_portfolios)

    inc = IncrementalSVD(R[:T0, :].copy(), rank=rank_k, tol=tol)
    update_times: List[float] = []
    rows: List[dict] = []
    n_refreshes = 0

    for step, idx in enumerate(range(T0, T), start=1):
        t0 = perf_counter()
        inc.add_row(R[idx, :])
        did_refresh = False
        if refresh_period is not None and should_refresh_periodic(step, refresh_period):
            inc.refresh()
            did_refresh = True
        elif bound_tau is not None and should_refresh_bound(inc, bound_tau):
            inc.refresh()
            did_refresh = True
        update_times.append(perf_counter() - t0)
        if did_refresh:
            n_refreshes += 1
        if did_refresh or step % eval_every == 0 or idx == T - 1:
            metrics = _evaluate_finance_step(inc, rank_k, w_eq, w_rand_list)
            metrics["t"] = idx + 1
            metrics["refreshed"] = did_refresh
            metrics["n_refreshes"] = n_refreshes
            rows.append(metrics)

    summary = {
        "mean_update_time": float(np.mean(update_times)) if update_times else 0.0,
        "total_update_time": float(np.sum(update_times)) if update_times else 0.0,
        "n_stream_events": float(T - T0),
        "n_refreshes": float(n_refreshes),
    }
    return pd.DataFrame(rows), summary
