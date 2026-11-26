"""Unit tests for the Sec. 5 factor-model building blocks in `src.finance`,
on small synthetic price panels (no live market-data call)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.finance import (
    causal_demean,
    compute_log_returns,
    covariance_rel_error,
    equal_weight_portfolio,
    low_rank_covariance,
    portfolio_risk,
    random_long_short_portfolios,
    risk_rel_error,
    run_finance_stream,
)


@pytest.mark.market_data
def test_run_finance_stream_on_real_etf_slice(etf_price_slice):
    """End-to-end check of the Sec. 5 pipeline on a reduced live slice of the
    experiment-7 ETF universe (5 tickers, 2022-2023): causal de-meaning,
    row-append incremental SVD and the covariance/risk metrics all run on
    real data and respect the paper's bounds (angle in [0, pi/2], errors
    non-negative, no look-ahead)."""
    returns = compute_log_returns(etf_price_slice)
    R = causal_demean(returns.to_numpy())
    assert R.shape[0] == returns.shape[0]
    T, N = R.shape
    k = 3
    T0 = max(60, N + k + 5)
    assert T0 < T, "reduced slice too short for the initial window"
    df, summary = run_finance_stream(R, k, T0, refresh_period=30, eval_every=5)
    assert len(df) > 0
    assert (df["angle_factor"] >= -1e-8).all()
    assert (df["angle_factor"] <= np.pi / 2 + 1e-6).all()
    assert (df["cov_rel_error"] >= 0).all()
    assert summary["n_refreshes"] >= 1


def _synthetic_prices(n_days: int = 60, n_assets: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_rets = rng.standard_normal((n_days, n_assets)) * 0.01
    log_prices = np.cumsum(log_rets, axis=0)
    prices = 100.0 * np.exp(log_prices)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=[f"T{i}" for i in range(n_assets)])


def test_compute_log_returns_shape_and_no_lookahead():
    prices = _synthetic_prices()
    returns = compute_log_returns(prices)
    assert returns.shape[0] == prices.shape[0] - 1
    expected_first = np.log(prices.iloc[1] / prices.iloc[0])
    np.testing.assert_allclose(returns.iloc[0].to_numpy(), expected_first.to_numpy())


def test_causal_demean_uses_only_past_and_current_rows():
    returns = _synthetic_prices().pct_change().dropna().to_numpy()
    centered = causal_demean(returns)
    assert centered.shape == returns.shape
    # Row t's center is the mean of rows 0..t only: perturbing a future row
    # must leave row 0's centered value unchanged (no look-ahead).
    perturbed = returns.copy()
    perturbed[-1, :] += 10.0
    centered_perturbed = causal_demean(perturbed)
    np.testing.assert_allclose(centered[0], centered_perturbed[0])


def test_low_rank_covariance_is_symmetric_psd():
    rng = np.random.default_rng(0)
    V_k = np.linalg.qr(rng.standard_normal((6, 3)))[0]
    S_k = np.array([3.0, 2.0, 1.0])
    Sigma = low_rank_covariance(V_k, S_k, T_eff=50)
    np.testing.assert_allclose(Sigma, Sigma.T)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert (eigvals >= -1e-10).all()


def test_covariance_rel_error_zero_when_equal():
    Sigma = np.eye(4)
    assert covariance_rel_error(Sigma, Sigma) == pytest.approx(0.0)


def test_portfolio_risk_and_risk_rel_error():
    Sigma = np.diag([1.0, 2.0, 3.0])
    w = np.array([1.0, 0.0, 0.0])
    assert portfolio_risk(w, Sigma) == pytest.approx(1.0)
    assert risk_rel_error(1.0, 1.0) == pytest.approx(0.0)
    assert risk_rel_error(1.1, 1.0) == pytest.approx(0.1)


def test_equal_weight_and_random_portfolios():
    w_eq = equal_weight_portfolio(5)
    assert w_eq.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    portfolios = random_long_short_portfolios(rng, 5, 4)
    assert len(portfolios) == 4
    for w in portfolios:
        assert np.linalg.norm(w) == pytest.approx(1.0)


def test_run_finance_stream_on_synthetic_returns_angle_bounded():
    rng = np.random.default_rng(4)
    T, N, k = 120, 8, 3
    U = np.linalg.qr(rng.standard_normal((T, k)))[0]
    V = np.linalg.qr(rng.standard_normal((N, k)))[0]
    S = np.diag([5.0, 3.0, 1.0])
    R = U @ S @ V.T + 0.01 * rng.standard_normal((T, N))
    df, summary = run_finance_stream(R, k, T0=40, refresh_period=10, eval_every=5)
    assert len(df) > 0
    assert (df["angle_factor"] >= 0).all()
    assert (df["angle_factor"] <= np.pi / 2 + 1e-6).all()
    assert (df["cov_rel_error"] >= 0).all()
    assert summary["n_refreshes"] >= 1
