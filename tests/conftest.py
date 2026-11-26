"""Shared pytest fixtures and marker registration."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test requires a live QuantX Dataloader/Massive call"
    )


@pytest.fixture(scope="session")
def etf_price_slice(request: pytest.FixtureRequest):
    """A small, cached slice of the experiment-7 ETF universe: 5 tickers over
    2022-2023, used only by market_data tests (never committed to the repo).
    """
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    cache = request.config.cache
    cached = cache.get("etf_price_slice/data", None)
    if cached is not None:
        import pandas as pd

        return pd.DataFrame(cached["prices"]).set_index(
            pd.to_datetime(cached["index"])
        )
    from data.data_loader import load_price_panel

    tickers = ["SPY", "QQQ", "TLT", "GLD", "IWM"]
    prices, _meta = load_price_panel(tickers, "2022-01-01", "2023-12-31", use_cache=True)
    cache.set(
        "etf_price_slice/data",
        {
            "prices": prices.reset_index(drop=True).to_dict(orient="list"),
            "index": [str(d) for d in prices.index],
        },
    )
    return prices
