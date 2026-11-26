"""Smoke tests for the experiment-4 mixed streaming pipeline (Sec. 4.4):
rank-1 entry updates interleaved with periodic row appends.

Exercises the underlying data-generation and streaming building blocks on a
small synthetic instance, under both refresh policies used by
`exp.exp4_mixed_streaming.run`, without invoking `main()`/`plot()` or writing
any files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.generate_synthetic import low_rank_matrix, make_rng, mixed_rank1_and_row_stream
from src.streaming import run_stream

_EXPECTED_COLUMNS = {
    "step",
    "E_inc",
    "E_opt",
    "frob_ratio",
    "angle_to_opt",
    "beta",
    "frob_norm",
    "beta_over_frob_norm",
    "beta_over_E_inc",
    "update_time",
    "refreshed",
    "n_refreshes",
    "m",
    "n",
}


def _small_stream():
    rng = make_rng(0)
    m, n, true_rank = 10, 8, 2
    A0, _U_true, V_true = low_rank_matrix(rng, m, n, true_rank)
    events = mixed_rank1_and_row_stream(
        rng,
        m,
        n,
        V_true,
        true_rank,
        n_steps=30,
        row_period=10,
        delta_std=0.05,
        row_noise_std=0.05,
    )
    return A0, events


def test_run_stream_no_refresh_has_expected_columns_and_valid_ratio():
    A0, events = _small_stream()
    df = run_stream(A0, k=2, events=events, refresh_period=None)
    assert _EXPECTED_COLUMNS.issubset(df.columns)
    assert len(df) > 0
    assert (df["frob_ratio"] >= 0).all()


def test_run_stream_with_periodic_refresh_has_expected_columns_and_valid_ratio():
    A0, events = _small_stream()
    df = run_stream(A0, k=2, events=events, refresh_period=10)
    assert _EXPECTED_COLUMNS.issubset(df.columns)
    assert len(df) > 0
    assert (df["frob_ratio"] >= 0).all()
    assert df["n_refreshes"].iloc[-1] >= 1


def test_row_appends_grow_the_row_dimension_over_the_stream():
    A0, events = _small_stream()
    df = run_stream(A0, k=2, events=events, refresh_period=None, log_every=1)
    assert df["m"].iloc[-1] > df["m"].iloc[0]
    assert (df["m"].diff().dropna() >= 0).all()
