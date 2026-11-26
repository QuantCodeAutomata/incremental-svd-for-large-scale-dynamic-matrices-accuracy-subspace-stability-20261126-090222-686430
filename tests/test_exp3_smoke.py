"""Smoke tests for the experiment-3 structural-growth pipeline (Sec. 4.3):
row and column appends into an incremental SVD.

Exercises the underlying data-generation and streaming building blocks on a
small synthetic instance, under both refresh policies, without invoking
`main()`/`plot()` or writing any files.
"""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import (
    col_append_stream,
    low_rank_matrix_noisy,
    make_rng,
    row_append_stream,
)
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


def _small_row_stream():
    rng = make_rng(0)
    m, n, true_rank, n_steps = 10, 8, 3, 20
    A0, _U_true, V_true = low_rank_matrix_noisy(rng, m, n, true_rank, 0.05)
    events = row_append_stream(rng, V_true, true_rank, n_steps, 0.05)
    return A0, events


def _small_col_stream():
    rng = make_rng(0)
    m, n, true_rank, n_steps = 10, 8, 3, 20
    A0, U_true, _V_true = low_rank_matrix_noisy(rng, m, n, true_rank, 0.05)
    events = col_append_stream(rng, U_true, true_rank, n_steps, 0.05)
    return A0, events


def test_row_stream_has_expected_columns_and_valid_ratio_under_both_policies():
    A0, events = _small_row_stream()
    for refresh_period in (None, 5):
        df = run_stream(A0, k=2, events=events, refresh_period=refresh_period)
        assert _EXPECTED_COLUMNS.issubset(df.columns)
        assert len(df) > 0
        assert (df["frob_ratio"] >= 0).all()
        assert np.isfinite(df["frob_ratio"]).all()


def test_col_stream_has_expected_columns_and_valid_ratio_under_both_policies():
    A0, events = _small_col_stream()
    for refresh_period in (None, 5):
        df = run_stream(A0, k=2, events=events, refresh_period=refresh_period)
        assert _EXPECTED_COLUMNS.issubset(df.columns)
        assert len(df) > 0
        assert (df["frob_ratio"] >= 0).all()
        assert np.isfinite(df["frob_ratio"]).all()


def test_row_appends_grow_the_row_dimension_over_the_stream():
    A0, events = _small_row_stream()
    df = run_stream(A0, k=2, events=events, refresh_period=None, log_every=1)
    assert df["m"].iloc[-1] > df["m"].iloc[0]
    assert (df["m"].diff().dropna() >= 0).all()
