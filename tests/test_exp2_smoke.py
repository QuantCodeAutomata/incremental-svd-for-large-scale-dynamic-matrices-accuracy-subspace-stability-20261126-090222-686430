"""Smoke tests for `exp.exp2_truncation_rank` on a small, manually built
stream (not via src/config.py), checking basic structural properties of
`run_stream`'s output for different truncation ranks k (Sec. 4.2).
"""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import low_rank_matrix, make_rng, rank1_entry_stream
from src.streaming import run_stream

EXPECTED_COLUMNS = {
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


def _small_stream(n_steps: int = 50):
    rng = make_rng(0)
    A0, _, _ = low_rank_matrix(rng, 12, 10, 3)
    events = rank1_entry_stream(rng, 12, 10, n_steps, 0.05)
    return A0, events


def test_run_stream_columns_and_nonnegative_frob_ratio():
    A0, events = _small_stream()
    df = run_stream(A0, 3, events, refresh_period=20, log_every=5)
    assert EXPECTED_COLUMNS.issubset(set(df.columns))
    assert len(df) > 0
    assert (df["frob_ratio"] >= 0).all()


def test_errors_finite_and_nonnegative_for_different_ranks():
    A0, events = _small_stream()
    for k in (3, 5):
        df = run_stream(A0, k, events, refresh_period=20, log_every=5)
        assert np.isfinite(df["E_inc"]).all()
        assert np.isfinite(df["E_opt"]).all()
        assert (df["E_inc"] >= 0).all()
        assert (df["E_opt"] >= 0).all()


def test_last_step_logged_and_matches_stream_length():
    A0, events = _small_stream(n_steps=30)
    df = run_stream(A0, 3, events, refresh_period=None, log_every=10)
    assert df["step"].iloc[-1] == len(events)
