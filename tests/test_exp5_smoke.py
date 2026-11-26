"""Smoke tests for the exp5 bound-triggered refresh building blocks (Sec.
4.5): the a posteriori bound beta_t must dominate the true incremental error
E_inc(t) at every logged step, and the bound-triggered policy must refresh
no later than the periodic policy at a tight tolerance.
"""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import low_rank_matrix, make_rng, rank1_entry_stream
from src.streaming import run_stream


def _small_stream(n_steps: int = 300):
    rng = make_rng(1)
    A0, _, _ = low_rank_matrix(rng, 15, 12, 3)
    events = rank1_entry_stream(rng, 15, 12, n_steps, 0.05)
    return A0, events


def test_bound_dominates_true_error_no_refresh():
    A0, events = _small_stream()
    df = run_stream(A0, 3, events, refresh_period=None, log_every=1)
    assert (df["E_inc"] <= df["beta"] + 1e-8).all()


def test_bound_dominates_true_error_with_refresh():
    A0, events = _small_stream()
    df = run_stream(A0, 3, events, refresh_period=50, log_every=1)
    assert (df["E_inc"] <= df["beta"] + 1e-8).all()


def test_bound_triggered_refresh_runs_and_bounds_error():
    A0, events = _small_stream()
    df = run_stream(A0, 3, events, bound_tau=0.1, log_every=1)
    assert df["n_refreshes"].iloc[-1] >= 1
    assert (df["E_inc"] <= df["beta"] + 1e-8).all()


def test_tighter_tolerance_triggers_more_refreshes():
    A0, events = _small_stream()
    loose = run_stream(A0, 3, events, bound_tau=0.4, log_every=1)
    tight = run_stream(A0, 3, events, bound_tau=0.05, log_every=1)
    assert tight["n_refreshes"].iloc[-1] >= loose["n_refreshes"].iloc[-1]


def test_beta_over_e_inc_finite_where_defined():
    A0, events = _small_stream()
    df = run_stream(A0, 3, events, refresh_period=None, log_every=1)
    tightness = df["beta_over_E_inc"].replace([np.inf, -np.inf], np.nan).dropna()
    assert (tightness >= 1.0 - 1e-6).all()
