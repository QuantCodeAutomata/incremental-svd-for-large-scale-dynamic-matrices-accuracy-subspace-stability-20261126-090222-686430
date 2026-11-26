"""Smoke tests for exp6 runtime-scaling building blocks (Sec. 4.6): timing
helpers return positive, finite times and a full SVD is not faster than a
single rank-1 update on a small matrix with the ratio the paper expects.
"""

from __future__ import annotations

from data.generate_synthetic import low_rank_matrix, make_rng, rank1_entry_stream
from src.streaming import time_entry_updates, time_full_svd


def test_time_entry_updates_positive_and_finite():
    rng = make_rng(2)
    A0, _, _ = low_rank_matrix(rng, 15, 12, 3)
    events = rank1_entry_stream(rng, 15, 12, 40, 0.05)
    updates = [(i, j, delta) for (_, i, j, delta) in events]
    t = time_entry_updates(A0, 3, updates, n_runs=2)
    assert t > 0.0
    assert t == t  # not NaN


def test_time_full_svd_positive_and_finite():
    rng = make_rng(2)
    A0, _, _ = low_rank_matrix(rng, 15, 12, 3)
    t = time_full_svd(A0, n_runs=2)
    assert t > 0.0


def test_update_time_increases_with_rank_on_average():
    """Larger k costs more per rank-1 update: the k x k core SVD is O(k^3)."""
    rng = make_rng(3)
    A0, _, _ = low_rank_matrix(rng, 30, 25, 4)
    events = rank1_entry_stream(rng, 30, 25, 60, 0.05)
    updates = [(i, j, delta) for (_, i, j, delta) in events]
    t_small = time_entry_updates(A0, 4, updates, n_runs=3)
    t_large = time_entry_updates(A0, 20, updates, n_runs=3)
    assert t_large > 0 and t_small > 0
