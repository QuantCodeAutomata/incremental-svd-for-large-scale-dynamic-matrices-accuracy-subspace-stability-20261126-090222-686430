"""Shared streaming runner: applies a sequence of update events to an
`IncrementalSVD` engine, optionally refreshing it, and logs accuracy metrics
against a full-SVD oracle. One implementation shared by experiments 1-6
(paper Secs. 4.1-4.6); experiment 7 uses the finance-specific runner in
`src/finance.py`, which reuses the same engine, policies and metrics.
"""

from __future__ import annotations

from time import perf_counter
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.isvd_core import IncrementalSVD
from src.isvd_metrics import evaluate_against_oracle
from src.isvd_policies import should_refresh_bound, should_refresh_periodic

Event = Union[Tuple[str, int, int, float], Tuple[str, np.ndarray]]


def apply_event(inc: IncrementalSVD, event: Event) -> None:
    """Dispatch one stream event onto the engine: ("rank1", i, j, delta),
    ("row", vector) or ("col", vector)."""
    kind = event[0]
    if kind == "rank1":
        _, i, j, delta = event
        inc.rank_one_entry_update(i, j, delta)
    elif kind == "row":
        _, vec = event
        inc.add_row(vec)
    elif kind == "col":
        _, vec = event
        inc.add_col(vec)
    else:
        raise ValueError(f"unknown stream event kind: {kind!r}")


def run_stream(
    A0: np.ndarray,
    k: int,
    events: Sequence[Event],
    refresh_period: Optional[int] = None,
    bound_tau: Optional[float] = None,
    log_every: int = 1,
    tol: float = 1e-10,
) -> pd.DataFrame:
    """Run an incremental SVD of rank `k` over `events`, under at most one
    refresh policy, logging accuracy metrics every `log_every` steps (always
    at the last step and at every refresh).

    Parameters
    ----------
    A0 : ndarray
        Initial dense matrix.
    k : int
        Truncation rank.
    events : sequence of event tuples
        See `apply_event` for the accepted event kinds.
    refresh_period : int or None
        Periodic refresh every `refresh_period` steps; mutually exclusive
        with `bound_tau`.
    bound_tau : float or None
        Bound-triggered refresh tolerance tau (eq. 16); mutually exclusive
        with `refresh_period`.
    log_every : int
        Steps between logged evaluations.
    tol : float
        Residual-norm tolerance passed to the engine.

    Returns
    -------
    DataFrame
        One row per logged step, with columns step, E_inc, E_opt, frob_ratio,
        angle_to_opt, beta, frob_norm, beta_over_frob_norm, beta_over_E_inc,
        update_time, refreshed, n_refreshes, m, n.
    """
    if refresh_period is not None and bound_tau is not None:
        raise ValueError("choose at most one of refresh_period, bound_tau")
    events = list(events)
    inc = IncrementalSVD(np.asarray(A0, dtype=float).copy(), rank=k, tol=tol)
    n_total = len(events)
    n_refreshes = 0
    rows: List[dict] = []
    for step, event in enumerate(events, start=1):
        t0 = perf_counter()
        apply_event(inc, event)
        did_refresh = False
        if refresh_period is not None and should_refresh_periodic(step, refresh_period):
            inc.refresh()
            did_refresh = True
        elif bound_tau is not None and should_refresh_bound(inc, bound_tau):
            inc.refresh()
            did_refresh = True
        update_time = perf_counter() - t0
        if did_refresh:
            n_refreshes += 1
        if did_refresh or step % log_every == 0 or step == n_total:
            metrics = evaluate_against_oracle(inc)
            metrics.update(
                step=step,
                update_time=update_time,
                refreshed=did_refresh,
                n_refreshes=n_refreshes,
                m=inc.A.shape[0],
                n=inc.A.shape[1],
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def time_entry_updates(
    A0: np.ndarray,
    k: int,
    updates: Sequence[Tuple[int, int, float]],
    tol: float = 1e-10,
    n_runs: int = 5,
) -> float:
    """Median, over `n_runs` independent runs, of the mean wall-clock time
    per rank-1 entry update (no refresh, no oracle evaluation), matching the
    runtime-measurement protocol of Secs. 4.2 and 4.6."""
    updates = list(updates)
    medians = []
    for _ in range(n_runs):
        inc = IncrementalSVD(np.asarray(A0, dtype=float).copy(), rank=k, tol=tol)
        t0 = perf_counter()
        for i, j, delta in updates:
            inc.rank_one_entry_update(i, j, delta)
        total = perf_counter() - t0
        medians.append(total / len(updates))
    return float(np.median(medians))


def time_full_svd(A: np.ndarray, n_runs: int = 5) -> float:
    """Median time of a thin full SVD (`numpy.linalg.svd(..., full_matrices=False)`)
    of a matrix of the same size, used as the full-recomputation baseline."""
    times = []
    for _ in range(n_runs):
        t0 = perf_counter()
        np.linalg.svd(A, full_matrices=False)
        times.append(perf_counter() - t0)
    return float(np.median(times))
