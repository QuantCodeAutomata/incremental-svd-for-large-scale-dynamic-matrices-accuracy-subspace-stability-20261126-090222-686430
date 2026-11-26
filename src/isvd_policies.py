"""Refresh policies: periodic and bound-triggered (paper Sec. 2.6, eq. 16).

`ruptures` (changepoint detection) was considered as a candidate for a
"when should the model be refreshed" decision rule; it detects changepoints
in an existing series and does not implement the paper's O(1)
bound-triggered refresh test, so these two policies are a custom
implementation of the paper's rules.
"""

from __future__ import annotations

from typing import Optional


def should_refresh_periodic(step: int, period: Optional[int]) -> bool:
    """True every `period` steps (paper Secs. 4.1-4.4); `period=None` means
    the no-refresh policy, which never triggers."""
    if period is None or period <= 0:
        return False
    return step % period == 0


def should_refresh_bound(inc, tau: float) -> bool:
    """Bound-triggered refresh rule of eq. (16): refresh once
    beta_t / ||A_t||_F > tau, using the a posteriori bound maintained in
    `inc.beta` and the incrementally maintained Frobenius norm `inc.N`."""
    frob_norm = inc.frob_norm()
    if frob_norm <= 1e-12:
        return False
    return (inc.beta / frob_norm) > tau
