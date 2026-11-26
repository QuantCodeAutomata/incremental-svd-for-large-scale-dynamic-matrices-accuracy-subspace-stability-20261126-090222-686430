"""Accuracy metrics comparing the incremental SVD to a full-SVD oracle.

Custom implementation of the metrics of Staykov (2026), Sec. 2.5 and Sec. 4
(error ratio eq. 10, principal angle, explained variance ratio).
`scipy.linalg.subspace_angles` provides principal angles between subspaces
and is used below instead of a hand-rolled arccos(sigma_min) formula for
numerical robustness (the two are mathematically equivalent for orthonormal
bases of equal dimension).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import subspace_angles


def optimal_rank_k(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oracle rank-k truncated SVD of the dense matrix `A` via
    ``numpy.linalg.svd(A, full_matrices=False)``.

    Returns (U_k, S_k, Vt_k) with U_k of shape (m, k), S_k of shape (k,) and
    Vt_k of shape (k, n).
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


def frob_opt_error(A: np.ndarray, S_opt: np.ndarray) -> float:
    """Optimal rank-k Frobenius error E_opt = sqrt(||A||_F^2 - ||Sigma_opt||_F^2),
    the Eckart-Young optimal low-rank approximation error."""
    frob_sq = float(np.sum(A ** 2))
    return float(np.sqrt(max(frob_sq - np.sum(S_opt ** 2), 0.0)))


def frob_ratio(E_inc: float, E_opt: float) -> float:
    """frob_ratio(t) = E_inc(t) / E_opt(t), eq. (10) of the paper."""
    if E_opt <= 1e-12:
        return 1.0 if E_inc <= 1e-9 else float(E_inc / 1e-12)
    return float(E_inc / E_opt)


def principal_angle(U1: np.ndarray, U2: np.ndarray) -> float:
    """Largest principal angle (in radians) between the column spaces of
    `U1` and `U2`: arccos(sigma_min(U1^T U2)) for orthonormal U1, U2."""
    angles = subspace_angles(U1, U2)
    return float(np.max(angles)) if len(angles) else 0.0


def explained_variance_ratio(S_full: np.ndarray, k: int) -> float:
    """EVR(k) = sum of the top-k squared singular values / total squared
    singular values, i.e. the share of Frobenius energy captured by rank k."""
    total = float(np.sum(S_full ** 2))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(S_full[:k] ** 2) / total)


def evaluate_against_oracle(inc, k: int | None = None) -> dict:
    """Metrics of `inc` (an `IncrementalSVD`) against the full-SVD oracle of
    its current dense matrix `inc.A`: incremental and optimal Frobenius
    error, their ratio (eq. 10), the principal angle to the optimal subspace,
    and the maintained a posteriori bound beta_t (Proposition 1)."""
    k = k if k is not None else inc.k
    A = inc.A
    U_opt, S_opt, _ = optimal_rank_k(A, k)
    E_opt = frob_opt_error(A, S_opt)
    E_inc = inc.frob_error()
    ratio = frob_ratio(E_inc, E_opt)
    angle = principal_angle(inc.U[:, :k], U_opt)
    frob_norm = inc.frob_norm()
    beta_over_frob = inc.beta / frob_norm if frob_norm > 1e-12 else np.nan
    beta_over_einc = inc.beta / E_inc if E_inc > 1e-12 else np.nan
    return {
        "E_inc": E_inc,
        "E_opt": E_opt,
        "frob_ratio": ratio,
        "angle_to_opt": angle,
        "beta": inc.beta,
        "frob_norm": frob_norm,
        "beta_over_frob_norm": beta_over_frob,
        "beta_over_E_inc": beta_over_einc,
    }
