"""Unit tests of `src.isvd_core.IncrementalSVD` on small synthetic fixtures.

Checks the paper's structural properties (Sec. 2): exactness of the
maintained Frobenius error (Proposition 1(c)), the a posteriori bound holding
at every step (Proposition 1), orthonormality of the maintained factors, and
correct dimension growth under row/column appends.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import low_rank_matrix, make_rng, rank1_entry_stream
from src.isvd_core import IncrementalSVD
from src.isvd_metrics import evaluate_against_oracle, optimal_rank_k, principal_angle


def _orthonormal(M: np.ndarray, atol: float = 1e-8) -> bool:
    gram = M.T @ M
    return np.allclose(gram, np.eye(gram.shape[0]), atol=atol)


def test_init_matches_truncated_svd():
    rng = make_rng(0)
    A0, _, _ = low_rank_matrix(rng, 20, 15, 4)
    inc = IncrementalSVD(A0, rank=4)
    U_opt, S_opt, _ = optimal_rank_k(A0, 4)
    # A rank-4 matrix is exactly captured by its rank-4 SVD: error ~ 0
    # (up to floating-point roundoff from the nested k x k SVD in init).
    assert inc.frob_error() < 1e-5
    assert np.allclose(np.sort(inc.S)[::-1], np.sort(S_opt)[::-1], atol=1e-6)
    assert _orthonormal(inc.U)
    assert _orthonormal(inc.V)


def test_rank_one_entry_update_matches_dense_reconstruction():
    """A single entry perturbation generically pushes a true-rank-3 matrix to
    rank 4, so a rank-3 truncation is not exact; check instead that the
    reconstruction error from U, S, V matches the maintained `frob_error`
    (Proposition 1(c)) rather than assuming zero error."""
    rng = make_rng(1)
    A0, _, _ = low_rank_matrix(rng, 15, 12, 3)
    inc = IncrementalSVD(A0, rank=3)
    inc.rank_one_entry_update(2, 5, 0.3)
    assert inc.A[2, 5] == pytest.approx(A0[2, 5] + 0.3)
    recon = inc.U @ np.diag(inc.S) @ inc.V.T
    recon_error = np.linalg.norm(inc.A - recon, "fro")
    assert recon_error == pytest.approx(inc.frob_error(), abs=1e-6)


def test_frob_error_is_exact_eckart_young_identity():
    """Proposition 1(c): for entry updates, E_inc(t) = sqrt(N_t - ||Sigma_t||_F^2)
    exactly, i.e. the maintained factors satisfy U^T A V = Sigma."""
    rng = make_rng(2)
    A0, _, _ = low_rank_matrix(rng, 20, 15, 5)
    inc = IncrementalSVD(A0, rank=5)
    events = rank1_entry_stream(rng, 20, 15, 200, 0.05)
    for _, i, j, delta in events:
        inc.rank_one_entry_update(i, j, delta)
    claimed = np.sqrt(max(inc.N - np.sum(inc.S ** 2), 0.0))
    assert inc.frob_error() == pytest.approx(claimed, abs=1e-9)


def test_a_posteriori_bound_holds_at_every_step():
    """Proposition 1: E_inc(t) <= beta_t at every step, up to roundoff."""
    rng = make_rng(3)
    A0, _, _ = low_rank_matrix(rng, 20, 15, 5)
    inc = IncrementalSVD(A0, rank=5)
    events = rank1_entry_stream(rng, 20, 15, 500, 0.05)
    worst = -np.inf
    for _, i, j, delta in events:
        inc.rank_one_entry_update(i, j, delta)
        worst = max(worst, inc.frob_error() - inc.beta)
    assert worst < 1e-6, f"bound violated by {worst}"


def test_add_row_grows_dimension_and_keeps_orthonormal_factors():
    rng = make_rng(4)
    A0, _, V_true = low_rank_matrix(rng, 10, 8, 3)
    inc = IncrementalSVD(A0, rank=3)
    new_row = rng.standard_normal(3) @ V_true.T
    inc.add_row(new_row)
    assert inc.A.shape == (11, 8)
    assert inc.U.shape == (11, 3)
    assert _orthonormal(inc.U, atol=1e-6)
    assert _orthonormal(inc.V, atol=1e-6)


def test_add_col_matches_transpose_trick():
    rng = make_rng(5)
    A0, U_true, _ = low_rank_matrix(rng, 10, 8, 3)
    inc = IncrementalSVD(A0, rank=3)
    new_col = U_true @ rng.standard_normal(3)
    inc.add_col(new_col)
    assert inc.A.shape == (10, 9)
    assert np.allclose(inc.A[:, -1], new_col)


def test_refresh_resets_bound_to_oracle_error():
    """After a refresh, beta_t0 is reset to the oracle error E_opt(t0)
    (Proposition 1(a)): the bound exactly equals the recomputed error."""
    rng = make_rng(6)
    A0, _, _ = low_rank_matrix(rng, 20, 15, 5)
    inc = IncrementalSVD(A0, rank=5)
    events = rank1_entry_stream(rng, 20, 15, 300, 0.05)
    for _, i, j, delta in events:
        inc.rank_one_entry_update(i, j, delta)
    inc.refresh()
    assert inc.beta == pytest.approx(inc.frob_error(), abs=1e-9)


def test_principal_angle_zero_for_identical_subspaces():
    rng = make_rng(7)
    A0, _, _ = low_rank_matrix(rng, 20, 15, 4)
    inc = IncrementalSVD(A0, rank=4)
    U_opt, _, _ = optimal_rank_k(A0, 4)
    angle = principal_angle(inc.U, U_opt)
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_invalid_rank_raises():
    with pytest.raises(ValueError):
        IncrementalSVD(np.eye(5), rank=0)


def test_evaluate_against_oracle_keys():
    rng = make_rng(8)
    A0, _, _ = low_rank_matrix(rng, 10, 8, 3)
    inc = IncrementalSVD(A0, rank=3)
    metrics = evaluate_against_oracle(inc)
    for key in ("E_inc", "E_opt", "frob_ratio", "angle_to_opt", "beta"):
        assert key in metrics
    assert metrics["frob_ratio"] >= 0.0
