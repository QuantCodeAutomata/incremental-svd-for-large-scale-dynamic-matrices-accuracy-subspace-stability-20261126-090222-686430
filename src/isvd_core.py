"""Incremental (rank-k truncated) SVD engine.

Custom implementation of the Brand-style incremental SVD updates described in
Section 2 of Staykov (2026), "Incremental SVD for Large-Scale Dynamic
Matrices" (arXiv:2605.24514): rank-1 entry updates, row appends and column
appends, with the O(1)-per-step a posteriori error bound of Proposition 1.

`scipy.sparse.linalg.svds` and scikit-learn's incremental-PCA equivalents
(`IncrementalPCA`, `TruncatedSVD`) were considered; none of them implement
rank-1 entry updates, the row/column append core-matrix algebra, or the a
posteriori error bound of Proposition 1, so the engine below is a custom
implementation of the paper's algorithm.
"""

from __future__ import annotations

import numpy as np


class IncrementalSVD:
    """Maintains a rank-k truncated SVD ``A ~= U @ diag(S) @ V.T`` of a dense,
    dynamically changing matrix under rank-1 entry updates and row/column
    appends (paper Sec. 2.2-2.4), together with the a posteriori Frobenius
    error bound beta_t of Proposition 1 (Sec. 2.5-2.6).

    Parameters
    ----------
    A0 : ndarray of shape (m, n)
        Initial dense matrix.
    rank : int
        Truncation rank k maintained by the engine.
    tol : float
        Residual-norm tolerance below which a new row/column is treated as
        already lying in the current subspace (no new direction added).
    """

    def __init__(self, A0: np.ndarray, rank: int, tol: float = 1e-10) -> None:
        A0 = np.asarray(A0, dtype=float)
        if A0.ndim != 2:
            raise ValueError("A0 must be a 2-D array")
        if rank < 1:
            raise ValueError("rank must be a positive integer")
        self.k = int(rank)
        self.tol = float(tol)
        self.A = A0.copy()
        self._init_factors()

    def _init_factors(self) -> None:
        U, S, Vt = np.linalg.svd(self.A, full_matrices=False)
        k = min(self.k, len(S))
        self.U = np.ascontiguousarray(U[:, :k])
        self.S = np.ascontiguousarray(S[:k])
        self.V = np.ascontiguousarray(Vt[:k, :].T)
        if k < self.k:
            # Pad with zero directions if the matrix rank is below k.
            pad_u = np.zeros((self.A.shape[0], self.k - k))
            pad_v = np.zeros((self.A.shape[1], self.k - k))
            self.U = np.hstack([self.U, pad_u])
            self.V = np.hstack([self.V, pad_v])
            self.S = np.concatenate([self.S, np.zeros(self.k - k)])
        self.N = float(np.sum(self.A ** 2))
        self._reset_bound()

    def _reset_bound(self) -> None:
        """Reset the a posteriori bound at a refresh point t0 (Proposition 1(a)):
        beta_{t0} = E_opt(t0) = sqrt(N_{t0} - ||Sigma_{t0}||_F^2)."""
        self.E_opt_t0 = self.frob_error()
        self.beta = self.E_opt_t0

    # ------------------------------------------------------------------
    # Error metrics maintained in O(1)/O(k) per step (paper Sec. 2.5)
    # ------------------------------------------------------------------
    def frob_error(self) -> float:
        """E_inc(t) = ||A_t - U_t Sigma_t V_t^T||_F = sqrt(N_t - ||Sigma_t||_F^2),
        exact because the maintained factors satisfy U_t^T A_t V_t = Sigma_t
        (paper Proposition 1(c))."""
        return float(np.sqrt(max(self.N - np.sum(self.S ** 2), 0.0)))

    def frob_norm(self) -> float:
        """||A_t||_F = sqrt(N_t)."""
        return float(np.sqrt(max(self.N, 0.0)))

    # ------------------------------------------------------------------
    # Rank-1 entry update (paper Sec. 2.2)
    # ------------------------------------------------------------------
    def rank_one_entry_update(self, i: int, j: int, delta: float) -> None:
        """Add `delta` to entry A[i, j] and update the truncated factors via
        the k x k core K = Sigma + delta * s_u * s_v^T (Brand's method)."""
        old_val = float(self.A[i, j])
        self.A[i, j] = old_val + delta
        u_i = self.U[i, :].copy()
        v_j = self.V[j, :].copy()
        a = float(u_i @ u_i)
        b = float(v_j @ v_j)
        core = np.diag(self.S) + delta * np.outer(u_i, v_j)
        Uk, Sk, Vkt = np.linalg.svd(core)
        k = self.k
        self.U = self.U @ Uk[:, :k]
        self.S = Sk[:k]
        self.V = self.V @ Vkt[:k, :].T
        eta = abs(delta) * np.sqrt(max(1.0 - a * b, 0.0))
        self.N += 2.0 * delta * old_val + delta ** 2
        self.beta += eta

    # ------------------------------------------------------------------
    # Row / column append (paper Sec. 2.3), transpose trick for columns
    # ------------------------------------------------------------------
    def _append_vector(self, U_role: np.ndarray, V_role: np.ndarray, vec: np.ndarray):
        """Brand-style append of `vec` as a new row of the matrix whose row
        factor is `U_role` and column factor is `V_role`. Returns the updated
        (U_role, S, V_role, eta) where eta is the discarded-energy term used
        to maintain the bound beta_t (Proposition 1(b))."""
        k = self.k
        p = vec @ V_role
        z = vec - p @ V_role.T
        rho = float(np.linalg.norm(z))
        m = U_role.shape[0]
        if rho > self.tol:
            z_hat = z / rho
            core = np.zeros((k + 1, k + 1))
            core[:k, :k] = np.diag(self.S)
            core[k, :k] = p
            core[k, k] = rho
            Uk, Sk, Vkt = np.linalg.svd(core)
            eta = float(Sk[k]) if k < len(Sk) else 0.0
            U_lifted = np.hstack([U_role, np.zeros((m, 1))])
            new_row = np.zeros((1, k + 1))
            new_row[0, k] = 1.0
            U_lifted = np.vstack([U_lifted, new_row])
            V_lifted = np.hstack([V_role, z_hat.reshape(-1, 1)])
            U_new = U_lifted @ Uk[:, :k]
            S_new = Sk[:k]
            V_new = V_lifted @ Vkt[:k, :].T
        else:
            core = np.zeros((k + 1, k))
            core[:k, :] = np.diag(self.S)
            core[k, :] = p
            Uk, Sk, Vkt = np.linalg.svd(core, full_matrices=False)
            eta = rho
            U_lifted = np.hstack([U_role, np.zeros((m, 1))])
            new_row = np.zeros((1, k + 1))
            new_row[0, k] = 1.0
            U_lifted = np.vstack([U_lifted, new_row])
            U_new = U_lifted @ Uk
            S_new = Sk
            V_new = V_role @ Vkt.T
        return U_new, S_new, V_new, eta

    def add_row(self, row: np.ndarray) -> None:
        """Append `row` (length n) as a new last row of A (m -> m+1)."""
        row = np.asarray(row, dtype=float)
        U_new, S_new, V_new, eta = self._append_vector(self.U, self.V, row)
        self.A = np.vstack([self.A, row.reshape(1, -1)])
        self.U, self.S, self.V = U_new, S_new, V_new
        self.N += float(np.sum(row ** 2))
        self.beta += eta

    def add_col(self, col: np.ndarray) -> None:
        """Append `col` (length m) as a new last column of A (n -> n+1), via
        the transpose trick: a column append on A is a row append on A^T."""
        col = np.asarray(col, dtype=float)
        V_new, S_new, U_new, eta = self._append_vector(self.V, self.U, col)
        self.A = np.hstack([self.A, col.reshape(-1, 1)])
        self.U, self.S, self.V = U_new, S_new, V_new
        self.N += float(np.sum(col ** 2))
        self.beta += eta

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Recompute the full truncated rank-k SVD of the current dense
        matrix A_t and reset the a posteriori bound (new segment t0 = t)."""
        self._init_factors()
