"""Synthetic data generators for the rank-1, row/column-append and mixed
update streams of Secs. 4.1-4.4 and 4.6 of Staykov (2026).

All randomness is injected through a caller-supplied `numpy.random.Generator`
so that, as in the authors' reference streams, the initial matrix and the
subsequent update stream are drawn from one continuous random sequence.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """A fresh default-bit-generator RNG for the given seed."""
    return np.random.default_rng(seed)


def low_rank_matrix(
    rng: np.random.Generator, m: int, n: int, true_rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A = U_true @ V_true.T with iid standard-normal factors (paper Sec.
    4.1: "a 50x40 matrix of true rank 5"). Returns (A, U_true, V_true)."""
    U_true = rng.standard_normal((m, true_rank))
    V_true = rng.standard_normal((n, true_rank))
    A = U_true @ V_true.T
    return A, U_true, V_true


def low_rank_matrix_noisy(
    rng: np.random.Generator, m: int, n: int, true_rank: int, noise_std: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-`true_rank` matrix plus additive Gaussian noise (paper Sec. 4.3:
    "sampled from a rank-6 model with additive noise"). Returns
    (A, U_true, V_true); U_true, V_true are the clean (noise-free) factors."""
    A, U_true, V_true = low_rank_matrix(rng, m, n, true_rank)
    if noise_std > 0:
        A = A + noise_std * rng.standard_normal((m, n))
    return A, U_true, V_true


def rank1_entry_stream(
    rng: np.random.Generator, m: int, n: int, n_steps: int, delta_std: float
) -> List[Tuple[str, int, int, float]]:
    """`n_steps` random rank-1 entry updates ("rank1", i, j, delta) with
    (i, j) uniform over all matrix entries and delta ~ N(0, delta_std^2)
    (paper Sec. 4.1)."""
    events = []
    for _ in range(n_steps):
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, n))
        delta = float(delta_std * rng.standard_normal())
        events.append(("rank1", i, j, delta))
    return events


def row_append_stream(
    rng: np.random.Generator,
    V_true: np.ndarray,
    true_rank: int,
    n_steps: int,
    noise_std: float,
) -> List[Tuple[str, np.ndarray]]:
    """`n_steps` new rows sampled from the rank-`true_rank` model with
    additive noise: row = u_new @ V_true.T + noise, u_new ~ N(0, I)
    (paper Sec. 4.3)."""
    events = []
    for _ in range(n_steps):
        u_new = rng.standard_normal(true_rank)
        row = u_new @ V_true.T
        if noise_std > 0:
            row = row + noise_std * rng.standard_normal(row.shape)
        events.append(("row", row))
    return events


def col_append_stream(
    rng: np.random.Generator,
    U_true: np.ndarray,
    true_rank: int,
    n_steps: int,
    noise_std: float,
) -> List[Tuple[str, np.ndarray]]:
    """`n_steps` new columns sampled from the rank-`true_rank` model with
    additive noise: col = U_true @ v_new + noise, v_new ~ N(0, I)
    (paper Sec. 4.3, column-streaming counterpart)."""
    events = []
    for _ in range(n_steps):
        v_new = rng.standard_normal(true_rank)
        col = U_true @ v_new
        if noise_std > 0:
            col = col + noise_std * rng.standard_normal(col.shape)
        events.append(("col", col))
    return events


def mixed_rank1_and_row_stream(
    rng: np.random.Generator,
    m: int,
    n: int,
    V_true: np.ndarray,
    true_rank: int,
    n_steps: int,
    row_period: int,
    delta_std: float,
    row_noise_std: float,
) -> List[Tuple]:
    """Mixed stream of rank-1 entry updates and periodic row appends (paper
    Sec. 4.4: "rank-1 perturbations with row additions, one new row every
    `row_period` steps"). At step t, a row is appended if t % row_period == 0,
    otherwise a random rank-1 entry update is applied. The row dimension m
    grows as rows are appended; entry updates are drawn over the current
    (growing) number of rows.
    """
    events: List[Tuple] = []
    m_current = m
    for t in range(1, n_steps + 1):
        if t % row_period == 0:
            u_new = rng.standard_normal(true_rank)
            row = u_new @ V_true.T
            if row_noise_std > 0:
                row = row + row_noise_std * rng.standard_normal(row.shape)
            events.append(("row", row))
            m_current += 1
        else:
            i = int(rng.integers(0, m_current))
            j = int(rng.integers(0, n))
            delta = float(delta_std * rng.standard_normal())
            events.append(("rank1", i, j, delta))
    return events
