"""Parametric spectral profile of the preconditioner, as implemented in Rust.

The profile multiplies the base preconditioner T by a positive Fourier multiplier on
both sides, T' = W^(1/2) T W^(1/2) with W = F^-1 diag(exp(tau(s))) F, so T' stays
Hermitian positive semidefinite and commutes with every symmetry of T. The variable
s = |k+G|^2 / |b_min|^2 is dimensionless; tau is piecewise linear in ln s through
KNOTS and constant outside them. tau = 0 reproduces T.
"""
from __future__ import annotations

import numpy as np

KNOTS = 4.0 ** (np.arange(8) - 2)  # 1/16 ... 1024
LOG_KNOTS = np.log(KNOTS)


def shortest_reciprocal_sq(b1, b2) -> float:
    """|b_min|^2 of the lattice spanned by b1 and b2 (Lagrange-Gauss reduction)."""
    u, v = np.asarray(b1, float), np.asarray(b2, float)
    if u @ u > v @ v:
        u, v = v, u
    for _ in range(64):
        v = v - np.round((u @ v) / (u @ u)) * u
        if v @ v >= u @ u:
            break
        u, v = v, u
    return float(u @ u)


def tau(s: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of the knot values in ln s, constant outside."""
    log_s = np.log(np.maximum(np.asarray(s, float), 1e-300))
    return np.interp(log_s, LOG_KNOTS, np.asarray(knots, float))


def sqrt_weights(q2: np.ndarray, b2min: float, knots) -> np.ndarray:
    """exp(tau / 2) per Fourier mode."""
    return np.exp(0.5 * tau(np.asarray(q2) / b2min, knots))


def profiled(apply_t, point, b2min: float, knots):
    """Wrap a numpy preconditioner apply (rows in spatial ordering) with the profile."""
    root = sqrt_weights(point.q2, b2min, knots)
    weight = lambda x: point.ifft(point.fft(x) * root)

    def apply(x: np.ndarray) -> np.ndarray:
        return weight(apply_t(weight(x)))
    return apply


def dense_profiled(t: np.ndarray, point, b2min: float, knots) -> np.ndarray:
    """W^(1/2) T W^(1/2) for a dense T in spatial ordering."""
    n = point.n
    root = sqrt_weights(point.q2, b2min, knots).reshape(point.ny, point.nx)
    eye = np.eye(n, dtype=complex)
    w = np.fft.ifft2(np.fft.fft2(eye.reshape(n, point.ny, point.nx)) * root).reshape(n, n).T
    return w @ t @ w
