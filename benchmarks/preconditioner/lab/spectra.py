"""Spectral quality measures of a preconditioner T for the pencil (A, B).

kappa_ta: spread of T A on the complement of its kernel; preconditioner quality
alone (for TM, where T is close to a multiple of A^-1, it is near 1).
kappa_eff: for band n of a block of size m, the spread of T P^H (A - lambda_n B) P
with P = I - V_m V_m^H B, dropping its m exact zeros. It predicts the
asymptotic per-iteration contraction of a preconditioned block eigensolver.
gap: lambda_n / lambda_(m+1), the spectral gap seen by a block of size m.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def eigenpairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ascending generalized eigenpairs with V^H B V = I."""
    return sla.eigh((a + a.conj().T) / 2, (b + b.conj().T) / 2)


def _positive_spectrum(product: np.ndarray, drop: int, rel_tol: float) -> np.ndarray:
    values = np.linalg.eigvals(product)
    if np.max(np.abs(values.imag), initial=0.0) > 1e-6 * np.max(np.abs(values.real)):
        raise ValueError("preconditioned spectrum is not real; T is not Hermitian positive semidefinite")
    real = values.real
    scale = np.max(np.abs(real))
    if drop:
        real = np.delete(real, np.argsort(np.abs(real))[:drop])
    return np.sort(real[real > rel_tol * scale])


def kappa_ta(t: np.ndarray, a: np.ndarray, rel_tol: float = 1e-10) -> tuple[float, float, float]:
    """(lambda_min, lambda_max, kappa) of T A over its nonzero spectrum."""
    s = _positive_spectrum(t @ a, 0, rel_tol)
    return float(s[0]), float(s[-1]), float(s[-1] / s[0])


def kappa_eff(t: np.ndarray, a: np.ndarray, b: np.ndarray, values: np.ndarray, vectors: np.ndarray,
              block: int, bands: list[int], rel_tol: float = 1e-9) -> list[dict]:
    vm = vectors[:, :block]
    projector = np.eye(a.shape[0]) - vm @ (vm.conj().T @ b)
    out = []
    for n in bands:
        shifted = projector.conj().T @ (a - values[n] * b) @ projector
        s = _positive_spectrum(t @ shifted, block, rel_tol)
        k = float(s[-1] / s[0])
        out.append({"band": n, "lambda": float(values[n]), "kappa_eff": k,
                    "gap": float(values[n] / values[block]) if values[block] > 0 else None,
                    "sd_rate": (k - 1) / (k + 1), "cg_rate": (np.sqrt(k) - 1) / (np.sqrt(k) + 1)})
    return out


def lanczos_kappa_ta(apply_a, apply_t, n: int, steps: int = 80, seed: int = 0,
                     kernel: np.ndarray | None = None) -> tuple[float, float, float]:
    """Extreme eigenvalues of T A by Lanczos in the A-inner product (T A is A-self-adjoint)."""
    def project(x: np.ndarray) -> np.ndarray:
        if kernel is None:
            return x
        return x - kernel * (kernel.conj() @ x) / (kernel.conj() @ kernel)

    rng = np.random.default_rng(seed)
    v = project(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    av = apply_a(v)
    norm = np.sqrt(np.real(v.conj() @ av))
    basis, abasis = [v / norm], [av / norm]
    for _ in range(steps):
        w = project(apply_t(abasis[-1]))
        aw = apply_a(w)
        for _ in range(2):
            for q, aq in zip(basis, abasis):
                c = aq.conj() @ w
                w, aw = w - c * q, aw - c * aq
        beta = np.sqrt(max(np.real(w.conj() @ aw), 0.0))
        if beta < 1e-12:
            break
        basis.append(w / beta)
        abasis.append(aw / beta)
    taq = [apply_t(aq) for aq in abasis]
    h = np.array([[aq.conj() @ t for t in taq] for aq in abasis])
    ritz = np.linalg.eigvalsh((h + h.conj().T) / 2)
    ritz = ritz[ritz > 1e-10 * ritz.max()]
    return float(ritz[0]), float(ritz[-1]), float(ritz[-1] / ritz[0])
