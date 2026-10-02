"""Numpy forms of the solver's preconditioners and of the candidates gated in E10.

Every operator acts on blocks of vectors (rows) in the solver's spatial ordering
(row-major ny x nx), with the forward FFT unnormalized and the inverse scaled by
1/N like the CPU backend. `PointData` holds the Fourier tables of one k-point as
reported by precond_lab, so the numpy forms can be checked against the Rust ones
(`parity`) before they are trusted.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

Apply = Callable[[np.ndarray], np.ndarray]


@dataclasses.dataclass
class PointData:
    ny: int
    nx: int
    polarization: str
    qx: np.ndarray
    qy: np.ndarray
    mask: np.ndarray
    is_gamma: bool
    eps: np.ndarray
    inv_eps: np.ndarray | None  # smoothed scalar inverse permittivity
    eta: np.ndarray  # (ny, nx, 2, 2) smoothed inverse permittivity tensors
    shift: float
    s_min: float

    @classmethod
    def from_ops(cls, ops) -> 'PointData':
        info, point = ops.info, ops.point
        shape = (ops.ny, ops.nx)
        return cls(ops.ny, ops.nx, info['polarization'],
                   np.asarray(point['k_plus_g_x'], float).reshape(shape),
                   np.asarray(point['k_plus_g_y'], float).reshape(shape),
                   np.asarray(point['kernel_mask'], bool).reshape(shape), bool(point['is_gamma']),
                   np.asarray(info['eps'], float).reshape(shape),
                   np.asarray(info['inv_eps'], float).reshape(shape) if 'inv_eps' in info else None,
                   np.asarray(info['inv_eps_tensors'], float).reshape(*shape, 2, 2),
                   float(point['preconditioner']['shift']), float(point['preconditioner']['s_min']))

    @property
    def q2(self) -> np.ndarray:
        return self.qx ** 2 + self.qy ** 2

    @property
    def n(self) -> int:
        return self.nx * self.ny

    def fft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.fft2(x.reshape(-1, self.ny, self.nx))

    def ifft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(x).reshape(-1, self.n)

    def fourier_diagonal(self, d: np.ndarray) -> Apply:
        return lambda x: self.ifft(self.fft(x) * d)


def fd_symbol(p: PointData, shift: float, eps_eff: float, mass: float = 0.0, gamma_mask: bool = True) -> np.ndarray:
    """eps_eff / (|q|^2 + mass + shift * eps_eff), the solver's Fourier diagonal."""
    q2 = np.maximum(p.q2, 0.0)
    d = eps_eff / (q2 + mass + shift * eps_eff)
    if gamma_mask and p.is_gamma:
        d = np.where(p.mask, 0.0, d)
    return d


def fd(p: PointData, shift: float | None = None, tm_mass_fraction: float = 0.01) -> Apply:
    """Fourier diagonal as built by the solver: arithmetic-mean eps for TM, harmonic-mean eta for TE."""
    shift = p.shift if shift is None else shift
    if p.polarization == 'TM':
        eps_eff = float(p.eps.mean())
        return p.fourier_diagonal(fd_symbol(p, shift, eps_eff, eps_eff * tm_mass_fraction))
    if p.inv_eps is None:
        raise ValueError('the TE Fourier diagonal needs inv_eps from a newer precond_lab')
    eps_eff = float(1.0 / np.mean(p.inv_eps))
    return p.fourier_diagonal(fd_symbol(p, shift, eps_eff))


def tp(p: PointData, shift: float | None = None, weight: str = 'eps') -> Apply:
    """D sum_cd q_c F E_cd F^-1 q_d D with D = 1/(|q|^2 + shift); E = eps (scalar) or eta^-1 (tensor)."""
    shift = p.shift if shift is None else shift
    d = np.where(p.mask, 0.0, 1.0 / np.maximum(p.q2 + shift, 1e-300))
    if weight == 'eps':
        e = np.zeros_like(p.eta)
        e[..., 0, 0] = e[..., 1, 1] = p.eps
    else:
        e = np.linalg.inv(p.eta)
    q = (p.qx, p.qy)

    def apply(x: np.ndarray) -> np.ndarray:
        r = p.fft(x) * d
        grads = [np.fft.ifft2(-1j * qc * r) for qc in q]
        out = 0
        for c in range(2):
            field = e[..., c, 0] * grads[0] + e[..., c, 1] * grads[1]
            out = out + 1j * q[c] * np.fft.fft2(field)
        return p.ifft(out * d)
    return apply


def te_operator(p: PointData) -> Apply:
    """A h = sum_cd (i q_c)^H F eta_cd F^-1 (i q_d) h, the TE operator with smoothed tensors."""
    q = (p.qx, p.qy)

    def apply(x: np.ndarray) -> np.ndarray:
        h = p.fft(x)
        grads = [np.fft.ifft2(1j * qc * h) for qc in q]
        out = 0
        for c in range(2):
            field = p.eta[..., c, 0] * grads[0] + p.eta[..., c, 1] * grads[1]
            out = out - 1j * q[c] * np.fft.fft2(field)
        return p.ifft(out)
    return apply


def dense(apply: Apply, n: int) -> np.ndarray:
    return apply(np.eye(n, dtype=complex)).T


def parity(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(reference - candidate)) / max(np.max(np.abs(reference)), 1e-300))


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def plane_waves(p: PointData, count: int, exclude_kernel: bool = True) -> np.ndarray:
    """Orthonormal coarse basis (n x m): the `count` lowest-|q| plane waves, completed to full shells."""
    q2 = p.q2.ravel()
    order = np.argsort(q2, kind='stable')
    if exclude_kernel and p.is_gamma:
        order = order[~p.mask.ravel()[order]]
    count = min(count, len(order))
    edge = q2[order[count - 1]]
    chosen = order[q2[order] <= edge * (1 + 1e-9) + 1e-12]
    basis = np.zeros((len(chosen), p.n), complex)
    for row, index in enumerate(chosen):
        spectrum = np.zeros(p.n, complex)
        spectrum[index] = np.sqrt(p.n)
        basis[row] = p.ifft(spectrum.reshape(1, p.ny, p.nx))[0]
    return basis.T


def two_level(smoother: Apply, a: np.ndarray, b: np.ndarray, basis: np.ndarray, shift: float = 0.0,
              form: str = 'additive') -> Apply:
    """Coarse correction Z (Z^H (A + shift B) Z)^-1 Z^H combined with a smoother T."""
    coarse = basis.conj().T @ (a + shift * b) @ basis
    coarse = 0.5 * (coarse + coarse.conj().T)
    factor = np.linalg.cholesky(coarse)

    def correction(x: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(factor.conj().T, np.linalg.solve(factor, basis.conj().T @ x.T))
        return (basis @ y).T

    if form == 'additive':
        return lambda x: smoother(x) + correction(x)
    k = a + shift * b

    def balanced(x: np.ndarray) -> np.ndarray:
        # (I - Q K) T (I - K Q) + Q, with Q the coarse correction.
        cx = correction(x)
        r = x - (k @ cx.T).T
        s = smoother(r)
        return s - correction((k @ s.T).T) + cx
    return balanced


def chebyshev(smoother: Apply, k: np.ndarray, interval: tuple[float, float], degree: int) -> Apply:
    """Chebyshev iteration for K y = x preconditioned by T, from y = 0: the linear map p(TK) T.

    `interval` bounds the spectrum of TK; degree d uses d + 1 smoother and d matrix applications.
    """
    lo, hi = interval
    theta, delta = (hi + lo) / 2, (hi - lo) / 2

    def apply(x: np.ndarray) -> np.ndarray:
        sigma = theta / delta
        rho = 1 / sigma
        y = np.zeros_like(x, dtype=complex)
        r = x.astype(complex)
        d = smoother(r) / theta
        for _ in range(degree):
            y = y + d
            r = r - (k @ d.T).T
            rho_next = 1 / (2 * sigma - rho)
            d = rho_next * rho * d + (2 * rho_next / delta) * smoother(r)
            rho = rho_next
        return y + d
    return apply
