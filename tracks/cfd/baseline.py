# Copyright (c) 2026 ORIQX AG. MIT licensed.
"""Pure-NumPy reference LBM step for CFD-track accuracy comparison.

Install: `pip install -e ".[cfd]"` (or just `numpy` is enough for this file).
"""

from __future__ import annotations

import numpy as np

D2Q9_C = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int32,
)
D2Q9_W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float64,
)
Q = 9


def equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium distribution: shape (nx, ny, Q)."""
    cu = np.einsum("kd,xyd->xyk", D2Q9_C.astype(np.float64), u)
    usq = (u * u).sum(axis=-1, keepdims=True)
    return D2Q9_W * rho[..., None] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usq)


def lbm_step(f: np.ndarray, tau: float, periodic_x: bool = True) -> np.ndarray:
    """One BGK collision + streaming step. f: (nx, ny, Q)."""
    rho = f.sum(axis=-1)
    u = np.einsum("xyk,kd->xyd", f, D2Q9_C.astype(np.float64)) / rho[..., None]
    f_eq = equilibrium(rho, u)
    f_post = f - (f - f_eq) / tau  # collision

    # Streaming
    f_new = np.zeros_like(f_post)
    nx, ny, _ = f.shape
    for k in range(Q):
        cx, cy = D2Q9_C[k]
        rolled = np.roll(f_post[..., k], shift=(cx, cy), axis=(0, 1))
        f_new[..., k] = rolled

    if not periodic_x:
        # Bounce-back at x walls
        opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        for k in range(Q):
            cx, _ = D2Q9_C[k]
            if cx == 1:
                f_new[0, :, k] = f_post[0, :, opposite[k]]
            elif cx == -1:
                f_new[-1, :, k] = f_post[-1, :, opposite[k]]

    return f_new


def initial_state(nx: int, ny: int, seed: int = 42) -> np.ndarray:
    """Equilibrium + small perturbation matching uniqx.cfd._build_initial_state."""
    rng = np.random.default_rng(seed)
    f = np.zeros((nx, ny, Q), dtype=np.float64)
    for k in range(Q):
        f[..., k] = D2Q9_W[k] * (1 + 0.001 * (rng.random((nx, ny)) - 0.5))
    return f


def kinetic_energy(f: np.ndarray) -> float:
    """Diagnostic matching the one in build_lbm_step_module."""
    return 0.5 * float((f * f).sum())


if __name__ == "__main__":
    nx, ny = 64, 16
    Re = 100.0
    u_char = 0.1
    nu = u_char * ny / Re
    tau = 3 * nu + 0.5
    f = initial_state(nx, ny)
    for _ in range(50):
        f = lbm_step(f, tau, periodic_x=True)
    print(f"NumPy reference after 50 steps: KE = {kinetic_energy(f):.6f}")
