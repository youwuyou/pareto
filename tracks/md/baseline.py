# Copyright (c) 2026 ORIQX AG. MIT licensed.
"""NumPy reference velocity-Verlet for the MD track.

Mirrors the harmonic step traced in `harmonic_step.py` so participants
can validate the gateway result, and includes an LJ path so they can
validate any anharmonic extension they build.
"""

from __future__ import annotations

import numpy as np
from lj_forces import lj_forces_numpy


def harmonic_step_numpy(
    x: np.ndarray, v: np.ndarray, K: np.ndarray, mass: float, dt: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """One velocity-Verlet step against a harmonic coupling matrix K.

    x, v are flat 3N vectors. Returns (x_new, v_new, kinetic_energy).
    """
    a = -(K @ x) / mass
    v_half = v + 0.5 * dt * a
    x_new = x + dt * v_half
    a_new = -(K @ x_new) / mass
    v_new = v_half + 0.5 * dt * a_new
    kinetic = 0.5 * mass * float(v_new @ v_new)
    return x_new, v_new, kinetic


def lj_step_numpy(
    positions: np.ndarray,
    velocities: np.ndarray,
    pair_indices: list[tuple[int, int]],
    sigma: float,
    epsilon: float,
    mass: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One velocity-Verlet step under a Lennard-Jones force field.

    positions, velocities: (N, 3) arrays.
    """
    forces = np.array(
        lj_forces_numpy(positions.tolist(), pair_indices, sigma, epsilon),
        dtype=np.float64,
    )
    a = forces / mass
    v_half = velocities + 0.5 * dt * a
    x_new = positions + dt * v_half
    forces_new = np.array(
        lj_forces_numpy(x_new.tolist(), pair_indices, sigma, epsilon),
        dtype=np.float64,
    )
    a_new = forces_new / mass
    v_new = v_half + 0.5 * dt * a_new
    return x_new, v_new


def total_energy_harmonic(x: np.ndarray, v: np.ndarray, K: np.ndarray, mass: float) -> float:
    """E = 0.5 m v.v + 0.5 x.K.x"""
    kinetic = 0.5 * mass * float(v @ v)
    potential = 0.5 * float(x @ K @ x)
    return kinetic + potential


if __name__ == "__main__":
    from harmonic_step import build_coupling_matrix, fcc_lattice

    positions = fcc_lattice(n_per_side=2, a=3.6)
    n = len(positions)
    print(f"FCC lattice: {n} atoms")
    K = np.array(build_coupling_matrix(positions, cutoff=5.0, spring_k=1.0))
    x = np.zeros(3 * n)  # zero displacement
    v = np.random.default_rng(42).standard_normal(3 * n) * 0.01
    mass = 40.0
    dt = 0.005

    E0 = total_energy_harmonic(x, v, K, mass)
    for _ in range(100):
        x, v, _ = harmonic_step_numpy(x, v, K, mass, dt)
    E1 = total_energy_harmonic(x, v, K, mass)
    print(f"Energy drift over 100 steps: {(E1 - E0) / E0 * 100:.4f}%")
