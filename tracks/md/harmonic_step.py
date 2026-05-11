# Copyright (c) 2026 ORIQX AG. MIT licensed.
"""Traced harmonic velocity-Verlet step for the MD starter.

The coupling matrix K is precomputed in pure Python from the equilibrium
geometry. The traced kernel applies force = -K @ x via two matmuls per step,
following the same pattern build_lbm_step_module uses for CFD.

This is intentionally a starting point. Participants are expected to replace
the harmonic force with a Lennard-Jones (or other) anharmonic force kernel.
"""

from __future__ import annotations

import math

from uniqx import ops
from uniqx.core import tracing
from uniqx.core.execution import fmt_mat, fmt_scalar, fmt_vec


def fcc_lattice(n_per_side: int, a: float) -> list[list[float]]:
    """Equilibrium positions on an n^3-unit FCC lattice (4 atoms per cell)."""
    basis = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    positions: list[list[float]] = []
    for i in range(n_per_side):
        for j in range(n_per_side):
            for k in range(n_per_side):
                for b in basis:
                    positions.append([a * (i + b[0]), a * (j + b[1]), a * (k + b[2])])
    return positions


def build_coupling_matrix(
    positions: list[list[float]], cutoff: float, spring_k: float
) -> list[list[float]]:
    """Build a (3N, 3N) harmonic coupling matrix K.

    Forces are F = -K @ x where x is the flattened displacement vector.
    Each pair within cutoff contributes a Hessian block along the bond
    direction with spring constant `spring_k`.

    Off-diagonal block (i, j) for pair (i, j) along unit vector û:
        H_ij = -k * û û^T
    Diagonal block (i, i): sum of -H_ij over neighbours j (Newton's third law).
    """
    n = len(positions)
    K = [[0.0] * (3 * n) for _ in range(3 * n)]

    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j][0] - positions[i][0]
            dy = positions[j][1] - positions[i][1]
            dz = positions[j][2] - positions[i][2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r > cutoff or r == 0.0:
                continue
            ux, uy, uz = dx / r, dy / r, dz / r
            block = [
                [spring_k * ux * ux, spring_k * ux * uy, spring_k * ux * uz],
                [spring_k * uy * ux, spring_k * uy * uy, spring_k * uy * uz],
                [spring_k * uz * ux, spring_k * uz * uy, spring_k * uz * uz],
            ]
            for a in range(3):
                for b in range(3):
                    # Off-diagonal i,j and j,i: -block
                    K[3 * i + a][3 * j + b] -= block[a][b]
                    K[3 * j + a][3 * i + b] -= block[a][b]
                    # Diagonal i,i and j,j: +block (Newton's 3rd)
                    K[3 * i + a][3 * i + b] += block[a][b]
                    K[3 * j + a][3 * j + b] += block[a][b]
    return K


def build_harmonic_step_module(
    positions: list[list[float]],
    velocities: list[list[float]],
    mass: float,
    coupling: list[list[float]],
    dt: float,
) -> tuple:
    """Trace one velocity-Verlet step:

        a    = -K @ x / m
        v_h  = v + 0.5 * dt * a
        x_n  = x + dt * v_h
        a_n  = -K @ x_n / m
        v_n  = v_h + 0.5 * dt * a_n

    Returns (module, runtime_inputs, metadata).
    """
    n_atoms = len(positions)
    dim = 3 * n_atoms

    x0 = [v for atom in positions for v in atom]
    v0 = [v for atom in velocities for v in atom]
    inv_m = 1.0 / mass
    half_dt = 0.5 * dt

    @tracing.to_module(name="harmonic_step")
    def harmonic_step(K, x, v):
        Kx = ops.matmul(K, x)
        a = ops.mul(ops.neg(inv_m), Kx)
        v_half = ops.add(v, ops.mul(half_dt, a))
        x_new = ops.add(x, ops.mul(dt, v_half))
        Kx_new = ops.matmul(K, x_new)
        a_new = ops.mul(ops.neg(inv_m), Kx_new)
        v_new = ops.add(v_half, ops.mul(half_dt, a_new))
        # Diagnostic: kinetic energy = 0.5 * m * v.v
        kinetic = ops.mul(0.5 * mass, ops.dot(v_new, v_new))
        return x_new, v_new, kinetic

    module = harmonic_step(coupling, x0, v0)

    runtime_inputs = [
        fmt_mat(coupling, dim, dim),
        fmt_vec(x0, dim),
        fmt_vec(v0, dim),
    ]

    metadata = {
        "n_atoms": n_atoms,
        "dim": dim,
        "mass": mass,
        "dt": dt,
    }

    return module, runtime_inputs, metadata
