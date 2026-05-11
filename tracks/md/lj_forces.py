# Copyright (c) 2026 ORIQX AG. MIT licensed.
"""Lennard-Jones force kernels — NumPy reference and a tracing scaffold.

The NumPy version is used by `baseline.py` for accuracy comparison.
The tracing scaffold is the obvious extension you'll want to make: replace
the harmonic step's `-K @ x` force with a per-pair LJ force traced through
`uniqx.ops`. The scaffold below shows the shape of that work; completing
it is part of the hackathon track.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# NumPy reference (used by baseline.py)
# ---------------------------------------------------------------------------


def lj_forces_numpy(
    positions: list[list[float]],
    pair_indices: list[tuple[int, int]],
    sigma: float,
    epsilon: float,
) -> list[list[float]]:
    """Lennard-Jones 6-12 forces. positions: list of [x, y, z]; pairs: precomputed.

    F_ij = 24 * eps * [(2 * (sig/r)^12 - (sig/r)^6) / r^2] * r_ij
    Force on i is sum over neighbours j of F_ij; force on j is -F_ij.
    """
    n = len(positions)
    forces = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i, j in pair_indices:
        dx = positions[j][0] - positions[i][0]
        dy = positions[j][1] - positions[i][1]
        dz = positions[j][2] - positions[i][2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 == 0.0:
            continue
        inv_r2 = 1.0 / r2
        sr2 = (sigma * sigma) * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        # Magnitude factor: -dU/dr / r, signs work out so positive value
        # pulls atoms apart when too close.
        f_mag = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r2
        # Force vector points from j to i for attractive pairs (positive r),
        # from i to j for repulsive (handled by sign of f_mag).
        forces[i][0] -= f_mag * dx
        forces[i][1] -= f_mag * dy
        forces[i][2] -= f_mag * dz
        forces[j][0] += f_mag * dx
        forces[j][1] += f_mag * dy
        forces[j][2] += f_mag * dz
    return forces


def lj_potential_energy_numpy(
    positions: list[list[float]],
    pair_indices: list[tuple[int, int]],
    sigma: float,
    epsilon: float,
) -> float:
    """Total LJ potential energy."""
    total = 0.0
    for i, j in pair_indices:
        dx = positions[j][0] - positions[i][0]
        dy = positions[j][1] - positions[i][1]
        dz = positions[j][2] - positions[i][2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 == 0.0:
            continue
        sr2 = (sigma * sigma) / r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        total += 4.0 * epsilon * (sr12 - sr6)
    return total


# ---------------------------------------------------------------------------
# Tracing scaffold — your work to complete
# ---------------------------------------------------------------------------
#
# The shape of the traced kernel is:
#
#   @tracing.to_module(name="lj_step")
#   def lj_step(positions, velocities, ...):
#       # 1. Reshape flat positions to (N, 3)
#       # 2. For each pair (i, j) in pair_indices (Python loop at trace time):
#       #      - Slice positions[i] and positions[j]
#       #      - Compute r_ij = positions[j] - positions[i] using ops.sub
#       #      - Compute r2 = ops.dot(r_ij, r_ij)
#       #      - Compute sr2, sr6, sr12 via ops.div, ops.mul, ops.pow
#       #      - Compute force magnitude and accumulate into a force tensor
#       # 3. Apply velocity-Verlet update using the accumulated forces
#       # 4. Return new (positions, velocities, energy_diagnostic)
#
# Tradeoffs to weigh as you build:
#   - Unrolling over N(N-1)/2 pairs at trace time produces a large IR graph.
#     For 64 atoms that's ~2k pairs — workable. For 256 atoms it's ~32k pairs.
#   - Cutoff-based neighbour lists reduce the unroll count but freeze the
#     topology — fine for short trajectories.
#   - Using ops.fori_loop or ops.scan to fold the pair loop into a single
#     control-flow op produces a much smaller graph but requires shaped
#     pair-index tensors.
#
# Available ops you'll need (from uniqx.ops):
#   add, sub, mul, div, pow, neg, dot, matmul, slice, reshape, concatenate
#
# Reference implementation against which to validate: lj_forces_numpy above.


if __name__ == "__main__":
    # Argon-like parameters for sanity-checking the NumPy reference.
    sigma = 3.4  # Angstrom
    epsilon = 0.01  # eV (rough — proper LJ for Ar uses ~0.0104 eV)
    positions = [[0.0, 0.0, 0.0], [3.6, 0.0, 0.0]]
    forces = lj_forces_numpy(positions, [(0, 1)], sigma, epsilon)
    energy = lj_potential_energy_numpy(positions, [(0, 1)], sigma, epsilon)
    print(f"LJ pair at r=3.6 A: F = {forces[0]}, U = {energy:.4f} eV")
