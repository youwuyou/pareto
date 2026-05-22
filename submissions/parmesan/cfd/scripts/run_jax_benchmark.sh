#!/usr/bin/env bash
# =============================================================================
# run_jax_benchmark.sh — JAX reference benchmark for the CFD track.
#
# Runs jax_main.py (the track's entry point) for both pressure solvers to
# produce the official output figure, then runs run_jax_benchmark.py to
# collect per-stage timing data and save it as JSON for notebook plotting.
#
# Usage (from any directory):
#   bash submissions/parmesan/cfd/scripts/run_jax_benchmark.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFD_DIR="$(realpath "$SCRIPT_DIR/../../../../tracks/cfd")"
ASSETS_DIR="$SCRIPT_DIR/../assets"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
VARIANT="N8-16-32_direct-cg"
RUN_DIR="$ASSETS_DIR/${TIMESTAMP}_${VARIANT}"
OUTPUT_JSON="$RUN_DIR/jax_benchmark.json"
LOG_FILE="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR"

# Redirect all output to both terminal and log file
exec > >(tee "$LOG_FILE") 2>&1

echo "CFD track : $CFD_DIR"
echo "Run dir   : $RUN_DIR"
echo ""

# ── Run jax_main.py (official entry point) for each grid size ─────────────────
# Grid sizes: N=8, 16, 32.
# Step A omits the advection term (u·∇u), so the only stability constraint is
# the explicit diffusion bound: dt = 0.25·dx²/ν (set by Grid.__post_init__).
# If advection is added, N=8 and N=16 would violate the convection CFL
# (dt > dx/U_lid by ×3 and ×1.6 respectively) and should be replaced by N ≥ 32.
#
# CG is included as a comparison point. Without a preconditioner the condition
# number of the discrete Poisson system grows as O(N²), so convergence slows
# with grid refinement and non-convergence warnings are expected at N=32.
STEPS=500

for N in 8 16 32; do
    for SOLVER in direct cg; do
        echo "=== jax_main.py --solver $SOLVER --n $N --steps $STEPS ==="
        python -u "$CFD_DIR/jax_main.py" \
            --solver "$SOLVER" \
            --steps "$STEPS" \
            --n "$N"
        # Copy the figure produced by jax_main.py into the timestamped run dir
        FIG_SRC="$ASSETS_DIR/results_jax.png"
        [ -f "$FIG_SRC" ] && cp "$FIG_SRC" "$RUN_DIR/results_jax_N${N}_${SOLVER}.png"
        echo ""
    done
done

# ── Run the benchmark script to collect timing data as JSON ───────────────────
echo "=== Collecting per-stage timing data ==="
python -u "$SCRIPT_DIR/run_jax_benchmark.py" "$OUTPUT_JSON"

# Copy per-solver convergence figures produced by run_jax_benchmark.py
for SOLVER in direct cg; do
    FIG="$ASSETS_DIR/results_jax_${SOLVER}.png"
    [ -f "$FIG" ] && cp "$FIG" "$RUN_DIR/"
done

# Update the 'latest' symlink for easy notebook access
ln -sfn "$TIMESTAMP" "$ASSETS_DIR/latest"

echo ""
echo "Done. Results saved to $RUN_DIR"
echo "      Symlink : $ASSETS_DIR/latest -> $TIMESTAMP"