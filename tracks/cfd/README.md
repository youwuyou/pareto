# Track: CFD

Run a 2-D Lattice-Boltzmann channel-flow step traced through the SDK. The platform places the matmul on whatever hardware the engine thinks is best; you set the grid, the Reynolds number, and the number of time steps.

## What ships

- [`starter.ipynb`](starter.ipynb) — 64 × 16 Poiseuille channel at Re=100, single LBM step traced as a matmul, full preflight workflow.
- [`baseline.py`](baseline.py) — Pure-NumPy reference LBM step. Used to validate that the gateway's matmul matches.

## SDK surface

```python
from uniqx.cfd import ChannelFlow, LidDrivenCavity
from uniqx.cfd import build_lbm_step_module, build_diffusion_step_module
```

`build_lbm_step_module(flow)` returns `(module, runtime_inputs, metadata)`:

- `module` — traced one-step operator `f_{n+1} = A @ f_n` with a kinetic-energy diagnostic
- `runtime_inputs` — pre-encoded buffer-view strings for the operator and initial state
- `metadata` — dict with `dim`, `nx`, `ny`, `tau`, `nu`, `Re`, `state_size`

Internally the engine pre-builds `A = S @ C` (streaming × collision) in Python and ships it as a `dim*Q × dim*Q` matrix. The traced module is therefore one matmul plus a dot product per step.

## Where to push beyond the starter

| Direction | What changes |
|---|---|
| Scale grid to 256 × 64 | State vector grows 16×; matmul is the dominant cost; GPU starts beating CPU |
| Cavity at Re=1000 vs Re=100 | `tau` shrinks; numerical stability tightens; `max_error_rate` grows |
| Multi-step trajectory | Submit N steps in a loop; the engine can cache the operator |
| Couple LBM with the diffusion step (`build_diffusion_step_module`) | Two traced kernels stitched together; thermal channel |
| Multi-relaxation-time (MRT) collision | Roll your own collision matrix; compare against BGK |

## Reference behaviour

For the starter channel (64 × 16, Re=100), the analytical Poiseuille profile is parabolic in `y` at the outlet. The baseline reproduces it. Your traced run should match within ~5% L2 error after a few hundred steps.
