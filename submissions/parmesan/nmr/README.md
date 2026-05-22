# Parmesan — NMR from First Principles to Spectrum

## Team
- Artemiy Burov
- WU, You

## Workload

Full NMR pipeline for **anti-3,4-difluoroheptane** (C₇H₁₄F₂, 23 atoms):

1. **Ab initio shieldings** — RHF/STO-3G + CPHF magnetic response via
   `shieldings_module`. Tested at two scales: H₂O (7 basis, works end-to-end)
   and difluoroheptane (59 basis, preflight analysis only due to compiled
   backend size limit).

2. **Spin dynamics** — ¹⁹F NMR spectrum simulation via `ops.expv` time evolution
   of the 16-spin NMR Hamiltonian in the rotating frame. Tested at 3–5 spin
   subsystems with scipy.expm baseline.

## Key Finding

The Pareto frontier reveals a clear hardware hierarchy: `cpu+gpu` delivers 21–30x
speedup over `cpu-only` with zero error and negligible carbon. QPU options appear
at larger scales but with unacceptable error (10–73%) and extreme carbon cost
(103 kg vs 0.3 g). The classical–quantum crossover for NMR spin dynamics lies
beyond the current 5-spin/dim=32 compiled backend limit, exactly where quantum
time evolution becomes native.

## Reproduce

```bash
export UNIQX_API_KEY="uxk_..."
export UNIQX_GATEWAY="api.oriqx.com:443"
jupyter notebook submissions/parmesan/nmr/submission.ipynb
```

Runs top-to-bottom. All gateway calls are wrapped in error handling; the notebook
completes even if the gateway is temporarily unavailable.
