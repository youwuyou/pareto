# Track: DFT

Density Functional Theory and Hartree–Fock compute the electronic ground state of a molecule by solving the self-consistent field (SCF) equations: an iterated eigenproblem coupled to a Fock matrix that depends on its own eigenvectors. A single SCF cycle stitches together several distinct computational patterns:

- **Dense linear algebra** — Fock diagonalization, orthonormalization
- **Tensor contractions** — two-electron repulsion integrals (ERIs) contracted with the density matrix to form Coulomb (J) and exchange (K)
- **Iterative solvers** — DIIS-accelerated fixed-point iteration over the density
- **Property evaluation** — NMR shieldings and J-couplings as post-SCF perturbation responses

Routing each stage to the hardware it runs fastest on is heterogeneous dispatch — and it is exactly what uniqx does. The full SCF + properties graph is submitted once; the engine schedules each subgraph to CPU, GPU, TPU, or QPU without the user managing data movement between devices.

You decide the algorithm, the basis, and the molecule. The platform decides where each block of the graph runs.

---

## Starting point

H₂O at STO-3G — the smallest closed-shell molecule that exercises every stage of the pipeline. The starter ships a full preflight → submit → compare-to-PySCF workflow.

### File map

| File | Role |
|------|------|
| [`starter.ipynb`](starter.ipynb) | End-to-end notebook: build geometry, extract basis, preflight, submit, compare to PySCF |
| [`baseline.py`](baseline.py) | PySCF RHF reference (the accuracy and runtime bar to beat) |

### SDK surface

```python
from uniqx.domains.chemistry.basis import extract_basis
from uniqx.domains.chemistry.hartree_fock import rhf_module
from uniqx.domains.chemistry.nmr_full import nmr_full_module, scf_module
```

`scf_module(geometry, basis_info, max_iter=N)` returns a traced iterative SCF; `rhf_module` returns the closed-form analytic module. `nmr_full_module` returns SCF energy plus isotropic shieldings plus Fermi-contact J-couplings.

All four take the same six runtime inputs derived from `basis_info`:

```python
runtime_inputs = [
    list(info.exps_flat),
    list(info.coeffs_flat),
    list(info.centers_flat),
    list(info.ang_flat),
    list(info.atom_coords_flat),
    list(info.charges_flat),
]
```

### Run

```bash
pip install -r requirements.txt

# PySCF baseline (local, no credentials needed)
python baseline.py

# uniqx (requires gateway access)
export UNIQX_GATEWAY=<host:port>
export UNIQX_API_KEY=<key>
jupyter notebook starter.ipynb
```

### Reference behaviour

- **H₂O at STO-3G**: RHF total energy ≈ −74.96 Ha (PySCF reference). The starter checks within 5 mHa.
- **H₂ at STO-3G**: RHF total energy ≈ −1.117 Ha — used by the SDK's own validation script.

---

## How to approach the problem

1. **Run the starter unmodified first.** Confirm H₂O/STO-3G reproduces within 5 mHa of PySCF and capture `preflight_log.txt`. This is your zero — every later change is measured against it. If this step doesn't reproduce on a clean machine, robustness scoring caps everything else (see [docs/judging.md](../../docs/judging.md)).
2. **Read the preflight table before submitting.** Each row is a candidate dispatch option with four columns: cost, runtime, accuracy (`max_error_rate`), and carbon. The recommended option is the one the gateway thinks balances the four. Picking a non-recommended option and defending it with numbers is worth more than picking the default — that is the entire point of the tradeoff-reasoning rubric.
3. **Move along one axis at a time.** Bigger basis, bigger molecule, more iterations, geometry optimization — each rotates the frontier differently. Changing two axes at once makes it hard to attribute which one moved the numbers.
4. **Always compare against PySCF on the same molecule and basis.** PySCF is the baseline for the DFT track. "Faster than PySCF at comparable accuracy" is the bar for full performance credit; matching it earns partial.
5. **Demonstrate scaling.** A submission that beats the baseline at one problem size scores lower than one that shows the trend across two. Pick a smaller and a larger size of the same workload and plot both.
6. **Write the interpretation, not just the number.** Three to six sentences citing specific table values ("the cuquantum option dropped runtime 3.2× at 0.4 mHa accuracy loss vs compiled") beat a paragraph of hand-waving. The judges read these.

### Where to push beyond the starter

| Direction | What changes |
|---|---|
| Bigger basis (6-31G, cc-pVDZ) | Cost column climbs; accuracy improves; graph fans out |
| Bigger molecule (methane → methanol → alanine) | Gateway splits the graph into more execution blocks |
| Geometry optimization | Wraps SCF in an outer minimization loop |
| Tighten `max_iter` / convergence | Trades accuracy for runtime — read the `max_error_rate` column |

---

## Challenges

### 1 — Benchmark against PySCF — 15 pts ★★☆☆☆

Run the starter as shipped and quantify where the cost lives. Profile the gateway submission against `baseline.py` on H₂O/STO-3G: total wall time, per-block runtime (read the execution log), and accuracy delta in mHa. Identify which stage dominates — ERI assembly, SCF iteration, or eigensolve — and explain why on this molecule.

**Deliverable:** timing breakdown chart (stage × runtime), accuracy delta vs PySCF, two-paragraph writeup naming the dominant stage and citing numbers from the preflight table.

---

### 2 — Scale the workload — 20 pts ★★★☆☆

Push to at least one larger problem size and document how the preflight frontier moves. Two valid directions:

- **Bigger basis** on the same molecule: STO-3G → 6-31G → cc-pVDZ
- **Bigger molecule** at the same basis: H₂O → methane → methanol → alanine

Run preflight at each size. Plot how cost, runtime, accuracy, and carbon trade off. Pick the size where the frontier kinks (e.g., where accuracy gains flatten but cost keeps climbing) and justify your stopping point in writing.

**Deliverable:** preflight tables across ≥ 2 sizes, frontier plot (one metric vs another, coloured by size), written interpretation tying the kink to the workload's structure.

---

### 3 — Geometry optimization or full NMR — 40 pts ★★★★☆

Pick one. Both wrap the SCF in an outer computation and stress the platform's block-splitting:

**Option A — Geometry optimization.** Wrap `scf_module` in an outer minimization loop (BFGS, steepest descent, or any optimizer that consumes the gradient w.r.t. nuclear coordinates). Converge to the optimized geometry of a small molecule (H₂O, CH₄, or NH₃) and compare bond lengths and angles against PySCF's `geomopt`. Track total energy at each step; the curve should monotonically decrease.

**Option B — Full NMR shieldings + J-couplings.** Use `nmr_full_module` to compute isotropic shieldings and Fermi-contact J-couplings on a 3-to-8-atom molecule (methane, methanol, ammonia). Validate each shielding against PySCF's NMR module within a documented tolerance (≤ 5 ppm is reasonable for STO-3G; tighter for cc-pVDZ). Tabulate the values per nucleus.

**Deliverable:** notebook section running the variation top-to-bottom, comparison plot or table against PySCF, ≥ 3-sentence interpretation tying the result to the preflight table — which dispatch option did you pick for the inner SCF and why?

---

### 4 — Non-obvious algorithmic twist — 25 pts ★★★★★

Originality with measurable evidence. Pick one direction not in the starter, defend it in a short design rationale (3–6 sentences citing preflight numbers), implement it, and benchmark it against your Challenge-2 result.

Examples — pick one or propose your own:

- **Basis-set extrapolation.** Run STO-3G, 6-31G, cc-pVDZ on the same molecule and extrapolate to the complete-basis-set (CBS) limit. Report the extrapolated energy and the uncertainty.
- **SCF convergence acceleration.** Compare DIIS variants, level-shifting schedules, or damping strategies. Plot convergence curves; quantify iteration count and total cost reduction.
- **Smarter initial guess.** Replace the default (core Hamiltonian) guess with a Hückel guess, superposition of atomic densities (SAD), or any learned predictor. Report iteration count reduction.
- **Mixed-basis / locally-dense basis.** Use a bigger basis on the chemically interesting atoms and a smaller basis on spectators. Quantify the cost/accuracy tradeoff vs uniform basis.
- **Functional or method comparison.** Run RHF vs an explicit DFT functional (LDA, PBE, B3LYP) on the same molecule. Compare against PySCF references and discuss which dispatch option each method favoured.

**Deliverable:** design rationale (3–6 sentences with specific preflight numbers), implementation, benchmark table vs the Challenge-2 baseline, plot showing the move on the cost/accuracy/carbon frontier.

---

## Scoring

| Challenge | Points | Difficulty |
|-----------|--------|------------|
| 1 — Benchmark against PySCF | 15 | ★★☆☆☆ |
| 2 — Scale the workload | 20 | ★★★☆☆ |
| 3 — Geometry opt or full NMR | 40 | ★★★★☆ |
| 4 — Non-obvious algorithmic twist | 25 | ★★★★★ |
| **Total** | **100** | |

Partial credit is awarded. Challenges build on each other but can be attempted independently — a working Challenge 3 without Challenge 4 still earns substantial credit. The four global judging dimensions (performance, tradeoff reasoning, creativity, robustness — see [docs/judging.md](../../docs/judging.md)) are applied across whichever challenges you complete. Tie-breaker is `total_carbon_g`.

Submit per [docs/submission.md](../../docs/submission.md): copy [templates/submission/](../../templates/submission/) into `submissions/<team-handle>/`, fill in `results.json` / `submission.ipynb` / `preflight_log.txt`, and open one PR per team against this repo before the deadline.
