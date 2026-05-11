# pareto

The ORIQX hackathon track. Write hardware-agnostic code. The platform measures the tradeoff.

---

## What this is

**pareto** is the starter repo for the ORIQX track at the Idearum hackathon. You write a workload — molecular dynamics, fluid dynamics, density functional theory, or something you design yourself — on top of the `uniqx` SDK. ORIQX's analytics engine produces a Pareto frontier of execution options (CPU, GPU, mixed) with measurable cost, runtime, accuracy, and carbon for each. You pick a point on the frontier, justify it, and run.

The winning team is the one whose algorithm × hardware choice delivers the strongest measurable performance against the baseline, with the clearest reasoning behind the choice.

The platform is hardware-agnostic and vendor-agnostic. Your code does not name a backend. The execution engine extends to QPUs when `preflight()` indicates a measurable advantage — but a strong submission can use only classical hardware.

---

## How `uniqx` fits

You write Python. The SDK traces your operations into an IR graph. You call `preflight()`, read the option table, and pick where on the frontier you want to land:

```python
import uniqx

client = uniqx.connect("api.oriqx.com:443")
module = my_workload(spec)             # traces — does not execute
options = uniqx.preflight(module, client=client)
print(options.summary())                # the Pareto table
choice = options.recommended            # or pick a different option
job_id = uniqx.submit(module, client=client, backend=choice["label"])
result = uniqx.get(job_id, client=client)
```

`options.summary()` prints a table with one row per execution plan: time, cost (USD), max error rate, carbon (g CO₂), and which nodes the engine plans to assign to CPU / GPU / QPU. That table is the artifact the judges read.

---

## Install

You need an API key from your hackathon organiser. The wheel lives behind a private index:

```bash
export UNIQX_API_KEY="uxk_...your-key..."
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url "https://uniqx:${UNIQX_API_KEY}@wheels.oriqx.com/simple/" uniqx
pip install -e .                        # installs pareto + baseline extras
```

Register and grab your key at [app.oriqx.com](https://app.oriqx.com). The full onboarding is at [docs.oriqx.com/getting-started-hackathon](https://docs.oriqx.com/getting-started-hackathon).

Set your gateway target once:

```bash
export UNIQX_GATEWAY="api.oriqx.com:443"
```

---

## Three tracks

Pick one. Or design your own — see [docs/tracks.md](docs/tracks.md) for the self-defined rubric.

| Track | What you optimize | Starter |
|---|---|---|
| **DFT** | SCF energy + NMR shieldings for a small molecule | [tracks/dft/](tracks/dft/) |
| **CFD** | 2-D Lattice-Boltzmann channel or cavity flow | [tracks/cfd/](tracks/cfd/) |
| **MD** | Lennard-Jones argon NVE dynamics | [tracks/md/](tracks/md/) |

Each track ships a working notebook, a NumPy/PySCF baseline, and a problem-extension prompt. You are free to change the workload, the basis, the grid, the integrator — anything. The starter is a floor, not a ceiling.

---

## Judging

Four criteria, weighted equally:

| Criterion | What we look for |
|---|---|
| **Performance** | Wall-clock runtime, numerical accuracy vs. baseline, scalability with problem size |
| **Tradeoff reasoning** | Quality of the justification for your point on the Pareto frontier — read the `preflight()` table, defend your pick |
| **Creativity** | Originality of the algorithm, the parameter choice, or (for self-defined tracks) the workload itself |
| **Robustness** | Code quality, reproducibility, clean submission template |

Full rubric: [docs/judging.md](docs/judging.md).

---

## Submit

Copy [templates/submission/](templates/submission/), fill in `results.json` and `submission.ipynb`, paste your `preflight().summary()` output into `preflight_log.txt`, push a fork, and open a PR against this repo before the deadline.

Schema and walk-through: [docs/submission.md](docs/submission.md).

---

## Help

- 5-minute walkthrough: [docs/quickstart.md](docs/quickstart.md)
- `preflight()` reference: [docs/preflight.md](docs/preflight.md)
- FAQ: [docs/faq.md](docs/faq.md)
- Hackathon support: your organiser's Slack workspace

---

License: MIT. Starter code in this repo is yours to fork, modify, and ship. The `uniqx` SDK installed via the private index remains proprietary and is licensed for hackathon use through your API key.
