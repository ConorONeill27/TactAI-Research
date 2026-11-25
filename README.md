# Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies

Conor O'Neill and Alec Vysotski, equal contribution.

TactAI combines a 1 kHz CBF-inspired wrench QP, a three-stage un-wedging controller, tactile contact invariants, and post-solve energy-tank scaling. The experiments use MuJoCo, analytical GelSight-style pressure fields, and a stochastic action-chunking policy proxy.

- [IEEEtran manuscript PDF](docs/research_paper.pdf)
- [Fixed-seed held-out results](data/phase3_validation/heldout_summary.json)
- [Tactile perturbation, tensor, and rim-guard audit](data/phase3_validation/contact_audit.json)
- [Architecture diagram](docs/figures/controller_architecture.pdf)

## Setup

The audit used Python 3.12.7, MuJoCo 3.15.0, NumPy 2.2.3, SciPy 1.18.1, and OSQP 1.1.3. A desktop Python installation with Tk is suitable for the SciFest UI. Run from the repository root.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS or Linux. Then install the dependencies for the manuscript pipeline and demo.

```sh
python -m pip install numpy==2.2.3 scipy==1.18.1 mujoco==3.15.0 osqp==1.1.3 matplotlib==3.11.2
```

For older experiments and the historical ReportLab generator, see [requirements.txt](requirements.txt). For the manuscript, install TeX Live or MiKTeX with `latexmk`, `IEEEtran`, `algorithm2e`, and `booktabs`, or use Tectonic. The snapshot and demo renderers require an OpenGL context. A headless Linux host can use `MUJOCO_GL=egl` with a functioning EGL driver.

## SciFest demo

```sh
python demo_scifest.py
```

Set initial tilt and lateral offset, then select **Run / restart**. **Compare** runs the same seeded initial condition side by side with the filter off and on. **OFF only** and **ON only** isolate a controller. Press **Space** to pause or resume and **R** to restart. The force charts share a scale and show the 15 N target and 40 N simulator jam threshold. Each panel shows the simulated left GelSight pressure and its center of pressure. The filtered panel shows Stages 1–3.

The demo advances a few 1 ms simulation ticks per display update so visitors can see the short contact event. At the default +10° and 0 mm, the baseline jams at 43.70 N and the filter inserts with a 15.24 N peak. The display reports excursions above 15 N.

```sh
python demo_scifest.py --smoke
```

## Reproduce the manuscript

```sh
python reproduce_all.py
```

The script runs the fixed-seed held-out comparison, Benchmarks A/B/C, robustness and contact audits, and the illustrative telemetry episode. It regenerates the figures and LaTeX tables and compiles `docs/research_paper.pdf`. LaTeX build intermediates are under `build/latex`. Use `--workers 4` to select contact-audit parallelism, `--compile-only` to reuse existing logs and figures, or `--tectonic PATH` to use a portable compiler. The full suite can take 15 minutes or more.

The held-out seed is `20261020`. The tactile and tensor panels use in-basin sampling seeds `73101` and `73102`. Guard perturbations are paired evaluations on the held-out conditions. The default controller remains frozen. Benchmark jam tallies also classify failed insertions above 22 N and below 10 mm final depth as wedged. The simulator stops for forces above 40 N or for a low jamming index combined with a force above 25 N.

The current held-out rerun gives **36/50 insertions**, **13/50 peaks at or below 15 N**, and a **27.25 N maximum**. Four of the 28 existing unit tests fail at their strict 15 N assertions on the audited dependency stack. Their force results match the pre-audit controller. Run the suite with `python -m unittest discover -v`.

## Repository map

| Path | Role |
| --- | --- |
| `src/envs/` | MuJoCo insertion, contour, and slip benchmarks |
| `src/controllers/` | Supervisory controller, policy proxy, and baselines |
| `src/sensors/` and `src/invariants/` | Tactile pressure fields and contact invariants |
| `scripts/run_phase3_validation.py` | Structured comparison, ablations, robustness, and figures |
| `scripts/run_heldout_validation.py` and `scripts/run_contact_audit.py` | Held-out comparison and paired audits |
| `docs/main.tex` and `docs/benchmark_results.tex` | IEEEtran source and generated result macros |

For arXiv source upload, use the two TeX files under `docs/` and the referenced files from `docs/figures/`, preserving their relative paths. The intended categories are `cs.RO` and `eess.SY`.
