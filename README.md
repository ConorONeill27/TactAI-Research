# Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies

Conor O'Neill and Alec Vysotski, equal contribution.

TactAI combines a 1 kHz CBF-inspired wrench QP, a three-stage un-wedging controller, tactile contact invariants, and post-solve energy-tank scaling. Benchmark A and the SciFest demo use a 20 Hz PyTorch action-chunking policy trained on nominal MuJoCo insertions.

- [IEEEtran manuscript PDF](docs/research_paper.pdf)
- [A1 held-out known grasp](data/phase3_validation/unified_a1.json) and [A2 held-out unknown grasp](data/phase3_validation/unified_a2.json)
- [Training log and nominal validation](data/learned_policy/training.json)
- [Frozen controller and evaluation manifest](data/phase3_validation/unified_grasp_manifest.json)
- [A2 tactile perturbation audit](data/phase3_validation/unified_a2_stress.json)
- [Architecture diagram](docs/figures/controller_architecture.pdf)

## Setup

The audit used Python 3.12.7, MuJoCo 3.15.0, NumPy 2.2.3, SciPy 1.18.1, and OSQP 1.1.3. A desktop Python installation with Tk is suitable for the SciFest UI. Run from the repository root.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS or Linux. Then install the dependencies for the manuscript pipeline and demo.

```sh
python -m pip install numpy==2.2.3 scipy==1.18.1 mujoco==3.15.0 osqp==1.1.3 matplotlib==3.11.2 torch==2.11.0 pymupdf
```

For older experiments and the historical ReportLab generator, see [requirements.txt](requirements.txt). For the manuscript, install TeX Live or MiKTeX with `latexmk`, `IEEEtran`, `algorithm2e`, and `booktabs`, or use Tectonic. The snapshot and demo renderers require an OpenGL context. A headless Linux host can use `MUJOCO_GL=egl` with a functioning EGL driver.

## SciFest demo

```sh
python demo_scifest.py
```

Set wrist tilt, lateral offset, and in-hand grasp tilt, then select **Run / restart**. The selector changes the left controller while the right runs TactAI on the same condition. The default +3? wrist and +7? grasp example demonstrates counter-tilting. Press **Space** to pause or resume and **R** to restart.

Run `python demo_scifest.py --smoke` to exercise the default comparison and selector.

## Reproduce the manuscript

Run `python reproduce_all.py` to rebuild the paper from the archived grasp evaluation and regenerate the contour, slip, and telemetry artifacts. Use `--compile-only` to reuse all logs and figures, or `--tectonic PATH` for a portable compiler. The pipeline verifies checkpoint and controller hashes, trial aggregates, generated tables, figure annotations, and the bibliography order.

The A1 held-out replay uses the original seed and the shared tactile observer. The A2 seed was drawn after freezing and executed once. Both results and their per-controller failure lists are archived, and the reproduction runner reuses them. A2 tilt sensing uses a tapered analytical pressure calibration from simulated geometry, and wrist moment sensing is simulated at the wrist origin.

To reproduce policy training without overwriting the frozen checkpoint, run `python train_policy.py --output build/retrained/weights.pt`. The training script collects its demonstrations in MuJoCo and writes the separate nominal validation results to the output directory.

## Repository map

| Path | Role |
| --- | --- |
| `src/envs/` | MuJoCo insertion, contour, and slip benchmarks |
| `src/controllers/` | Supervisory controller, learned chunk policy, and baselines |
| `src/sensors/` and `src/invariants/` | Tactile pressure fields and contact invariants |
| `scripts/run_phase3_validation.py` | Structured comparison, ablations, robustness, and figures |
| `scripts/run_heldout_validation.py` and `scripts/run_contact_audit.py` | Held-out comparison and paired audits |
| `docs/main.tex` and `docs/benchmark_results.tex` | IEEEtran source and generated result macros |

For arXiv source upload, use the two TeX files under `docs/` and the referenced files from `docs/figures/`, preserving their relative paths. The intended categories are `cs.RO` and `eess.SY`.
