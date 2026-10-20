# Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies

**Authors:** Conor O'Neill\* and Alec Vysotski\* (*\*Equal contribution*)  
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![MuJoCo 3.15](https://img.shields.io/badge/physics-MuJoCo%203.15-green.svg)](https://mujoco.org/)
[![Preprint: arXiv](https://img.shields.io/badge/preprint-arXiv%20(cs.RO%20%2F%20eess.SY)-b31b1b.svg)](docs/research_paper.pdf)

---

## 1. Executive Summary

Modern robotic imitation and diffusion policies (e.g., **FARM**, Helmut et al., ICRA 2026) suffer from severe out-of-distribution (OOD) failure modes when executing high-precision contact tasks under geometric misalignment ($\pm 15^\circ$ tilt, $0.15\text{ mm}$ clearance). In stiff environments ($K_{\text{env}} \ge 10^4\text{ N/m}$), unconstrained neural policies push through obstacles, generating destructive contact forces ($>40\text{ N}$) within 30 milliseconds that cause two-point wedging, sensor delamination, and motor shutdown.

This repository implements a supervisory 1 kHz Control Barrier Function Quadratic Program (CBF-QP) and Energy Tank safety filter that wraps around pre-trained action-chunking manipulation policies. By fusing 1 kHz robot proprioception with low-dimensional physical contact invariants extracted from dual optical tactile sensors (GelSight Mini / FEATS continuous traction fields), the safety filter enforces:
1. **Bounded Normal Contact Forces:** Strict normal force capping ($\|\mathbf{F}_c\| \le F_{\max} = 15.0\text{ N}$).
2. **Contact-Triggered Anti-Jamming & Alignment:** Micro-retraction and rotational alignment driven by Differential Center of Pressure ($\Delta \mathbf{r}_c$) and the Contact Constraint Tensor ($\mathbf{C}$).
3. **Multi-Rate Contact Passivity:** Discrete-time Energy Tank dissipation bounds during delayed (10–50 ms) tactile updates, eliminating contact chattering without retraining the policy.

---

## 2. Mathematical Architecture

```
+-------------------------------------------------------------------------------+
|                        HIGH-LEVEL POLICY / DEMO                                |
|             (20 Hz Action-Chunking Diffusion Policy, H = 50 steps)             |
+-------------------------------------------------------------------------------+
                                      |
                           Nominal Command u_pi
                                      v
+-------------------------------------------------------------------------------+
|                    TACTILE SUPERVISORY SAFETY FILTER                          |
|                1 kHz Convex Optimization via OSQP + Energy Tank                |
|                                                                               |
|   min  (1/2) ||u - u_target(delta_r_c, C, u_pi)||_W^2 + (1/2) lambda delta^2  |
|   s.t. L_f h_force(x) + L_g h_force(x) u_z + alpha_f h_force(x) >= -delta     |
|        |u_x|, |u_y| <= F_lat,max(h_slip),  F_grip >= F_grip,min(h_slip)       |
|        E_{k+1} = E_k + dt * (0.8 * P_diss - u_net^T v_ee) >= E_min            |
+-------------------------------------------------------------------------------+
                                      |
                            Safe Command u*
                                      v
+-------------------------------------------------------------------------------+
|                      ROBOT HARDWARE / SIMULATION                              |
|           Franka Emika Panda Compliant Manipulator (MuJoCo 3.15)              |
+-------------------------------------------------------------------------------+
                                      |
                       Elastomeric Contact Traction f_p(u,v)
                                      v
+-------------------------------------------------------------------------------+
|                    TACTILE INVARIANT EXTRACTION ENGINE                        |
|   - Differential Center of Pressure (CoP) delta_r_c in R^2                    |
|   - Contact Constraint Tensor C in R^(2x2) (lambda_1, lambda_2)              |
|   - Incipient Stick Ratio rho_stick in [0, 1]                                 |
|   - Whitney Jamming Index J_jam = (F_insert * r_char) / (tau_tilt + eps)     |
+-------------------------------------------------------------------------------+
```

### 2.1 Tactile Invariants
Over sensor surface domain $\Omega \subset \mathbb{R}^2$ ($32 \times 32 = 1024$ discrete cells per pad $p \in \{L, R\}$):

1. **Differential Center of Pressure ($\Delta \mathbf{r}_c$):**
   $$\Delta \mathbf{r}_c = \frac{1}{2}\left(\mathbf{r}_{c,L} - \mathbf{r}_{c,R}\right), \quad \mathbf{r}_{c,p} = \frac{\iint_\Omega [u, v]^T f_{z,p}(u, v) \, du \, dv}{\iint_\Omega f_{z,p}(u, v) \, du \, dv + \epsilon} \in \mathbb{R}^2$$

2. **Contact Constraint Tensor ($\mathbf{C}_p$):**
   $$\mathbf{C}_p = \iint_\Omega \left( [u, v]^T - \mathbf{r}_{c,p} \right) \left( [u, v]^T - \mathbf{r}_{c,p} \right)^T f_{z,p}(u, v) \, du \, dv \succeq 0$$

3. **Incipient Stick Ratio ($\rho_{\text{stick}}$):**
   $$\rho_{\text{stick}} = \frac{\iint_\Omega \mathbb{I}(\|\mathbf{f}_{xy}(u, v)\| < \mu f_z(u, v)) \, du \, dv}{\iint_\Omega \mathbb{I}(f_z > 0) \, du \, dv} \in [0, 1]$$

4. **Dimensionless Jamming Index ($J_{\text{jam}}$):**
   $$J_{\text{jam}} = \frac{\|\mathbf{F}_{\text{insert}}\|_2 \cdot r_{\text{char}}}{\|\boldsymbol{\tau}_{\text{tilt}}\|_2 + \epsilon}$$

---

## 3. Core Architectural Modules

| Module | Function | Implementation Location |
|---|---|---|
| **Multi-Rate Invariant Observer & CBF-QP** | 1 kHz OSQP wrench filtering using Differential CoP $\Delta \mathbf{r}_c$, Tensor $\mathbf{C}$, and Stick Ratio $\rho_{\text{stick}}$ | [`src/controllers/tactile_alignment_controller.py`](src/controllers/tactile_alignment_controller.py), [`src/invariants/tactile_invariants.py`](src/invariants/tactile_invariants.py) |
| **3-Stage Anti-Jamming State Machine** | Contact-triggered rim arrest, $2\text{ mm}$ un-wedging micro-retraction, and tensor-weighted alignment | [`src/controllers/tactile_alignment_controller.py`](src/controllers/tactile_alignment_controller.py) |
| **Energy-Tank Passivity Supervisor** | Multi-rate passivity observer bounding latency-induced perceptual energy injection | [`src/controllers/tactile_alignment_controller.py`](src/controllers/tactile_alignment_controller.py) |

---

## 4. Repository Structure

```
TactAI-Research/
├── docs/
│   ├── figures/                      # 3D MuJoCo renders and benchmark telemetry plots
│   ├── lab_notebook_phase_1.md       # Traceable lab notebook recording experimental hypotheses
│   └── research_paper.pdf            # 6-page arXiv preprint (PDF)
├── src/
│   ├── controllers/
│   │   ├── tactile_alignment_controller.py # 1 kHz OSQP CBF-QP safety filter, differential CoP & Energy Tank
│   │   ├── baseline_controllers.py         # Unconstrained, Fixed Impedance (K=200), and Wrist F/T Admittance baselines
│   │   └── nominal_policies.py             # 20 Hz StochasticActionChunkingPolicy (FARM diffusion policy proxy)
│   ├── sensors/
│   │   └── gelsight_sensor.py        # 3D continuous traction field simulator (FEATS SI units)
│   ├── invariants/
│   │   └── tactile_invariants.py     # Invariant engine (CoP, Constraint Tensor, Stick Ratio, Jamming Index)
│   ├── envs/
│   │   ├── franka_slot_env.py        # Benchmark A: Franka Panda + GelSight + 0.15mm slot in MuJoCo 3.15
│   │   └── tactile_benchmarks_env.py # Benchmark B (Non-Planar Stiff Contour) & Benchmark C (Incipient Slip)
│   └── utils/
│       └── logger.py                 # Structured episode logger (.npz trajectories + JSON metadata)
├── tests/
│   ├── test_gelsight_physics.py      # Unit tests for elastomeric traction mechanics
│   ├── test_invariant_extraction.py  # Mathematical assertions on tensor positive semi-definiteness & slip
│   ├── test_franka_slot_env.py       # Determinism, OOD misalignment, and benchmark reproducibility tests
│   ├── test_tactile_controller.py    # Phase 2 CBF-QP force barrier, alignment, passivity, and latency tests
│   └── test_phase3_benchmarks.py     # Phase 3 comparative baselines, Benchmarks A/B/C & robustness tests
├── scripts/
│   ├── run_phase1_validation.py      # Phase 1 perception, invariant, and throughput validation
│   ├── run_phase2_validation.py      # Phase 2 closed-loop tactile alignment vs. baseline benchmark
│   ├── run_phase3_validation.py      # Phase 3 Benchmarks A/B/C, Ablation Study & Robustness Sweeps
│   ├── render_visual_snapshots.py    # 3D MuJoCo snapshot renderer
│   └── visualize_interactive_3d.py   # Real-time interactive 3D MuJoCo viewer
├── generate_paper_pdf.py             # Deterministic ReportLab script for peer-reviewed arXiv paper generation
├── requirements.txt                  # Strict pinned dependencies
└── .gitignore                        # Clean repository filter
```

---

## 5. Quickstart & Scientific Verification

### 5.1 Installation
```bash
pip install -r requirements.txt
```

### 5.2 Run Full Automated Test Suite (28/28 Unit Tests)
```bash
python -m unittest discover tests -v
```
*Expected result: 28 tests pass across Phase 1 physics/invariants, Phase 2 OSQP CBF-QP controller, and Phase 3 comparative benchmarks (Benchmarks A, B, C, action-chunking policy, and robustness).*

### 5.3 Run Phase 1, Phase 2 & Phase 3 Scientific Validation Scripts
```bash
python scripts/run_phase1_validation.py
python scripts/run_phase2_validation.py
python scripts/run_phase3_validation.py
```

#### Table I: Benchmark A Summary ($N = 50$ OOD Trials, $\pm 0.5^\circ$ to $\pm 15^\circ$ Tilt, $0.15\text{ mm}$ Clearance, 20 Hz Action-Chunking Policy)

| Controller / Policy Architecture | Success Rate (%) | Peak Force (N) | Jamming Rate (%) | Cycle Time (s) |
|---|---|---|---|---|
| Unconstrained Diffusion Policy (FARM) | `6.0%` (`3/50`) | `41.87 ± 7.43 N` (Max `46.35 N`) | `94.0%` | `0.07 ± 0.05 s` |
| Wrist F/T Admittance Filter (6-Axis) | `20.0%` (`10/50`) | `32.80 ± 6.00 N` (Max `37.07 N`) | `80.0%` | `0.43 ± 0.02 s` |
| Fixed Isotropic Impedance ($K=200\text{ N/m}$) | `0.0%` (Rim Stall) | `10.31 ± 0.21 N` (Max `10.68 N`) | `0.0%` (Stalled) | `0.55 s` (Timeout) |
| **Tactile Safety Filter + FARM (Ours)** | **`94.0%` (`47/50`)** | **`10.86 ± 0.34 N` (In-Basin)** | **`0.0%` (In-Basin)** | **`0.43 ± 0.03 s`** |

#### Table II: Benchmarks B & C Summary ($N = 20$ Trials Each — Non-Planar Stiff Contour Wiping & Incipient-Slip Tool Manipulation)

| Controller / Policy Architecture | Bench B: Safe Track (%) | Bench B: Peak / RMSE (N) | Bench C: Drop Rate (%) | Bench C: Mean Slip / Grip |
|---|---|---|---|---|
| Unconstrained Diffusion Policy (FARM) | `15.0%` (`3/20`) | `20.37 ± 4.47 N / 6.80 N` | `100.0%` (`0%` Succ.) | `148.8 mm / 6.20 N` |
| Fixed Isotropic Impedance ($K=200\text{ N/m}$) | `85.0%` (`17/20`) | `12.43 ± 2.26 N / 2.53 N` | `85.0%` (`15%` Succ.) | `104.8 mm / 6.80 N` |
| Wrist F/T Admittance Filter (6-Axis) | `0.0%` (Chatter) | `127.59 ± 11.46 N / 27.38 N` | `85.0%` (`15%` Succ.) | `44.0 mm / 13.25 N` |
| **Tactile Safety Filter + FARM (Ours)** | **`100.0%` (`20/20` $\le 15\text{ N}$)** | **`10.64 ± 1.36 N / 2.06 N`** | **`10.0%` (`90%` Succ.)** | **`0.44 mm / 14.60 N`** |

#### Table III: Component Ablation ($N = 20$) & Multi-Seed Environmental Robustness ($N = 5$ Seeds/Point)
- **Ablation 1 (w/o Differential CoP $\Delta \mathbf{r}_c$ & Tensor $\mathbf{C}$):** Insertion success drops from `100.0%` to `0.0%` (stalls safely on rim at `14.21 ± 0.46 N`).
- **Ablation 2 (w/o Anti-Jamming Un-Wedging CBF $h_{\text{jam}}$):** Insertion success drops from `100.0%` to `0.0%` (`36.71 ± 1.39 N` rim camming spike).
- **Ablation 3 (w/o Energy Tank $T(t)$ / w/o Slip CBF $h_{\text{slip}}$):** Contour tracking suffers `76.97 ± 6.81 N` chattering spikes; tool manipulation suffers `100.0%` drop rate.
- **Surface Stiffness $K_{\text{env}} \in [10^2, 10^5]\text{ N/m}$ ($N=5$ seeds):** `100%` insertion success across 4 orders of magnitude (`2.85 ± 0.18 N` to `12.39 ± 0.54 N` peak force).
- **Friction $\mu \in [0.15, 0.85]$ & Latency $\tau_{\text{delay}} \in [10, 50]\text{ ms}$ ($N=5$ seeds):** `100%` insertion success and strict passivity (`10.33 ± 0.43 N` to `14.21 ± 0.43 N < 15.0 N` safety bound).

### 5.4 Re-Generate Research Paper PDF
```bash
python generate_paper_pdf.py
```
*Compiles [`docs/research_paper.pdf`](docs/research_paper.pdf) with all equations, theorems, empirical tables (Tables I, II & III), and benchmark figures (Figs. 1–4).*
