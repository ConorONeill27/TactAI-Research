# Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies

**Authors:** Conor O'Neill\* and Alec Vysotski\* (*\*Equal contribution*)  
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![MuJoCo 3.15](https://img.shields.io/badge/physics-MuJoCo%203.15-green.svg)](https://mujoco.org/)
[![Preprint: arXiv](https://img.shields.io/badge/preprint-arXiv%20(cs.RO%20%2F%20eess.SY)-b31b1b.svg)](docs/research_paper.pdf)

---

## 1. Executive Summary

Modern robotic imitation and action-chunking diffusion policies (such as tactile-conditioned variants like **FARM**, Helmut et al., 2025) generalize well visually but provide no physical safety guarantee when contact geometry is out of distribution (OOD, e.g., up to $\pm 15^\circ$ per-axis tilt during $0.15\text{ mm}$-clearance peg insertion). In stiff environments ($K_{\text{env}} \ge 10^4\text{ N/m}$), unconstrained policies extrapolate and drive the part into the receptacle, generating force spikes above $40\text{ N}$ within tens of milliseconds and causing two-point wedging or object drop.

This repository implements **TactAI**, a 1 kHz supervisory safety filter combining a 7-variable barrier-inspired hybrid impedance Quadratic Program (OSQP), a decoupled incipient-slip margin controller, a 3-stage anti-jamming automaton, and a discrete-time Energy Tank passivity supervisor. By fusing 1 kHz robot proprioception with delayed (10–50 ms) dual-pad optical-tactile traction invariants (GelSight Mini / FEATS traction fields), the filter enforces:
1. **Bounded Contact Forces:** Strict normal force capping ($\|\mathbf{F}_{\text{ext}}\|_2 \le F_{\max} = 15.0\text{ N}$).
2. **Contact-Triggered Un-Wedging & Alignment:** Micro-retraction ($z_{\text{clear}} = 2\text{ mm}$) before rotational alignment driven by Differential Center of Pressure ($\Delta \mathbf{r}_c$) and the Contact Constraint Tensor ($\mathbf{C}$).
3. **Multi-Rate Contact Passivity:** Discrete-time Energy Tank dissipation bounds during delayed (10–50 ms) tactile updates, eliminating contact chattering without retraining the policy.

---

## 2. Mathematical Architecture

```
+-------------------------------------------------------------------------------+
|                        HIGH-LEVEL POLICY / DEMO                                |
|        (20 Hz Action-Chunking Policy Proxy, H = 50 steps)                     |
+-------------------------------------------------------------------------------+
                                      |
                           Nominal Command u_pi
                                      v
+-------------------------------------------------------------------------------+
|                    TACTILE SUPERVISORY SAFETY FILTER                          |
|         1 kHz 7-Variable Convex QP via OSQP + Energy Tank Passivity           |
|                                                                               |
|   min  (1/2) ||u - u_target(delta_r_c, C, u_pi)||_W^2 + (1/2) lambda delta^2  |
|   s.t. u_z + delta >= u_z,cbf(h_force, h_jam),  |u_x|, |u_y| <= F_lat,max     |
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
|   - Whitney Tip-Moment Ratio J_jam = ||tau_tilt||_2 / (|F_z,ext| r_char + eps)|
+-------------------------------------------------------------------------------+
```

### 2.1 Tactile Invariants
Over sensor surface domain $\Omega \subset \mathbb{R}^2$ ($32 \times 32 = 1024$ discrete cells per pad $p \in \{L, R\}$):

1. **Differential Center of Pressure ($\Delta \mathbf{r}_c$):**
   $$\Delta \mathbf{r}_c = \frac{1}{2}\left(\mathbf{r}_{c,L} - \mathbf{r}_{c,R}\right), \quad \mathbf{r}_{c,p} = F_{z,p}^{-1} \int_\Omega \mathbf{u} f_{z,p}(u, v) \, \mathrm{d}u \, \mathrm{d}v \in \mathbb{R}^2$$

2. **Contact Constraint Tensor ($\mathbf{C}_p$):**
   $$\mathbf{C}_p = \int_\Omega \left( \mathbf{u} - \mathbf{r}_{c,p} \right) \left( \mathbf{u} - \mathbf{r}_{c,p} \right)^\top f_{z,p}(u, v) \, \mathrm{d}u \, \mathrm{d}v \succeq \mathbf{0}$$

3. **Incipient Stick Ratio ($\rho_{\text{stick}}$):**
   $$\rho_{\text{stick}} = \frac{\int_\Omega \mathbb{I}(\|\mathbf{f}_{xy}(u, v)\|_2 < \mu f_z(u, v)) \, \mathrm{d}u \, \mathrm{d}v}{\int_\Omega \mathbb{I}(f_z(u, v) > 0) \, \mathrm{d}u \, \mathrm{d}v} \in [0, 1]$$

4. **Whitney-Inspired Tip-Moment Ratio ($J_{\text{jam}}$):**
   $$J_{\text{jam}} = \frac{\|\hat{\boldsymbol{\tau}}_{\text{tilt}}\|_2}{|\hat{F}_{z,\text{ext}}| \, r_{\text{char}} + \epsilon}, \quad \hat{\boldsymbol{\tau}}_{\text{tilt}} = \hat{\boldsymbol{\tau}}_{\text{ext}} - \mathbf{r}_{\text{tip}} \times \hat{\mathbf{F}}_{\text{ext}}, \quad h_{\text{jam}} = 1 - J_{\text{jam}}$$

---

## 3. Core Architectural Modules

| Module | Function | Implementation Location |
|---|---|---|
| **Multi-Rate Invariant Observer & Hybrid Impedance QP** | 1 kHz OSQP wrench filtering using Differential CoP $\Delta \mathbf{r}_c$, Tensor $\mathbf{C}$, and Stick Ratio $\rho_{\text{stick}}$ | [`src/controllers/tactile_alignment_controller.py`](src/controllers/tactile_alignment_controller.py), [`src/invariants/tactile_invariants.py`](src/invariants/tactile_invariants.py) |
| **3-Stage Anti-Jamming Automaton** | Contact-triggered rim arrest ($G_{1\to 2}$), $2\text{ mm}$ un-wedging micro-retraction, and aligned insertion ($G_{2\to 3}$) | [`src/controllers/tactile_alignment_controller.py`](src/controllers/tactile_alignment_controller.py) |
| **Energy-Tank Passivity Supervisor** | Multi-rate passivity tank $T(t)$ bounding energy injected by delayed tactile updates | [`src/controllers/tactile_alignment_controller.py`](src/controllers/tactile_alignment_controller.py) |

---

## 4. Repository Structure

```
TactAI-Research/
├── docs/
│   ├── figures/                      # 3D MuJoCo renders, equations, and benchmark telemetry plots
│   ├── lab_notebook_phase_1.md       # Traceable lab notebook recording experimental hypotheses
│   └── research_paper.pdf            # 6-page research preprint (PDF)
├── src/
│   ├── controllers/
│   │   ├── tactile_alignment_controller.py # 1 kHz OSQP safety filter, differential CoP & Energy Tank
│   │   ├── baseline_controllers.py         # Unconstrained, Fixed Impedance (K=200), and Wrist F/T Admittance baselines
│   │   └── nominal_policies.py             # 20 Hz StochasticActionChunkingPolicy (action-chunking policy proxy)
│   ├── sensors/
│   │   └── gelsight_sensor.py        # 3D continuous traction field simulator (Cattaneo-Mindlin patch model)
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
│   ├── test_tactile_controller.py    # Phase 2 force barrier, alignment, passivity, and latency tests
│   └── test_phase3_benchmarks.py     # Phase 3 comparative baselines, Benchmarks A/B/C & robustness tests
├── scripts/
│   ├── run_phase1_validation.py      # Phase 1 perception, invariant, and throughput validation
│   ├── run_phase2_validation.py      # Phase 2 closed-loop tactile alignment vs. baseline benchmark
│   ├── run_phase3_validation.py      # Phase 3 Benchmarks A/B/C, Ablation Study & Robustness Sweeps
│   ├── render_visual_snapshots.py    # 3D MuJoCo snapshot renderer
│   └── visualize_interactive_3d.py   # Real-time interactive 3D MuJoCo viewer
├── generate_paper_pdf.py             # Deterministic ReportLab script for 6-page research preprint generation
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
*Expected result: 28 tests pass across Phase 1 physics/invariants, Phase 2 OSQP controller, and Phase 3 comparative benchmarks (Benchmarks A, B, C, action-chunking policy proxy, and robustness).*

### 5.3 Run Phase 1, Phase 2 & Phase 3 Scientific Validation Scripts
```bash
python scripts/run_phase1_validation.py
python scripts/run_phase2_validation.py
python scripts/run_phase3_validation.py
```

#### Table I: Benchmark A: $0.15\text{ mm}$-Clearance Insertion (Held-Out Random Set Primary, $N = 50$; Structured Stress Set, $N = 50$)

| Controller / Policy | Held-Out Succ. ($N=50$) | Held-Out In-Basin | Held-Out Jam / Peak | Structured Succ. ($N=50$) | Structured Jam / Peak |
|---|---|---|---|---|---|
| Unconstrained Policy (Proxy) | `0.0% (0/50) [0–7%]` | `0/32 (0.0%)` | `100% / 42.96 ± 2.61 N` | `6.0% (3/50)` | `94.0% / 41.87 ± 7.43 N` |
| Wrist F/T Admittance | `8.0% (4/50) [3–19%]` | `3/32 (9.4%)` | `92.0% / 33.54 ± 3.82 N` | `20.0% (10/50)` | `80.0% / 32.80 ± 6.00 N` |
| Fixed Impedance ($K=200$) | `0.0% (Rim Stall)` | `0/32 (0.0%)` | `0.0% / 9.99 ± 0.17 N` | `0.0% (Rim Stall)` | `0.0% / 10.31 ± 0.21 N` |
| **Tactile Safety Filter (Ours)** | **`74.0% (37/50) [60–84%]`** | **`31/32 (96.9%)`** | **`0.0% / 10.15 ± 2.29 N`** | **`94.0% (47/50)`** | **`0.0% / 8.83 ± 2.58 N`** |

#### Table II: Benchmarks B & C ($N = 20$ Trials Each — Non-Planar Contour Tracking & Incipient-Slip Tool Manipulation)

| Controller Architecture | Bench B: Safe Track (%) | Bench B: Force RMSE | Bench C: Drop Rate (%) | Bench C: Mean Slip / Grip |
|---|---|---|---|---|
| Unconstrained Policy (Proxy) | `15.0% (3/20)` | `6.80 N (Peak 20.37 N)` | `100.0% (0% Succ.)` | `148.8 mm / 6.20 N` |
| Fixed Impedance ($K=200$) | `85.0% (17/20)` | `2.53 N (Peak 12.43 N)` | `85.0% (15% Succ.)` | `104.8 mm / 6.80 N` |
| Wrist F/T Admittance | `0.0% (Chatter)` | `27.38 N (Peak 127.59 N)` | `85.0% (15% Succ.)` | `44.0 mm / 13.25 N` |
| **Tactile Safety Filter (Ours)** | **`100.0% (20/20 <= 15 N)`** | **`2.06 N (Peak 10.64 N)`** | **`10.0% (90% Succ.)`** | **`0.44 mm / 14.60 N`** |

#### Table III: Component Ablations on In-Basin OOD Trials ($N = 20$) & Multi-Seed Environmental Sweeps ($N = 5$ Seeds/Point)
- **Ablation 1 (w/o CoP $\Delta \mathbf{r}_c$ & Tensor $\mathbf{C}$):** `0.0%` Insertion (Stalls on Rim at `11.79 N`) vs. `100.0%` Succ. (Sub-Degree Alignment).
- **Ablation 2 (w/o Stage 2 Un-Wedging $h_{\text{jam}}$):** `0.0%` Insertion (`30.22 ± 2.02 N` Camming) vs. `100.0%` Succ. (`0.0%` Jamming, `< 15 N`).
- **Ablation 3 (w/o Energy Tank Passivity $T(t)$):** Bench A `35%` Succ. / Bench B `77.0 N` Chatter vs. Bench A `100%` (`11.8 N`) / Bench B `10.64 N`.
- **Ablation 4 (w/o Incipient Slip Controller $h_{\text{slip}}$):** `100.0%` Tool Drop (`148.8 mm` Gross Slip) vs. `90.0%` Retention (`0.44 mm` Mean Slip).
- **Surface Stiffness $K_{\text{env}} \in [10^2, 10^5]\text{ N/m}$ ($N=5$ seeds):** `10^2`: Safe (`3.2 N`, No Insert); `10^3–10^5`: `100%` Succ. (`< 15 N`).
- **Friction $\mu \in [0.15, 0.85]$ & Latency $\tau_{\text{delay}} \in [10, 50]\text{ ms}$ ($N=5$ seeds):** `100%` Succ. / Safe (`8.63–11.91 N` Peaks).

### 5.4 Re-Generate Research Paper PDF
```bash
python generate_paper_pdf.py
```
*Compiles [`docs/research_paper.pdf`](docs/research_paper.pdf) with all equations, propositions, empirical tables (Tables I, II & III), and benchmark figures (Figs. 1–4).*
