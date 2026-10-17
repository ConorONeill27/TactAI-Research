# TACTAI — SCIENTIFIC LAB NOTEBOOK (PHASES 1, 2 & 3)
**Project Title:** Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies  
**Authors:** Alec Vysotski and Conor O'Neill  
**Collaboration Target:** Institute for Intelligent Autonomous Systems (IAS), TU Darmstadt  
**Date of Entry:** October 16, 2026  
**Document Version:** 3.0.0-ARXIV-PREPRINT  

---

## 1. RESEARCH MOTIVATION & PROBLEM STATEMENT
During the preliminary adversarial audit of tactile robot learning (October 2026), three conventional paradigms were evaluated and falsified:
1. *Latent World Model Friction Parameter Adaptation:* Failed due to gradient inversion across non-smooth Coulomb friction boundaries in unconstrained MLPs (Experiment 0).
2. *Active Bayesian Probing for Scalar Friction:* Shown to provide zero statistical benefit over uniform random excitation in unconstrained 1D contact (Experiment 1).
3. *Single-Contact In-Hand Pivoting without Environmental Support:* Shown to be solvable by trivial constant normal force heuristics, offering no novel research contribution (Experiment 2).

### The Core Scientific Frontier: Contact Safety under Geometric Constraint
In high-precision contact-rich tasks (sub-millimeter clearance insertion, non-planar stiff contour following, and dynamic shear manipulation), modern foundation/diffusion policies (such as **FARM**, Helmut et al., ICRA 2026) suffer from severe out-of-distribution (OOD) failure modes:
- When geometric misalignment occurs ($\pm 5^\circ$ to $\pm 15^\circ$ tilt, $0.15\text{ mm}$ clearance), the unconstrained policy extrapolates across non-smooth contact boundaries and pushes directly into rigid obstacles.
- In stiff environments ($K_{\text{env}} \ge 10^4\text{ N/m}$), contact forces exceed $40\text{ N}$ within $30\text{ ms}$, causing two-point mechanical wedging and elastomeric membrane delamination.

---

## 2. MATHEMATICAL ARCHITECTURE & INVARIANT FORMULATION (PHASE 1)

### 2.1 Sensor Coordinate Frame & SI Discretization
Let $\Omega = [-L_x/2, L_x/2] \times [-L_y/2, L_y/2]$ be the contact surface of each GelSight Mini sensor pad ($L_x = 18.6\text{ mm}, L_y = 15.4\text{ mm}$), discretized into a $32 \times 32 = 1024$ grid with elemental area $\Delta A = \Delta u \cdot \Delta v$.
At each control step, the sensor outputs a continuous 3D Cattaneo-Mindlin traction field:
$$\mathbf{f}(u_i, v_j) = [f_x(u_i, v_j), f_y(u_i, v_j), f_z(u_i, v_j)]^T \quad \text{in Pascals (N/m}^2\text{)}$$

### 2.2 Invariant 1: Total Contact Wrench & Unit Normal
$$\mathbf{F}_c = \sum_{i,j} \mathbf{f}(u_i, v_j) \, \Delta A \in \mathbb{R}^3, \quad \hat{\mathbf{n}}_c = \frac{\mathbf{F}_c}{\|\mathbf{F}_c\|_2 + \epsilon}$$

### 2.3 Invariant 2: Differential Traction Center of Pressure ($\Delta \mathbf{r}_c$)
For a single pad $p \in \{L, R\}$:
$$\mathbf{r}_{c,p} = \frac{\sum_{i,j} [u_i, v_j]^T f_{z,p}(u_i, v_j) \, \Delta A}{\sum_{i,j} f_{z,p}(u_i, v_j) \, \Delta A + \epsilon} \in \mathbb{R}^2$$
Across dual opposing GelSight pads in a parallel-jaw gripper, the **Differential Center of Pressure**
$$\Delta \mathbf{r}_c = \frac{1}{2}\left(\mathbf{r}_{c,L} - \mathbf{r}_{c,R}\right)$$
cancels common-mode translational shift and uniquely encodes the 2D angular deflection $\hat{\boldsymbol{\theta}}_{\text{tactile}} = \arcsin(\text{clip}(\Delta \mathbf{r}_c / r_{\text{pad}}, -1, 1))$ of the grasped workpiece.

### 2.4 Invariant 3: Contact Constraint Tensor ($\mathbf{C}$)
$$\mathbf{C} = \sum_{i,j} \left( [u_i, v_j]^T - \mathbf{r}_c \right) \left( [u_i, v_j]^T - \mathbf{r}_c \right)^T f_z(u_i, v_j) \, \Delta A \in \mathbb{R}^{2 \times 2} \succeq 0$$
The eigenvalue anisotropy ratio $\lambda_1 / \lambda_2$ distinguishes uniform planar seating ($\lambda_1 / \lambda_2 \approx 1$) from line/edge rim contact ($\lambda_1 / \lambda_2 > 2.5$).

### 2.5 Invariant 4: Incipient Stick Ratio ($\rho_{\text{stick}}$) & Whitney Jamming Index ($J_{\text{jam}}$)
$$\rho_{\text{stick}} = \frac{\sum_{i,j} \mathbb{I}(\|\mathbf{f}_{xy,ij}\|_2 < \mu \, f_{z,ij}) \cdot \mathbb{I}(f_{z,ij} > 0)}{\sum_{i,j} \mathbb{I}(f_{z,ij} > 0) + \epsilon} \in [0, 1], \qquad J_{\text{jam}} = \frac{\|\mathbf{F}_{\text{insert}}\|_2 \cdot r_{\text{char}}}{\|\boldsymbol{\tau}_{\text{tilt}}\|_2 + \epsilon}$$

---

## 3. 1 kHz SUPERVISORY SAFETY FILTER & OSQP CBF-QP (PHASE 2)

Implemented in [`src/controllers/tactile_alignment_controller.py`](../src/controllers/tactile_alignment_controller.py):
- Solves a sparse 7-variable convex Quadratic Program via `OSQP` (`warm_starting=True`, mean solve time `0.04 ms`) at 1 kHz:
  $$\min_{\mathbf{u} \in \mathbb{R}^6, \delta \in \mathbb{R}} \frac{1}{2} \|\mathbf{u} - \mathbf{u}_{\text{target}}(\Delta \mathbf{r}_c, \mathbf{C})\|_{\mathbf{W}}^2 + \frac{1}{2}\lambda_{\text{slack}}\delta^2$$
  subject to:
  1. **Normal Force Barrier ($h_{\text{force}}$):** $h_{\text{force}}(x) = F_{\max} - \|\mathbf{F}_{\text{ext}}\|_2 \ge 0$ ($F_{\max} = 15.0\text{ N}$).
  2. **Contact-Triggered Anti-Jamming Un-Wedging Barrier ($h_{\text{jam}}$):** Zero rotational torque in free space ($\|\mathbf{F}_{\text{ext}}\|_2 < 0.1\text{ N}$); upon rim contact, arrests downward motion below $15\text{ N}$ and executes a $2\text{ mm}$ un-wedging micro-retraction before engaging Differential CoP rotational alignment.
  3. **Incipient Slip Barrier ($h_{\text{slip}}$):** $h_{\text{slip}}(x) = \rho_{\text{stick}} - \rho_{\min} \ge 0$ ($\rho_{\min} = 0.20$), scaling lateral shear bounds $F_{\text{lat,max}}(h_{\text{slip}})$ and normal grip force.
  4. **Energy Tank Passivity Supervisor ($T(t)$):**
     $$E_{k+1} = E_k + \Delta t \left(0.8 P_{\text{diss}} - \mathbf{u}_{\text{net}}^T \mathbf{v}_{\text{ee}}\right) \ge E_{\min}$$

---

## 4. MULTI-BENCHMARK & ABLATION VALIDATION LOG (PHASE 3)

Implemented in [`scripts/run_phase3_validation.py`](../scripts/run_phase3_validation.py), [`src/controllers/baseline_controllers.py`](../src/controllers/baseline_controllers.py), [`src/controllers/nominal_policies.py`](../src/controllers/nominal_policies.py), and [`src/envs/tactile_benchmarks_env.py`](../src/envs/tactile_benchmarks_env.py):
- **Benchmark A (20 OOD Trials, $\pm 5^\circ$ to $\pm 15^\circ$ Tilt, $0.15\text{ mm}$ Clearance, 20 Hz Action-Chunking Nominal Policy):**
  - `Unconstrained Diffusion Policy (FARM)`: `0.0%` success, `100.0%` jamming, peak force `43.51 ± 1.47 N`.
  - `Wrist F/T Admittance Filter`: `20.0%` success (mild $\pm 5^\circ$ tilts only), `80.0%` jamming, peak force `32.54 ± 5.67 N`.
  - `Fixed Isotropic Impedance (K=200 N/m)`: `0.0%` success (stalls on slot rim), peak force `10.36 ± 0.19 N`.
  - `Tactile Safety Filter + FARM (Ours)`: `100.0%` success (`20/20`), `0.0%` jamming, peak force `11.69 ± 0.37 N` (`< 15.0 N` CBF bound).
- **Benchmark B (Non-Planar Stiff Contour Wiping, $K_{\text{env}} = 10^4\text{ N/m}$, $25\text{ ms}$ Delay):**
  - `Tactile Safety Filter + Tank (Ours)`: `100.0%` safe contour tracking with zero chattering, whereas unconstrained and delayed admittance baselines violate the $15\text{ N}$ safety bound.
- **Benchmark C (Dynamic Extrinsic Tool Manipulation under Incipient Slip, $\mu \in [0.22, 0.48]$):**
  - `Tactile Safety Filter + Slip-CBF (Ours)`: `100.0%` success (`0.0%` tool drop, `< 15.0 N` normal grip force), whereas baselines without incipient slip sensing suffer macroscopic tool drop.
- **Component Ablation Study:** Confirms that ablating Differential CoP/Tensor $\mathbf{C}$ or Anti-Jamming Un-Wedging CBF $h_{\text{jam}}$ collapses OOD insertion success to `0.0%`, while ablating the Energy Tank $T(t)$ or Slip CBF $h_{\text{slip}}$ causes force-bound violations or tool drop in Benchmarks B and C.
