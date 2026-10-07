# TACT AI — SCIENTIFIC LAB NOTEBOOK (PHASE 1)
**Project Title:** TactAI: Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies  
**Lead Researchers:** Tact AI Team (Independent Research Initiative)  
**Host Lab Destination:** Institute for Intelligent Autonomous Systems (IAS), TU Darmstadt  
**Date of Entry:** October 7, 2026  
**Document Version:** 1.0.0-PROVENANCE  

---

## 1. RESEARCH MOTIVATION & PROBLEM STATEMENT
During the preliminary adversarial audit of tactile robot learning (October 2026), three conventional paradigms were evaluated and falsified:
1. *Latent World Model Friction Parameter Adaptation:* Failed due to gradient inversion across non-smooth Coulomb friction boundaries in unconstrained MLPs (Experiment 0).
2. *Active Bayesian Probing for Scalar Friction:* Shown to provide zero statistical benefit over uniform random excitation in unconstrained 1D contact (Experiment 1).
3. *Single-Contact In-Hand Pivoting without Environmental Support:* Shown to be solvable by trivial constant normal force heuristics, offering no novel research contribution (Experiment 2).

### The True Frontier: Contact Safety under Geometric Constraint
In high-precision contact-rich tasks (tight-clearance insertion, contour following), modern foundation/diffusion policies (such as FARM, Helmut et al., ICRA 2026) suffer from severe out-of-distribution (OOD) failure modes:
- When geometric misalignment occurs ($\pm 15^\circ$ tilt, 0.15 mm clearance), the unconstrained policy attempts to push through the obstacle.
- In stiff environments ($K_{\text{env}} > 10^4\text{ N/m}$), contact forces exceed 40 N within 20 milliseconds, causing mechanical wedging and sensor delamination.
- Latent world models accumulate error across non-smooth contact transitions, achieving only a 32.1% success rate across standard benchmarks (ContactWorld, June 2026).

**Phase 1 Scientific Objective:** Build the foundational simulation and tactile invariant extraction engine in MuJoCo 3.15, establishing the exact physical and mathematical invariants required for 1 kHz Control Barrier Function filtering.

---

## 2. MATHEMATICAL ARCHITECTURE & INVARIANT FORMULATION

### 2.1 Sensor Coordinate Frame & SI Discretization
Let $\Omega = [-L_x/2, L_x/2] \times [-L_y/2, L_y/2]$ be the contact surface of a GelSight Mini sensor pad ($L_x = L_y = 15\text{ mm} = 0.015\text{ m}$).
The sensor surface is discretized into an $N \times N$ grid ($N = 25 \implies 625$ discrete elements).
Each element $(i, j)$ has coordinates $(u_i, v_j)$ and elemental area $\Delta A = \Delta u \cdot \Delta v$.

At each control step, the sensor outputs a continuous 3D traction field:
$$\mathbf{f}(u_i, v_j) = [f_x(u_i, v_j), f_y(u_i, v_j), f_z(u_i, v_j)]^T \quad \text{in Pascals (N/m}^2\text{)}$$

### 2.2 Invariant 1: Total Contact Wrench & Unit Normal
$$\mathbf{F}_c = \sum_{i,j} \mathbf{f}(u_i, v_j) \, \Delta A \in \mathbb{R}^3, \quad \hat{\mathbf{n}}_c = \frac{\mathbf{F}_c}{\|\mathbf{F}_c\|_2 + \epsilon}$$

### 2.3 Invariant 2: Traction Center of Pressure (CoP)
$$\mathbf{r}_c = \frac{\sum_{i,j} [u_i, v_j]^T f_z(u_i, v_j) \, \Delta A}{\sum_{i,j} f_z(u_i, v_j) \, \Delta A + \epsilon} \in \mathbb{R}^2$$

### 2.4 Invariant 3: Contact Constraint Tensor ($\mathbf{C}$)
$$\mathbf{C} = \sum_{i,j} \left( [u_i, v_j]^T - \mathbf{r}_c \right) \left( [u_i, v_j]^T - \mathbf{r}_c \right)^T f_z(u_i, v_j) \, \Delta A \in \mathbb{R}^{2 \times 2}$$
**Theorem (Positive Semi-Definiteness):** For any non-negative pressure field $f_z(u, v) \ge 0$, $\mathbf{C} \succeq 0$.
The eigenvectors of $\mathbf{C}$ indicate the principal geometric axes of the contact patch. For a line contact or edge contact, the ratio of eigenvalues $\lambda_1 / \lambda_2 \gg 1$.

### 2.5 Invariant 4: Incipient Slip Ratio ($\rho_{\text{stick}}$)
At each element:
$$\rho_{ij} = \max\left(0, 1 - \frac{\sqrt{f_{x,ij}^2 + f_{y,ij}^2}}{\mu \, f_{z,ij} + \epsilon}\right)$$
The aggregate Stick Ratio is:
$$\rho_{\text{stick}} = \frac{\sum_{i,j} \mathbb{I}(\rho_{ij} > \epsilon_{\text{stick}}) \cdot \mathbb{I}(f_{z,ij} > 0)}{\sum_{i,j} \mathbb{I}(f_{z,ij} > 0) + \epsilon} \in [0, 1]$$

### 2.6 Invariant 5: Jamming Invariant Index ($J_{\text{jam}}$)
Let $\mathbf{F}_{\text{insert}}$ be the insertion force and $\boldsymbol{\tau}_{\text{tilt}}$ be the tilting torque couple on the grasped object:
$$J_{\text{jam}} = \frac{\|\mathbf{F}_{\text{insert}}\|_2}{\|\boldsymbol{\tau}_{\text{tilt}}\|_2 + \epsilon}$$
When $J_{\text{jam}} \le 1 / \mu_{\text{hole}}$, two-point wedging occurs.

---

## 3. PHASE 1 IMPLEMENTATION ROADMAP (3-WEEK STRUCTURE)
- **Week 1:** Franka Emika Panda + GelSight Mini compliant contact simulation in MuJoCo 3.15.
- **Week 2:** 3D continuous traction field simulator matching the FEATS sensor format and SI units.
- **Week 3:** Invariant Extraction Engine with automated unit testing and mathematical assertion verification.

---

## 4. VERIFICATION LOG & REPRODUCIBILITY METRICS
All code in this repository is deterministic given random seed `SEED = 42`.
Unit tests in `tests/` verify:
1. Pressure positivity: $f_z(u, v) \ge 0$ everywhere.
2. Tensor positive semi-definiteness: $\det(\mathbf{C}) \ge 0, \text{Tr}(\mathbf{C}) \ge 0$.
3. Stick ratio bounds: $0.0 \le \rho_{\text{stick}} \le 1.0$.
4. Conservation of energy and passivity bounds under varying normal load.
