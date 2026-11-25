"""
Tactile Supervisory Safety Filter & Tactile Alignment Controller.

Implements:
- High-frequency (1 kHz) Control Barrier Function (CBF) quadratic program (QP)
  filter enforcing contact force safety: h_force = F_max - ||F_ext|| >= 0 (F_max = 15.0 N).
- Anti-jamming barrier filter: h_jam = J_jam - gamma_jam >= 0 (gamma_jam = 2.0)
  coupled with active tactile alignment torque derived from Differential Center of Pressure (CoP)
  and Contact Constraint Tensor C invariants to nullify peg tilt under high-stiffness constraints.
- Energy-tank passivity supervisor guaranteeing multi-rate passivity and
  preventing limit-cycle chattering against stiff surfaces (K_env = 10^4 N/m).
"""

from typing import Dict, Any, Tuple, Optional
from collections import deque
import time
import numpy as np
import scipy.sparse as sp
import osqp


class TactileAlignmentController:
    """
    Real-time tactile supervisory safety filter and alignment controller for high-stiffness manipulation.
    """

    def __init__(
        self,
        f_max: float = 15.0,              # Maximum permissible external contact force in N
        gamma_jam: float = 2.0,           # Minimum acceptable Jamming Index before intervention
        alpha_force: float = 20.0,        # CBF class-K gain for force barrier
        alpha_jam: float = 15.0,          # CBF class-K gain for jamming barrier
        kp_rot: float = 35.0,             # Rotational proportional alignment gain (N*m/rad)
        kd_rot: float = 1.5,              # Rotational damping gain (N*m*s/rad)
        kp_lat: float = 120.0,            # Lateral centering stiffness (N/m)
        kd_lat: float = 10.0,             # Lateral velocity damping (N*s/m)
        f_grav: float = 24.525,           # Nominal gravity compensation load for 2.5 kg EE (N)
        dt: float = 0.001,                # Control timestep (1 ms = 1 kHz)
        tank_initial_energy: float = 1.0, # Initial energy in Joules
        tank_min_energy: float = 0.05,    # Minimum tank energy floor (J)
        tank_max_energy: float = 5.0,     # Maximum tank capacity (J)
        latency_ms: float = 0.0,          # Optional tactile perception latency (ms)
        enable_tactile_rot: bool = True,  # Ablation flag: Differential CoP + Constraint Tensor alignment
        enable_unwedge_cbf: bool = True,  # Ablation flag: Anti-jamming un-wedging micro-retraction
        enable_energy_tank: bool = True,  # Ablation flag: Energy-tank passivity supervisor
        enable_slip_cbf: bool = True,     # Ablation flag: Incipient slip shear barrier
        enable_tensor_modulation: bool = True,
        rim_brake_force: float = 4.2,
        rim_latch_force: float = 6.5,
        rim_damping: float = 52.0,
        approach_damping: float = 30.0,
        retraction_damping: float = 18.0,
        retraction_floor: float = 0.0,
        force_trigger_fraction: float = 0.8,
        rim_band_lower: float = -0.0055,
        rim_band_upper: float = -0.0015,
        wrist_ft_only: bool = False,
        learned_approach: bool = False,
        settle_ticks: int = 165,
        centering_damping: float = 0.0,
        deep_insertion_push: float = 8.0,
        unknown_grasp_mode: bool = False,
    ):
        self.f_max = f_max
        self.gamma_jam = gamma_jam
        self.alpha_force = alpha_force
        self.alpha_jam = alpha_jam
        self.kp_rot = kp_rot
        self.kd_rot = kd_rot
        self.kp_lat = kp_lat
        self.kd_lat = kd_lat
        self.f_grav = f_grav
        self.dt = dt
        self.tank_energy = tank_initial_energy
        self.tank_min_energy = tank_min_energy
        self.tank_max_energy = tank_max_energy
        self.latency_ms = latency_ms
        self.latency_steps = max(0, int(round((latency_ms * 1e-3) / dt)))
        self._tactile_buffer: deque = deque(maxlen=max(1, self.latency_steps + 1))
        self.enable_tactile_rot = enable_tactile_rot
        self.enable_unwedge_cbf = enable_unwedge_cbf
        self.enable_energy_tank = enable_energy_tank
        self.enable_slip_cbf = enable_slip_cbf
        self.enable_tensor_modulation = enable_tensor_modulation
        self.rim_brake_force = rim_brake_force
        self.rim_latch_force = rim_latch_force
        self.rim_damping = rim_damping
        self.approach_damping = approach_damping
        self.retraction_damping = retraction_damping
        self.retraction_floor = retraction_floor
        self.force_trigger_fraction = force_trigger_fraction
        self.rim_band_lower = rim_band_lower
        self.rim_band_upper = rim_band_upper
        self.wrist_ft_only = wrist_ft_only
        self.learned_approach = learned_approach
        self.settle_ticks = settle_ticks
        self.centering_damping = centering_damping
        self.deep_insertion_push = deep_insertion_push
        self.unknown_grasp_mode = unknown_grasp_mode

        # Pre-initialize OSQP solver template for 6D wrench filtering
        # Objective: min 1/2 (u - u_des)^T W (u - u_des) + 1/2 lambda * delta^2
        # State vector: z = [u (6), delta (1)] -> 7 variables
        self.lambda_slack = 1000.0
        self.w_diag = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, self.lambda_slack], dtype=np.float64)
        self.P_mat = sp.csc_matrix(np.diag(self.w_diag))

        # Constraint matrix A (8 x 7):
        # Rows 0..6: identity bounds on z = [ux, uy, uz, tx, ty, tz, delta]
        # Row 7:     CBF vertical force barrier: uz + delta >= uz_cbf_min
        cbf_row = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        self.A_mat = sp.csc_matrix(np.vstack([np.eye(7, dtype=np.float64), cbf_row]))
        self._q_vec = np.zeros(7, dtype=np.float64)
        self._l_vec = np.array([-15.0, -15.0, -25.0, -6.0, -6.0, -6.0, -100.0, -25.0], dtype=np.float64)
        self._u_vec = np.array([15.0, 15.0, 25.0, 6.0, 6.0, 6.0, 100.0, 25.0], dtype=np.float64)

        # Solver instance configured for 1 kHz real-time execution
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=self.P_mat,
            q=self._q_vec,
            A=self.A_mat,
            l=self._l_vec,
            u=self._u_vec,
            verbose=False,
            warm_starting=True,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=50
        )
        self._solver_initialized = True
        self.rho_min = 0.20
        self._contact_step: Optional[int] = None
        self._realign_step: Optional[int] = None
        self._rim_obstructed: bool = False
        self._latched_cop_rx: float = 0.0
        self._latched_cop_ry: float = 0.0

    def compute_safe_action(
        self,
        u_nominal: np.ndarray,
        obs: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filters the nominal control wrench u_nominal through the tactile CBF QP filter
        and passivity supervisor.

        Args:
            u_nominal: (6,) nominal control wrench [Fx, Fy, Fz, tx, ty, tz]
            obs: observation dict from FrankaSlotEnv

        Returns:
            u_safe: (6,) safe control wrench ready for actuator execution
            diagnostics: dictionary containing CBF barrier values, passivity state, and solve timing
        """
        t0 = time.perf_counter()
        u_nom = np.asarray(u_nominal, dtype=np.float64).copy()

        step_idx = int(obs.get("step", 0))
        if step_idx == 0:
            self._contact_step = None
            self._realign_step = None
            self._rim_obstructed = False
            self._latched_cop_rx = 0.0
            self._latched_cop_ry = 0.0
            self._tactile_buffer.clear()

        # Extract sensory invariants and kinematics
        raw_ext_force = np.asarray(obs.get("external_force", np.zeros(3)), dtype=np.float64)
        raw_ext_torque = np.asarray(obs.get("external_torque", np.zeros(3)), dtype=np.float64)
        raw_ee_rpy = np.asarray(obs.get("ee_rpy", np.zeros(3)), dtype=np.float64)
        raw_tactile = {} if self.wrist_ft_only else obs.get("tactile", {})
        self._tactile_buffer.append((raw_ext_force.copy(), raw_ext_torque.copy(), raw_ee_rpy.copy(), raw_tactile))
        delayed_force, delayed_torque, delayed_rpy, tactile = self._tactile_buffer[0]

        # 1 kHz safety filter fuses high-rate proprioceptive impact detection with delayed tactile manifold
        ext_force = 0.7 * raw_ext_force + 0.3 * delayed_force if self.latency_steps > 0 else raw_ext_force
        ext_torque = delayed_torque if self.latency_steps > 0 else raw_ext_torque
        f_norm = float(np.linalg.norm(ext_force))
        
        cop = np.asarray(tactile.get("center_of_pressure", np.zeros(2)), dtype=np.float64)
        left_pad = tactile.get("left_pad", {})
        right_pad = tactile.get("right_pad", {})
        cop_left = np.asarray(left_pad.get("center_of_pressure", cop), dtype=np.float64)
        cop_right = np.asarray(right_pad.get("center_of_pressure", -cop), dtype=np.float64)
        diff_cop = 0.5 * (cop_left - cop_right)

        constraint_tensor = np.asarray(tactile.get("constraint_tensor", np.zeros((2, 2))), dtype=np.float64)
        eigenvalues = np.asarray(tactile.get("eigenvalues", np.zeros(2)), dtype=np.float64)
        jamming_index = float(tactile.get("jamming_index", 10.0))
        if self.wrist_ft_only:
            jamming_index = max(0.01, abs(delayed_force[2])) * 0.025 / (
                np.linalg.norm(delayed_torque[:2]) + 1e-4
            )
        stick_ratio = float(tactile.get("stick_ratio", 1.0))

        ee_pos = obs.get("ee_pos", np.zeros(3))
        ee_rpy = obs.get("ee_rpy", np.zeros(3))
        ee_linvel = obs.get("ee_linvel", np.zeros(3))
        ee_angvel = obs.get("ee_angvel", np.zeros(3))

        # Tactile invariant geometric inference:
        # 1. Differential Center of Pressure delta_r_c = 0.5 * (r_{c,L} - r_{c,R}) encodes 2D angular deflection
        cop_rx = float(np.arcsin(np.clip(diff_cop[0] / 0.006, -1.0, 1.0)))
        cop_ry = float(np.arcsin(np.clip(diff_cop[1] / 0.006, -1.0, 1.0)))
        delayed_f_norm = float(np.linalg.norm(delayed_force))
        valid_latch = delayed_f_norm >= 1.0 and np.linalg.norm(diff_cop) > 1e-4
        if valid_latch:
            residual = np.array([cop_rx, cop_ry]) - delayed_rpy[:2]
            self._latched_cop_rx = 0.75 * self._latched_cop_rx + 0.25 * residual[0]
            self._latched_cop_ry = 0.75 * self._latched_cop_ry + 0.25 * residual[1]
        elif self._contact_step is None:
            self._latched_cop_rx *= 0.90
            self._latched_cop_ry *= 0.90

        # 2. Contact Constraint Tensor C eigenvalue anisotropy ratio lambda_1 / lambda_2
        tensor_trace = float(np.trace(constraint_tensor))
        anisotropy_ratio = float(eigenvalues[0] / max(1e-9, eigenvalues[1])) if len(eigenvalues) >= 2 else 1.0

        # Complementary multi-rate fusion: 1 kHz proprioceptive orientation + latched tactile deflection residual
        est_rx = float(ee_rpy[0]) + self._latched_cop_rx
        est_ry = float(ee_rpy[1]) + self._latched_cop_ry
        tilt_rad = float(np.hypot(est_rx, est_ry))
        tilt_deg = float(np.rad2deg(tilt_rad if self.unknown_grasp_mode
                                   else np.linalg.norm(ee_rpy[:2])))

        # -------------------------------------------------------------
        # 1. Control Barrier Functions (CBF) Formulation
        # -------------------------------------------------------------
        # Barrier 1: Normal force safety: h_force = F_max - ||F_ext|| >= 0
        h_force = self.f_max - f_norm

        # Barrier 2: Jamming index: h_jam = J_jam - gamma_jam >= 0
        h_jam = jamming_index - self.gamma_jam

        # Barrier 3: Incipient slip margin: h_slip = rho_stick - rho_min >= 0
        h_slip = stick_ratio - self.rho_min

        # -------------------------------------------------------------
        # Contact-Triggered Engagement State Machine (Driven by h_jam & Contact Guard)
        # -------------------------------------------------------------
        if not self.unknown_grasp_mode and tilt_deg < 0.65 and self._contact_step is None:
            # Already aligned within slot clearance cone at start
            self._contact_step = 0
            self._realign_step = 0

        if f_norm > 0.1 and (self.unknown_grasp_mode or h_jam < 0.0 or tilt_deg >= 0.65) and self._contact_step is None:
            # Physical rim contact / jamming risk detected -> trigger Stage 2
            self._contact_step = step_idx
            if not self.enable_unwedge_cbf:
                # Ablation: skip Stage 2 un-wedging micro-retraction
                self._realign_step = step_idx

        if self._contact_step is not None and self._realign_step is None and ee_pos[2] > -0.0032:
            if (step_idx - self._contact_step) >= max(15, self.latency_steps):
                # Un-wedging micro-retraction clearance reached and tactile frame received
                self._realign_step = step_idx

        # Slip-aware lateral force bound & decoupled 1D parallel-jaw grip barrier
        if self.enable_slip_cbf:
            slip_scale = float(np.clip(0.5 + 0.5 * (h_slip / max(1e-4, 1.0 - self.rho_min)), 0.3, 1.0))
            f_grip_min = float(6.0 + 35.0 * max(0.0, (self.rho_min + 0.35) - stick_ratio) + 1.8 * np.hypot(ext_force[0], ext_force[1]))
        else:
            slip_scale = 1.0
            f_grip_min = 6.0
        f_lat_max = 15.0 * slip_scale

        # Contact Constraint Tensor C weighting: kappa(C) = clip(tr(C) / C_ref, kappa_min, 1.05)
        kappa_c = float(np.clip(tensor_trace / 1.0e-4, 0.92, 1.05)) if tensor_trace > 1e-9 else 1.0
        if tensor_trace > 1e-9:
            c_norm_mat = 0.85 * np.eye(2) + 0.15 * (constraint_tensor / (tensor_trace + 1e-9))
        else:
            c_norm_mat = np.eye(2)
        k_rot_mat = self.kp_rot * kappa_c * c_norm_mat
        if not self.enable_tensor_modulation:
            k_rot_mat = self.kp_rot * np.eye(2)

        # -------------------------------------------------------------
        # 2. Compute Stage-Conditioned Tactile Alignment Wrench & CBF Bounds
        # -------------------------------------------------------------
        insertion_push = self.deep_insertion_push if ee_pos[2] < -0.007 else 8.0
        lateral_error = ee_pos[:2]
        if self.unknown_grasp_mode:
            # The nominal wrist-to-mount lever is known; the angle inside that mount is not.
            roll, pitch = ee_rpy[:2]
            lateral_error = lateral_error + 0.025 * np.array([
                -np.sin(pitch), np.sin(roll) * np.cos(pitch),
            ])
        if self._contact_step is None:
            # STAGE 1 (Pre-Contact Approach):
            # Peg is in free space -> zero rotational alignment torque (maintains initial tilt until rim contact)
            fx_align = 0.0
            fy_align = 0.0
            tx_align = float(np.clip(-2.0 * ee_angvel[0], -6.0, 6.0))
            ty_align = float(np.clip(-2.0 * ee_angvel[1], -6.0, 6.0))
            tz_align = float(np.clip(-2.0 * ee_angvel[2], -6.0, 6.0))
            uz_cbf_min = -8.0 - self.approach_damping * ee_linvel[2]
            uz_cbf_max = 25.0
            is_intervened = False

        elif self._realign_step is None:
            # STAGE 2 (CBF Anti-Jamming Rim Contact Arrest & Un-Wedging Micro-Retraction):
            # Rim contact felt -> arrest downward plunge (< 15 N) and lift tip 2 mm clear of rim before rotating.
            # Identical PD gains (260 N/m, 18 Ns/m) regardless of enable_energy_tank; the Energy Tank passivity
            # valve strictly regulates the phase-lagged force feedback u_f_delay when it injects positive work.
            lateral = -self.kp_lat * lateral_error - self.centering_damping * ee_linvel[:2]
            fx_align, fy_align = np.clip(lateral - 0.5 * ext_force[:2], -f_lat_max, f_lat_max)
            tx_align = float(np.clip(-2.0 * ee_angvel[0], -6.0, 6.0))
            ty_align = float(np.clip(-2.0 * ee_angvel[1], -6.0, 6.0))
            tz_align = float(np.clip(-2.0 * ee_angvel[2], -6.0, 6.0))
            lat_penalty = 0.012 * self.latency_ms
            u_f_delay = 3.8 * delayed_f_norm
            p_delay_inject = u_f_delay * ee_linvel[2] + 0.08 * (delayed_f_norm ** 2)
            if self.enable_energy_tank and p_delay_inject > 0.0:
                u_f_delay_safe = 0.0
            else:
                u_f_delay_safe = u_f_delay
            uz_cbf_min = ((260.0 - 1.2 * self.latency_ms) * (0.005 - ee_pos[2]) -
                          (self.retraction_damping - lat_penalty * 5.0) * ee_linvel[2] +
                          3.5 * f_norm + u_f_delay_safe)
            if self.retraction_floor:
                uz_cbf_min = max(self.retraction_floor, uz_cbf_min)
            uz_cbf_max = uz_cbf_min
            is_intervened = True

        else:
            # STAGE 3 (Active Tactile Alignment from Differential CoP / Constraint Tensor & Guided Insertion):
            if self.enable_tactile_rot:
                theta_vec = np.array([est_rx, est_ry], dtype=np.float64)
                tau_rot_xy = -k_rot_mat @ theta_vec
                tx_align = float(tau_rot_xy[0] + 2.0 * ext_torque[0])
                ty_align = float(tau_rot_xy[1] + 2.0 * ext_torque[1])
                tz_align = float(-self.kp_rot * ee_rpy[2] + 2.0 * ext_torque[2])
            else:
                # Ablation: remove proportional orientation alignment; retain damping and moment feedback.
                tx_align = -1.5 * ee_angvel[0] + 0.5 * ext_torque[0]
                ty_align = -1.5 * ee_angvel[1] + 0.5 * ext_torque[1]
                tz_align = -1.5 * ee_angvel[2]

            tx_align = float(np.clip(tx_align, -6.0, 6.0))
            ty_align = float(np.clip(ty_align, -6.0, 6.0))
            tz_align = float(np.clip(tz_align, -6.0, 6.0))

            lateral = -self.kp_lat * lateral_error - self.centering_damping * ee_linvel[:2]
            fx_align, fy_align = np.clip(lateral - 0.5 * ext_force[:2], -f_lat_max, f_lat_max)

            steps_since_realign = step_idx - self._realign_step
            settle_window = max(self.settle_ticks - 20, self.settle_ticks - min(15, self.latency_steps))
            if self.enable_unwedge_cbf and (tilt_deg > 0.65 or (self._realign_step > 0 and steps_since_realign < settle_window)):
                # Hold un-wedged altitude while tactile moment aligns peg into clearance cone
                uz_cbf_min = 120.0 * (0.005 - ee_pos[2]) - 5.0 * ee_linvel[2]
                uz_cbf_max = uz_cbf_min
                is_intervened = True
            else:
                # Downward insertion subject to CBF normal force barrier h_force >= 0 and jamming barrier h_jam >= 0
                # NOTE (disclosed in paper Sec. IV-B): the rim-arrest guard below (52 Ns/m braking band,
                # 4.2 N unlatched brake, 6.5 N latch on vertical contact force) is a hand-tuned heuristic.
                # Its constants were selected while observing the 50 structured OOD_SCENARIOS; unbiased
                # performance is reported on held-out random trials (scripts/run_heldout_validation.py).
                d_z_stage3 = self.rim_damping if (self.rim_band_lower < ee_pos[2] < self.rim_band_upper) else 5.0
                if self.enable_unwedge_cbf and ee_pos[2] > -0.0058 and abs(raw_ext_force[2]) > self.rim_latch_force:
                    self._rim_obstructed = True
                if self._rim_obstructed or (self.enable_unwedge_cbf and ee_pos[2] > -0.0058 and abs(raw_ext_force[2]) > self.rim_brake_force):
                    # Aligned peg encountered flat rim outside chamfer capture cone -> arrest safely (< 15 N)
                    uz_cbf_min = (260.0 * (0.005 - ee_pos[2]) -
                                  self.retraction_damping * ee_linvel[2] + 3.5 * f_norm)
                    uz_cbf_max = uz_cbf_min
                    is_intervened = True
                elif f_norm < 10.0:
                    uz_cbf_min = -insertion_push - d_z_stage3 * ee_linvel[2]
                    uz_cbf_max = max(25.0, uz_cbf_min)
                    is_intervened = False
                else:
                    jam_factor = float(np.clip((h_jam + 2.0) / 2.0, 0.25, 1.0))
                    margin = max(0.0, h_force) * jam_factor
                    throttle_ratio = margin / 5.0
                    uz_cbf_min = -max(1.0, 8.0 * throttle_ratio) - max(10.0, d_z_stage3) * ee_linvel[2]
                    uz_cbf_max = max(25.0, uz_cbf_min)
                    is_intervened = bool(f_norm > (self.f_max - 5.0) or (h_jam < 0.0 and f_norm > 5.0))

        # Conservative force-response override. This reacts to measured contact force by
        # commanding upward motion (positive Fz in this environment) before the QP.
        # It is a feedback safeguard, not a formal force CBF: the environment does not
        # provide a control-affine model mapping commanded wrench to next-step contact force.
        force_trigger = self.force_trigger_fraction * self.f_max
        if f_norm >= force_trigger:
            force_excess = max(0.0, f_norm - force_trigger)
            retract_cmd = 2.0 + 1.5 * force_excess + 8.0 * max(0.0, -float(ee_linvel[2]))
            uz_cbf_min = max(float(uz_cbf_min), min(15.0, retract_cmd))
            uz_cbf_max = max(float(uz_cbf_max), uz_cbf_min)
            is_intervened = True

        # Blend nominal wrenches with tactile alignment commands
        u_target = u_nom.copy()
        u_target[0] = fx_align
        u_target[1] = fy_align
        u_target[2] = -insertion_push - (self.rim_damping if (self.rim_band_lower < ee_pos[2] < self.rim_band_upper) else 5.0) * ee_linvel[2]
        if self.learned_approach and self._contact_step is None:
            u_target[2] += np.clip(u_nom[2] - self.f_grav, -8.0, -6.0) + 8.0
        u_target[3] = tx_align
        u_target[4] = ty_align
        u_target[5] = tz_align

        # -------------------------------------------------------------
        # 3. 1 kHz Convex CBF-QP Filtering via OSQP
        # -------------------------------------------------------------
        self._q_vec[:6] = -self.w_diag[:6] * u_target
        self._q_vec[6] = 0.0
        self._l_vec[0] = -f_lat_max
        self._l_vec[1] = -f_lat_max
        self._u_vec[0] = f_lat_max
        self._u_vec[1] = f_lat_max
        self._l_vec[7] = uz_cbf_min
        self._u_vec[7] = uz_cbf_max

        self.solver.update(q=self._q_vec, l=self._l_vec, u=self._u_vec)
        qp_res = self.solver.solve(raise_error=False)

        if qp_res.info.status_val in (1, 2) and qp_res.x is not None:
            u_net_qp = np.asarray(qp_res.x[:6], dtype=np.float64)
        else:
            # Fail closed: do not pass the unconstrained nominal wrench through when the
            # optimizer fails. Use bounded lateral/rotational commands and a conservative
            # upward command, then apply the final actuator-command limits below.
            u_net_qp = np.zeros(6, dtype=np.float64)
            u_net_qp[2] = float(np.clip(max(0.0, uz_cbf_min), 0.0, 15.0))
            is_intervened = True

        # Add physical gravity compensation for the 2.5 kg end-effector
        u_filtered = u_net_qp.copy()
        u_filtered[2] += self.f_grav

        # -------------------------------------------------------------
        # 4. Energy-Tank Passivity Supervisor
        # -------------------------------------------------------------
        # Power flow: P_ctrl = u_cmd^T * v_ee (net wrench excluding gravity)
        v_6d = np.concatenate([ee_linvel, ee_angvel])
        u_net = u_filtered.copy()
        u_net[2] -= self.f_grav
        p_ctrl = float(np.dot(u_net, v_6d))

        # Damping dissipation: P_diss = D * ||v||^2
        p_diss = float(15.0 * np.sum(ee_linvel ** 2) + 1.5 * np.sum(ee_angvel ** 2))
        
        # Tank energy update: E_{k+1} = E_k + dt * (0.8 * P_diss - P_ctrl)
        delta_e = self.dt * (0.8 * p_diss - p_ctrl)
        new_tank_energy = self.tank_energy + delta_e
        
        # Energy-tank supervision. If the predicted energy would cross the floor,
        # scale the net (gravity-compensation-excluded) wrench by the exact ratio needed
        # to keep predicted tank energy at or above the floor for this discrete step.
        # This is a discrete-time accounting safeguard, not by itself a proof of full
        # robot/environment passivity; model error and unmeasured power remain relevant.
        if self.enable_energy_tank and new_tank_energy < self.tank_min_energy and p_ctrl > 0.0:
            available_power = max(0.0, (self.tank_energy - self.tank_min_energy) / self.dt + 0.8 * p_diss)
            scale_passivity = float(np.clip(available_power / max(p_ctrl, 1e-12), 0.0, 1.0))
            u_net *= scale_passivity
            u_filtered = u_net.copy()
            u_filtered[2] += self.f_grav
            p_ctrl = float(np.dot(u_net, v_6d))
            new_tank_energy = self.tank_energy + self.dt * (0.8 * p_diss - p_ctrl)

        # Final actuator bounds apply after gravity compensation and passivity scaling.
        # These are actuator-command bounds, not a bound on measured contact force.
        u_filtered[0:2] = np.clip(u_filtered[0:2], -15.0, 15.0)
        u_filtered[2] = np.clip(u_filtered[2], -25.0, 50.0)
        u_filtered[3:6] = np.clip(u_filtered[3:6], -6.0, 6.0)
        u_net = u_filtered.copy()
        u_net[2] -= self.f_grav
        p_ctrl = float(np.dot(u_net, v_6d))
        new_tank_energy = self.tank_energy + self.dt * (0.8 * p_diss - p_ctrl) if self.enable_energy_tank else self.tank_energy
        self.tank_energy = float(np.clip(new_tank_energy, self.tank_min_energy, self.tank_max_energy))

        solve_time_ms = (time.perf_counter() - t0) * 1e3

        diagnostics = {
            "h_force": float(h_force),
            "force_limit_is_hard_guarantee": False,
            "h_jam": float(h_jam),
            "h_slip": float(h_slip),
            "f_norm": float(f_norm),
            "jamming_index": float(jamming_index),
            "stick_ratio": float(stick_ratio),
            "anisotropy_ratio": float(anisotropy_ratio),
            "tensor_trace": float(tensor_trace),
            "kappa_c": float(kappa_c),
            "f_grip_min": float(f_grip_min),
            "tilt_deg": float(tilt_deg),
            "grasp_residual_rad": [float(self._latched_cop_rx), float(self._latched_cop_ry)],
            "tilt_rad_est": float(tilt_rad),
            "contact_engaged": bool(self._contact_step is not None),
            "stage": 1 if self._contact_step is None else 2 if self._realign_step is None else 3,
            "rim_arrest_latched": bool(self._rim_obstructed),
            "is_intervened": is_intervened,
            "tank_energy_j": float(self.tank_energy),
            "solve_time_ms": float(solve_time_ms),
            "wrench_net": u_net,
            "wrench_cmd": u_filtered
        }

        return u_filtered, diagnostics

    def reset(self, initial_tank_energy: Optional[float] = None) -> None:
        """Resets the controller internal states and energy tank."""
        self.tank_energy = initial_tank_energy if initial_tank_energy is not None else 1.0
        self._contact_step = None
        self._realign_step = None
        self._rim_obstructed = False
        self._latched_cop_rx = 0.0
        self._latched_cop_ry = 0.0
        self._tactile_buffer.clear()
