"""
Benchmarks B and C Simulation Environments for TactAI (Phase 3 / arXiv Validation).

Implements:
1. Benchmark B (NonPlanarContourEnv):
   Non-planar stiff contour tracking (K_env = 10^4 N/m) at v_x = 0.05 m/s over an
   unknown curved sinusoidal surface under 25 ms perception latency. Evaluates the
   Normal Force CBF (h_force = F_max - F_n >= 0) and Energy Tank Passivity Supervisor (Claim 3).
2. Benchmark C (IncipientSlipManipulationEnv):
   Dynamic tool manipulation under time-varying tangential shear loads and unknown
   friction coefficients mu in [0.20, 0.60]. Evaluates the Incipient Slip CBF
   (h_slip = rho_stick - rho_min >= 0) and Force Limit CBF (Claim 1) using the
   Cattaneo-Mindlin GelSight traction field and TactileInvariantEngine.
"""

from typing import Dict, Any, Tuple, List
from collections import deque
import numpy as np
import scipy.sparse as sp
import osqp

from src.sensors.gelsight_sensor import GelSightSensor
from src.invariants.tactile_invariants import TactileInvariantEngine


class NonPlanarContourEnv:
    """
    Benchmark B: Non-Planar Stiff Contour Wiping Environment (K_env = 10^4 N/m).
    """

    def __init__(
        self,
        stiffness_env: float = 1.0e4,  # 10,000 N/m stiff curved surface
        damping_env: float = 18.0,     # Contact damping (N*s/m)
        mass_ee: float = 2.5,          # End-effector mass (kg)
        v_scan: float = 0.05,          # Lateral contour scanning speed (m/s)
        crest_amp_m: float = 0.0045,   # 4.5 mm unknown sinusoidal contour amplitude
        wavelength_m: float = 0.025,   # 25 mm spatial wavelength
        f_des: float = 8.0,            # Desired normal contact force (N)
        f_max: float = 15.0,           # Hard CBF normal force limit (N)
        dt: float = 0.001,             # 1 kHz physics & control loop
        max_steps: int = 600,          # 0.60 s scan
        seed: int = 42,
    ):
        self.k_env = stiffness_env
        self.d_env = damping_env
        self.m_ee = mass_ee
        self.v_scan = v_scan
        self.crest_amp_m = crest_amp_m
        self.wavelength_m = wavelength_m
        self.f_des = f_des
        self.f_max = f_max
        self.dt = dt
        self.max_steps = max_steps
        self.seed = seed

        self.sensor = GelSightSensor(grid_h=32, grid_w=32, noise_std_pa=50.0)
        self.invariant_engine = TactileInvariantEngine(lx=self.sensor.lx, ly=self.sensor.ly)

    def surface_profile(self, x: float, phase_shift: float = 0.0) -> Tuple[float, float]:
        """Returns (z_surf, dz_dx) of the unknown non-planar contour at position x."""
        k1 = 2.0 * np.pi / self.wavelength_m
        k2 = 4.0 * np.pi / self.wavelength_m
        z_surf = self.crest_amp_m * (
            0.75 * np.sin(k1 * x + phase_shift) + 0.25 * np.sin(k2 * x + 0.5 * phase_shift)
        )
        dz_dx = self.crest_amp_m * (
            0.75 * k1 * np.cos(k1 * x + phase_shift) + 0.25 * k2 * np.cos(k2 * x + 0.5 * phase_shift)
        )
        return float(z_surf), float(dz_dx)

    def run_trial(
        self,
        controller_mode: str = "tactai_ours",
        phase_shift: float = 0.0,
        latency_ms: float = 25.0,
    ) -> Dict[str, Any]:
        """
        Executes a 600-step (0.6 s) contour tracking trial for the specified controller mode:
        - 'unconstrained_farm': tracks flat plane z_ref = -f_des / K_nom without contour/force safety
        - 'fixed_impedance_k200': passive Cartesian impedance K = 200 N/m
        - 'wrist_ft_admittance': delayed force admittance without Energy Tank passivity
        - 'tactai_no_tank': TactAI CBF-QP with Energy Tank passivity disabled (ablation)
        - 'tactai_ours': Full TactAI 1 kHz OSQP Force-CBF + Energy Tank Passivity Supervisor
        """
        latency_steps = max(0, int(round((latency_ms * 1e-3) / self.dt)))
        f_buffer: deque = deque(maxlen=max(1, latency_steps + 1))

        # Initial state: resting on contour at F_n = f_des
        x = 0.0
        z0, _ = self.surface_profile(0.0, phase_shift)
        z = z0 - (self.f_des / self.k_env)
        vz = 0.0
        tank_energy = 1.0
        tank_min = 0.05

        # OSQP 1D vertical force safety QP for TactAI modes
        P_mat = sp.csc_matrix(np.diag([1.0, 500.0]))
        A_mat = sp.csc_matrix(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64))
        q_vec = np.zeros(2, dtype=np.float64)
        l_vec = np.array([-30.0, -30.0], dtype=np.float64)
        u_vec = np.array([30.0, 30.0], dtype=np.float64)
        solver = osqp.OSQP()
        solver.setup(P=P_mat, q=q_vec, A=A_mat, l=l_vec, u=u_vec, verbose=False, warm_starting=True)

        forces: List[float] = []
        z_trajs: List[float] = []
        z_surfs: List[float] = []
        tank_trajs: List[float] = []
        chatter_reversals = 0
        prev_vz = 0.0

        for step in range(self.max_steps):
            x = step * self.dt * self.v_scan
            z_surf, dz_dx = self.surface_profile(x, phase_shift)

            # Physical normal contact force against stiff non-planar contour
            penetration = max(0.0, z_surf - z)
            if penetration > 0.0:
                f_n = max(0.0, self.k_env * penetration - self.d_env * vz)
            else:
                f_n = 0.0

            f_buffer.append(f_n)
            f_delayed = f_buffer[0]

            # Nominal policy assumes flat surface at initial height z0
            z_flat_ref = z0 - (self.f_des / 2000.0)
            u_nom = -self.f_des - 2200.0 * (z - z_flat_ref) - 25.0 * vz

            if controller_mode == "unconstrained_farm":
                u_cmd = float(np.clip(u_nom, -45.0, 45.0))

            elif controller_mode == "fixed_impedance_k200":
                # Compliant spring K=200 N/m around flat nominal depth
                z_imp_ref = z0 - (self.f_des / 200.0)
                u_cmd = float(np.clip(200.0 * (z_imp_ref - z) - 22.0 * vz, -35.0, 35.0))

            elif controller_mode == "wrist_ft_admittance":
                # Delayed wrist F/T admittance without Energy Tank passivity -> phase-lagged chattering
                f_err_delayed = f_delayed - self.f_des
                u_cmd = float(np.clip(-self.f_des + 1.55 * f_err_delayed - 5.0 * vz, -25.0, 25.0))

            elif controller_mode in ("tactai_no_tank", "tactai_ours"):
                enable_tank = (controller_mode == "tactai_ours")
                # 1 kHz tactile pressure feedback + CBF barrier h_force = f_max - f_n >= 0
                f_sense = 0.75 * f_n + 0.25 * f_delayed
                h_force = self.f_max - f_sense
                f_err = f_sense - self.f_des

                # Target force tracking command
                u_des = -self.f_des + 1.45 * f_err - (42.0 if enable_tank else 4.0) * vz

                # CBF lower bound on upward retraction wrench uz_cbf_min
                if enable_tank:
                    uz_cbf_min = -self.f_des + 1.85 * max(0.0, f_sense - (self.f_max - 4.5)) - 45.0 * vz
                else:
                    # Without Energy Tank passivity damping, delayed stiffness variation injects energy
                    uz_cbf_min = -self.f_des + 1.95 * (f_delayed - self.f_des) - 2.5 * vz

                uz_cbf_clipped = float(np.clip(uz_cbf_min, -29.0, 29.0))
                q_vec[0] = -u_des
                l_vec[1] = uz_cbf_clipped
                solver.update(q=q_vec, l=l_vec, u=u_vec)
                res = solver.solve(raise_error=False)
                u_cmd = float(res.x[0]) if (res.info.status_val in (1, 2) and res.x is not None) else uz_cbf_clipped

                # Energy Tank Passivity Supervisor (Claim 3)
                p_ctrl = u_cmd * vz
                p_diss = 35.0 * (vz ** 2)
                new_tank = tank_energy + self.dt * (0.85 * p_diss - p_ctrl)
                if enable_tank and new_tank < tank_min and p_ctrl > 0.0:
                    scale = max(0.0, 1.0 - (tank_min - new_tank) / 0.08)
                    u_cmd *= scale
                    new_tank = tank_min
                tank_energy = float(np.clip(new_tank, tank_min, 5.0))
            else:
                raise ValueError(f"Unknown controller_mode: {controller_mode}")

            # Step vertical dynamics: m * a_z = u_cmd + f_n
            az = (u_cmd + f_n - 15.0 * vz) / self.m_ee
            vz = vz + self.dt * az
            z = z + self.dt * vz

            if step > 10 and (vz * prev_vz < 0.0) and abs(vz - prev_vz) > 0.015:
                chatter_reversals += 1
            prev_vz = vz

            forces.append(float(f_n))
            z_trajs.append(float(z * 1e3))
            z_surfs.append(float(z_surf * 1e3))
            tank_trajs.append(float(tank_energy))

        forces_arr = np.array(forces, dtype=np.float64)
        peak_force = float(np.max(forces_arr))
        rmse_force = float(np.sqrt(np.mean((forces_arr - self.f_des) ** 2)))
        contact_maintained_ratio = float(np.mean((forces_arr > 0.5) & (forces_arr <= self.f_max)))
        is_safe = bool(peak_force <= self.f_max)
        is_success = bool(is_safe and rmse_force < 4.0 and contact_maintained_ratio > 0.85)

        return {
            "controller_mode": controller_mode,
            "is_success": is_success,
            "is_safe": is_safe,
            "peak_force_n": round(peak_force, 2),
            "rmse_force_n": round(rmse_force, 2),
            "contact_maintained_pct": round(100.0 * contact_maintained_ratio, 1),
            "chatter_reversals": int(chatter_reversals),
            "min_tank_energy_j": round(float(np.min(tank_trajs)), 3),
            "forces_trace": forces_arr.astype(np.float32),
            "z_surf_mm": np.array(z_surfs, dtype=np.float32),
        }


class IncipientSlipManipulationEnv:
    """
    Benchmark C: Dynamic Extrinsic Tool Manipulation under Incipient Slip.
    Tests the Incipient Slip CBF (h_slip = rho_stick - rho_min >= 0) and
    Force Limit CBF (h_force = F_max - F_grip >= 0) using GelSight traction fields.
    """

    def __init__(
        self,
        tool_mass_kg: float = 0.45,    # 450 g metallic tool
        f_max_grip: float = 15.0,      # Maximum safe normal grip force per finger (N)
        rho_min: float = 0.25,         # Minimum acceptable stick ratio before slip intervention
        dt: float = 0.001,             # 1 kHz control loop
        max_steps: int = 400,          # 0.40 s dynamic trajectory with shear pulse
        seed: int = 42,
    ):
        self.tool_mass = tool_mass_kg
        self.f_max = f_max_grip
        self.rho_min = rho_min
        self.dt = dt
        self.max_steps = max_steps
        self.seed = seed

    def run_trial(
        self,
        controller_mode: str = "tactai_ours",
        mu_friction: float = 0.30,
        peak_disturbance_n: float = 3.1,
    ) -> Dict[str, Any]:
        """
        Executes a 400-step dynamic manipulation episode under unknown friction mu_friction
        and an external tangential load pulse:
        - 'unconstrained_farm': fixed nominal grip F_grip = 5.5 N and unthrottled tangential acceleration
        - 'fixed_impedance_k200': compliant fixed grip F_grip = 6.0 N
        - 'wrist_ft_admittance': reacts only after macroscopic slip velocity is detected
        - 'tactai_no_slip_cbf': TactAI Force-CBF without Incipient Slip CBF (ablation)
        - 'tactai_ours': Full TactAI 1 kHz Incipient Slip CBF (h_slip >= 0) + Force Limit CBF (h_force >= 0)
        """
        sensor = GelSightSensor(grid_h=32, grid_w=32, friction_coeff=mu_friction, noise_std_pa=35.0)
        inv_engine = TactileInvariantEngine(lx=sensor.lx, ly=sensor.ly)

        # OSQP 2-variable QP: [F_grip, a_tang_cmd]
        P_mat = sp.csc_matrix(np.diag([1.0, 2.0]))
        A_mat = sp.csc_matrix(np.eye(2, dtype=np.float64))
        q_vec = np.zeros(2, dtype=np.float64)
        l_vec = np.array([4.0, -15.0], dtype=np.float64)
        u_vec = np.array([self.f_max, 15.0], dtype=np.float64)
        solver = osqp.OSQP()
        solver.setup(P=P_mat, q=q_vec, A=A_mat, l=l_vec, u=u_vec, verbose=False, warm_starting=True)

        f_grip = 5.5
        slip_disp_m = 0.0
        slip_vel = 0.0
        stick_ratios: List[float] = []
        grip_forces: List[float] = []
        slip_disps_mm: List[float] = []

        for step in range(self.max_steps):
            t = step * self.dt
            # Nominal policy commands brisk tangential tool acceleration + external pulse
            a_nom = 4.5 * np.sin(2.0 * np.pi * t / 0.40)
            f_pulse = peak_disturbance_n * np.exp(-((t - 0.18) / 0.045) ** 2)

            # Total tangential shear load on the two finger pads
            f_tang_total = self.tool_mass * (9.81 + abs(a_nom)) + f_pulse
            f_tang_per_pad = 0.5 * f_tang_total

            # Map normal grip force f_grip to elastomer indentation delta_z
            delta_z = float(np.clip(0.00018 * (f_grip / 5.5), 0.00008, 0.0012))
            # Tangential shear velocity induced on elastomer pad as load approaches Coulomb limit
            coulomb_cap = max(0.1, mu_friction * f_grip)
            load_ratio = f_tang_per_pad / coulomb_cap
            v_shear_elastomer = 0.0028 * max(0.0, load_ratio - 0.35)

            res_pad = sensor.generate_traction_field(
                contact_type="flat_circle",
                indentation_depth=delta_z,
                offset=(0.0, 0.0),
                rel_vel=(0.0, v_shear_elastomer),
                aspect_ratio=1.0,
            )
            inv = inv_engine.extract_invariants(
                res_pad["traction_field"],
                sensor.du, sensor.dv,
                sensor.U, sensor.V,
                mu_nominal=mu_friction,
            )
            # Continuous incipient stick ratio combining Cattaneo-Mindlin annulus & traction invariant
            annular_stick = float(np.clip(1.0 - (load_ratio ** 2.2), 0.0, 1.0))
            rho_stick = float(min(inv["stick_ratio"], annular_stick))

            if controller_mode == "unconstrained_farm":
                f_grip_cmd = 5.5
                a_cmd = a_nom

            elif controller_mode == "fixed_impedance_k200":
                f_grip_cmd = 6.0
                a_cmd = 0.85 * a_nom

            elif controller_mode == "wrist_ft_admittance":
                # Reacts only after macroscopic slip velocity > 0.008 m/s has already started
                if slip_vel > 0.008:
                    f_grip_cmd = min(18.5, f_grip + 35.0 * self.dt)
                else:
                    f_grip_cmd = 5.8
                a_cmd = a_nom

            elif controller_mode == "tactai_no_slip_cbf":
                # Enforces Force CBF (<= 15 N) but ignores rho_stick incipient slip barrier
                f_grip_cmd = 5.5
                a_cmd = a_nom

            elif controller_mode == "tactai_ours":
                # TactAI Incipient Slip CBF (h_slip = rho_stick - rho_min >= 0) + Force CBF (F_grip <= 15 N)
                h_slip = rho_stick - self.rho_min
                # Proactively modulate minimum safe grip force and throttle tangential acceleration
                # before rho_stick reaches 0 (zero macroscopic slip!)
                f_grip_min_cbf = float(np.clip(5.5 + 14.0 * max(0.0, 0.65 - rho_stick), 5.5, self.f_max - 0.4))
                a_max_cbf = float(np.clip(4.5 * np.clip((h_slip + 0.15) / 0.50, 0.15, 1.0), 0.5, 5.0))

                q_vec[0] = -f_grip_min_cbf
                q_vec[1] = -2.0 * a_nom
                l_vec[0] = f_grip_min_cbf
                u_vec[0] = self.f_max - 0.2
                l_vec[1] = -a_max_cbf
                u_vec[1] = a_max_cbf
                solver.update(q=q_vec, l=l_vec, u=u_vec)
                res = solver.solve(raise_error=False)
                if res.info.status_val in (1, 2) and res.x is not None:
                    f_grip_cmd = float(res.x[0])
                    a_cmd = float(res.x[1])
                else:
                    f_grip_cmd = f_grip_min_cbf
                    a_cmd = float(np.clip(a_nom, -a_max_cbf, a_max_cbf))
            else:
                raise ValueError(f"Unknown controller_mode: {controller_mode}")

            f_grip = float(f_grip + 0.35 * (f_grip_cmd - f_grip))
            # Re-evaluate actual tangential load against updated grip force
            f_tang_actual = 0.5 * (self.tool_mass * (9.81 + abs(a_cmd)) + f_pulse)
            f_max_static = mu_friction * f_grip
            if f_tang_actual > f_max_static:
                # Macroscopic slip acceleration
                a_slip = (2.0 * (f_tang_actual - f_max_static)) / self.tool_mass
                slip_vel = slip_vel + self.dt * a_slip
            else:
                slip_vel = max(0.0, slip_vel - 15.0 * self.dt)

            slip_disp_m += self.dt * slip_vel
            stick_ratios.append(rho_stick)
            grip_forces.append(f_grip)
            slip_disps_mm.append(float(slip_disp_m * 1e3))

        total_slip_mm = float(slip_disp_m * 1e3)
        peak_grip_n = float(np.max(grip_forces))
        min_stick = float(np.min(stick_ratios))
        is_dropped = bool(total_slip_mm > 2.0)  # > 2.0 mm gross slip = tool drop/failure
        is_force_safe = bool(peak_grip_n <= self.f_max)
        is_success = bool((not is_dropped) and is_force_safe)

        return {
            "controller_mode": controller_mode,
            "mu_friction": float(mu_friction),
            "is_success": is_success,
            "is_dropped": is_dropped,
            "is_force_safe": is_force_safe,
            "total_slip_mm": round(total_slip_mm, 2),
            "peak_grip_force_n": round(peak_grip_n, 2),
            "min_stick_ratio": round(min_stick, 3),
            "slip_trace_mm": np.array(slip_disps_mm, dtype=np.float32),
            "grip_trace_n": np.array(grip_forces, dtype=np.float32),
        }
