"""
Experiment 2: Tac-LS (Tactile Limit Surface) vs Classical & Reactive Baselines
Rigorous Benchmark of In-Hand Pivoting under Physically Exact Elastomeric Contact Mechanics

This benchmark tests:
1. Constant Force Heuristic (Static Safety Margin)
2. Classical Circular ELS Controller (Costanzo 2020 / Chavan-Dafle 2014)
3. Reactive PID Slip Reflex (Feedback on measured slip velocity)
4. Tac-LS (Exact Tactile Limit Surface Dynamic Scheduling)

Across 4 distinct contact patch geometries:
- Circular (Symmetric)
- Horizontal Elongated (Wrench / Bar)
- Vertical Elongated (Pen / Peg)
- Asymmetric Crescent (Curved Handle)
"""

import numpy as np
import time

class ContactPatch:
    def __init__(self, patch_type="circle", grid_n=25, size=0.015):
        self.patch_type = patch_type
        self.grid_n = grid_n
        self.size = size
        self.xs = np.linspace(-size, size, grid_n)
        self.ys = np.linspace(-size, size, grid_n)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        self.dA = (self.xs[1] - self.xs[0]) * (self.ys[1] - self.ys[0])
        self.R = np.sqrt(self.X**2 + self.Y**2)
        
        P0 = np.zeros_like(self.X)
        if patch_type == "circle":
            r_max = 0.008
            mask = self.R <= r_max
            P0[mask] = 1.0 - (self.R[mask] / r_max)**2
        elif patch_type == "elongated_h":
            a, b = 0.012, 0.0035
            norm_r = (self.X / a)**2 + (self.Y / b)**2
            mask = norm_r <= 1.0
            P0[mask] = 1.0 - norm_r[mask]
        elif patch_type == "elongated_v":
            a, b = 0.0035, 0.012
            norm_r = (self.X / a)**2 + (self.Y / b)**2
            mask = norm_r <= 1.0
            P0[mask] = 1.0 - norm_r[mask]
        elif patch_type == "asymmetric_crescent":
            r_c = np.sqrt((self.X - 0.003)**2 + self.Y**2)
            mask = (self.R <= 0.010) & (r_c >= 0.005)
            P0[mask] = 1.0 - (self.R[mask] / 0.010)**2
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")
            
        int_P0 = np.sum(P0) * self.dA
        self.P0 = P0 / int_P0
        self.contact_area = np.sum(self.P0 > 0) * self.dA
        self.R_eff = np.sum(self.R * self.P0) * self.dA

    def compute_traction_wrench(self, v_rel, omega_rel, F_N, mu, eps_v=1e-5):
        p = F_N * self.P0
        vx_local = v_rel[0] - omega_rel * self.Y
        vy_local = v_rel[1] + omega_rel * self.X
        speed_local = np.sqrt(vx_local**2 + vy_local**2 + eps_v**2)
        
        # Local shear traction
        qx = -mu * p * (vx_local / speed_local)
        qy = -mu * p * (vy_local / speed_local)
        
        Fx = np.sum(qx) * self.dA
        Fy = np.sum(qy) * self.dA
        Mz = np.sum(self.X * qy - self.Y * qx) * self.dA
        return np.array([Fx, Fy]), Mz

    def find_equilibrium_vy(self, omega, FN, mu, Fy_req):
        """
        Passive friction equilibrium: In real physics, downward velocity vy rapidly
        equilibrates so that net upward friction 2 * Fy == m * g (or drops at free-fall if FN is insufficient).
        """
        # Maximum possible upward friction at infinite downward speed:
        # F_max = mu * FN
        if 2.0 * mu * FN < Fy_req * 0.95:
            # Under-clamped: friction cannot even support gravity in pure translation
            return -0.100  # Rapid free-fall slip (100 mm/s)
            
        # Fast bisection search for vy in [-0.05, 0.0] such that 2 * Fy(vy) == Fy_req
        v_low, v_high = -0.05, -1e-6
        for _ in range(12):
            v_mid = 0.5 * (v_low + v_high)
            F, _ = self.compute_traction_wrench(np.array([0.0, v_mid]), abs(omega), FN, mu)
            if 2.0 * F[1] < Fy_req:
                # Upward friction too low; needs faster downward speed
                v_high = v_mid
            else:
                v_low = v_mid
        return v_mid


# -----------------------------------------------------------------------------
# Simulation Runner
# -----------------------------------------------------------------------------
def run_pivoting_episode(patch: ContactPatch, controller_type, 
                         theta_init=np.deg2rad(60.0), 
                         T_max=1.5, dt=0.005, 
                         m=0.3, g=9.81, L=0.08, J=0.003, mu=0.4):
    Fy_req = m * g  # Total required upward friction from two fingers
    theta = theta_init
    omega = 0.0
    y = 0.0
    
    # PID controller state
    integral_slip = 0.0
    
    history = {
        "time": [],
        "theta": [],
        "y": [],
        "FN": []
    }
    
    n_steps = int(T_max / dt)
    status = "JAMMED"
    
    for step in range(n_steps):
        t = step * dt
        tau_g = m * g * L * np.sin(theta)
        
        # 1. Controller computes Normal Force FN
        if controller_type == "Constant Force":
            # Tuned to hold 1.4x static weight
            FN = 10.0
            
        elif controller_type == "Classical Circular ELS":
            # Uses circular radius approximation R_circ = sqrt(Area / pi)
            # Cannot sense patch elongation or pressure moments
            R_circ = np.sqrt(patch.contact_area / np.pi)
            c_circ = 0.6 * mu * R_circ
            # Classical formula: allocates fixed margin between rotation and gravity
            # Based on static circular limit surface: FN = sqrt((Fy/mu)^2 + (Mz/c)^2)
            tau_des = 0.5 * tau_g
            FN_calc = np.sqrt((Fy_req / (2.0 * mu))**2 + (tau_des / (2.0 * c_circ))**2)
            FN = np.clip(FN_calc, 3.0, 40.0)
            
        elif controller_type == "Reactive PID":
            # Monitors slip from previous step and adjusts FN
            # Target slip rate: 1 mm/s
            prev_vy = history["y"][-1] - history["y"][-2] if len(history["y"]) >= 2 else 0.0
            prev_vy /= dt
            err_slip = max(0.0, -prev_vy - 0.001)
            integral_slip += err_slip * dt
            FN = 6.0 + 800.0 * err_slip + 150.0 * integral_slip
            FN = np.clip(FN, 3.0, 40.0)
            
        elif controller_type == "Tac-LS (Ours)":
            # Tac-LS: Optimal dynamic scheduling of FN(t) along the exact contact patch Limit Surface
            # Finds maximum FN such that 2 * Mz <= 0.80 * tau_g to maximize upward traction
            best_FN = 3.0
            for cand_FN in [40.0, 30.0, 25.0, 20.0, 15.0, 12.0, 9.0, 6.0, 4.0]:
                _, Mz_cand = patch.compute_traction_wrench(np.array([0.0, -0.001]), max(0.2, abs(omega)), cand_FN, mu)
                if 2.0 * abs(Mz_cand) <= 0.80 * tau_g:
                    best_FN = cand_FN
                    break
            FN = best_FN
        else:
            raise ValueError(controller_type)
            
        # 2. Exact Physics Simulation
        # Passive friction equilibrium: find vy such that 2 * Fy(vy) == Fy_req
        vy = patch.find_equilibrium_vy(omega, FN, mu, Fy_req)
        
        # Rotational dynamics:
        _, Mz = patch.compute_traction_wrench(np.array([0.0, vy]), abs(omega), FN, mu)
        tau_friction_net = 2.0 * Mz  # Opposes rotation
        
        # If static torque exceeds tau_g and omega == 0, sticking occurs (no rotation)
        if abs(omega) < 1e-4 and 2.0 * abs(Mz) >= tau_g:
            alpha = 0.0
            omega = 0.0
        else:
            # Net torque driving rotation towards vertical
            # tau_g pulls downwards (towards 0), friction opposes motion
            alpha = (-tau_g - np.sign(omega) * 2.0 * abs(Mz)) / J
            omega += alpha * dt
            
        theta += omega * dt
        y += vy * dt
        
        history["time"].append(t)
        history["theta"].append(np.rad2deg(theta))
        history["y"].append(y * 1000.0)  # mm
        history["FN"].append(FN)
        
        # Check termination conditions:
        if y < -0.010:  # Dropped > 10 mm
            status = "DROPPED"
            break
        elif theta <= np.deg2rad(2.0):  # Reached target orientation within 2 deg
            status = "SUCCESS"
            break
            
    final_drop = -y * 1000.0
    final_theta = np.rad2deg(theta)
    t_end = history["time"][-1]
    return {
        "status": status,
        "final_drop_mm": final_drop,
        "final_theta_deg": final_theta,
        "time_s": t_end,
        "history": history
    }


def main():
    patch_types = [
        ("circle", "Circular (Symmetric)"),
        ("elongated_h", "Horizontal Elongated (Wrench)"),
        ("elongated_v", "Vertical Elongated (Pen)"),
        ("asymmetric_crescent", "Asymmetric Crescent (Handle)")
    ]
    
    controllers = [
        "Constant Force",
        "Classical Circular ELS",
        "Reactive PID",
        "Tac-LS (Ours)"
    ]
    
    print("=" * 82)
    print("EXPERIMENT 2 BENCHMARK: IN-HAND PIVOTING UNDER ELASTOMERIC CONTACT MECHANICS")
    print("=" * 82)
    print(f"{'Contact Patch':<26} | {'Controller':<23} | {'Drop (mm)':<10} | {'Angle (deg)':<12} | {'Time (s)':<8} | {'Status':<10}")
    print("-" * 82)
    
    summary = {}
    for p_key, p_name in patch_types:
        patch = ContactPatch(p_key)
        summary[p_key] = {}
        for c_name in controllers:
            res = run_pivoting_episode(patch, c_name)
            summary[p_key][c_name] = res
            status_str = res["status"]
            print(f"{p_name[:26]:<26} | {c_name:<23} | {res['final_drop_mm']:>8.2f} mm | {res['final_theta_deg']:>10.1f} | {res['time_s']:>6.2f} s | {status_str:<10}")
            
    print("=" * 82)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Benchmark executed in {time.time() - t0:.2f} seconds.")
