"""
Experiment 2: Tac-LS (Tactile Limit Surface) vs Classical & Reactive Baselines
In-Hand Pivoting & Controlled Extrinsic Dexterity under Elastomeric Contact Mechanics

This experiment tests whether computing the exact contact-specific Limit Surface
from high-resolution tactile pressure distributions p(x, y) enables zero-drop in-hand
pivoting of arbitrary unknown objects, where classical circular ELS and reactive
reflexes either jam or drop the object.
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------------------------------
# 1. Contact Patch & Pressure Distribution Models
# -----------------------------------------------------------------------------
class ContactPatch:
    def __init__(self, patch_type="circle", grid_n=25, size=0.015):
        """
        grid_n: resolution along each axis (e.g. 25x25 = 625 traction elements)
        size: half-width of the tactile sensor pad (0.015 m = 15 mm, like GelSight Mini)
        """
        self.patch_type = patch_type
        self.grid_n = grid_n
        self.size = size
        self.xs = np.linspace(-size, size, grid_n)
        self.ys = np.linspace(-size, size, grid_n)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        self.dA = (self.xs[1] - self.xs[0]) * (self.ys[1] - self.ys[0])
        self.R = np.sqrt(self.X**2 + self.Y**2)
        
        # Base normalized pressure shape P0(x, y) such that integral P0 dA = 1
        P0 = np.zeros_like(self.X)
        if patch_type == "circle":
            r_max = 0.008
            mask = self.R <= r_max
            P0[mask] = 1.0 - (self.R[mask] / r_max)**2
        elif patch_type == "elongated_h":
            # Elongated horizontally (aspect ratio 3.5:1)
            a, b = 0.012, 0.0035
            norm_r = (self.X / a)**2 + (self.Y / b)**2
            mask = norm_r <= 1.0
            P0[mask] = 1.0 - norm_r[mask]
        elif patch_type == "elongated_v":
            # Elongated vertically (aspect ratio 1:3.5)
            a, b = 0.0035, 0.012
            norm_r = (self.X / a)**2 + (self.Y / b)**2
            mask = norm_r <= 1.0
            P0[mask] = 1.0 - norm_r[mask]
        elif patch_type == "asymmetric_crescent":
            # Curved crescent / tool handle shape
            r_c = np.sqrt((self.X - 0.003)**2 + self.Y**2)
            mask = (self.R <= 0.010) & (r_c >= 0.005)
            P0[mask] = 1.0 - (self.R[mask] / 0.010)**2
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")
            
        int_P0 = np.sum(P0) * self.dA
        if int_P0 > 0:
            self.P0 = P0 / int_P0
        else:
            self.P0 = P0
            
        # Compute exact physical properties of this contact patch
        self.contact_area = np.sum(self.P0 > 0) * self.dA
        # R_eff = integral ||r|| * P0 dA (torsional moment arm)
        self.R_eff = np.sum(self.R * self.P0) * self.dA
        # Moments of inertia
        self.I_xx = np.sum(self.Y**2 * self.P0) * self.dA
        self.I_yy = np.sum(self.X**2 * self.P0) * self.dA
        self.I_polar = self.I_xx + self.I_yy

    def get_pressure(self, F_N):
        """Total pressure distribution p(x, y) for a given total normal force F_N."""
        return F_N * self.P0

    def compute_traction_wrench(self, v_rel, omega_rel, F_N, mu, eps_v=1e-4):
        """
        Computes 2D friction force [Fx, Fy] and torsional moment Mz on the object
        exerted by ONE finger with pressure distribution p(x, y).
        v_rel: [vx, vy] translational velocity of object relative to sensor
        omega_rel: angular velocity of object relative to sensor (rad/s)
        F_N: normal force per finger (N)
        mu: friction coefficient
        """
        p = self.get_pressure(F_N)
        
        # Local relative velocity at each point (X, Y):
        # v_local = v_rel + omega_rel * (-Y, X)
        vx_local = v_rel[0] - omega_rel * self.Y
        vy_local = v_rel[1] + omega_rel * self.X
        
        speed_local = np.sqrt(vx_local**2 + vy_local**2 + eps_v**2)
        
        # Local shear traction q(x, y) opposing local slip velocity
        # Coulomb law: ||q|| = mu * p
        qx = -mu * p * (vx_local / speed_local)
        qy = -mu * p * (vy_local / speed_local)
        
        # Net forces and moment:
        # Fx = sum qx dA, Fy = sum qy dA
        # Mz = sum (X * qy - Y * qx) dA
        Fx = np.sum(qx) * self.dA
        Fy = np.sum(qy) * self.dA
        Mz = np.sum(self.X * qy - self.Y * qx) * self.dA
        
        # Incipient slip / stick ratio: elements where speed < 2 * eps_v
        contact_mask = p > 0
        stick_mask = contact_mask & (speed_local < 3.0 * eps_v)
        stick_ratio = np.sum(stick_mask) / max(1, np.sum(contact_mask))
        
        return np.array([Fx, Fy]), Mz, stick_ratio


# -----------------------------------------------------------------------------
# 2. Controllers for In-Hand Pivoting
# -----------------------------------------------------------------------------
class ClassicalCircularELSController:
    """
    Classical baseline (Costanzo et al. 2020, Chavan-Dafle & Rodriguez 2014):
    Assumes contact patch is a circular disk with area equal to nominal area.
    Cannot perceive contact patch orientation, aspect ratio, or spatial anisotropy.
    Uses circular approximation for required normal force.
    """
    def __init__(self, mu_nominal=0.4, R_circ_nominal=0.005):
        self.mu = mu_nominal
        self.R_circ = R_circ_nominal
        self.c = 0.6 * self.mu * self.R_circ

    def compute_normal_force(self, theta, omega, vy, m, g, L):
        Fy_req = (m * g) / 2.0
        # Circular approximation: ignores directional anisotropy of patch
        # Assumes rotation at omega decreases Fy by sqrt(1 + (omega * R_circ / v_tol)^2)
        v_tol = 0.001  # 1 mm/s tolerance
        FN_circ = (Fy_req / self.mu) * np.sqrt(1.0 + (np.abs(omega) * self.R_circ / v_tol)**2)
        return np.clip(FN_circ, 2.0, 45.0)


class ReactivePIDReflexController:
    """
    Reactive baseline:
    Starts at nominal holding force. Uses high-gain feedback on measured slip velocity (vy)
    to reactively clamp down when slipping, and relaxes when not slipping.
    """
    def __init__(self, Kp=2500.0, Ki=300.0, FN_base=5.0):
        self.Kp = Kp
        self.Ki = Ki
        self.FN_base = FN_base
        self.integral_drop = 0.0

    def reset(self):
        self.integral_drop = 0.0

    def compute_normal_force(self, vy, dt):
        drop_speed = max(0.0, -vy)
        self.integral_drop += drop_speed * dt
        # If dropping, react aggressively
        delta_FN = self.Kp * drop_speed + self.Ki * self.integral_drop
        FN = self.FN_base + delta_FN
        return np.clip(FN, 2.0, 45.0)


class ConstantForceController:
    """
    Heuristic baseline: Constant normal force tuned for static grasp (1.5x static threshold).
    """
    def __init__(self, FN_const=7.5):
        self.FN_const = FN_const

    def compute_normal_force(self):
        return self.FN_const


class TacLSController:
    """
    Proposed Method: Tac-LS (Tactile Limit Surface Controller)
    Uses the real-time contact pressure distribution p(x, y) directly from tactile sensing.
    Computes the exact directional velocity field coupling:
    v_local = [vx - omega * Y, vy + omega * X]
    Solves for the exact minimum FN*(p, omega) required to guarantee:
    Fy(p, FN, vy=-v_tol, omega) >= Fy_req = mg / 2
    taking into account the true 2D spatial shape and moment tensors of the patch.
    """
    def __init__(self, patch: ContactPatch, mu_est=0.4, v_tol=0.001):
        self.patch = patch
        self.mu = mu_est
        self.v_tol = v_tol  # allowable micro-slip rate (1 mm/s)
        
        # Precompute normalized traction efficiency curve as a function of omega
        # for this specific contact patch:
        # eta_y(omega) = integral (P0 * (v_tol) / sqrt((-omega*Y)^2 + (v_tol + omega*X)^2)) dA
        self.omegas_tab = np.linspace(0.0, 3.0, 61)
        self.eta_tab = np.zeros_like(self.omegas_tab)
        for i, om in enumerate(self.omegas_tab):
            vx_loc = -om * self.patch.Y
            vy_loc = -self.v_tol + om * self.patch.X
            speed_loc = np.sqrt(vx_loc**2 + vy_loc**2 + 1e-8)
            # Upward traction per unit normal force:
            qy_norm = self.mu * self.patch.P0 * (self.v_tol / speed_loc)
            self.eta_tab[i] = np.sum(qy_norm) * self.patch.dA

    def compute_normal_force(self, theta, omega, vy, m, g, L):
        Fy_req = (m * g) / 2.0
        
        # Look up exact traction efficiency for current omega:
        om_abs = np.abs(omega)
        eta = np.interp(om_abs, self.omegas_tab, self.eta_tab)
        eta = max(1e-4, eta)
        
        # Exact normal force required to guarantee Fy >= Fy_req:
        FN_exact = Fy_req / eta
        
        # Incipient slip feedback: if vy drops below tolerance, apply predictive boost
        if vy < -self.v_tol:
            FN_exact += 1000.0 * (-vy - self.v_tol)
            
        return np.clip(FN_exact, 2.0, 45.0)


# -----------------------------------------------------------------------------
# 3. Simulation Environment: In-Hand Pivoting Dynamics
# -----------------------------------------------------------------------------
def simulate_pivoting(patch: ContactPatch, controller, controller_name, 
                      theta_init=np.pi/3.0, T_total=2.0, dt=0.002, 
                      mu=0.4, m=0.3, L=0.08, J=0.003):
    """
    Simulates in-hand pivoting of an object with:
    - Mass m = 0.3 kg (mg = 2.943 N)
    - Length L = 0.08 m (CoM offset)
    - Inertia J = 0.003 kg*m^2
    - Initial angle theta_init = 60 deg (0 = hanging vertical)
    - Total time T_total = 2.0 s (1000 steps at dt = 2 ms)
    """
    g = 9.81
    n_steps = int(T_total / dt)
    
    # State: [x, y, vx, vy, theta, omega]
    x, y = 0.0, 0.0
    vx, vy = 0.0, 0.0
    theta = theta_init
    omega = 0.0
    
    # History logs
    history = {
        "time": [],
        "theta": [],
        "y": [],
        "FN": [],
        "stick_ratio": [],
        "omega": [],
        "vy": []
    }
    
    if hasattr(controller, "reset"):
        controller.reset()
        
    for step in range(n_steps):
        t = step * dt
        
        # Controller decides normal force FN
        if controller_name == "Tac-LS":
            FN = controller.compute_normal_force(theta, omega, vy, m, g, L)
        elif controller_name == "Classical Circular ELS":
            FN = controller.compute_normal_force(theta, omega, vy, m, g, L)
        elif controller_name == "Reactive PID":
            FN = controller.compute_normal_force(vy, dt)
        elif controller_name == "Constant Force":
            FN = controller.compute_normal_force()
        else:
            raise ValueError(controller_name)
            
        # Physics simulation: Contact patch computes friction wrench from two fingers
        v_rel = np.array([vx, vy])
        F_finger, Mz_finger, stick_ratio = patch.compute_traction_wrench(v_rel, omega, FN, mu)
        
        # Two symmetric fingers:
        F_friction = 2.0 * F_finger
        Mz_friction = 2.0 * Mz_finger
        
        # Gravitational wrench on object:
        # Gravitational force downwards: [0, -mg]
        # Gravitational torque about grasp center: tau_g = -m * g * L * sin(theta)
        tau_g = -m * g * L * np.sin(theta)
        
        # Accelerations:
        ax = F_friction[0] / m
        ay = (F_friction[1] - m * g) / m
        alpha = (Mz_friction + tau_g) / J
        
        # Numerical integration (Semi-implicit Euler):
        vx += ax * dt
        vy += ay * dt
        omega += alpha * dt
        
        # Velocity damping from ambient / internal gel viscoelasticity:
        vx *= 0.995
        vy *= 0.995
        omega *= 0.998
        
        x += vx * dt
        y += vy * dt
        theta += omega * dt
        
        # Store logs
        history["time"].append(t)
        history["theta"].append(theta)
        history["y"].append(y)
        history["FN"].append(FN)
        history["stick_ratio"].append(stick_ratio)
        history["omega"].append(omega)
        history["vy"].append(vy)
        
        # Early termination if object completely drops out (> 15 mm drop)
        if y < -0.015:
            # Dropped! Fill remaining steps
            rem = n_steps - step - 1
            for r in range(rem):
                history["time"].append((step + 1 + r) * dt)
                history["theta"].append(theta)
                history["y"].append(y)
                history["FN"].append(FN)
                history["stick_ratio"].append(0.0)
                history["omega"].append(0.0)
                history["vy"].append(vy)
            break
            
    for k in history:
        history[k] = np.array(history[k])
        
    return history


# -----------------------------------------------------------------------------
# 4. Comprehensive Benchmark Suite across Contact Geometries
# -----------------------------------------------------------------------------
def run_benchmark():
    patch_types = [
        ("circle", "Circular Contact (Symmetric)"),
        ("elongated_h", "Horizontal Elongated (Wrench / Bar)"),
        ("elongated_v", "Vertical Elongated (Pen / Peg)"),
        ("asymmetric_crescent", "Asymmetric Crescent (Curved Handle)")
    ]
    
    controllers = [
        "Classical Circular ELS",
        "Reactive PID",
        "Constant Force",
        "Tac-LS"
    ]
    
    results = {}
    
    print("=" * 78)
    print("EXPERIMENT 2: TACTILE LIMIT SURFACE (Tac-LS) IN-HAND PIVOTING BENCHMARK")
    print("=" * 78)
    print(f"{'Patch Geometry':<25} | {'Controller':<22} | {'Drop (mm)':<10} | {'Final Angle (deg)':<18} | {'Status':<10}")
    print("-" * 78)
    
    for p_key, p_name in patch_types:
        patch = ContactPatch(patch_type=p_key)
        results[p_key] = {}
        
        for c_name in controllers:
            if c_name == "Classical Circular ELS":
                ctrl = ClassicalCircularELSController(R_circ_nominal=np.sqrt(patch.contact_area / np.pi))
            elif c_name == "Reactive PID":
                ctrl = ReactivePIDReflexController()
            elif c_name == "Constant Force":
                ctrl = ConstantForceController(FN_const=11.0)
            elif c_name == "Tac-LS":
                ctrl = TacLSController(patch=patch)
            else:
                raise ValueError(c_name)
                
            hist = simulate_pivoting(patch, ctrl, c_name)
            results[p_key][c_name] = hist
            
            final_drop_mm = -hist["y"][-1] * 1000.0
            final_theta_deg = np.rad2deg(hist["theta"][-1])
            
            # Status classification:
            # Target is theta = 0 deg. Started at 60 deg.
            if final_drop_mm > 10.0:
                status = "DROPPED"
            elif np.abs(final_theta_deg) > 10.0:
                status = "JAMMED"
            else:
                status = "SUCCESS"
                
            print(f"{p_name[:25]:<25} | {c_name:<22} | {final_drop_mm:>8.2f} mm | {final_theta_deg:>12.1f} deg | {status:<10}")
            
    print("=" * 78)
    return results


if __name__ == "__main__":
    t0 = time.time()
    res = run_benchmark()
    print(f"Benchmark finished in {time.time() - t0:.2f} seconds.")
