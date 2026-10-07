"""
GelSight Mini Tactile Sensor Model
Simulates the 3D continuous traction field f(u, v) = [f_x, f_y, f_z]^T (in Pa / N/m^2)
over an elastomeric pad according to Cattaneo-Mindlin elastomeric contact mechanics.

Matches the FEATS (NeurIPS 2024 / IROS 2025; Helmut et al.) tactile representation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class GelSightSensor:
    """
    Physical simulation of a GelSight Mini optical tactile sensor pad.
    
    Pad dimensions:
        Lx = 18.6 mm (0.0186 m)
        Ly = 15.4 mm (0.0154 m)
        Gel thickness = 2.0 mm (0.002 m)
        Elastomer Young's Modulus E = 150 kPa (soft silicone gel)
        Poisson's ratio nu = 0.48 (nearly incompressible elastomer)
    """

    def __init__(
        self,
        grid_h: int = 32,
        grid_w: int = 32,
        lx: float = 0.0186,
        ly: float = 0.0154,
        youngs_modulus: float = 1.5e5,
        friction_coeff: float = 0.40,
        noise_std_pa: float = 50.0
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.lx = lx
        self.ly = ly
        self.E = youngs_modulus
        self.mu = friction_coeff
        self.noise_std = noise_std_pa

        # Spatial coordinates in sensor frame: origin at pad center
        self.xs = np.linspace(-self.lx / 2.0, self.lx / 2.0, self.grid_w)
        self.ys = np.linspace(-self.ly / 2.0, self.ly / 2.0, self.grid_h)
        self.U, self.V = np.meshgrid(self.xs, self.ys)

        self.du = self.xs[1] - self.xs[0]
        self.dv = self.ys[1] - self.ys[0]
        self.dA = self.du * self.dv  # elemental area in m^2
        self.total_area = self.lx * self.ly

    def generate_traction_field(
        self,
        contact_type: str = "flat_circle",
        indentation_depth: float = 0.0005,  # 0.5 mm
        offset: Tuple[float, float] = (0.0, 0.0),
        orientation_rad: float = 0.0,
        rel_vel: Tuple[float, float] = (0.0, 0.0),  # (vx, vy) in m/s
        rel_omega: float = 0.0,  # rotational slip in rad/s
        aspect_ratio: float = 1.0,
        major_axis: float = 0.004,  # 4 mm nominal contact radius
        eps_v: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Computes the continuous 3D traction field across the sensor grid.
        
        Args:
            contact_type: 'flat_circle', 'elliptical', 'line_edge', 'two_point'
            indentation_depth: delta in meters
            offset: (u0, v0) contact center in meters
            orientation_rad: angle of principal axis in radians
            rel_vel: translational slip velocity (vx, vy) in m/s
            rel_omega: torsional slip velocity around normal in rad/s
            aspect_ratio: major / minor axis ratio
            major_axis: semi-major contact length in meters
            eps_v: regularization velocity for micro-adhesion / Stribeck
            
        Returns:
            Dictionary containing:
                'traction_field': (H, W, 3) ndarray of [fx, fy, fz] in Pascals
                'normal_pressure': (H, W) ndarray of fz in Pascals
                'shear_traction': (H, W, 2) ndarray of [fx, fy] in Pascals
                'net_force': (3,) ndarray [Fx, Fy, Fz] in Newtons
                'net_moment': (3,) ndarray [Mx, My, Mz] in N*m
                'contact_mask': (H, W) bool array indicating active contact
        """
        u0, v0 = offset
        # Coordinate transformation to contact frame
        u_rel = self.U - u0
        v_rel = self.V - v0
        cos_th = np.cos(orientation_rad)
        sin_th = np.sin(orientation_rad)
        u_rot = cos_th * u_rel + sin_th * v_rel
        v_rot = -sin_th * u_rel + cos_th * v_rel

        # Compute normal pressure distribution fz(u, v)
        fz = np.zeros_like(self.U)
        a = max(1e-4, major_axis)
        b = a / max(1.0, aspect_ratio)

        if contact_type in ["flat_circle", "elliptical"]:
            norm_rad_sq = (u_rot / a)**2 + (v_rot / b)**2
            mask = norm_rad_sq <= 1.0
            # Hertzian / parabolic elastomeric pressure profile
            # Peak pressure scales with indentation depth E * delta / a
            p_max = (1.5 * self.E * indentation_depth) / max(1e-4, a)
            fz[mask] = p_max * (1.0 - norm_rad_sq[mask])

        elif contact_type == "line_edge":
            # Edge / chamfer contact: narrow rectangular contact
            mask = (np.abs(u_rot) <= a) & (np.abs(v_rot) <= b)
            p_max = (2.0 * self.E * indentation_depth) / max(1e-4, b)
            if np.any(mask):
                fz[mask] = p_max * (1.0 - (v_rot[mask] / b)**2)

        elif contact_type == "two_point":
            # Two-point contact couple (characteristic of jamming)
            d_sep = a  # separation distance
            r1_sq = ((u_rot - d_sep / 2.0) / b)**2 + (v_rot / b)**2
            r2_sq = ((u_rot + d_sep / 2.0) / b)**2 + (v_rot / b)**2
            mask1 = r1_sq <= 1.0
            mask2 = r2_sq <= 1.0
            p_max = (1.5 * self.E * indentation_depth) / max(1e-4, b)
            fz[mask1] += p_max * (1.0 - r1_sq[mask1])
            fz[mask2] += p_max * 0.8 * (1.0 - r2_sq[mask2])
            mask = mask1 | mask2
        else:
            raise ValueError(f"Unknown contact_type: {contact_type}")

        # Local relative velocity at each point (U, V):
        # v_loc = v_rel + omega x (r - r0)
        vx_loc = rel_vel[0] - rel_omega * (self.V - v0)
        vy_loc = rel_vel[1] + rel_omega * (self.U - u0)
        speed_loc = np.sqrt(vx_loc**2 + vy_loc**2 + eps_v**2)

        # Local shear traction q = -mu * fz * (v / ||v||)
        # Coulomb friction limit applied locally
        qx = -self.mu * fz * (vx_loc / speed_loc)
        qy = -self.mu * fz * (vy_loc / speed_loc)

        # Add physical sensor noise (Gaussian in Pascals)
        if self.noise_std > 0:
            noise_z = np.random.normal(0, self.noise_std, fz.shape)
            noise_x = np.random.normal(0, self.noise_std, qx.shape)
            noise_y = np.random.normal(0, self.noise_std, qy.shape)
            # Only add noise to active or near-contact regions
            active_region = fz > 0
            fz[active_region] = np.maximum(0.0, fz[active_region] + noise_z[active_region])
            qx[active_region] = qx[active_region] + noise_x[active_region]
            qy[active_region] = qy[active_region] + noise_y[active_region]

        # Stack into 3D traction tensor (H, W, 3) in Pascals
        traction_field = np.stack([qx, qy, fz], axis=-1)

        # Integrate total net forces and moments in Newtons and N*m
        Fx = float(np.sum(qx) * self.dA)
        Fy = float(np.sum(qy) * self.dA)
        Fz = float(np.sum(fz) * self.dA)
        net_force = np.array([Fx, Fy, Fz])

        # Net moments about pad center (0, 0, 0):
        # Mx = integral (v * fz) dA
        # My = integral (-u * fz) dA
        # Mz = integral (u * qy - v * qx) dA
        Mx = float(np.sum(self.V * fz) * self.dA)
        My = float(np.sum(-self.U * fz) * self.dA)
        Mz = float(np.sum(self.U * qy - self.V * qx) * self.dA)
        net_moment = np.array([Mx, My, Mz])

        return {
            "traction_field": traction_field,
            "normal_pressure": fz,
            "shear_traction": np.stack([qx, qy], axis=-1),
            "net_force": net_force,
            "net_moment": net_moment,
            "contact_mask": fz > 0
        }
