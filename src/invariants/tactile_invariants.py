"""
Tactile Invariant Extraction Engine
Extracts mathematically invariant geometric properties from 3D tactile traction fields:
- Total Force F_c and Unit Normal n_c
- Traction Center of Pressure (CoP) r_c
- Contact Constraint Tensor C in R^{2x2} (eigenvalues lambda_1, lambda_2, orientation theta)
- Incipient Stick Ratio rho_stick in [0, 1]
- Jamming Invariant Index J_jam
- Contact Mode Categorization

Directly implements Equations (1)-(3) of the Research Paper.
"""

from enum import Enum
from typing import Dict, Any, Tuple
import numpy as np


class ContactMode(Enum):
    FREE_SPACE = 0        # No contact (Fz < eps)
    POINT_CONTACT = 1     # Concentrated circular/elliptic contact (lambda1 ~ lambda2)
    LINE_CONTACT = 2      # Edge / chamfer contact (lambda1 >> lambda2)
    TWO_POINT_JAMMING = 3 # Two separated contact points creating a force couple
    GROSS_SLIP = 4        # Macro-slip (rho_stick < 0.1)


class TactileInvariantEngine:
    """
    Real-time extractor of contact-manifold invariants from continuous traction fields.
    """

    def __init__(
        self,
        lx: float = 0.0186,
        ly: float = 0.0154,
        force_threshold_n: float = 0.05,
        stick_threshold: float = 0.20,
        aspect_ratio_line_threshold: float = 2.5
    ):
        self.lx = lx
        self.ly = ly
        self.f_thresh = force_threshold_n
        self.stick_thresh = stick_threshold
        self.line_thresh = aspect_ratio_line_threshold

    def extract_invariants(
        self,
        traction_field: np.ndarray,  # (H, W, 3) [fx, fy, fz] in Pascals
        du: float,
        dv: float,
        U: np.ndarray,  # (H, W) spatial x-coordinates in meters
        V: np.ndarray,  # (H, W) spatial y-coordinates in meters
        mu_nominal: float = 0.40,
        eps: float = 1e-8
    ) -> Dict[str, Any]:
        """
        Extracts all geometric contact invariants from the traction field.
        
        Args:
            traction_field: (H, W, 3) ndarray of [fx, fy, fz] in Pa
            du, dv: grid cell dimensions in meters
            U, V: spatial coordinate grids
            mu_nominal: nominal friction coefficient
            eps: numerical stabilizer
            
        Returns:
            Dictionary containing:
                'net_force': (3,) array [Fx, Fy, Fz] in Newtons
                'contact_normal': (3,) unit vector n_c
                'center_of_pressure': (2,) array [uc, vc] in meters
                'constraint_tensor': (2, 2) positive semi-definite matrix C
                'eigenvalues': (2,) array [lambda_1, lambda_2] sorted descending
                'principal_axis': (2,) unit vector of major contact axis
                'orientation_rad': angle of major axis in radians
                'stick_ratio': scalar rho_stick in [0.0, 1.0]
                'jamming_index': scalar J_jam
                'contact_mode': ContactMode enum
                'in_contact': bool
        """
        dA = du * dv
        qx = traction_field[..., 0]
        qy = traction_field[..., 1]
        fz = np.maximum(0.0, traction_field[..., 2])  # Non-negative normal pressure

        # 1. Total Contact Force & Unit Normal
        Fx = float(np.sum(qx) * dA)
        Fy = float(np.sum(qy) * dA)
        Fz = float(np.sum(fz) * dA)
        net_force = np.array([Fx, Fy, Fz])

        if Fz < self.f_thresh:
            # Free space
            return {
                "in_contact": False,
                "net_force": net_force,
                "contact_normal": np.array([0.0, 0.0, 1.0]),
                "center_of_pressure": np.array([0.0, 0.0]),
                "constraint_tensor": np.zeros((2, 2)),
                "eigenvalues": np.zeros(2),
                "principal_axis": np.array([1.0, 0.0]),
                "orientation_rad": 0.0,
                "stick_ratio": 1.0,
                "jamming_index": 100.0,
                "contact_mode": ContactMode.FREE_SPACE
            }

        contact_normal = net_force / max(eps, np.linalg.norm(net_force))

        # 2. Traction Center of Pressure (CoP) r_c = [uc, vc]
        uc = float(np.sum(U * fz) * dA / (Fz + eps))
        vc = float(np.sum(V * fz) * dA / (Fz + eps))
        r_c = np.array([uc, vc])

        # 3. Contact Constraint Tensor C in R^{2x2}
        U_rel = U - uc
        V_rel = V - vc

        C_11 = float(np.sum(U_rel**2 * fz) * dA)
        C_22 = float(np.sum(V_rel**2 * fz) * dA)
        C_12 = float(np.sum(U_rel * V_rel * fz) * dA)
        C = np.array([[C_11, C_12], [C_12, C_22]])

        # Strict positive semi-definiteness assertion
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        # Sort descending: lambda_1 >= lambda_2
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(0.0, eigvals[idx])
        eigvecs = eigvecs[:, idx]

        principal_axis = eigvecs[:, 0]
        orientation_rad = float(np.arctan2(principal_axis[1], principal_axis[0]))

        # 4. Incipient Slip Ratio rho_stick in [0, 1]
        shear_mag = np.sqrt(qx**2 + qy**2)
        local_coulomb_limit = mu_nominal * fz
        
        # Local adhesion margin
        stick_mask = (fz > 0.01 * np.max(fz)) & (shear_mag < (local_coulomb_limit - self.stick_thresh * local_coulomb_limit))
        active_contact_mask = fz > 0.01 * np.max(fz)
        
        n_active = np.sum(active_contact_mask)
        if n_active > 0:
            rho_stick = float(np.sum(stick_mask) / n_active)
        else:
            rho_stick = 1.0
        rho_stick = np.clip(rho_stick, 0.0, 1.0)

        # 5. Net Moments & Jamming Index
        # Tilting torque about sensor pad center:
        # tau_tilt = integral r x f_z
        Mx_tilt = float(np.sum(V * fz) * dA)
        My_tilt = float(np.sum(-U * fz) * dA)
        tau_tilt_mag = np.sqrt(Mx_tilt**2 + My_tilt**2)
        
        # Dimensionless Jamming Index:
        # J_jam = (F_insert * r_char) / (tau_tilt + eps)
        # Higher values (> 5.0) indicate safe centered contact; low values (< 2.5) indicate high tilting couple (wedging risk)
        r_char = min(self.lx, self.ly) / 2.0
        jamming_index = float((Fz * r_char) / (tau_tilt_mag + eps * 1e-4))

        # 6. Contact Mode Categorization
        aspect = float(eigvals[0] / max(1e-9, eigvals[1]))
        if rho_stick < 0.10:
            contact_mode = ContactMode.GROSS_SLIP
        elif jamming_index < 2.5 and aspect > 1.5:
            contact_mode = ContactMode.TWO_POINT_JAMMING
        elif aspect >= self.line_thresh:
            contact_mode = ContactMode.LINE_CONTACT
        else:
            contact_mode = ContactMode.POINT_CONTACT

        return {
            "in_contact": True,
            "net_force": net_force,
            "contact_normal": contact_normal,
            "center_of_pressure": r_c,
            "constraint_tensor": C,
            "eigenvalues": eigvals,
            "principal_axis": principal_axis,
            "orientation_rad": orientation_rad,
            "stick_ratio": rho_stick,
            "jamming_index": jamming_index,
            "contact_mode": contact_mode
        }
