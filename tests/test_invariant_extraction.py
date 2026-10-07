"""
Unit tests for TactileInvariantEngine.
Verifies mathematical properties of Center of Pressure, Contact Constraint Tensor (positive semi-definiteness),
Incipient Stick Ratio bounds, and Jamming Invariant classification.
"""

import unittest
import numpy as np
from src.sensors.gelsight_sensor import GelSightSensor
from src.invariants.tactile_invariants import TactileInvariantEngine, ContactMode


class TestInvariantExtraction(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.sensor = GelSightSensor(grid_h=32, grid_w=32, noise_std_pa=0.0)
        self.engine = TactileInvariantEngine(lx=self.sensor.lx, ly=self.sensor.ly)

    def test_free_space_invariants(self):
        """Verify that zero contact yields FREE_SPACE mode and safe defaults."""
        zero_traction = np.zeros((32, 32, 3))
        inv = self.engine.extract_invariants(
            zero_traction, self.sensor.du, self.sensor.dv, self.sensor.U, self.sensor.V
        )
        self.assertFalse(inv["in_contact"])
        self.assertEqual(inv["contact_mode"], ContactMode.FREE_SPACE)
        self.assertEqual(inv["stick_ratio"], 1.0)
        self.assertEqual(float(np.linalg.norm(inv["center_of_pressure"])), 0.0)

    def test_tensor_positive_semi_definiteness(self):
        """Verify that Contact Constraint Tensor C is positive semi-definite (lambda1 >= lambda2 >= 0)."""
        res = self.sensor.generate_traction_field(
            contact_type="flat_circle",
            indentation_depth=0.0005,
            offset=(0.002, -0.001)
        )
        inv = self.engine.extract_invariants(
            res["traction_field"], self.sensor.du, self.sensor.dv, self.sensor.U, self.sensor.V
        )
        self.assertTrue(inv["in_contact"])
        C = inv["constraint_tensor"]
        eigvals = inv["eigenvalues"]

        # Symmetry
        np.testing.assert_allclose(C, C.T, rtol=1e-5, atol=1e-7)
        # Positive eigenvalues
        self.assertGreaterEqual(eigvals[1], 0.0, "Eigenvalues of C must be non-negative.")
        self.assertGreaterEqual(eigvals[0], eigvals[1], "Eigenvalues must be sorted descending.")
        # Trace and determinant
        self.assertGreaterEqual(np.trace(C), 0.0)
        self.assertGreaterEqual(np.linalg.det(C), -1e-9)

    def test_center_of_pressure_tracking(self):
        """Verify that Center of Pressure r_c tracks contact offset accurately."""
        target_offset = (0.003, -0.002)
        res = self.sensor.generate_traction_field(
            contact_type="flat_circle",
            indentation_depth=0.0004,
            offset=target_offset
        )
        inv = self.engine.extract_invariants(
            res["traction_field"], self.sensor.du, self.sensor.dv, self.sensor.U, self.sensor.V
        )
        uc, vc = inv["center_of_pressure"]
        # Error must be less than 0.5 grid cell (~0.25 mm)
        self.assertAlmostEqual(uc, target_offset[0], delta=self.sensor.du)
        self.assertAlmostEqual(vc, target_offset[1], delta=self.sensor.dv)

    def test_line_contact_eigenvalue_anisotropy(self):
        """Verify that line/edge contact creates strong eigenvalue anisotropy (lambda1 >> lambda2)."""
        res = self.sensor.generate_traction_field(
            contact_type="line_edge",
            indentation_depth=0.0004,
            aspect_ratio=4.0,
            orientation_rad=np.deg2rad(45.0)
        )
        inv = self.engine.extract_invariants(
            res["traction_field"], self.sensor.du, self.sensor.dv, self.sensor.U, self.sensor.V
        )
        self.assertTrue(inv["in_contact"])
        l1, l2 = inv["eigenvalues"]
        self.assertGreater(l1 / max(1e-9, l2), 2.5, "Line contact must produce eigenvalue ratio > 2.5.")
        self.assertEqual(inv["contact_mode"], ContactMode.LINE_CONTACT)
        # Orientation angle should be close to 45 deg or -135 deg (modulo pi)
        angle_diff = np.abs((inv["orientation_rad"] - np.deg2rad(45.0)) % np.pi)
        self.assertLess(min(angle_diff, np.pi - angle_diff), np.deg2rad(10.0))

    def test_stick_ratio_bounds(self):
        """Verify stick ratio bounds: near 1.0 under static adhesion, near 0.0 under high slip."""
        # Static contact (zero slip velocity)
        res_static = self.sensor.generate_traction_field(
            contact_type="flat_circle",
            indentation_depth=0.0005,
            rel_vel=(0.0, 0.0)
        )
        inv_static = self.engine.extract_invariants(
            res_static["traction_field"], self.sensor.du, self.sensor.dv, self.sensor.U, self.sensor.V
        )
        self.assertGreater(inv_static["stick_ratio"], 0.85, "Zero slip must yield high stick ratio.")

        # High slip velocity (vx = 0.1 m/s)
        res_slip = self.sensor.generate_traction_field(
            contact_type="flat_circle",
            indentation_depth=0.0005,
            rel_vel=(0.10, 0.0)
        )
        inv_slip = self.engine.extract_invariants(
            res_slip["traction_field"], self.sensor.du, self.sensor.dv, self.sensor.U, self.sensor.V
        )
        self.assertLess(inv_slip["stick_ratio"], 0.15, "High slip must yield low stick ratio.")
        self.assertEqual(inv_slip["contact_mode"], ContactMode.GROSS_SLIP)

    def test_two_point_jamming_detection(self):
        """Verify that asymmetric two-point contact couple produces low jamming index and triggers TWO_POINT_JAMMING mode."""
        res_jam = self.sensor.generate_traction_field(
            contact_type="two_point",
            indentation_depth=0.0006,
            offset=(0.0035, 0.0),
            major_axis=0.007,
            aspect_ratio=2.5
        )
        inv_jam = self.engine.extract_invariants(
            res_jam["traction_field"], self.sensor.du, self.sensor.dv, self.sensor.U, self.sensor.V
        )
        self.assertTrue(inv_jam["in_contact"])
        self.assertLess(inv_jam["jamming_index"], 2.5, "Two-point eccentric couple must yield J_jam < 2.5.")
        self.assertEqual(inv_jam["contact_mode"], ContactMode.TWO_POINT_JAMMING)


if __name__ == "__main__":
    unittest.main()
