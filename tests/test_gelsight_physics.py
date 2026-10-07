"""
Unit tests for GelSightSensor physics model.
Verifies SI units, Coulomb friction bounds, indentation scaling, and contact geometry profiles.
"""

import unittest
import numpy as np
from src.sensors.gelsight_sensor import GelSightSensor


class TestGelSightPhysics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.sensor = GelSightSensor(grid_h=32, grid_w=32, noise_std_pa=0.0)

    def test_dimensions_and_mesh(self):
        """Verify grid dimensions and spatial coordinates."""
        self.assertEqual(self.sensor.U.shape, (32, 32))
        self.assertEqual(self.sensor.V.shape, (32, 32))
        self.assertAlmostEqual(self.sensor.U[0, -1] - self.sensor.U[0, 0], self.sensor.lx, places=5)
        self.assertAlmostEqual(self.sensor.V[-1, 0] - self.sensor.V[0, 0], self.sensor.ly, places=5)

    def test_normal_indentation_scaling(self):
        """Verify that deeper indentation monotonically increases normal force Fz."""
        d1 = 0.0002  # 0.2 mm
        d2 = 0.0006  # 0.6 mm
        res1 = self.sensor.generate_traction_field(contact_type="flat_circle", indentation_depth=d1)
        res2 = self.sensor.generate_traction_field(contact_type="flat_circle", indentation_depth=d2)

        Fz1 = res1["net_force"][2]
        Fz2 = res2["net_force"][2]

        self.assertGreater(Fz1, 0.0)
        self.assertGreater(Fz2, Fz1, "Deeper indentation must produce higher normal force Fz.")
        # Pressure must be strictly non-negative
        self.assertTrue(np.all(res1["normal_pressure"] >= 0.0))

    def test_coulomb_friction_saturation(self):
        """Verify that total shear force obeys Coulomb limit: ||F_xy|| <= mu * Fz."""
        mu = self.sensor.mu
        # Strong translational slip: vx = 0.05 m/s, vy = 0.02 m/s
        res = self.sensor.generate_traction_field(
            contact_type="flat_circle",
            indentation_depth=0.0005,
            rel_vel=(0.05, 0.02)
        )
        Fx, Fy, Fz = res["net_force"]
        shear_mag = np.sqrt(Fx**2 + Fy**2)
        max_allowable_shear = mu * Fz * 1.01  # 1% numerical tolerance

        self.assertLessEqual(
            shear_mag, max_allowable_shear,
            f"Shear force {shear_mag:.4f} N exceeds Coulomb limit {max_allowable_shear:.4f} N."
        )

    def test_contact_geometries(self):
        """Verify that flat_circle, line_edge, and two_point contacts generate active contact masks."""
        for c_type in ["flat_circle", "line_edge", "two_point"]:
            res = self.sensor.generate_traction_field(contact_type=c_type, indentation_depth=0.0004)
            n_active = np.sum(res["contact_mask"])
            self.assertGreater(n_active, 10, f"Contact type {c_type} must activate contact grid cells.")
            self.assertGreater(res["net_force"][2], 0.1, f"Contact type {c_type} must generate normal force.")


if __name__ == "__main__":
    unittest.main()
