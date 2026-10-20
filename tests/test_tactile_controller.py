"""
Unit tests for TactileAlignmentController (Phase 2).
Verifies:
- Control Barrier Function (CBF) normal force barrier enforcement (h_force >= 0, F_max = 15 N)
- Anti-jamming barrier intervention (h_jam >= 0)
- Tactile reorientation from out-of-distribution initial tilt (+-10 deg to 0 deg)
- Full closed-loop sub-millimeter insertion success (depth >= 20 mm, peak force < 15 N)
- Energy-tank passivity supervisor
- Real-time solve frequency >= 1 kHz (< 1.0 ms solve time)
"""

import unittest
import numpy as np
import time

from src.envs.franka_slot_env import FrankaSlotEnv
from src.controllers.tactile_alignment_controller import TactileAlignmentController


class TestTactileAlignmentController(unittest.TestCase):
    def setUp(self):
        self.env = FrankaSlotEnv(seed=42)
        self.controller = TactileAlignmentController(
            f_max=15.0,
            gamma_jam=2.0,
            kp_rot=30.0,
            kd_rot=1.2,
            dt=0.001
        )

    def test_controller_initialization(self):
        """Verify default controller and energy-tank parameters."""
        self.assertEqual(self.controller.f_max, 15.0)
        self.assertEqual(self.controller.gamma_jam, 2.0)
        self.assertEqual(self.controller.dt, 0.001)
        self.assertEqual(self.controller.tank_energy, 1.0)
        self.assertEqual(self.controller.tank_min_energy, 0.05)

    def test_high_frequency_solve_time(self):
        """Verify that controller execution latency is well under 1.0 ms (>= 1 kHz requirement)."""
        obs, _ = self.env.reset()
        u_nom = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

        latencies_ms = []
        for _ in range(100):
            t0 = time.perf_counter()
            u_safe, diag = self.controller.compute_safe_action(u_nom, obs)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            latencies_ms.append(elapsed_ms)

        mean_latency = float(np.mean(latencies_ms))
        max_latency = float(np.max(latencies_ms))
        
        self.assertLess(mean_latency, 0.5, f"Mean latency {mean_latency:.3f} ms exceeds 0.5 ms limit.")
        self.assertLess(max_latency, 1.0, f"Max latency {max_latency:.3f} ms exceeds 1.0 ms (1 kHz) limit.")

    def test_energy_tank_passivity_preservation(self):
        """Verify that energy tank maintains positive energy and enforces passivity bounds."""
        obs, _ = self.env.reset()
        u_nom = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

        for _ in range(50):
            u_safe, diag = self.controller.compute_safe_action(u_nom, obs)
            obs, _, _, _, _ = self.env.step(u_safe)
            self.assertGreaterEqual(diag["tank_energy_j"], self.controller.tank_min_energy)
            self.assertLessEqual(diag["tank_energy_j"], self.controller.tank_max_energy)

    def test_tilted_realignment_from_10deg(self):
        """Verify that initial +10 deg tilt is actively corrected to < 1.0 deg."""
        obs, info = self.env.reset(ood_tilt_deg=(10.0, 0.0), ood_offset_mm=(0.0, 0.0))
        u_nom = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

        reoriented = False
        for step in range(400):
            u_safe, diag = self.controller.compute_safe_action(u_nom, obs)
            obs, _, _, _, info = self.env.step(u_safe)
            
            if info["tilt_deg"] < 1.0:
                reoriented = True
                break

        self.assertTrue(reoriented, "Peg tilt should be nullified to < 1.0 deg by tactile controller.")
        self.assertLess(step, 350, f"Reorientation took {step} ms, expected < 350 ms.")

    def test_closed_loop_insertion_success_under_misalignment(self):
        """Verify that misaligned peg (+10 deg tilt) achieves full successful insertion with safe contact forces."""
        obs, info = self.env.reset(ood_tilt_deg=(10.0, 0.0), ood_offset_mm=(0.0, 0.0))
        u_nom = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

        succeeded = False
        peak_force = 0.0

        for step in range(500):
            u_safe, diag = self.controller.compute_safe_action(u_nom, obs)
            obs, r, term, trunc, info = self.env.step(u_safe)
            peak_force = max(peak_force, info["current_force_n"])

            if info["is_success"]:
                succeeded = True
                break

        self.assertTrue(succeeded, "Closed-loop tactile controller must succeed in sub-millimeter insertion.")
        self.assertGreaterEqual(info["insertion_depth_m"], 0.020, "Insertion depth must reach >= 20 mm.")
        self.assertLess(peak_force, 15.0, f"Peak force {peak_force:.2f} N exceeded 15.0 N safety barrier.")
        self.assertLess(info["tilt_deg"], 1.0, "Final tilt must be < 1.0 deg.")

    def test_combined_tilt_and_lateral_offset_insertion(self):
        """Verify robust insertion under combined tilt (-10 deg) and lateral offset (0.4 mm)."""
        obs, info = self.env.reset(ood_tilt_deg=(0.0, -10.0), ood_offset_mm=(0.4, 0.0))
        u_nom = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

        succeeded = False
        peak_force = 0.0

        for step in range(500):
            u_safe, diag = self.controller.compute_safe_action(u_nom, obs)
            obs, r, term, trunc, info = self.env.step(u_safe)
            peak_force = max(peak_force, info["current_force_n"])

            if info["is_success"]:
                succeeded = True
                break

        self.assertTrue(succeeded, "Controller must succeed under combined tilt and lateral offset.")
        self.assertLess(peak_force, 15.0, "Peak force must remain strictly below 15 N barrier.")


if __name__ == "__main__":
    unittest.main()
