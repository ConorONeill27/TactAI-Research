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
from scripts.run_phase3_validation import run_single_episode
from src.controllers.insertion_protocol import tactai


class TestTactileAlignmentController(unittest.TestCase):
    def setUp(self):
        self.env = FrankaSlotEnv(seed=42)
        self.controller = tactai(self.env.dt)

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
        result = run_single_episode(self.env, self.controller, (10.0, 0.0), (0.0, 0.0))
        self.assertTrue(result["is_success"])
        self.assertGreaterEqual(result["final_depth_mm"], 20.0)
        self.assertLessEqual(result["peak_force_n"], 15.0)
        self.assertLess(result["final_tilt_deg"], 1.0)

    def test_combined_tilt_and_lateral_offset_insertion(self):
        """Verify robust insertion under combined tilt (-10 deg) and lateral offset (0.4 mm)."""
        result = run_single_episode(self.env, self.controller, (0.0, -10.0), (0.4, 0.0))
        self.assertTrue(result["is_success"])
        self.assertLessEqual(result["peak_force_n"], 15.0)


if __name__ == "__main__":
    unittest.main()
