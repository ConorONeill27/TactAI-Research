"""
Unit tests for FrankaSlotEnv and EpisodeLogger.
Verifies simulation determinism, OOD misalignment handling, tactile perception integration,
and structured logging reproducibility.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np

from src.envs.franka_slot_env import FrankaSlotEnv
from src.utils.logger import EpisodeLogger
from src.invariants.tactile_invariants import ContactMode


class TestFrankaSlotEnv(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.env = FrankaSlotEnv(seed=42)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reset_in_free_space(self):
        """Verify that reset places peg above slot in FREE_SPACE with zero external force."""
        obs, info = self.env.reset()
        self.assertEqual(info["step"], 0)
        self.assertAlmostEqual(info["insertion_depth_m"], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(obs["external_force"])), 0.0, places=4)
        self.assertEqual(obs["tactile"]["contact_mode"], ContactMode.FREE_SPACE)

    def test_deterministic_reproducibility(self):
        """Verify that identical seeds and action sequences produce identical trajectories."""
        actions = [np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0]) for _ in range(20)]
        
        # Run 1
        obs1, _ = self.env.reset(seed=123)
        pos_traj_1 = []
        for a in actions:
            obs1, _, _, _, _ = self.env.step(a)
            pos_traj_1.append(obs1["ee_pos"].copy())

        # Run 2
        obs2, _ = self.env.reset(seed=123)
        pos_traj_2 = []
        for a in actions:
            obs2, _, _, _, _ = self.env.step(a)
            pos_traj_2.append(obs2["ee_pos"].copy())

        np.testing.assert_allclose(pos_traj_1, pos_traj_2, rtol=1e-7, atol=1e-7)

    def test_ood_misalignment_injection(self):
        """Verify that OOD tilt and lateral offset are applied correctly to initial state."""
        target_tilt_deg = (12.0, -8.0)
        target_offset_mm = (0.75, -0.50)
        
        obs, info = self.env.reset(
            ood_tilt_deg=target_tilt_deg,
            ood_offset_mm=target_offset_mm
        )
        
        # Check initial lateral offset (mm -> m)
        self.assertAlmostEqual(obs["ee_pos"][0], 0.75 * 1e-3, places=5)
        self.assertAlmostEqual(obs["ee_pos"][1], -0.50 * 1e-3, places=5)

        # Check initial tilt angles (deg -> rad)
        self.assertAlmostEqual(obs["ee_rpy"][0], np.deg2rad(12.0), places=5)
        self.assertAlmostEqual(obs["ee_rpy"][1], np.deg2rad(-8.0), places=5)

    def test_aligned_insertion_success(self):
        """Verify that aligned insertion under downward push successfully enters slot."""
        obs, info = self.env.reset(ood_tilt_deg=(0.0, 0.0), ood_offset_mm=(0.0, 0.0))
        down_wrench = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])
        
        succeeded = False
        for _ in range(200):
            obs, r, term, trunc, info = self.env.step(down_wrench)
            if info["is_success"]:
                succeeded = True
                break
        
        self.assertTrue(succeeded, "Aligned peg should successfully insert into slot.")
        self.assertGreaterEqual(info["insertion_depth_m"], 0.020)
        self.assertLess(info["peak_force_n"], 20.0)

    def test_tilted_insertion_wedging_failure(self):
        """Verify that unconstrained push with 10 deg tilt causes jamming / high force."""
        obs, info = self.env.reset(ood_tilt_deg=(10.0, 0.0), ood_offset_mm=(0.0, 0.0))
        down_wrench = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])
        
        jammed = False
        for _ in range(100):
            obs, r, term, trunc, info = self.env.step(down_wrench)
            if info["is_jammed"] or term:
                jammed = info["is_jammed"]
                break

        self.assertTrue(jammed, "Tilted insertion must trigger jamming / force threshold.")
        self.assertGreater(info["peak_force_n"], 40.0)

    def test_episode_logger_io(self):
        """Verify that EpisodeLogger writes valid .npz and .json files with correct contents."""
        logger = EpisodeLogger(log_dir=self.temp_dir)
        logger.start_episode(
            episode_id="test_001",
            controller_name="test_controller",
            seed=42,
            env_params={"slot_clearance": 0.00015}
        )
        
        obs, info = self.env.reset()
        action = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])
        
        for _ in range(10):
            obs, reward, term, trunc, info = self.env.step(action)
            logger.log_step(obs, action, reward, info)

        summary = logger.finish_episode(info, save_prefix="test_output")
        
        npz_file = os.path.join(self.temp_dir, "test_output.npz")
        json_file = os.path.join(self.temp_dir, "test_output_summary.json")

        self.assertTrue(os.path.exists(npz_file), "NPZ file must exist.")
        self.assertTrue(os.path.exists(json_file), "JSON summary must exist.")

        # Test loading .npz
        loaded_npz = np.load(npz_file)
        self.assertEqual(len(loaded_npz["steps"]), 10)
        self.assertEqual(loaded_npz["actions"].shape, (10, 6))

        # Test summary dictionary
        self.assertEqual(summary["metadata"]["episode_id"], "test_001")
        self.assertEqual(summary["kpis"]["total_steps"], 10)


if __name__ == "__main__":
    unittest.main()
