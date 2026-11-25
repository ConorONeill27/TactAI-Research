"""Observer and simulator interface checks for A2."""

import unittest

import numpy as np

from src.controllers.insertion_protocol import tactai
from src.envs.franka_slot_env import FrankaSlotEnv
from src.envs.unknown_grasp_env import UnknownGraspEnv


class TestUnknownGrasp(unittest.TestCase):
    def test_shared_cop_calibration_across_taxel_boundaries(self):
        env = FrankaSlotEnv()
        for tilt in ((12, 0), (13, 0), (15, 0), (10, -6), (-14, 12)):
            with self.subTest(tilt=tilt):
                env.reset(ood_tilt_deg=tilt)
                pads = env._compute_tactile_invariants(
                    np.zeros(6), np.zeros(6), np.array([0, 0, -2.]), np.zeros(3)
                )
                cop = (pads["left_pad"]["center_of_pressure"] -
                       pads["right_pad"]["center_of_pressure"]) / 2
                np.testing.assert_allclose(
                    np.rad2deg(np.arcsin(cop / .006)), np.rad2deg(env.peg_tilt_rad()), atol=.1
                )

    def test_unit_gain_residual_and_unloaded_hold(self):
        controller = tactai(unknown_grasp_mode=True)
        cop = np.array([.006 * np.sin(np.deg2rad(10)), 0])
        obs = {
            "ee_rpy": np.deg2rad([3., 0., 0.]),
            "ee_pos": np.array([0., 0., -.004]),
            "external_force": np.array([0., 0., -2.]),
            "tactile": {"left_pad": {"center_of_pressure": cop},
                        "right_pad": {"center_of_pressure": -cop}},
        }
        for step in range(32):
            obs["step"] = step
            _, diag = controller.compute_safe_action(np.zeros(6), obs)
        self.assertEqual(diag["stage"], 2)
        np.testing.assert_allclose(np.rad2deg(diag["grasp_residual_rad"]), [7, 0], atol=.001)
        residual = diag["grasp_residual_rad"]
        obs["external_force"] = np.zeros(3)
        obs["ee_pos"] = np.zeros(3)
        obs["ee_rpy"] = np.deg2rad([-7., 0., 0.])
        for step in range(32, 90):
            obs["step"] = step
            _, diag = controller.compute_safe_action(np.zeros(6), obs)
        self.assertEqual(diag["stage"], 3)
        self.assertEqual(diag["grasp_residual_rad"], residual)
        self.assertLess(diag["tilt_deg"], .001)

    def test_grasp_rest_pose_without_privileged_observations(self):
        env = UnknownGraspEnv()
        obs, _ = env.reset(grasp_tilt_deg=(7, 0), ood_tilt_deg=(3, 0))
        for key in ("grasp_tilt_deg", "peg_deflection", "peg_pos"):
            self.assertNotIn(key, obs)
        self.assertAlmostEqual(np.rad2deg(obs["ee_rpy"][0]), 3)
        self.assertAlmostEqual(np.rad2deg(env.peg_tilt_rad()[0]), 10, places=2)
        index = env.model.joint("peg_drx").qposadr[0]
        self.assertAlmostEqual(np.rad2deg(env.model.qpos_spring[index]), 7)
        env.reset(grasp_tilt_deg=(-5, 0), ood_tilt_deg=(0, 0))
        self.assertAlmostEqual(np.rad2deg(env.model.qpos_spring[index]), -5)
