"""
Unit tests for Phase 3 Baseline Controllers, Stochastic Action-Chunking Nominal Policy,
Benchmarks A, B, and C, Component Ablations, and Robustness Sweeps.
"""

import unittest
import numpy as np

from src.envs.franka_slot_env import FrankaSlotEnv
from src.envs.tactile_benchmarks_env import NonPlanarContourEnv, IncipientSlipManipulationEnv
from scripts.run_phase3_validation import run_single_episode
from src.controllers.insertion_protocol import tactai
from src.controllers.baseline_controllers import (
    UnconstrainedPolicyBaseline,
    FixedIsotropicImpedanceController,
    WristFTAdmittanceController,
)
from src.controllers.nominal_policies import StochasticActionChunkingPolicy


class TestPhase3Benchmarks(unittest.TestCase):
    """Tests for Phase 3 comparative controllers, Benchmarks A/B/C, and ablations."""

    def setUp(self):
        self.env = FrankaSlotEnv(seed=42)
        self.u_nom = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

    def test_baseline_controller_interfaces(self):
        """All three baseline controllers must return valid (6,) wrenches and diagnostics."""
        obs, _ = self.env.reset(ood_tilt_deg=(8.0, 0.0), ood_offset_mm=(0.1, 0.0))

        controllers = [
            UnconstrainedPolicyBaseline(dt=self.env.dt, latency_ms=10.0),
            FixedIsotropicImpedanceController(k_trans=200.0, dt=self.env.dt, latency_ms=10.0),
            WristFTAdmittanceController(dt=self.env.dt, latency_ms=10.0),
        ]

        for ctrl in controllers:
            ctrl.reset()
            u_cmd, diag = ctrl.compute_safe_action(self.u_nom, obs)
            self.assertEqual(u_cmd.shape, (6,))
            self.assertTrue(np.all(np.isfinite(u_cmd)))
            self.assertIn("f_norm", diag)
            self.assertIn("solve_time_ms", diag)
            self.assertLess(diag["solve_time_ms"], 5.0)

    def test_stochastic_action_chunking_policy(self):
        """StochasticActionChunkingPolicy must produce deterministic 50-step chunks at 20 Hz."""
        obs, _ = self.env.reset(ood_tilt_deg=(10.0, -5.0))
        policy = StochasticActionChunkingPolicy(seed=101)
        policy.reset(seed=101)
        actions = [policy.get_action(obs) for _ in range(120)]
        self.assertEqual(actions[0].shape, (6,))
        self.assertTrue(np.all(np.isfinite(actions[0])))

        # Reproducibility with same seed
        policy.reset(seed=101)
        actions_repeat = [policy.get_action(obs) for _ in range(120)]
        np.testing.assert_allclose(actions[0], actions_repeat[0])
        np.testing.assert_allclose(actions[75], actions_repeat[75])

    def test_comparative_10deg_ood_insertion(self):
        """Compare the same learned checkpoint with and without supervision."""
        initial = ((10.0, 0.0), (0.0, 0.0))
        baseline = run_single_episode(self.env, UnconstrainedPolicyBaseline(), *initial)
        supervised = run_single_episode(self.env, tactai(self.env.dt), *initial)
        self.assertFalse(baseline["is_success"])
        self.assertGreater(baseline["peak_force_n"], 15.0)
        self.assertTrue(supervised["is_success"])
        self.assertLessEqual(supervised["peak_force_n"], 15.0)

    def test_benchmark_b_nonplanar_contour_tracking(self):
        """Benchmark B: TactAI must bound contour force <= 15 N and outperform unconstrained baseline."""
        contour_env = NonPlanarContourEnv(stiffness_env=1.0e4, seed=42)
        res_base = contour_env.run_trial(controller_mode="unconstrained_farm", phase_shift=0.2, latency_ms=25.0)
        res_ours = contour_env.run_trial(controller_mode="tactai_ours", phase_shift=0.2, latency_ms=25.0)

        self.assertFalse(res_base["is_safe"])
        self.assertGreater(res_base["peak_force_n"], 20.0)
        self.assertTrue(res_ours["is_success"])
        self.assertTrue(res_ours["is_safe"])
        self.assertLessEqual(res_ours["peak_force_n"], 15.0)

    def test_benchmark_c_incipient_slip_manipulation(self):
        """Benchmark C: TactAI Slip-CBF must prevent tool drop and keep grip force <= 15 N."""
        slip_env = IncipientSlipManipulationEnv(seed=42)
        res_base = slip_env.run_trial(controller_mode="unconstrained_farm", mu_friction=0.28, peak_disturbance_n=3.8)
        res_ours = slip_env.run_trial(controller_mode="tactai_ours", mu_friction=0.28, peak_disturbance_n=3.8)

        self.assertTrue(res_base["is_dropped"])
        self.assertTrue(res_ours["is_success"])
        self.assertFalse(res_ours["is_dropped"])
        self.assertLessEqual(res_ours["peak_grip_force_n"], 15.0)

    def test_tactile_controller_latency_and_stiffness_robustness(self):
        """TactileAlignmentController must remain passive and succeed under 25 ms latency and K_env = 10^5 N/m."""
        env_stiff = FrankaSlotEnv(stiffness_env=1.0e5, friction_env=0.60, seed=42)
        controller = tactai(env_stiff.dt, latency_ms=25.0)
        result = run_single_episode(env_stiff, controller, (10.0, 0.0), (0.0, 0.0))
        self.assertTrue(result["is_success"])
        self.assertLessEqual(result["peak_force_n"], 15.0)
        self.assertGreaterEqual(result["min_tank_energy_j"], controller.tank_min_energy)


if __name__ == "__main__":
    unittest.main()
