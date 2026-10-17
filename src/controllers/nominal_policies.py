"""
Nominal Learned Policy Generators for TactAI Benchmarking.

Models a 20 Hz Action-Chunking / Diffusion Policy (e.g., FARM; Helmut et al., ICRA 2026)
operating under Out-of-Distribution (OOD) geometric misalignments.
At 20 Hz (every H_chunk = 50 steps at 1 kHz), the policy predicts a temporal chunk
of nominal 6D Cartesian wrenches u_pi(t). Under OOD contact states, the unconstrained
policy extrapolates downward insertion commands accompanied by realistic multimodal
action-chunk drift and sensory noise.
"""

from typing import Dict, Any, Optional
import numpy as np


class StochasticActionChunkingPolicy:
    """
    20 Hz Action-Chunking Nominal Policy Simulator (FARM-style Diffusion Policy proxy).
    Generates 50-ms action chunks at 20 Hz with OOD extrapolation bias and smooth
    intra-chunk temporal correlation.
    """

    def __init__(
        self,
        nominal_fz: float = -8.0,
        chunk_horizon: int = 50,      # 50 steps @ 1 kHz = 20 Hz policy inference rate
        lateral_drift_std: float = 0.6,
        vertical_noise_std: float = 0.4,
        torque_drift_std: float = 0.15,
        seed: int = 42,
    ):
        self.nominal_fz = nominal_fz
        self.chunk_horizon = chunk_horizon
        self.lateral_drift_std = lateral_drift_std
        self.vertical_noise_std = vertical_noise_std
        self.torque_drift_std = torque_drift_std
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._current_chunk: Optional[np.ndarray] = None
        self._chunk_step: int = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(self.seed)
        self._current_chunk = None
        self._chunk_step = 0

    def _sample_new_chunk(self, obs: Dict[str, Any]) -> np.ndarray:
        """Samples a temporally smooth (H, 6) action chunk conditioned on OOD state."""
        ee_rpy = np.asarray(obs.get("ee_rpy", np.zeros(3)), dtype=np.float64)
        ext_force = np.asarray(obs.get("external_force", np.zeros(3)), dtype=np.float64)
        f_norm = float(np.linalg.norm(ext_force))

        # OOD extrapolation bias: when tilted and in contact, an unconstrained policy
        # pushes along its tilted body axis rather than un-wedging
        ood_bias_fx = float(np.clip(-1.5 * np.sin(ee_rpy[1]), -1.5, 1.5))
        ood_bias_fy = float(np.clip(1.5 * np.sin(ee_rpy[0]), -1.5, 1.5))
        # When blocked at rim, unconstrained policy slightly increases downward push
        ood_push_boost = -1.2 if f_norm > 5.0 else 0.0

        # Smooth temporal basis over the 50-step chunk
        t_grid = np.linspace(0.0, np.pi, self.chunk_horizon)
        envelope = np.sin(t_grid)

        dfx = ood_bias_fx + self.rng.normal(0.0, self.lateral_drift_std) * envelope
        dfy = ood_bias_fy + self.rng.normal(0.0, self.lateral_drift_std) * envelope
        dfz = (self.nominal_fz + ood_push_boost) + self.rng.normal(0.0, self.vertical_noise_std) * envelope
        dtx = self.rng.normal(0.0, self.torque_drift_std) * envelope
        dty = self.rng.normal(0.0, self.torque_drift_std) * envelope
        dtz = self.rng.normal(0.0, 0.5 * self.torque_drift_std) * envelope

        chunk = np.stack([dfx, dfy, dfz, dtx, dty, dtz], axis=-1)
        return chunk.astype(np.float64)

    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns the (6,) nominal policy wrench u_pi for the current 1 kHz step."""
        if self._current_chunk is None or self._chunk_step >= self.chunk_horizon:
            self._current_chunk = self._sample_new_chunk(obs)
            self._chunk_step = 0

        u_pi = self._current_chunk[self._chunk_step].copy()
        self._chunk_step += 1
        return u_pi
