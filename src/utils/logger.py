"""
Structured Episode Logger for TactAI Simulation Benchmarks.
Provides scientific provenance, deterministic logging, and structured trajectory storage.

Saves:
1. Compressed binary trajectories (.npz) containing full states, actions, wrenches, and tactile invariants.
2. Formatted summary reports (.json) containing key performance indicators (KPIs),
   failure modes, contact distributions, and provenance metadata.
"""

from typing import Dict, Any, List, Optional
import os
import json
import datetime
import subprocess
import numpy as np

from src.invariants.tactile_invariants import ContactMode


class EpisodeLogger:
    """
    High-frequency structured episode logger for robotic manipulation benchmarks.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.reset()

    def reset(self):
        """Clears buffers for a new episode."""
        self.metadata: Dict[str, Any] = {}
        self.times: List[float] = []
        self.steps: List[int] = []
        self.ee_poses: List[np.ndarray] = []
        self.ee_rpys: List[np.ndarray] = []
        self.ee_linvels: List[np.ndarray] = []
        self.ee_angvels: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.ext_forces: List[np.ndarray] = []
        self.ext_torques: List[np.ndarray] = []
        self.cops: List[np.ndarray] = []
        self.eigenvalues: List[np.ndarray] = []
        self.stick_ratios: List[float] = []
        self.jamming_indices: List[float] = []
        self.contact_modes: List[int] = []
        self.insertion_depths: List[float] = []
        self.rewards: List[float] = []

    def start_episode(
        self,
        episode_id: str,
        controller_name: str,
        seed: int,
        env_params: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None
    ):
        """Initializes episode metadata and provenance tracking."""
        self.reset()
        
        # Query current git commit if in a git repository
        git_hash = "unknown"
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode("ascii").strip()
        except Exception:
            pass

        self.metadata = {
            "episode_id": episode_id,
            "controller_name": controller_name,
            "seed": seed,
            "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_commit": git_hash,
            "env_params": env_params or {},
            "extra_meta": extra_meta or {}
        }

    def log_step(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        info: Dict[str, Any]
    ):
        """Logs a single control step."""
        self.steps.append(int(obs["step"]))
        self.times.append(float(obs["time"]))
        self.ee_poses.append(obs["ee_pos"].copy())
        self.ee_rpys.append(obs["ee_rpy"].copy())
        self.ee_linvels.append(obs["ee_linvel"].copy())
        self.ee_angvels.append(obs["ee_angvel"].copy())
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.ext_forces.append(obs["external_force"].copy())
        self.ext_torques.append(obs["external_torque"].copy())
        
        tactile = obs["tactile"]
        self.cops.append(tactile["center_of_pressure"].copy())
        self.eigenvalues.append(tactile["eigenvalues"].copy())
        self.stick_ratios.append(float(tactile["stick_ratio"]))
        self.jamming_indices.append(float(tactile["jamming_index"]))
        
        mode = tactile["contact_mode"]
        mode_val = mode.value if isinstance(mode, ContactMode) else int(mode)
        self.contact_modes.append(mode_val)
        
        self.insertion_depths.append(float(info.get("insertion_depth_m", 0.0)))
        self.rewards.append(float(reward))

    def finish_episode(
        self,
        final_info: Dict[str, Any],
        save_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compiles summary statistics, writes .npz trajectory and .json summary.
        
        Returns:
            summary_dict containing key metrics
        """
        prefix = save_prefix or f"{self.metadata.get('controller_name', 'ep')}_{self.metadata.get('episode_id', '0')}"
        
        # Convert lists to NumPy arrays
        np_steps = np.array(self.steps, dtype=np.int32)
        np_times = np.array(self.times, dtype=np.float64)
        np_ee_poses = np.array(self.ee_poses, dtype=np.float64)
        np_ee_rpys = np.array(self.ee_rpys, dtype=np.float64)
        np_ee_linvels = np.array(self.ee_linvels, dtype=np.float64)
        np_ee_angvels = np.array(self.ee_angvels, dtype=np.float64)
        np_actions = np.array(self.actions, dtype=np.float64)
        np_ext_forces = np.array(self.ext_forces, dtype=np.float64)
        np_ext_torques = np.array(self.ext_torques, dtype=np.float64)
        np_cops = np.array(self.cops, dtype=np.float64)
        np_eigenvalues = np.array(self.eigenvalues, dtype=np.float64)
        np_stick_ratios = np.array(self.stick_ratios, dtype=np.float64)
        np_jamming_indices = np.array(self.jamming_indices, dtype=np.float64)
        np_contact_modes = np.array(self.contact_modes, dtype=np.int32)
        np_depths = np.array(self.insertion_depths, dtype=np.float64)
        np_rewards = np.array(self.rewards, dtype=np.float64)

        # Compute force magnitudes
        force_mags = np.linalg.norm(np_ext_forces, axis=1) if len(np_ext_forces) > 0 else np.array([0.0])
        peak_force = float(np.max(force_mags))
        
        # Contact active filter (force > 0.5 N)
        contact_active = force_mags > 0.5
        mean_contact_force = float(np.mean(force_mags[contact_active])) if np.any(contact_active) else 0.0

        # Contact mode distribution
        total_steps = max(1, len(self.steps))
        mode_counts = {mode.name: 0 for mode in ContactMode}
        for m in self.contact_modes:
            try:
                name = ContactMode(m).name
                mode_counts[name] += 1
            except ValueError:
                pass
        mode_distribution = {k: float(v / total_steps) for k, v in mode_counts.items()}

        # Summary KPIs
        summary = {
            "metadata": self.metadata,
            "kpis": {
                "is_success": bool(final_info.get("is_success", False)),
                "is_jammed": bool(final_info.get("is_jammed", False)),
                "is_force_exceeded": bool(final_info.get("is_force_exceeded", False)),
                "total_steps": len(self.steps),
                "cycle_time_s": float(np_times[-1]) if len(np_times) > 0 else 0.0,
                "peak_force_n": peak_force,
                "mean_contact_force_n": mean_contact_force,
                "final_insertion_depth_mm": float(np_depths[-1] * 1e3) if len(np_depths) > 0 else 0.0,
                "min_stick_ratio": float(np.min(np_stick_ratios)) if len(np_stick_ratios) > 0 else 1.0,
                "min_jamming_index": float(np.min(np_jamming_indices)) if len(np_jamming_indices) > 0 else 100.0,
                "cumulative_reward": float(np.sum(np_rewards))
            },
            "contact_mode_distribution": mode_distribution
        }

        # Save binary .npz file
        npz_path = os.path.join(self.log_dir, f"{prefix}.npz")
        np.savez_compressed(
            npz_path,
            steps=np_steps,
            times=np_times,
            ee_poses=np_ee_poses,
            ee_rpys=np_ee_rpys,
            ee_linvels=np_ee_linvels,
            ee_angvels=np_ee_angvels,
            actions=np_actions,
            ext_forces=np_ext_forces,
            ext_torques=np_ext_torques,
            cops=np_cops,
            eigenvalues=np_eigenvalues,
            stick_ratios=np_stick_ratios,
            jamming_indices=np_jamming_indices,
            contact_modes=np_contact_modes,
            insertion_depths=np_depths,
            rewards=np_rewards
        )

        # Save JSON summary
        json_path = os.path.join(self.log_dir, f"{prefix}_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
