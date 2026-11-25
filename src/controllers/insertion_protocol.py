"""Controller constructors for the learned-policy insertion protocol."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.controllers.baseline_controllers import (
    FixedIsotropicImpedanceController, UnconstrainedPolicyBaseline,
)
from src.controllers.tactile_alignment_controller import TactileAlignmentController

DATA = Path(__file__).resolve().parents[2] / "data" / "learned_policy"


class ImpedanceSpiralController:
    """Rim-triggered planar Archimedean search with compliant attitude."""

    def __init__(self, dt=.001):
        self.dt = dt
        self.reset()

    def reset(self):
        self.contact_time = None

    def compute_safe_action(self, u_nominal, obs):
        pos, vel = obs["ee_pos"], obs["ee_linvel"]
        force = obs["external_force"]
        if self.contact_time is None and np.linalg.norm(force) > 1:
            self.contact_time = obs["time"]
        reference = np.zeros(2)
        if self.contact_time is not None and pos[2] > -.010:
            angle = 2 * np.pi * 5 * (obs["time"] - self.contact_time)
            radius = min(.0025, .00015 * angle)
            reference = radius * np.array([np.cos(angle), np.sin(angle)])
        lateral = np.clip(1400 * (reference - pos[:2]) - 40 * vel[:2], -12, 12)
        axial = np.clip(-5 - 55 * vel[2] + max(0, abs(force[2]) - 6), -12, 12)
        torque = np.clip(-12 * obs["ee_rpy"] - .7 * obs["ee_angvel"], -2, 2)
        return np.r_[lateral, axial + 24.525, torque], {}


@lru_cache(maxsize=1)
def frozen_parameters():
    with (DATA / "development_tuning.json").open(encoding="utf-8") as stream:
        return json.load(stream)["selected_parameters"]


def tactai(dt=.001, **overrides):
    parameters = frozen_parameters() | overrides
    return TactileAlignmentController(dt=dt, **parameters)


def insertion_controllers(dt):
    return {
        "unconstrained_policy": ("Unconstrained learned policy", UnconstrainedPolicyBaseline(dt=dt)),
        "fixed_impedance_k200": ("Fixed impedance", FixedIsotropicImpedanceController(dt=dt)),
        "ft_cbf_qp": ("Wrist F/T-only QP", tactai(dt, wrist_ft_only=True)),
        "impedance_spiral": ("Impedance + spiral", ImpedanceSpiralController(dt)),
        "tactile_safety_filter_ours": ("TactAI", tactai(dt)),
    }
