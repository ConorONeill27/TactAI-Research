"""A2 insertion with an unknown compliant grasp and wrist-origin loading."""

import numpy as np

from src.envs.franka_slot_env import FrankaSlotEnv


class UnknownGraspEnv(FrankaSlotEnv):
    """The grasp offset is a hinge spring rest pose and is unavailable to controllers."""

    def reset(self, *args, grasp_tilt_deg=(0.0, 0.0), **kwargs):
        self._grasp_rest = np.deg2rad(grasp_tilt_deg)
        return super().reset(*args, **kwargs)

    def _reset_peg_mount(self):
        super()._reset_peg_mount()
        for joint, angle in zip(("peg_drx", "peg_dry"), self._grasp_rest, strict=True):
            index = self.model.joint(joint).qposadr[0]
            self.model.qpos_spring[index] = angle
            self.data.qpos[index] = angle

    def _insertion_pose(self):
        vertical = self.data.geom_xmat[self.peg_geom_id, 8]
        tip = self.data.geom_xpos[self.peg_geom_id, 2] - self.peg_half_l * vertical
        return float(tip), float(np.rad2deg(np.arccos(np.clip(vertical, -1, 1))))

    def _compute_external_contact_wrench(self):
        force, peg_torque = super()._compute_external_contact_wrench()
        wrist = self.data.body("ee_base")
        lever = self.data.geom_xpos[self.peg_geom_id] - wrist.xpos
        # Equivalently, sum the moments of every peg contact about the wrist.
        # The Cartesian QP uses world components of the wrist-origin wrench.
        return force, peg_torque + np.cross(lever, force)

    def _get_observation(self):
        obs = super()._get_observation()
        # Keep privileged simulator state out of the policy and controller interface.
        obs.pop("peg_deflection")
        obs.pop("peg_pos")
        return obs
