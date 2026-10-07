"""
Interactive 3D Simulation Viewer for TactAI.
Launches the native MuJoCo Simulate GUI with smooth animated insertion cycles.

Controls:
- Left Mouse Drag:           Rotate 3D camera
- Right Mouse Drag / Scroll: Zoom in / out
- Middle Mouse Drag:         Pan camera
- Tab:                       Switch to closeup camera angle
- Ctrl + Right Mouse Drag:   Grab and pull peg / robot with 3D force
- Spacebar:                  Pause / resume simulation
- Backspace:                 Reset simulation
"""

import os
import sys
import argparse
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv


def main():
    parser = argparse.ArgumentParser(description="TactAI 3D Interactive Viewer")
    parser.add_argument("--tilt", type=float, default=10.0, help="Initial tilt in degrees (default: 10.0 deg)")
    parser.add_argument("--offset", type=float, default=0.0, help="Initial lateral offset in mm (default: 0.0 mm)")
    parser.add_argument("--aligned", action="store_true", help="Run aligned insertion (0 deg tilt)")
    args = parser.parse_args()

    tilt_deg = 0.0 if args.aligned else args.tilt
    offset_mm = args.offset

    env = FrankaSlotEnv(seed=42)
    obs, info = env.reset(ood_tilt_deg=(tilt_deg, 0.0), ood_offset_mm=(offset_mm, 0.0))

    print("=" * 70)
    print("  TACTAI: INTERACTIVE 3D MUJOCO SIMULATOR")
    print("=" * 70)
    print(f"  Mode:       {'ALIGNED (0° Tilt)' if tilt_deg == 0.0 else f'MISALIGNED (+{tilt_deg}° Tilt)'}")
    print(f"  Clearance:  {env.slot_clearance * 1e3:.2f} mm per side (0.30 mm total clearance)")
    print(f"  Stiffness:  {env.stiffness_env:.0f} N/m")
    print("=" * 70)
    print("Controls in 3D Window:")
    print("  - Left click + drag:         Rotate 3D camera")
    print("  - Right click + drag:        Zoom camera in/out (or scroll wheel)")
    print("  - Center click + drag:       Pan camera")
    print("  - Tab:                       Switch between Free Cam and Closeup Cam")
    print("  - Ctrl + Right click + drag: Apply external 3D force with mouse")
    print("  - Spacebar:                  Pause / Resume physics")
    print("=" * 70)

    # Smooth cyclic insertion controller
    cycle_period = 6.0  # seconds per complete cycle

    def controller(m, d):
        t = d.time % cycle_period

        # Target vertical trajectory relative to base starting height
        if t < 1.5:
            # Phase 1: Smooth approach downwards
            s = t / 1.5
            z_des = -0.010 * s
            vz_des = -0.010 / 1.5
            # PD position control
            d.ctrl[2] = 200.0 * (z_des - d.qpos[2]) + 20.0 * (vz_des - d.qvel[2])
        elif t < 3.5:
            # Phase 2: Insertion push
            # Downward constant force of -8 N (pushes into slot or wedges against rim)
            d.ctrl[2] = -8.0
        elif t < 4.5:
            # Phase 3: Hold and observe contact state
            d.ctrl[2] = 0.0
        else:
            # Phase 4: Smooth retract back up to ready pose
            s = (t - 4.5) / 1.5
            z_des = -0.020 * (1.0 - s)
            vz_des = 0.020 / 1.5
            d.ctrl[2] = 200.0 * (z_des - d.qpos[2]) + 20.0 * (vz_des - d.qvel[2])

    mujoco.set_mjcb_control(controller)

    try:
        mujoco.viewer.launch(env.model, env.data)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
