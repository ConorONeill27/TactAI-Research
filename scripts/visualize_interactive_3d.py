"""
Interactive 3D Simulation Viewer for TactAI.
Launches the native MuJoCo Simulate GUI on the main thread for full interactive 3D rendering.

Controls:
- Left Mouse Drag:         Rotate 3D camera
- Right Mouse Drag / Scroll: Zoom in/out
- Middle Mouse Drag:       Pan camera
- Double Click:            Select body
- Ctrl + Right Drag:       Apply external 3D force directly on peg or gripper
- Spacebar:                Pause / resume physics
- Backspace:               Reset simulation
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
    parser.add_argument("--aligned", action="store_true", help="Launch in aligned mode (0 deg tilt)")
    args = parser.parse_args()

    tilt_deg = 0.0 if args.aligned else args.tilt
    offset_mm = args.offset

    env = FrankaSlotEnv(seed=42)
    obs, info = env.reset(ood_tilt_deg=(tilt_deg, 0.0), ood_offset_mm=(offset_mm, 0.0))

    print("=" * 70)
    print("  TACTAI: INTERACTIVE 3D MUJOCO SIMULATE GUI")
    print("=" * 70)
    print(f"  Mode:       {'ALIGNED' if tilt_deg == 0.0 else f'MISALIGNED (+{tilt_deg}° Tilt)'}")
    print(f"  Clearance:  {env.slot_clearance * 1e3:.2f} mm per side")
    print(f"  Stiffness:  {env.stiffness_env:.0f} N/m")
    print("=" * 70)
    print("Controls:")
    print("  - Left click + drag:         Rotate 3D camera")
    print("  - Right click + drag:        Zoom in/out (or scroll wheel)")
    print("  - Middle click + drag:       Pan camera")
    print("  - Double click body:         Select body")
    print("  - Ctrl + Right click + drag: Apply external 3D perturbation force")
    print("  - Space:                     Pause / Play simulation")
    print("  - Backspace:                 Reset simulation state")
    print("=" * 70)

    # Controller callback called at every physics step by the viewer
    def controller(m, d):
        # Downward push command on Z axis
        d.ctrl[2] = -8.0

    mujoco.set_mjcb_control(controller)

    # Launch full interactive Simulate GUI on main thread
    try:
        mujoco.viewer.launch(env.model, env.data)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
