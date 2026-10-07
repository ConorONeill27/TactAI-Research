"""
Interactive 3D Simulation Viewer for TactAI.
Launches the native MuJoCo 3D viewer window on your desktop.

Controls:
- Left Mouse Drag:  Rotate camera view
- Right Mouse Drag: Zoom camera in/out
- Scroll Wheel:     Zoom camera
- Middle Mouse:     Pan camera
- Spacebar:         Pause / resume simulation
- ESC / Close:      Exit viewer
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv


def run_interactive_viewer():
    print("=" * 70)
    print("  TACTAI: INTERACTIVE 3D MUJOCO VIEWER")
    print("=" * 70)
    print("Controls:")
    print("  - Left click + drag:   Rotate 3D camera")
    print("  - Right click + drag:  Zoom in / out")
    print("  - Center click + drag: Pan camera")
    print("  - Spacebar:            Pause / unpause")
    print("=" * 70)

    env = FrankaSlotEnv(seed=42)

    # Scenarios to cycle through
    scenarios = [
        {"name": "Scenario 1: Aligned Insertion (0° Tilt) — SUCCESS", "tilt": (0.0, 0.0), "offset": (0.0, 0.0)},
        {"name": "Scenario 2: Tilted Push (+10° Tilt) — RIM WEDGING", "tilt": (10.0, 0.0), "offset": (0.0, 0.0)},
        {"name": "Scenario 3: Lateral Offset (0.8 mm) — WALL JAMMING", "tilt": (0.0, 0.0), "offset": (0.8, 0.0)},
    ]

    action_down = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Initial camera view centered on slot opening
        viewer.cam.lookat = [0.0, 0.0, 0.030]
        viewer.cam.distance = 0.22
        viewer.cam.elevation = -22
        viewer.cam.azimuth = 135

        scenario_idx = 0
        while viewer.is_running():
            sc = scenarios[scenario_idx % len(scenarios)]
            print(f"\n>>> Running {sc['name']}...")
            obs, info = env.reset(ood_tilt_deg=sc["tilt"], ood_offset_mm=sc["offset"])
            viewer.sync()
            time.sleep(0.5)

            step_count = 0
            while viewer.is_running() and step_count < 300:
                step_start = time.perf_counter()

                obs, reward, terminated, truncated, info = env.step(action_down)
                viewer.sync()
                step_count += 1

                if terminated or truncated:
                    status = "SUCCESS" if info["is_success"] else ("JAMMED" if info["is_jammed"] else "STOPPED")
                    print(f"    Result: {status} | Depth: {info['insertion_depth_m']*1e3:.2f} mm | Peak Force: {info['peak_force_n']:.1f} N")
                    time.sleep(1.2)  # Pause to observe final contact state
                    break

                # Sleep to match real-time 60 FPS visual rate
                elapsed = time.perf_counter() - step_start
                if elapsed < env.dt:
                    time.sleep(env.dt - elapsed)

            scenario_idx += 1


if __name__ == "__main__":
    run_interactive_viewer()
