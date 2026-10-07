"""
Interactive 3D Simulation Viewer for TactAI.
Demonstrates the Franka Emika Panda robotic manipulator operating under the
Phase 2 Tactile Supervisory Safety Filter and Tactile Alignment Controller.

Features:
- Official Franka Emika Panda articulated robot arm on industrial workstation
- Dual GelSight Mini optical tactile sensor pads with continuous traction perception
- Transparent acrylic slot receptacle with 0.15 mm sub-millimeter clearance
- Active closed-loop tactile alignment from out-of-distribution tilt (+10 deg -> 0 deg)
- Real-time Control Barrier Function (CBF) force safety filter (F_max <= 15 N)
- Continuous demonstration cycle (Approach -> Align -> Insert -> Hold -> Retract)

Controls:
- Left Mouse Drag:           Rotate 3D camera
- Right Mouse Drag / Scroll: Zoom in / out
- Middle Mouse Drag:         Pan camera
- Tab:                       Toggle between Robot View and Closeup View
- Ctrl + Right Mouse Drag:   Grab and pull peg / robot with 3D mouse force
- Spacebar:                  Pause / resume simulation
"""

import os
import sys
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv
from src.controllers.tactile_alignment_controller import TactileAlignmentController


def main():
    parser = argparse.ArgumentParser(description="TactAI 3D Interactive Robot Viewer")
    parser.add_argument("--tilt", type=float, default=10.0, help="Initial misalignment tilt in degrees (default: 10.0 deg)")
    parser.add_argument("--offset", type=float, default=0.0, help="Initial lateral offset in mm (default: 0.0 mm)")
    parser.add_argument("--cycle", type=float, default=7.0, help="Demonstration cycle duration in seconds (default: 7.0 s)")
    args = parser.parse_args()

    tilt_deg = args.tilt
    offset_mm = args.offset
    cycle_period = args.cycle

    env = FrankaSlotEnv(seed=42)
    controller = TactileAlignmentController(
        f_max=15.0,
        gamma_jam=2.0,
        kp_rot=35.0,
        kd_rot=1.5,
        dt=env.dt
    )

    env.reset(ood_tilt_deg=(tilt_deg, 0.0), ood_offset_mm=(offset_mm, 0.0))

    print("=" * 78)
    print("      TACTAI: FRANKA EMIKA PANDA TACTILE MANIPULATION SIMULATOR")
    print("=" * 78)
    print(f"  Robot:             Franka Emika Panda (Official DeepMind Menagerie Model)")
    print(f"  Sensors:           Dual GelSight Mini (Continuous 3D Traction Fields)")
    print(f"  Fixture:           Precision Receptacle with {env.slot_clearance * 1e3:.2f} mm Clearance (0.30 mm Total)")
    print(f"  Environment:       High Stiffness K_env = {env.stiffness_env:.0f} N/m")
    print(f"  Initial Tilt:      +{tilt_deg:.1f}° Out-of-Distribution Misalignment")
    print(f"  Safety Filter:     Active CBF Force Throttle (F_max = {controller.f_max:.1f} N)")
    print("=" * 78)
    print("Controls in 3D Viewer:")
    print("  - Left click + drag:         Rotate camera around robot / workspace")
    print("  - Right click + drag / wheel:Zoom in / out")
    print("  - Center click + drag:       Pan camera")
    print("  - Tab:                       Toggle between Robot View and Closeup View")
    print("  - Ctrl + Right click:        Apply external 3D force perturbation with mouse")
    print("  - Spacebar:                  Pause / Resume simulation")
    print("=" * 78)

    # Telemetry throttle
    last_print_time = [0.0]

    def physics_callback(m, d):
        t_cycle = d.time % cycle_period

        # -------------------------------------------------------------
        # 5-Phase Cyclic Demonstration Schedule
        # -------------------------------------------------------------
        # Phase 1: Approach (0.0 to 1.5s)
        # Phase 2: Tactile Realignment & Insertion (1.5 to 4.5s)
        # Phase 3: Hold Seated in Cavity (4.5 to 5.5s)
        # Phase 4: Smooth Retract (5.5 to 7.0s)
        # -------------------------------------------------------------

        if t_cycle < 1.5:
            # Phase 1: Smooth Approach with Initial Tilt maintained
            phase_name = "APPROACH"
            s = t_cycle / 1.5
            z_des = 0.005 - 0.005 * s
            vz_des = -0.005 / 1.5
            rx_des = np.deg2rad(tilt_deg)

            # Hold lateral and orientation at initial misaligned state
            fx = -60.0 * (d.qpos[0] - offset_mm * 1e-3) - 10.0 * d.qvel[0]
            fy = -60.0 * d.qpos[1] - 10.0 * d.qvel[1]
            fz = controller.f_grav + 150.0 * (z_des - d.qpos[2]) + 15.0 * (vz_des - d.qvel[2])
            tx = -30.0 * (d.qpos[3] - rx_des) - 1.5 * d.qvel[3]
            ty = -30.0 * d.qpos[4] - 1.5 * d.qvel[4]
            tz = 0.0

            action = np.array([fx, fy, fz, tx, ty, tz])
            is_active_safety = False

        elif t_cycle < 4.5:
            # Phase 2: Active Closed-Loop Tactile Alignment & Insertion
            phase_name = "TACTILE ALIGN & INSERT"
            obs = env._get_observation()
            u_nom = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])
            u_safe, diag = controller.compute_safe_action(u_nom, obs)
            action = u_safe
            is_active_safety = diag["is_intervened"]

        elif t_cycle < 5.5:
            # Phase 3: Hold Fully Seated
            phase_name = "HOLD INSERTED (SUCCESS)"
            fx = -80.0 * d.qpos[0] - 10.0 * d.qvel[0]
            fy = -80.0 * d.qpos[1] - 10.0 * d.qvel[1]
            fz = controller.f_grav - 3.0 - 15.0 * d.qvel[2]
            tx = -40.0 * d.qpos[3] - 2.0 * d.qvel[3]
            ty = -40.0 * d.qpos[4] - 2.0 * d.qvel[4]
            tz = 0.0

            action = np.array([fx, fy, fz, tx, ty, tz])
            is_active_safety = False

        else:
            # Phase 4: Retract back to start height and restore initial tilt
            phase_name = "RETRACT & RE-ARM"
            s = (t_cycle - 5.5) / 1.5
            z_des = -0.020 + 0.025 * s
            vz_des = 0.025 / 1.5
            rx_des = np.deg2rad(tilt_deg) * s

            fx = -60.0 * d.qpos[0] - 10.0 * d.qvel[0]
            fy = -60.0 * d.qpos[1] - 10.0 * d.qvel[1]
            fz = controller.f_grav + 120.0 * (z_des - d.qpos[2]) + 15.0 * (vz_des - d.qvel[2])
            tx = -30.0 * (d.qpos[3] - rx_des) - 1.5 * d.qvel[3]
            ty = -30.0 * d.qpos[4] - 1.5 * d.qvel[4]
            tz = 0.0

            action = np.array([fx, fy, fz, tx, ty, tz])
            is_active_safety = False

        # Apply action to actuators
        d.ctrl[:] = action

        # Print real-time dashboard at 10 Hz
        now = time.time()
        if now - last_print_time[0] >= 0.10:
            last_print_time[0] = now
            peg_z_bottom = d.xpos[env.peg_geom_id][2] - env.peg_half_l
            depth_mm = max(0.0, (env.slot_depth - peg_z_bottom) * 1e3)
            tilt_curr_deg = np.rad2deg(np.linalg.norm(d.qpos[3:5]))
            
            # Query external contact force
            ext_force, _ = env._compute_external_contact_wrench()
            f_mag = float(np.linalg.norm(ext_force))
            
            cbf_tag = "[CBF ACTIVE]" if is_active_safety else "[SAFE]"
            sys.stdout.write(
                f"\r  [{phase_name:<23}] "
                f"Time: {t_cycle:4.1f}s | "
                f"Tilt: {tilt_curr_deg:5.2f}° | "
                f"Depth: {depth_mm:5.2f} mm | "
                f"Force: {f_mag:5.2f} N | "
                f"{cbf_tag:<12} "
            )
            sys.stdout.flush()

    # Register controller callback
    mujoco.set_mjcb_control(physics_callback)

    try:
        mujoco.viewer.launch(env.model, env.data)
    finally:
        mujoco.set_mjcb_control(None)
        print("\n\nSimulation viewer closed.")


if __name__ == "__main__":
    main()
