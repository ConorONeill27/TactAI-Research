"""
Interactive 3D Simulation Viewer for TactAI.
Demonstrates the Franka Emika Panda robotic manipulator operating under the
Phase 2 Tactile Supervisory Safety Filter and Tactile Alignment Controller.

Features:
- Official Franka Emika Panda 7-DoF articulated robot arm on industrial workstation
- Dual GelSight Mini optical tactile sensor pads with continuous traction perception
- Transparent acrylic slot receptacle with 0.15 mm sub-millimeter clearance
- Active closed-loop tactile alignment from out-of-distribution tilt (+10 deg -> 0 deg)
- Real-time Control Barrier Function (CBF) force safety filter (F_max <= 15 N)
- Continuous demonstration cycle (Approach -> Align -> Insert -> Hold -> Retract)

Controls:
- Left Mouse Drag:           Rotate 3D camera
- Right Mouse Drag / Scroll: Zoom in / out
- Middle Mouse Drag:         Pan camera
- Tab:                       Toggle camera (Robot View -> Closeup View -> Free Camera)
- Spacebar:                  Pause / resume simulation
- R:                         Reset demonstration cycle
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
    print("  - Tab:                       Toggle camera view (Workstation -> Closeup -> Free)")
    print("  - Spacebar:                  Pause / Resume simulation")
    print("  - R:                         Reset demonstration cycle")
    print("=" * 78)

    # State flags
    is_paused = [False]
    cam_modes = [0, 1, -1]  # 0: robot_view, 1: closeup_view, -1: free
    cam_names = ["Robot Workstation View", "Closeup Cavity View", "Free Orbit Camera"]
    cam_idx = [0]
    sim_time = [0.0]
    last_print_time = [0.0]
    last_cycle_idx = [0]

    def key_callback(keycode):
        if keycode == 32:  # Space
            is_paused[0] = not is_paused[0]
            status = "PAUSED" if is_paused[0] else "RUNNING"
            print(f"\n  [Simulation {status}]")
        elif keycode == 258:  # Tab
            cam_idx[0] = (cam_idx[0] + 1) % len(cam_modes)
            mode = cam_modes[cam_idx[0]]
            if mode == -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mode
            print(f"\n  [Camera Mode: {cam_names[cam_idx[0]]}]")
        elif keycode in (82, 114):  # R or r
            env.reset(ood_tilt_deg=(tilt_deg, 0.0), ood_offset_mm=(offset_mm, 0.0))
            sim_time[0] = 0.0
            print("\n  [Demonstration Reset to Start]")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        # Default camera: robot_view
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = 0

        # Substeps per graphics sync (e.g. 2 steps of 1 ms = 2 ms per frame, 500 fps physics)
        substeps = 2
        dt_frame = substeps * env.dt

        while viewer.is_running():
            frame_start = time.time()

            if not is_paused[0]:
                for _ in range(substeps):
                    t_cycle = sim_time[0] % cycle_period
                    current_cycle = int(sim_time[0] // cycle_period)

                    # Cycle reset hook
                    if current_cycle != last_cycle_idx[0]:
                        last_cycle_idx[0] = current_cycle
                        env.reset(ood_tilt_deg=(tilt_deg, 0.0), ood_offset_mm=(offset_mm, 0.0))

                    # -------------------------------------------------------------
                    # 5-Phase Cyclic Demonstration Schedule
                    # -------------------------------------------------------------
                    if t_cycle < 1.5:
                        # Phase 1: Smooth Approach with Initial Tilt maintained
                        phase_name = "APPROACH"
                        s = t_cycle / 1.5
                        z_des = 0.005 - 0.005 * s
                        vz_des = -0.005 / 1.5
                        rx_des = np.deg2rad(tilt_deg)

                        fx = -60.0 * (env.data.qpos[0] - offset_mm * 1e-3) - 10.0 * env.data.qvel[0]
                        fy = -60.0 * env.data.qpos[1] - 10.0 * env.data.qvel[1]
                        fz = controller.f_grav + 150.0 * (z_des - env.data.qpos[2]) + 15.0 * (vz_des - env.data.qvel[2])
                        tx = -30.0 * (env.data.qpos[3] - rx_des) - 1.5 * env.data.qvel[3]
                        ty = -30.0 * env.data.qpos[4] - 1.5 * env.data.qvel[4]
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
                        fx = -80.0 * env.data.qpos[0] - 10.0 * env.data.qvel[0]
                        fy = -80.0 * env.data.qpos[1] - 10.0 * env.data.qvel[1]
                        fz = controller.f_grav - 3.0 - 15.0 * env.data.qvel[2]
                        tx = -40.0 * env.data.qpos[3] - 2.0 * env.data.qvel[3]
                        ty = -40.0 * env.data.qpos[4] - 2.0 * env.data.qvel[4]
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

                        fx = -60.0 * env.data.qpos[0] - 10.0 * env.data.qvel[0]
                        fy = -60.0 * env.data.qpos[1] - 10.0 * env.data.qvel[1]
                        fz = controller.f_grav + 120.0 * (z_des - env.data.qpos[2]) + 15.0 * (vz_des - env.data.qvel[2])
                        tx = -30.0 * (env.data.qpos[3] - rx_des) - 1.5 * env.data.qvel[3]
                        ty = -30.0 * env.data.qpos[4] - 1.5 * env.data.qvel[4]
                        tz = 0.0

                        action = np.array([fx, fy, fz, tx, ty, tz])
                        is_active_safety = False

                    # Step environment physics and update kinematic robot arm
                    env.step(action)
                    sim_time[0] += env.dt

            # Sync 3D viewer graphics
            viewer.sync()

            # Real-time dashboard telemetry at 10 Hz
            now = time.time()
            if now - last_print_time[0] >= 0.10:
                last_print_time[0] = now
                t_c = sim_time[0] % cycle_period
                peg_z_bottom = env.data.geom_xpos[env.peg_geom_id][2] - env.peg_half_l
                depth_mm = max(0.0, (env.slot_depth - peg_z_bottom) * 1e3)
                tilt_curr_deg = np.rad2deg(np.linalg.norm(env.data.qpos[3:5]))
                ext_force, _ = env._compute_external_contact_wrench()
                f_mag = float(np.linalg.norm(ext_force))
                cbf_tag = "[CBF ACTIVE]" if is_active_safety else "[SAFE]"

                sys.stdout.write(
                    f"\r  [{phase_name:<23}] "
                    f"Time: {t_c:4.1f}s | "
                    f"Tilt: {tilt_curr_deg:5.2f}° | "
                    f"Depth: {depth_mm:5.2f} mm | "
                    f"Force: {f_mag:5.2f} N | "
                    f"{cbf_tag:<12} "
                )
                sys.stdout.flush()

            # Regulate frame rate to real-time wall clock
            time_spent = time.time() - frame_start
            sleep_time = dt_frame - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\n\nSimulation viewer closed.")


if __name__ == "__main__":
    main()
