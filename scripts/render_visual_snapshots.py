"""
Render 3D Visual Snapshots of the TactAI Simulation Environment.
Generates high-resolution images of:
1. 3D Setup Overview (Franka gripper, dual GelSight pads, peg, 0.15mm slot).
2. Side-by-side comparison: Aligned insertion vs. Misaligned (tilted) wedging.
3. Tactile 3D traction distribution on the GelSight sensor pad.
4. Composite visual dashboard.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv


def render_scene(env, camera_distance=0.22, azimuth=135, elevation=-22,
                 lookat=(0.0, 0.0, 0.030), width=960, height=720, show_contacts=False):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = list(lookat)
    camera.distance = camera_distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    
    env.model.geom_rgba[env.peg_geom_id] = (0.02, 0.65, 0.83, 1.0)
    with mujoco.Renderer(env.model, height, width) as renderer:
        renderer.update_scene(env.data, camera)
        if show_contacts:
            for i in range(env.data.ncon):
                contact = env.data.contact[i]
                if env.peg_geom_id not in (contact.geom1, contact.geom2):
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, i, wrench)
                if wrench[0] <= 0 or renderer.scene.ngeom >= renderer.scene.maxgeom:
                    continue
                normal = contact.frame[:3]
                if contact.geom1 == env.peg_geom_id:
                    normal = -normal
                arrow = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW,
                                    np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                    np.array([1.0, 0.65, 0.05, 1.0], dtype=np.float32))
                mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.001,
                                     contact.pos - normal * 0.012, contact.pos)
                renderer.scene.ngeom += 1
        return renderer.render()


def main():
    figures_dir = os.path.join("docs", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print("Rendering 3D visual snapshots...")

    # 1. Aligned Insertion State
    env_aligned = FrankaSlotEnv(seed=42)
    obs_a, info_a = env_aligned.reset(ood_tilt_deg=(0.0, 0.0), ood_offset_mm=(0.0, 0.0))
    action_down = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])
    for _ in range(70):
        obs_a, _, term, _, info_a = env_aligned.step(action_down)
        if term:
            break
    img_aligned = render_scene(env_aligned, camera_distance=0.18, azimuth=140, elevation=-20, lookat=(0.0, 0.0, 0.025))

    # 2. Tilted Misaligned Jammed State
    env_tilted = FrankaSlotEnv(seed=42)
    obs_t, info_t = env_tilted.reset(ood_tilt_deg=(10.0, 0.0), ood_offset_mm=(0.0, 0.0))
    for _ in range(35):
        obs_t, _, term, _, info_t = env_tilted.step(action_down)
        if term:
            break
    img_tilted = render_scene(env_tilted, camera_distance=0.16, azimuth=125,
                              elevation=-14, lookat=(0.0, 0.0, 0.020), show_contacts=True)

    # 3. Overview Perspective (Peg above slot in ready position)
    env_ready = FrankaSlotEnv(seed=42)
    env_ready.reset()
    img_overview = render_scene(env_ready, camera_distance=0.28, azimuth=130, elevation=-25, lookat=(0.0, 0.0, 0.040))

    # Extract tactile traction distribution from the tilted pad
    tactile_data = obs_t["tactile"]["left_pad"]
    # Normal pressure
    fz_pad = env_tilted.left_pad_pressure / 1e3

    # Create composite 2x2 dashboard
    fig = plt.figure(figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    # Panel 1: 3D Environment Setup
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_overview)
    ax1.set_title("1. 3D Simulation Workbench (Franka Arm, GelSight Mini Pads, Slot)", color="#38BDF8", fontsize=13, weight="bold", pad=10)
    ax1.axis("off")
    # Annotations
    ax1.text(0.04, 0.92, "Clearance: 0.15 mm per side\nStiffness: K_env = 10,000 N/m (10 N/mm)\nFrequency: 1,000 Hz",
             transform=ax1.transAxes, color="#F8FAFC", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#38BDF8", alpha=0.9))

    # Panel 2: Aligned Insertion (Success)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img_aligned)
    ax2.set_title("2. Aligned Insertion (0° Tilt) — SUCCESS", color="#4ADE80", fontsize=13, weight="bold", pad=10)
    ax2.axis("off")
    ax2.text(0.04, 0.92, f"Insertion depth: {info_a['insertion_depth_m']*1e3:.1f} mm\nPeak force: {info_a['peak_force_n']:.1f} N",
             transform=ax2.transAxes, color="#F8FAFC", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#14532D", edgecolor="#4ADE80", alpha=0.9))

    # Panel 3: Tilted Insertion (Jamming Failure)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(img_tilted)
    ax3.set_title("3. Tilted Push (+10° Tilt) — WEDGING / FORCE SURGE", color="#F87171", fontsize=13, weight="bold", pad=10)
    ax3.axis("off")
    ax3.text(0.04, 0.92, f"Peak force: {info_t['peak_force_n']:.1f} N\nGold arrows: peg contact normals\nDepth: {info_t['insertion_depth_m']*1e3:.2f} mm",
             transform=ax3.transAxes, color="#F8FAFC", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#7F1D1D", edgecolor="#F87171", alpha=0.9))

    # Panel 4: GelSight 3D Continuous Tactile Pressure Field
    ax4 = fig.add_subplot(2, 2, 4)
    U_mm = env_tilted.sensor_left.U * 1e3
    V_mm = env_tilted.sensor_left.V * 1e3
    cf = ax4.contourf(U_mm, V_mm, fz_pad, levels=30, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Normal Pressure $f_z(u,v)$ (kPa)", color="#F8FAFC", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#F8FAFC")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#F8FAFC')
    ax4.set_facecolor("#1E293B")
    ax4.set_title("4. GelSight Tactile Pad Pressure Field $f_z(u,v)$", color="#FBBF24", fontsize=13, weight="bold", pad=10)
    ax4.set_xlabel("Pad Width $u$ (mm)", color="#F8FAFC", fontsize=10)
    ax4.set_ylabel("Pad Height $v$ (mm)", color="#F8FAFC", fontsize=10)
    ax4.tick_params(colors="#94A3B8")
    for spine in ax4.spines.values():
        spine.set_color("#475569")
    
    # Mark Center of Pressure
    cop_u, cop_v = tactile_data["center_of_pressure"] * 1e3
    ax4.plot(cop_u, cop_v, "r*", markersize=14, label="Center of Pressure (CoP)")
    ax4.legend(loc="upper left", facecolor="#0F172A", edgecolor="#FBBF24", labelcolor="#F8FAFC", fontsize=9)

    plt.tight_layout()
    dashboard_path = os.path.join(figures_dir, "simulation_3d_overview.png")
    plt.savefig(dashboard_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print(f"Saved 3D simulation dashboard to: {dashboard_path}")
    return dashboard_path


if __name__ == "__main__":
    main()
