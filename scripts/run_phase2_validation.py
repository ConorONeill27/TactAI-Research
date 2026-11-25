"""
Illustrative paired 10-degree comparison of the learned policy with and without TactAI.

Outputs:
- Binary episode trajectories (.npz) and JSON summaries in data/phase2_validation/
- High-resolution comparative telemetry plot in docs/figures/phase2_validation_benchmark.png
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv
from src.controllers.insertion_protocol import tactai
from src.controllers.learned_policy import LearnedChunkPolicy
from src.utils.logger import EpisodeLogger


def run_benchmark():
    out_dir = "data/phase2_validation"
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = "docs/figures"
    os.makedirs(fig_dir, exist_ok=True)

    logger = EpisodeLogger(log_dir=out_dir)
    env = FrankaSlotEnv(seed=42)
    controller = tactai(env.dt)
    policy = LearnedChunkPolicy()

    print("=" * 80)
    print("  TACTAI: PHASE 2 SCIENTIFIC VALIDATION BENCHMARK")
    print("  Controlled Tactile Alignment vs. Unconstrained Jamming Baseline")
    print("=" * 80)

    # -------------------------------------------------------------
    # Case 1: Unconstrained Baseline (+10 deg tilt)
    # -------------------------------------------------------------
    print("\n>>> RUNNING CASE 1: Baseline Unconstrained Push (+10° Tilt)")
    logger.start_episode(
        episode_id="baseline_tilt_10deg",
        controller_name="unconstrained_baseline",
        seed=42,
        env_params={"slot_clearance": env.slot_clearance, "initial_tilt_deg": 10.0}
    )
    obs, info = env.reset(ood_tilt_deg=(10.0, 0.0), ood_offset_mm=(0.0, 0.0))
    b_times, b_forces, b_depths, b_tilts = [], [], [], []
    for step in range(550):
        u_down = policy.get_action(obs)
        obs, r, term, trunc, info = env.step(u_down)
        logger.log_step(obs, u_down, r, info)
        b_times.append((step + 1) * env.dt * 1e3)
        b_forces.append(info["current_force_n"])
        b_depths.append(info["insertion_depth_m"] * 1e3)
        b_tilts.append(info["tilt_deg"])
        if term or trunc:
            break

    b_summary = logger.finish_episode(info, save_prefix="case1_baseline_jamming")
    print(f"  [RESULT] Status: {'JAMMED' if info['is_jammed'] else 'ENDED'}")
    print(f"  Peak Force:    {b_summary['kpis']['peak_force_n']:.2f} N")
    print(f"  Final Depth:   {b_summary['kpis']['final_insertion_depth_mm']:.2f} mm")
    print(f"  Duration:      {b_summary['kpis']['cycle_time_s']*1e3:.1f} ms")

    # -------------------------------------------------------------
    # Case 2: Phase 2 Tactile Controller (+10 deg tilt)
    # -------------------------------------------------------------
    print("\n>>> RUNNING CASE 2: TactAI Tactile Alignment Controller (+10° Tilt)")
    logger.start_episode(
        episode_id="tactai_controller_tilt_10deg",
        controller_name="tactai_frozen",
        seed=42,
        env_params={"slot_clearance": env.slot_clearance, "initial_tilt_deg": 10.0}
    )
    obs, info = env.reset(ood_tilt_deg=(10.0, 0.0), ood_offset_mm=(0.0, 0.0))
    controller.reset()
    policy.reset()

    c_times, c_forces, c_depths, c_tilts = [], [], [], []
    for step in range(550):
        u_nom = policy.get_action(obs)
        u_safe, diag = controller.compute_safe_action(u_nom, obs)
        obs, r, term, trunc, info = env.step(u_safe)
        logger.log_step(obs, u_safe, r, info)
        c_times.append((step + 1) * env.dt * 1e3)
        c_forces.append(info["current_force_n"])
        c_depths.append(info["insertion_depth_m"] * 1e3)
        c_tilts.append(info["tilt_deg"])
        if term or trunc:
            break

    c_summary = logger.finish_episode(info, save_prefix="case2_tactai_success")
    print(f"  [RESULT] Status: {'SUCCESS' if info['is_success'] else 'COMPLETED'}")
    print(f"  Peak Force:    {c_summary['kpis']['peak_force_n']:.2f} N")
    print(f"  Final Depth:   {c_summary['kpis']['final_insertion_depth_mm']:.2f} mm (Deep Clearance Entry)")
    print(f"  Final Tilt:    {info['tilt_deg']:.3f}° (Aligned from +10.0°)")
    print(f"  Duration:      {c_summary['kpis']['cycle_time_s']*1e3:.1f} ms")

    # -------------------------------------------------------------
    # Comparative Visual Telemetry Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    plt.subplots_adjust(hspace=0.35)

    # Subplot 1: Contact Force
    axes[0].plot(b_times, b_forces, "r-", linewidth=2.2, label="Unconstrained learned policy")
    axes[0].plot(c_times, c_forces, "b-", linewidth=2.2, label="TactAI")
    axes[0].axhline(40.0, color="red", linestyle=":", linewidth=1.5, label="Simulator force stop (40 N)")
    axes[0].axhline(controller.f_max, color="green", linestyle="--", linewidth=1.5, label=f"Force target ({controller.f_max:.0f} N)")
    axes[0].set_ylabel("Contact Force (N)", fontsize=11, fontweight="bold")
    tilt = c_summary["metadata"]["env_params"]["initial_tilt_deg"]
    axes[0].set_title(f"Tactile alignment vs. wedging ({tilt:+g}° tilt)", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", framealpha=0.9)

    # Subplot 2: Tilt Angle Correction
    axes[1].plot(b_times, b_tilts, "r-", linewidth=2.2, label="Unconstrained baseline")
    axes[1].plot(c_times, c_tilts, "b-", linewidth=2.2, label="TactAI reorientation")
    axes[1].axhline(1.0, color="green", linestyle="--", linewidth=1.5, label="Sub-Degree Alignment (< 1.0°)")
    axes[1].set_ylabel("Peg Tilt Angle (°)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", framealpha=0.9)

    # Subplot 3: Insertion Depth
    axes[2].plot(b_times, b_depths, "r-", linewidth=2.2,
                 label=f"Unconstrained baseline ({b_depths[-1]:.2f} mm)")
    axes[2].plot(c_times, c_depths, "b-", linewidth=2.2,
                 label=f"TactAI insertion ({c_depths[-1]:.2f} mm)")
    axes[2].axhline(20.0, color="green", linestyle="--", linewidth=1.5, label="Task Success Depth (20.0 mm)")
    axes[2].set_xlabel("Elapsed Time (ms)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Insertion Depth (mm)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper left", framealpha=0.9)

    for ax, values in zip(axes, (b_forces, b_tilts, b_depths)):
        ax.plot(b_times[-1], values[-1], "rx", markersize=9, markeredgewidth=2)
    end_label = "Jam termination" if b_summary["kpis"]["is_jammed"] else "Baseline timeout"
    axes[0].annotate(end_label, (b_times[-1], b_forces[-1]), xytext=(0.48, 0.46),
                     textcoords="axes fraction", zorder=7,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1),
                     arrowprops=dict(arrowstyle="-", color="#555555"))

    fig_path = os.path.join(fig_dir, "phase2_validation_benchmark.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(fig_path)[0] + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"\n[SAVED] Benchmark telemetry chart saved to: {fig_path}")

    print("\n" + "=" * 80)
    print("  PHASE 2 SCIENTIFIC VALIDATION SUMMARY")
    print("=" * 80)
    print(f"  Metric                     | Baseline (Unconstrained) | TactAI (Phase 2)")
    print(f"  ---------------------------+--------------------------+--------------------")
    print(f"  Insertion success          | {str(b_summary['kpis']['is_success']):<24} | {str(c_summary['kpis']['is_success']):<18}")
    print(f"  Peak Contact Force         | {b_summary['kpis']['peak_force_n']:<24.2f} N | {c_summary['kpis']['peak_force_n']:<18.2f} N")
    print(f"  Final Insertion Depth      | {b_summary['kpis']['final_insertion_depth_mm']:<24.2f} mm| {c_summary['kpis']['final_insertion_depth_mm']:<18.2f} mm")
    print(f"  Final Tilt Angle           | {b_tilts[-1]:<24.2f}° | {info['tilt_deg']:<18.3f}°")
    print(f"  Simulator jam              | {str(b_summary['kpis']['is_jammed']):<24} | {str(c_summary['kpis']['is_jammed']):<18}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
