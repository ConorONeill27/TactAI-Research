"""
Phase 1 Scientific Validation Script: MuJoCo 3.15 Simulation, Perception & Invariant Engine
Project: Tac-Guard (Tactile Contact-Manifold Safety Filtering for Learned Manipulation Policies)

Executes:
1. Benchmark A Baseline Evaluation (Sub-millimeter slot insertion under OOD tilt)
2. Mathematical invariant assertions across continuous traction fields
3. High-frequency execution rate benchmarking (target >= 500 Hz)
4. Structured logging of scientific provenance and trajectory data (.npz & .json)
"""

import os
import sys
import time
import json
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv
from src.utils.logger import EpisodeLogger
from src.invariants.tactile_invariants import ContactMode


def run_phase1_validation():
    print("=" * 80)
    print("  TAC-GUARD: PHASE 1 SCIENTIFIC VALIDATION BENCHMARK")
    print("  Target Destination: IAS, TU Darmstadt (Belousov & Peters)")
    print("=" * 80)

    log_dir = os.path.join("data", "phase1_validation")
    os.makedirs(log_dir, exist_ok=True)
    logger = EpisodeLogger(log_dir=log_dir)

    env = FrankaSlotEnv(seed=42)
    print(f"\n[Environment Config]")
    print(f"  Slot clearance:     {env.slot_clearance * 1e3:.2f} mm per side (total width {2.0 * env.slot_half_w * 1e3:.2f} mm)")
    print(f"  Peg dimensions:     {env.peg_width * 1e3:.1f} x {env.peg_depth * 1e3:.1f} x {env.peg_length * 1e3:.1f} mm")
    print(f"  Contact stiffness:  {env.stiffness_env:.1e} N/m")
    print(f"  Control frequency:  {1.0 / env.dt:.0f} Hz (dt = {env.dt * 1e3:.1f} ms)")
    print(f"  Sensor resolution:  {env.grid_size} x {env.grid_size} ({env.grid_size**2} traction vectors / pad)")

    # ---------------------------------------------------------
    # TEST 1: Unconstrained Baseline under OOD Misalignment (+10 deg tilt)
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> TEST CASE 1: Unconstrained Baseline under OOD Tilt (+10 deg)")
    print("-" * 60)
    
    logger.start_episode(
        episode_id="baseline_ood_tilt_10deg",
        controller_name="unconstrained_baseline",
        seed=42,
        env_params={
            "slot_clearance_mm": env.slot_clearance * 1e3,
            "stiffness_env_npm": env.stiffness_env,
            "tilt_deg": 10.0
        }
    )

    obs, info = env.reset(ood_tilt_deg=(10.0, 0.0), ood_offset_mm=(0.0, 0.0))
    down_wrench = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0]) # Downward push action

    t_start = time.perf_counter()
    step_count = 0
    while True:
        obs, reward, term, trunc, info = env.step(down_wrench)
        logger.log_step(obs, down_wrench, reward, info)
        step_count += 1
        if term or trunc:
            break
    duration_s = time.perf_counter() - t_start

    summary_ood = logger.finish_episode(info, save_prefix="baseline_ood_tilt_10deg")
    kpis_ood = summary_ood["kpis"]

    print(f"  Termination status:  {'JAMMED / WEDGED' if kpis_ood['is_jammed'] else 'STOPPED'}")
    print(f"  Duration:            {kpis_ood['cycle_time_s'] * 1e3:.1f} ms ({step_count} steps)")
    print(f"  Insertion depth:     {kpis_ood['final_insertion_depth_mm']:.2f} mm / 25.0 mm")
    print(f"  Peak contact force:  {kpis_ood['peak_force_n']:.2f} N (Failure threshold: >40.0 N)")
    print(f"  Min stick ratio:     {kpis_ood['min_stick_ratio']:.4f}")
    print(f"  Min jamming index:   {kpis_ood['min_jamming_index']:.4f}")
    print(f"  Success:             {kpis_ood['is_success']}")

    assert kpis_ood["is_jammed"], "Test 1 assertion failed: OOD tilt must cause jamming!"
    assert kpis_ood["peak_force_n"] > 40.0, "Test 1 assertion failed: Force spike must exceed 40 N!"
    print("  [PASS] Replicated exact OOD failure mode described in Research Paper Section V.")

    # ---------------------------------------------------------
    # TEST 2: Aligned Compliant Insertion (0 deg tilt)
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> TEST CASE 2: Aligned Compliant Insertion (0 deg tilt)")
    print("-" * 60)

    logger.start_episode(
        episode_id="aligned_insertion",
        controller_name="aligned_baseline",
        seed=42,
        env_params={
            "slot_clearance_mm": env.slot_clearance * 1e3,
            "stiffness_env_npm": env.stiffness_env,
            "tilt_deg": 0.0
        }
    )

    obs, info = env.reset(ood_tilt_deg=(0.0, 0.0), ood_offset_mm=(0.0, 0.0))
    step_count = 0
    t_start = time.perf_counter()
    while True:
        obs, reward, term, trunc, info = env.step(down_wrench)
        logger.log_step(obs, down_wrench, reward, info)
        step_count += 1
        if term or trunc:
            break
    duration_s = time.perf_counter() - t_start

    summary_aligned = logger.finish_episode(info, save_prefix="aligned_insertion")
    kpis_aligned = summary_aligned["kpis"]

    print(f"  Termination status:  {'SUCCESS' if kpis_aligned['is_success'] else 'STOPPED'}")
    print(f"  Duration:            {kpis_aligned['cycle_time_s'] * 1e3:.1f} ms ({step_count} steps)")
    print(f"  Insertion depth:     {kpis_aligned['final_insertion_depth_mm']:.2f} mm / 25.0 mm")
    print(f"  Peak contact force:  {kpis_aligned['peak_force_n']:.2f} N (Safe limit: <20.0 N)")
    print(f"  Success:             {kpis_aligned['is_success']}")

    assert kpis_aligned["is_success"], "Test 2 assertion failed: Aligned insertion must succeed!"
    assert kpis_aligned["final_insertion_depth_mm"] >= 20.0, "Test 2 assertion failed: Peg must insert >= 20 mm!"
    assert kpis_aligned["peak_force_n"] < 20.0, "Test 2 assertion failed: Contact force must remain low!"
    print("  [PASS] Verified clearance insertion within physical tolerance.")

    # ---------------------------------------------------------
    # TEST 3: High-Frequency Rate & Mathematical Invariant Assertions
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> TEST CASE 3: 1 kHz Throughput & Invariant Integrity Assertions")
    print("-" * 60)

    n_bench_steps = 200
    obs, info = env.reset()
    bench_action = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])

    t0 = time.perf_counter()
    for _ in range(n_bench_steps):
        obs, _, _, _, _ = env.step(bench_action)
        # Check invariants
        tac = obs["tactile"]
        # Positivity of eigenvalues
        l1, l2 = tac["eigenvalues"]
        assert l1 >= l2 >= -1e-8, f"Invariant violation: C eigenvalues not PSD: {l1}, {l2}"
        # Stick ratio bounded
        rho = tac["stick_ratio"]
        assert 0.0 <= rho <= 1.0, f"Invariant violation: stick ratio out of bounds: {rho}"
        # Jamming index non-negative
        assert tac["jamming_index"] >= 0.0, f"Invariant violation: J_jam negative: {tac['jamming_index']}"

    total_bench_time = time.perf_counter() - t0
    step_hz = n_bench_steps / total_bench_time
    step_ms = (total_bench_time / n_bench_steps) * 1e3

    print(f"  Steps evaluated:     {n_bench_steps}")
    print(f"  Total wall time:     {total_bench_time:.3f} s")
    print(f"  Loop step time:      {step_ms:.2f} ms")
    print(f"  Effective frequency: {step_hz:.0f} Hz (Claim 1 requirement: >= 500 Hz)")
    assert step_hz >= 500.0, f"Throughput requirement not met: {step_hz:.0f} Hz < 500 Hz"
    print("  [PASS] Achieved high-frequency execution rate compliant with Patent Claim 1.")

    # ---------------------------------------------------------
    # SUMMARY & PROVENANCE REPORT
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PHASE 1 VALIDATION COMPLETE — ALL CRITERIA SATISFIED")
    print("=" * 80)
    print(f"  Saved artifacts:")
    print(f"    - {os.path.join(log_dir, 'baseline_ood_tilt_10deg.npz')}")
    print(f"    - {os.path.join(log_dir, 'baseline_ood_tilt_10deg_summary.json')}")
    print(f"    - {os.path.join(log_dir, 'aligned_insertion.npz')}")
    print(f"    - {os.path.join(log_dir, 'aligned_insertion_summary.json')}")
    print("\n  Summary JSON Excerpt (OOD Failure):")
    print(json.dumps(kpis_ood, indent=4))
    print("\n  Summary JSON Excerpt (Aligned Success):")
    print(json.dumps(kpis_aligned, indent=4))


if __name__ == "__main__":
    run_phase1_validation()
