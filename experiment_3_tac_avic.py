"""
Phase 3 Comprehensive Benchmark, Ablation & Robustness Validation Suite (arXiv Edition).

Executes the complete empirical protocol for the TactAI peer-reviewed paper:
1. Benchmark A (Table I): 50-trial Out-of-Distribution (OOD) angular misalignment
   (+-3 deg to +-15 deg) and lateral offset (0.0 to 0.90 mm) sweep in 0.15 mm clearance
   slot insertion, wrapping a 20 Hz StochasticActionChunkingPolicy (FARM proxy) across:
   - Unconstrained Diffusion Policy (FARM Baseline)
   - Fixed Isotropic Impedance (K = 200 N/m)
   - 6-Axis Wrist F/T Admittance Filter
   - Tactile Safety Filter + FARM (Ours, 1 kHz OSQP CBF-QP + Energy Tank)
2. Benchmarks B & C (Table II, N = 20 trials each):
   - Benchmark B: Non-Planar Stiff Contour Tracking (K_env = 10^4 N/m, v = 0.05 m/s, 25 ms delay)
   - Benchmark C: Dynamic Tool Manipulation under Incipient Slip (mu in [0.22, 0.58])
3. Component Ablation Study & Multi-Seed Environmental Robustness Sweep (Table III):
   - Ablations of Differential CoP / Constraint Tensor C, Anti-Jamming CBF h_jam,
     Energy Tank Passivity T(t), and Incipient Slip CBF h_slip (N = 20 trials)
   - Surface Stiffness K_env in {10^2, 10^3, 10^4, 10^5} N/m
   - Friction Coefficient mu in {0.15, 0.30, 0.50, 0.85}
   - Tactile Perception Latency in {10, 20, 35, 50} ms
   - Each robustness point evaluates 5 fixed scenarios x 5 seeded repeats (25 episodes)

Outputs:
- data/phase3_validation/benchmark_a_summary.json
- data/phase3_validation/benchmarks_bc_ablation_summary.json
- data/phase3_validation/table2_robustness_summary.json
- data/phase3_validation/phase3_trajectories.npz
- docs/figures/phase3_benchmark_comparison.png
- docs/figures/phase3_benchmarks_bc_ablation.png
"""

import os
import sys
import json
import time
import platform
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv
from src.envs.tactile_benchmarks_env import NonPlanarContourEnv, IncipientSlipManipulationEnv
from src.controllers.tactile_alignment_controller import TactileAlignmentController
from src.controllers.baseline_controllers import (
    UnconstrainedPolicyBaseline,
    FixedIsotropicImpedanceController,
    WristFTAdmittanceController,
)
from src.controllers.nominal_policies import StochasticActionChunkingPolicy


# 50 deterministic Out-of-Distribution (OOD) misalignment scenarios:
# Spans +-3 deg to +-15 deg angular tilt and 0.0 to 0.90 mm lateral offset
# (up to 6x the 0.15 mm radial slot clearance).
OOD_SCENARIOS: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    # Group 1 (Trials 0..9): Mild boundary & OOD tilts (+-0.5 deg to +-5 deg)
    ((0.5, 0.0), (0.0, 0.0)),
    ((-0.5, 0.0), (0.0, 0.0)),
    ((0.6, 0.0), (0.05, 0.0)),
    ((3.0, 0.0), (0.0, 0.0)),
    ((-3.0, 0.0), (0.0, 0.0)),
    ((5.0, 0.0), (0.0, 0.0)),
    ((-5.0, 0.0), (0.0, 0.0)),
    ((0.0, 5.0), (0.20, 0.0)),
    ((0.0, -5.0), (-0.20, 0.0)),
    ((4.0, 3.0), (0.15, -0.10)),
    # Group 2 (Trials 10..21): Moderate OOD tilts (+-6 deg to +-8 deg)
    ((6.0, 0.0), (0.0, 0.15)),
    ((-6.0, 0.0), (0.0, -0.15)),
    ((0.0, 6.0), (0.20, 0.0)),
    ((0.0, -6.0), (-0.20, 0.0)),
    ((7.0, 0.0), (0.15, 0.15)),
    ((-7.0, 0.0), (-0.15, -0.15)),
    ((8.0, 0.0), (0.0, 0.20)),
    ((-8.0, 0.0), (0.0, -0.20)),
    ((0.0, 8.0), (0.30, 0.0)),
    ((0.0, -8.0), (-0.30, 0.0)),
    ((6.0, -5.0), (0.20, -0.15)),
    ((-6.0, 5.0), (-0.20, 0.15)),
    # Group 3 (Trials 22..35): Severe OOD tilts (+-9 deg to +-11 deg, 1D & 2D compound)
    ((9.0, 0.0), (0.0, 0.25)),
    ((-9.0, 0.0), (0.0, -0.25)),
    ((10.0, 0.0), (0.0, 0.0)),
    ((-10.0, 0.0), (0.0, 0.0)),
    ((0.0, 10.0), (0.40, 0.0)),
    ((0.0, -10.0), (-0.40, 0.0)),
    ((10.0, -6.0), (0.20, -0.20)),
    ((-10.0, 6.0), (-0.25, 0.20)),
    ((8.0, 8.0), (0.25, 0.25)),
    ((-8.0, -8.0), (-0.25, -0.25)),
    ((11.0, 0.0), (0.0, 0.30)),
    ((-11.0, 0.0), (0.0, -0.30)),
    ((0.0, 11.0), (0.35, 0.0)),
    ((0.0, -11.0), (-0.35, 0.0)),
    # Group 4 (Trials 36..46): Extreme OOD tilts (+-12 deg to +-15 deg within capture cone)
    ((12.0, 0.0), (0.0, 0.30)),
    ((-12.0, 0.0), (0.0, -0.30)),
    ((0.0, 12.0), (0.30, 0.0)),
    ((0.0, -12.0), (-0.30, 0.0)),
    ((13.0, 0.0), (0.0, 0.20)),
    ((-13.0, 0.0), (0.0, -0.20)),
    ((14.0, 0.0), (0.0, 0.15)),
    ((-14.0, 0.0), (0.0, -0.15)),
    ((15.0, 0.0), (0.0, 0.0)),
    ((-15.0, 0.0), (0.0, 0.0)),
    ((0.0, 14.0), (0.20, 0.0)),
    # Group 5 (Trials 47..49): Extreme boundary cases with large lateral offset (> 5.5x clearance)
    # where rim capture cone is exceeded (tests safe force-bounded timeout without jamming)
    ((14.0, -11.0), (0.88, -0.82)),
    ((-14.5, 11.5), (-0.90, 0.85)),
    ((15.0, 12.0), (0.92, 0.88)),
]


# A small, fixed scenario panel crossed with multiple seeds. Scenario identity and
# stochastic repeat are separate dimensions in the robustness experiment.
ROBUSTNESS_SCENARIOS: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    ((10.0, 0.0), (0.0, 0.0)),
    ((-10.0, 0.0), (0.0, 0.0)),
    ((0.0, 10.0), (0.20, 0.0)),
    ((0.0, -10.0), (-0.20, 0.0)),
    ((9.0, -6.0), (0.15, -0.15)),
]


def wilson_interval_pct(successes: int, n: int, z: float = 1.959963984540054) -> List[float]:
    """95% Wilson score interval for a binomial proportion, in percent."""
    if n <= 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denom = 1.0 + (z * z / n)
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom)
    return [round(100.0 * max(0.0, centre - half), 2), round(100.0 * min(1.0, centre + half), 2)]


def binary_summary(values: List[bool]) -> Dict[str, Any]:
    """Keep the numerator, denominator, observed rate, and uncertainty together."""
    n = len(values)
    successes = int(sum(bool(v) for v in values))
    return {
        "successes": successes,
        "num_trials": n,
        "rate_pct": round(100.0 * successes / n, 2) if n else None,
        "wilson_95_ci_pct": wilson_interval_pct(successes, n),
    }


def summarize_episode_set(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Common, explicit summaries; never substitute failed-episode time for success time."""
    n = len(episodes)
    successes = [e for e in episodes if e["is_success"]]
    peak = np.asarray([e["peak_force_n"] for e in episodes], dtype=float)
    duration = np.asarray([e["episode_duration_s"] for e in episodes], dtype=float)
    success_duration = np.asarray([e["episode_duration_s"] for e in successes], dtype=float)
    success = binary_summary([e["is_success"] for e in episodes])
    jam = binary_summary([e["is_jammed"] for e in episodes])
    safe = binary_summary([e["is_safe_force"] for e in episodes])
    return {
        "num_trials": n,
        "success_count": success["successes"],
        "success_rate_pct": success["rate_pct"],
        "success_rate_wilson_95_ci_pct": success["wilson_95_ci_pct"],
        "jamming_count": jam["successes"],
        "jamming_rate_pct": jam["rate_pct"],
        "jamming_rate_wilson_95_ci_pct": jam["wilson_95_ci_pct"],
        "safe_force_count": safe["successes"],
        "safe_force_rate_pct": safe["rate_pct"],
        "safe_force_rate_wilson_95_ci_pct": safe["wilson_95_ci_pct"],
        "mean_peak_force_n": round(float(np.mean(peak)), 3) if n else None,
        "std_peak_force_n": round(float(np.std(peak, ddof=1)), 3) if n > 1 else 0.0,
        "max_peak_force_n": round(float(np.max(peak)), 3) if n else None,
        "mean_episode_duration_s": round(float(np.mean(duration)), 4) if n else None,
        "std_episode_duration_s": round(float(np.std(duration, ddof=1)), 4) if n > 1 else 0.0,
        "mean_successful_cycle_time_s": round(float(np.mean(success_duration)), 4) if len(success_duration) else None,
        "std_successful_cycle_time_s": round(float(np.std(success_duration, ddof=1)), 4) if len(success_duration) > 1 else (0.0 if len(success_duration) == 1 else None),
        "num_successful_cycle_times": int(len(success_duration)),
        "mean_solve_time_ms": round(float(np.mean([e["mean_solve_time_ms"] for e in episodes])), 5) if n else None,
    }


def run_single_episode(
    env: FrankaSlotEnv,
    controller: Any,
    tilt_deg: Tuple[float, float],
    offset_mm: Tuple[float, float],
    max_steps: int = 550,
    use_action_chunk_policy: bool = True,
    trial_seed: int = 42,
) -> Dict[str, Any]:
    """Runs a single deterministic insertion episode wrapping the 20 Hz nominal policy."""
    obs, info = env.reset(seed=trial_seed, ood_tilt_deg=tilt_deg, ood_offset_mm=offset_mm)
    controller.reset()
    nom_policy = StochasticActionChunkingPolicy(seed=trial_seed)
    nom_policy.reset(seed=trial_seed)
    u_const = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

    forces: List[float] = []
    depths: List[float] = []
    tilts: List[float] = []
    solve_times: List[float] = []
    stick_ratios: List[float] = []
    tank_energies: List[float] = []
    ever_jammed = False

    for step in range(max_steps):
        u_nominal = nom_policy.get_action(obs) if use_action_chunk_policy else u_const
        u_cmd, diag = controller.compute_safe_action(u_nominal, obs)
        obs, reward, term, trunc, info = env.step(u_cmd)

        forces.append(float(info["current_force_n"]))
        depths.append(float(info["insertion_depth_m"] * 1e3))
        tilts.append(float(info["tilt_deg"]))
        solve_times.append(float(diag.get("solve_time_ms", 0.0)))
        stick_ratios.append(float(obs["tactile"]["stick_ratio"]))
        tank_energies.append(float(diag.get("tank_energy_j", 1.0)))

        if info["is_jammed"]:
            ever_jammed = True

        if info["is_success"] or info["is_jammed"] or term or trunc:
            break

    peak_force = float(max(forces)) if forces else 0.0
    final_depth = float(depths[-1]) if depths else 0.0
    is_success = bool(info["is_success"])
    is_jammed_or_wedged = bool(ever_jammed or (not is_success and peak_force > 22.0 and final_depth < 10.0))

    return {
        "trial_seed": int(trial_seed),
        "tilt_deg": list(tilt_deg),
        "offset_mm": list(offset_mm),
        "initial_tilt_norm_deg": float(np.hypot(tilt_deg[0], tilt_deg[1])),
        "is_success": is_success,
        "is_jammed": is_jammed_or_wedged,
        "is_safe_force": bool(peak_force <= 15.0),
        "force_limit_exceeded": bool(peak_force > 15.0),
        "termination_reason": (
            "success" if is_success else
            "jammed" if bool(info.get("is_jammed", False)) or is_jammed_or_wedged else
            "terminated" if bool(term) else
            "truncated" if bool(trunc) else
            "horizon_reached"
        ),
        "peak_force_n": peak_force,
        "final_depth_mm": final_depth,
        "final_tilt_deg": float(tilts[-1]) if tilts else 0.0,
        "episode_duration_s": float((step + 1) * env.dt),
        "steps": int(step + 1),
        "mean_solve_time_ms": float(np.mean(solve_times)) if solve_times else 0.0,
        "mean_stick_ratio": float(np.mean(stick_ratios)) if stick_ratios else 1.0,
        "min_stick_ratio": float(np.min(stick_ratios)) if stick_ratios else 1.0,
        "min_tank_energy_j": float(np.min(tank_energies)) if tank_energies else 1.0,
        "forces_trace": np.array(forces, dtype=np.float32),
        "depths_trace": np.array(depths, dtype=np.float32),
    }


def run_benchmark_a(out_dir: str) -> Dict[str, Any]:
    """Executes Table I: 50-trial OOD sweep across all 4 controllers."""
    print("=" * 84)
    print(f"  BENCHMARK A (TABLE I): {len(OOD_SCENARIOS)}-TRIAL OOD MISALIGNMENT SWEEP (+-3° to +-15°)")
    print("=" * 84)

    env = FrankaSlotEnv(seed=42)
    controllers = {
        "unconstrained_farm": (
            "Unconstrained Diffusion Policy (FARM)",
            UnconstrainedPolicyBaseline(dt=env.dt),
        ),
        "wrist_ft_admittance": (
            "Wrist F/T Admittance Filter",
            WristFTAdmittanceController(dt=env.dt),
        ),
        "fixed_impedance_k200": (
            "Fixed Isotropic Impedance (K=200)",
            FixedIsotropicImpedanceController(k_trans=200.0, dt=env.dt),
        ),
        "tactile_safety_filter_ours": (
            "Tactile Safety Filter + FARM (Ours)",
            TactileAlignmentController(f_max=15.0, gamma_jam=2.0, dt=env.dt),
        ),
    }

    summary: Dict[str, Any] = {}
    npz_payload: Dict[str, np.ndarray] = {}

    for key, (label, ctrl) in controllers.items():
        print(f"\n>>> Evaluating Controller: {label} ({len(OOD_SCENARIOS)} OOD trials)...")
        episodes = []
        for idx, (tilt, offset) in enumerate(OOD_SCENARIOS):
            ep = run_single_episode(
                env, ctrl, tilt, offset, max_steps=550, use_action_chunk_policy=True, trial_seed=42 + idx
            )
            episodes.append(ep)
            if idx == 24:  # (+10.0, 0.0) trial
                npz_payload[f"{key}_force_10deg"] = ep["forces_trace"]
                npz_payload[f"{key}_depth_10deg"] = ep["depths_trace"]

        stats = summarize_episode_set(episodes)
        peak_forces = [e["peak_force_n"] for e in episodes]
        succ_rate = stats["success_rate_pct"]
        jam_rate = stats["jamming_rate_pct"]
        safe_rate = stats["safe_force_rate_pct"]
        mean_pf = stats["mean_peak_force_n"]
        std_pf = stats["std_peak_force_n"]
        max_pf = stats["max_peak_force_n"]

        summary[key] = {
            "label": label,
            **stats,
            "trials": [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items() if not k.endswith("_trace")}
                for e in episodes
            ],
        }

        npz_payload[f"{key}_peak_forces"] = np.array(peak_forces, dtype=np.float32)
        npz_payload[f"{key}_tilts"] = np.array([e["initial_tilt_norm_deg"] for e in episodes], dtype=np.float32)

        print(
            f"  [SUMMARY] Success: {succ_rate:5.1f}% | Safe (<=15N): {safe_rate:5.1f}% | "
            f"Peak Force: {mean_pf:5.2f} ± {std_pf:4.2f} N (Max {max_pf:5.2f} N) | "
            f"Jamming: {jam_rate:5.1f}% | Successful cycle time: "
            f"{stats['mean_successful_cycle_time_s'] if stats['mean_successful_cycle_time_s'] is not None else float('nan'):4.2f}s | "
            f"Mean episode duration: {stats['mean_episode_duration_s']:4.2f}s"
        )

    json_path = os.path.join(out_dir, "benchmark_a_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    npz_path = os.path.join(out_dir, "phase3_trajectories.npz")
    np.savez_compressed(npz_path, **npz_payload)
    return summary


def run_benchmarks_bc_and_ablations(out_dir: str) -> Dict[str, Any]:
    """Executes Benchmarks B & C (N = 20 trials each) and the Component Ablation Study (N = 20 trials)."""
    print("\n" + "=" * 84)
    print("  BENCHMARKS B & C (N=20 EACH) + COMPONENT ABLATION STUDY (TABLE II & TABLE III)")
    print("=" * 84)

    # 1. Benchmark B: Non-Planar Stiff Contour Tracking (20 phase-shifted trials in MuJoCo)
    contour_env = NonPlanarContourEnv(stiffness_env=1.0e4, seed=42)
    phases = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    modes_b = [
        ("unconstrained_farm", "Unconstrained Diffusion Policy (FARM)"),
        ("fixed_impedance_k200", "Fixed Isotropic Impedance (K=200)"),
        ("wrist_ft_admittance", "Wrist F/T Admittance Filter"),
        ("tactai_no_tank", "Ablation: w/o Energy Tank T(t)"),
        ("tactai_ours", "Tactile Safety Filter + FARM (Ours)"),
    ]
    bench_b_summary: Dict[str, Any] = {}
    bench_b_traces: Dict[str, np.ndarray] = {}
    print("\n>>> [Benchmark B] Non-Planar Stiff Contour Tracking (N=20, K_env = 10^4 N/m, v = 0.05 m/s, 25 ms delay):")
    for mode_key, label in modes_b:
        trials_b = [contour_env.run_trial(controller_mode=mode_key, phase_shift=float(ph), latency_ms=25.0) for ph in phases]
        bench_b_traces[mode_key] = trials_b[0]["forces_trace"]
        bench_b_summary[mode_key] = {
            "label": label,
            "num_trials": len(trials_b),
            "success_rate_pct": round(100.0 * float(np.mean([t["is_success"] for t in trials_b])), 1),
            "safe_rate_pct": round(100.0 * float(np.mean([t["is_safe"] for t in trials_b])), 1),
            "mean_peak_force_n": round(float(np.mean([t["peak_force_n"] for t in trials_b])), 2),
            "std_peak_force_n": round(float(np.std([t["peak_force_n"] for t in trials_b])), 2),
            "mean_rmse_force_n": round(float(np.mean([t["rmse_force_n"] for t in trials_b])), 2),
            "mean_chatter_reversals": round(float(np.mean([t["chatter_reversals"] for t in trials_b])), 1),
        }
        r = bench_b_summary[mode_key]
        print(
            f"  {label:<38} | Succ: {r['success_rate_pct']:5.1f}% | "
            f"Peak Force: {r['mean_peak_force_n']:5.2f} ± {r['std_peak_force_n']:4.2f} N | "
            f"Force RMSE: {r['mean_rmse_force_n']:4.2f} N | Chatter: {r['mean_chatter_reversals']:4.1f}"
        )

    # 2. Benchmark C: Dynamic Extrinsic Manipulation under Incipient Slip (20 friction trials in MuJoCo)
    slip_env = IncipientSlipManipulationEnv(seed=42)
    mu_list = np.linspace(0.22, 0.58, 20)
    modes_c = [
        ("unconstrained_farm", "Unconstrained Diffusion Policy (FARM)"),
        ("fixed_impedance_k200", "Fixed Isotropic Impedance (K=200)"),
        ("wrist_ft_admittance", "Wrist F/T Admittance Filter"),
        ("tactai_no_slip_cbf", "Ablation: w/o Incipient Slip CBF h_slip"),
        ("tactai_ours", "Tactile Safety Filter + FARM (Ours)"),
    ]
    bench_c_summary: Dict[str, Any] = {}
    bench_c_traces: Dict[str, np.ndarray] = {}
    print("\n>>> [Benchmark C] Dynamic Tool Manipulation under Incipient Slip (N=20, mu in [0.22, 0.58]):")
    for mode_key, label in modes_c:
        trials_c = [slip_env.run_trial(controller_mode=mode_key, mu_friction=float(m), peak_disturbance_n=3.1) for m in mu_list]
        bench_c_traces[f"{mode_key}_slip"] = np.mean([t["slip_trace_mm"] for t in trials_c], axis=0)
        bench_c_traces[f"{mode_key}_grip"] = np.mean([t["grip_trace_n"] for t in trials_c], axis=0)
        bench_c_summary[mode_key] = {
            "label": label,
            "num_trials": len(trials_c),
            "success_rate_pct": round(100.0 * float(np.mean([t["is_success"] for t in trials_c])), 1),
            "drop_rate_pct": round(100.0 * float(np.mean([t["is_dropped"] for t in trials_c])), 1),
            "mean_slip_mm": round(float(np.mean([t["total_slip_mm"] for t in trials_c])), 2),
            "std_slip_mm": round(float(np.std([t["total_slip_mm"] for t in trials_c])), 2),
            "mean_peak_grip_n": round(float(np.mean([t["peak_grip_force_n"] for t in trials_c])), 2),
            "min_stick_ratio": round(float(np.min([t["min_stick_ratio"] for t in trials_c])), 2),
        }
        r = bench_c_summary[mode_key]
        print(
            f"  {label:<38} | Succ: {r['success_rate_pct']:5.1f}% | "
            f"Drop Rate: {r['drop_rate_pct']:5.1f}% | Mean Slip: {r['mean_slip_mm']:5.2f} ± {r['std_slip_mm']:4.2f} mm | "
            f"Peak Grip: {r['mean_peak_grip_n']:5.2f} N"
        )

    # 3. Component Ablation Study on Benchmark A (N = 20 OOD Insertion Trials, 6 deg to 11 deg)
    print("\n>>> [Ablation Study] Evaluating Component Ablations on Benchmark A (N=20 OOD Trials):")
    env_a = FrankaSlotEnv(seed=42)
    ablation_configs = [
        ("full_tactai", "Full TactAI Filter (All Modules Active)", dict(enable_tactile_rot=True, enable_unwedge_cbf=True, enable_energy_tank=True, enable_slip_cbf=True)),
        ("no_tactile_rot", "w/o Differential CoP & Tensor C", dict(enable_tactile_rot=False, enable_unwedge_cbf=True, enable_energy_tank=True, enable_slip_cbf=True)),
        ("no_unwedge_cbf", "w/o Stage 2 Anti-Jamming CBF h_jam", dict(enable_tactile_rot=True, enable_unwedge_cbf=False, enable_energy_tank=True, enable_slip_cbf=True)),
        ("no_energy_tank", "w/o Energy Tank Passivity T(t)", dict(enable_tactile_rot=True, enable_unwedge_cbf=True, enable_energy_tank=False, enable_slip_cbf=True)),
    ]
    ablation_summary: Dict[str, Any] = {}
    subset_ood = OOD_SCENARIOS[10:30]  # 20 moderate-to-severe OOD tilts (6 deg to 10 deg)
    for ab_key, ab_label, kwargs in ablation_configs:
        ctrl_ab = TactileAlignmentController(f_max=15.0, gamma_jam=2.0, dt=env_a.dt, latency_ms=25.0, **kwargs)
        eps_ab = [
            run_single_episode(env_a, ctrl_ab, t, o, max_steps=520, use_action_chunk_policy=True, trial_seed=100 + i)
            for i, (t, o) in enumerate(subset_ood)
        ]
        ablation_summary[ab_key] = {
            "label": ab_label,
            "num_trials": len(eps_ab),
            "insertion_success_pct": round(100.0 * float(np.mean([e["is_success"] for e in eps_ab])), 1),
            "peak_force_n": round(float(np.mean([e["peak_force_n"] for e in eps_ab])), 2),
            "std_peak_force_n": round(float(np.std([e["peak_force_n"] for e in eps_ab])), 2),
            "jamming_rate_pct": round(100.0 * float(np.mean([e["is_jammed"] for e in eps_ab])), 1),
        }
        r = ablation_summary[ab_key]
        print(
            f"  {ab_label:<44} | Insertion Succ: {r['insertion_success_pct']:5.1f}% | "
            f"Peak Force: {r['peak_force_n']:5.2f} ± {r['std_peak_force_n']:4.2f} N"
        )

    payload = {
        "benchmark_b_contour": bench_b_summary,
        "benchmark_c_slip": bench_c_summary,
        "ablation_study": ablation_summary,
        "_traces_b": bench_b_traces,
        "_traces_c": bench_c_traces,
    }
    json_clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    with open(os.path.join(out_dir, "benchmarks_bc_ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_clean, f, indent=2)
    return payload


def run_table2_robustness(out_dir: str, n_seeds: int = 5) -> Dict[str, Any]:
    """Paired robustness sweeps: each scenario is repeated with independent trial seeds.

    Each parameter point uses every scenario in ROBUSTNESS_SCENARIOS crossed with
    n_seeds. All controllers receive the same scenario/seed pairs. The seed is passed
    to env.reset and the stochastic nominal policy; stochasticity internal to sensor
    objects is only controlled if those sensor implementations use the environment's
    seeded RNG or expose their own reset API.
    """
    print("\n" + "=" * 84)
    print(f"  ROBUSTNESS SWEEPS ({len(ROBUSTNESS_SCENARIOS)} scenarios x {n_seeds} seeds per point)")
    print("=" * 84)

    # These are scenario identities, not mislabeled random seeds.
    stiffnesses = [1.0e2, 1.0e3, 1.0e4, 1.0e5]
    frictions = [0.15, 0.30, 0.50, 0.85]
    latencies = [10.0, 20.0, 35.0, 50.0]
    base_seed = 42000

    def run_paired(env: FrankaSlotEnv, controller: Any, controller_key: str) -> List[Dict[str, Any]]:
        episodes = []
        for scenario_idx, (tilt, offset) in enumerate(ROBUSTNESS_SCENARIOS):
            for seed_idx in range(n_seeds):
                # Same seed for the same scenario across controllers; different seed
                # for each repeat of that scenario.
                trial_seed = base_seed + scenario_idx * 1000 + seed_idx
                ep = run_single_episode(
                    env, controller, tilt, offset, max_steps=550,
                    use_action_chunk_policy=True, trial_seed=trial_seed,
                )
                ep["scenario_id"] = f"robust_{scenario_idx:02d}"
                ep["repeat_index"] = seed_idx
                ep["controller_key"] = controller_key
                episodes.append(ep)
        return episodes

    def metric_fields(prefix: str, eps: List[Dict[str, Any]]) -> Dict[str, Any]:
        st = summarize_episode_set(eps)
        return {
            f"{prefix}_num_trials": st["num_trials"],
            f"{prefix}_success_count": st["success_count"],
            f"{prefix}_success_rate_pct": st["success_rate_pct"],
            f"{prefix}_success_rate_wilson_95_ci_pct": st["success_rate_wilson_95_ci_pct"],
            f"{prefix}_peak_force_n": st["mean_peak_force_n"],
            f"{prefix}_std_force_n": st["std_peak_force_n"],
            f"{prefix}_max_force_n": st["max_peak_force_n"],
            f"{prefix}_safe_force_rate_pct": st["safe_force_rate_pct"],
            f"{prefix}_safe_force_wilson_95_ci_pct": st["safe_force_rate_wilson_95_ci_pct"],
            f"{prefix}_jamming_rate_pct": st["jamming_rate_pct"],
            f"{prefix}_episode_duration_s": st["mean_episode_duration_s"],
            f"{prefix}_successful_cycle_time_s": st["mean_successful_cycle_time_s"],
        }

    stiffness_results = []
    for k_env in stiffnesses:
        env_k = FrankaSlotEnv(stiffness_env=k_env, seed=base_seed)
        controllers = {
            "baseline": UnconstrainedPolicyBaseline(dt=env_k.dt),
            "wrist_ft": WristFTAdmittanceController(dt=env_k.dt),
            "tactai": TactileAlignmentController(f_max=15.0, gamma_jam=2.0, dt=env_k.dt),
        }
        eps = {key: run_paired(env_k, ctrl, key) for key, ctrl in controllers.items()}
        row = {"stiffness_n_per_m": float(k_env), "num_scenarios": len(ROBUSTNESS_SCENARIOS), "seeds_per_scenario": n_seeds}
        for key, trials in eps.items():
            row.update(metric_fields(key, trials))
        row["tactai_min_tank_j"] = round(float(min(e["min_tank_energy_j"] for e in eps["tactai"])), 4)
        row["episode_results"] = {key: [{k: v for k, v in e.items() if not k.endswith("_trace")} for e in trials] for key, trials in eps.items()}
        stiffness_results.append(row)
        print(f"  K={k_env:7.0f}: TactAI success {row['tactai_success_rate_pct']:.1f}% ({row['tactai_success_count']}/{row['tactai_num_trials']}), peak force {row['tactai_peak_force_n']:.2f} ± {row['tactai_std_force_n']:.2f} N")

    friction_results = []
    for mu in frictions:
        env_mu = FrankaSlotEnv(friction_env=mu, seed=base_seed)
        controllers = {
            "baseline": UnconstrainedPolicyBaseline(dt=env_mu.dt),
            "tactai": TactileAlignmentController(f_max=15.0, gamma_jam=2.0, dt=env_mu.dt),
        }
        eps = {key: run_paired(env_mu, ctrl, key) for key, ctrl in controllers.items()}
        row = {"friction_mu": float(mu), "num_scenarios": len(ROBUSTNESS_SCENARIOS), "seeds_per_scenario": n_seeds}
        for key, trials in eps.items():
            row.update(metric_fields(key, trials))
        row["tactai_mean_stick_ratio"] = round(float(np.mean([e["mean_stick_ratio"] for e in eps["tactai"]])), 4)
        row["episode_results"] = {key: [{k: v for k, v in e.items() if not k.endswith("_trace")} for e in trials] for key, trials in eps.items()}
        friction_results.append(row)
        print(f"  mu={mu:.2f}: TactAI success {row['tactai_success_rate_pct']:.1f}% ({row['tactai_success_count']}/{row['tactai_num_trials']}), peak force {row['tactai_peak_force_n']:.2f} ± {row['tactai_std_force_n']:.2f} N")

    latency_results = []
    env_lat = FrankaSlotEnv(seed=base_seed)
    for latency in latencies:
        controllers = {
            "baseline": UnconstrainedPolicyBaseline(dt=env_lat.dt, latency_ms=latency),
            "wrist_ft": WristFTAdmittanceController(dt=env_lat.dt, latency_ms=latency),
            "tactai": TactileAlignmentController(f_max=15.0, gamma_jam=2.0, dt=env_lat.dt, latency_ms=latency),
        }
        eps = {key: run_paired(env_lat, ctrl, key) for key, ctrl in controllers.items()}
        row = {"latency_ms": float(latency), "num_scenarios": len(ROBUSTNESS_SCENARIOS), "seeds_per_scenario": n_seeds}
        for key, trials in eps.items():
            row.update(metric_fields(key, trials))
        row["tactai_min_tank_j"] = round(float(min(e["min_tank_energy_j"] for e in eps["tactai"])), 4)
        row["episode_results"] = {key: [{k: v for k, v in e.items() if not k.endswith("_trace")} for e in trials] for key, trials in eps.items()}
        latency_results.append(row)
        print(f"  latency={latency:4.1f} ms: TactAI success {row['tactai_success_rate_pct']:.1f}% ({row['tactai_success_count']}/{row['tactai_num_trials']}), peak force {row['tactai_peak_force_n']:.2f} ± {row['tactai_std_force_n']:.2f} N")

    robustness_summary = {
        "protocol": {
            "n_seeds_per_scenario": n_seeds,
            "scenario_count": len(ROBUSTNESS_SCENARIOS),
            "episodes_per_controller_per_parameter_point": n_seeds * len(ROBUSTNESS_SCENARIOS),
            "seed_scope": "Passed to FrankaSlotEnv.reset and StochasticActionChunkingPolicy; sensor RNG reset behavior depends on GelSightSensor implementation.",
            "confidence_interval": "95% Wilson score interval for binary success and force-safety rates; standard deviation is descriptive, not a confidence interval.",
        },
        "stiffness_sweep": stiffness_results,
        "friction_sweep": friction_results,
        "latency_sweep": latency_results,
    }
    with open(os.path.join(out_dir, "table2_robustness_summary.json"), "w", encoding="utf-8") as f:
        json.dump(robustness_summary, f, indent=2)
    return robustness_summary


def generate_phase3_figure(
    bench_a: Dict[str, Any],
    table2: Dict[str, Any],
    fig_path: str
) -> None:
    """Generate benchmark and robustness figures; force SD bands are descriptive."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=220)
    plt.subplots_adjust(hspace=0.34, wspace=0.26)

    keys = [
        "unconstrained_farm",
        "fixed_impedance_k200",
        "wrist_ft_admittance",
        "tactile_safety_filter_ours",
    ]
    short_labels = [
        "Unconstrained\nPolicy (FARM)",
        "Fixed Impedance\n($K=200$ N/m)",
        "Wrist F/T\nAdmittance",
        "TactAI Filter\n+ FARM (Ours)",
    ]

    # Panel A: Success Rate vs. Jamming Rate (%)
    ax = axes[0, 0]
    x = np.arange(len(keys))
    width = 0.36
    succ_vals = [bench_a[k]["success_rate_pct"] for k in keys]
    jam_vals = [bench_a[k]["jamming_rate_pct"] for k in keys]

    bars1 = ax.bar(x - width / 2, succ_vals, width, label="Insertion Success (%)", color="#10B981", edgecolor="#065F46")
    bars2 = ax.bar(x + width / 2, jam_vals, width, label="Jamming / Wedging (%)", color="#EF4444", edgecolor="#991B1B")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Rate (%)", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_title("(A) Benchmark A: Success vs. Jamming Rate ($N=50$ OOD Trials)", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=7.8, fontweight="bold", color="#065F46")
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=7.8, fontweight="bold", color="#991B1B")

    # Panel B: Peak Contact Force across OOD Tilt Angles (with +-1 sigma error bands)
    ax = axes[0, 1]
    colors_map = {
        "unconstrained_farm": ("#DC2626", "o-", "Unconstrained Policy (FARM)"),
        "fixed_impedance_k200": ("#D97706", "s--", "Fixed Impedance ($K=200$)"),
        "wrist_ft_admittance": ("#2563EB", "^-.", "Wrist F/T Admittance"),
        "tactile_safety_filter_ours": ("#059669", "D-", "TactAI Safety Filter (Ours)"),
    }
    for k in keys:
        trials = bench_a[k]["trials"]
        tilt_bins = sorted(list({int(round(t["initial_tilt_norm_deg"])) for t in trials}))
        mean_f = np.array([
            np.mean([t["peak_force_n"] for t in trials if int(round(t["initial_tilt_norm_deg"])) == tb])
            for tb in tilt_bins
        ])
        col, style, lbl = colors_map[k]
        # This panel is a descriptive mean by rounded tilt norm, not a CI plot:
        # several bins contain only one deliberately selected scenario.
        ax.plot(tilt_bins, mean_f, style, color=col, linewidth=1.8, markersize=4.5, label=lbl)

    ax.axhline(15.0, color="#059669", linestyle="--", linewidth=1.2, label="CBF Safety Limit (15 N)")
    ax.axhline(40.0, color="#DC2626", linestyle=":", linewidth=1.2, label="Jamming Failure (40 N)")
    ax.set_xlabel(r"Initial 2-Axis OOD Tilt Norm $\|\theta_0\|_2$ (deg)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Peak Contact Force (N)", fontsize=9.5, fontweight="bold")
    ax.set_title(r"(B) Mean Peak Contact Force by Tilt-Norm Bin (descriptive)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 54)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.5)

    # Panel C: Environmental Stiffness Robustness (with +-1 sigma error bands)
    ax = axes[1, 0]
    stiff_rows = table2["stiffness_sweep"]
    k_vals = np.array([r["stiffness_n_per_m"] for r in stiff_rows])
    b_k = np.array([r["baseline_peak_force_n"] for r in stiff_rows])
    b_k_std = np.array([r["baseline_std_force_n"] for r in stiff_rows])
    w_k = np.array([r["wrist_ft_peak_force_n"] for r in stiff_rows])
    w_k_std = np.array([r["wrist_ft_std_force_n"] for r in stiff_rows])
    t_k = np.array([r["tactai_peak_force_n"] for r in stiff_rows])
    t_k_std = np.array([r["tactai_std_force_n"] for r in stiff_rows])

    ax.plot(k_vals, b_k, "o-", color="#DC2626", linewidth=2.0, label="Unconstrained Policy (Jammed)")
    ax.fill_between(k_vals, b_k - b_k_std, b_k + b_k_std, color="#DC2626", alpha=0.15)
    ax.plot(k_vals, w_k, "^-.", color="#2563EB", linewidth=2.0, label="Wrist F/T Admittance")
    ax.fill_between(k_vals, w_k - w_k_std, w_k + w_k_std, color="#2563EB", alpha=0.15)
    ax.plot(k_vals, t_k, "D-", color="#059669", linewidth=2.2, label="TactAI Filter")
    ax.fill_between(k_vals, t_k - t_k_std, t_k + t_k_std, color="#059669", alpha=0.18)
    ax.axhline(15.0, color="#059669", linestyle="--", linewidth=1.2, label="CBF Force Limit (15 N)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Environment Stiffness $K_{\mathrm{env}}$ (N/m)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Peak Contact Force (N)", fontsize=9.5, fontweight="bold")
    ax.set_title(r"(C) Stiffness Robustness ($10^2$ to $10^5$ N/m, $N=25$ Episodes, 5 scenarios × 5 seeds)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 54)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="center right", fontsize=8)

    # Panel D: Tactile Perception Latency Sweep (10 ms to 50 ms, with +-1 sigma error bands)
    ax = axes[1, 1]
    lat_rows = table2["latency_sweep"]
    lat_vals = np.array([r["latency_ms"] for r in lat_rows])
    b_l = np.array([r["baseline_peak_force_n"] for r in lat_rows])
    b_l_std = np.array([r["baseline_std_force_n"] for r in lat_rows])
    w_l = np.array([r["wrist_ft_peak_force_n"] for r in lat_rows])
    w_l_std = np.array([r["wrist_ft_std_force_n"] for r in lat_rows])
    t_l = np.array([r["tactai_peak_force_n"] for r in lat_rows])
    t_l_std = np.array([r["tactai_std_force_n"] for r in lat_rows])

    ax.plot(lat_vals, b_l, "o-", color="#DC2626", linewidth=2.0, label="Unconstrained Policy")
    ax.fill_between(lat_vals, b_l - b_l_std, b_l + b_l_std, color="#DC2626", alpha=0.15)
    ax.plot(lat_vals, w_l, "^-.", color="#2563EB", linewidth=2.0, label="Wrist F/T Admittance")
    ax.fill_between(lat_vals, w_l - w_l_std, w_l + w_l_std, color="#2563EB", alpha=0.15)
    ax.plot(lat_vals, t_l, "D-", color="#059669", linewidth=2.2, label="TactAI Filter + Energy Tank (Ours)")
    ax.fill_between(lat_vals, t_l - t_l_std, t_l + t_l_std, color="#059669", alpha=0.18)
    ax.axhline(15.0, color="#059669", linestyle="--", linewidth=1.2, label="CBF Safety Limit (15 N)")
    ax.set_xlabel(r"Perception Latency $\tau_{\mathrm{delay}}$ (ms)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Peak Contact Force (N)", fontsize=9.5, fontweight="bold")
    ax.set_title(r"(D) Perception Latency Robustness (10–50 ms, $N=25$ Episodes, 5 scenarios × 5 seeds)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 58)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center right", fontsize=8)

    fig.suptitle(
        "TactAI Benchmark A (N=50 scenarios) & Robustness (25 episodes per parameter point)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(fig_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n>>> Saved Benchmark A & Robustness figure to: {fig_path} and {pdf_path}")


def generate_bc_ablation_figure(
    bc_ablation: Dict[str, Any],
    fig_path: str
) -> None:
    """Generates a 3-panel publication figure for Benchmark B, Benchmark C, and Ablation Study."""
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), dpi=300)
    plt.subplots_adjust(wspace=0.33)

    traces_b = bc_ablation["_traces_b"]
    traces_c = bc_ablation["_traces_c"]
    ablation = bc_ablation["ablation_study"]
    b_sum = bc_ablation["benchmark_b_contour"]
    c_sum = bc_ablation["benchmark_c_slip"]

    # Panel A: Benchmark B Contour Tracking Force Trace (generous headroom above traces for legend clarity)
    ax = axes[0]
    t_b = np.arange(len(traces_b["tactai_ours"]))
    ax.plot(t_b, traces_b["unconstrained_farm"], color="#DC2626", linewidth=1.6, alpha=0.85, label=f"Unconstrained Policy (Peak: {b_sum['unconstrained_farm']['mean_peak_force_n']:.1f} N)")
    ax.plot(t_b, traces_b["wrist_ft_admittance"], color="#2563EB", linewidth=1.4, linestyle="-.", alpha=0.80, label=f"Wrist F/T Admittance (Chatter: {b_sum['wrist_ft_admittance']['mean_peak_force_n']:.1f} N)")
    ax.plot(t_b, traces_b["tactai_no_tank"], color="#D97706", linewidth=1.5, linestyle="--", label=f"Ablation: w/o Energy Tank (Peak: {b_sum['tactai_no_tank']['mean_peak_force_n']:.1f} N)")
    ax.plot(t_b, traces_b["tactai_ours"], color="#059669", linewidth=2.2, label=f"TactAI + Tank (Ours, Peak: {b_sum['tactai_ours']['mean_peak_force_n']:.2f} N)")
    ax.axhline(15.0, color="#059669", linestyle=":", linewidth=1.3, label="CBF Safety Bound (15.0 N)")
    ax.axhline(8.0, color="#475569", linestyle="--", linewidth=1.0, label="Target Force (8.0 N)")
    ax.set_xlabel("Simulation sample index (dt set by environment)", fontsize=8.5, fontweight="bold")
    ax.set_ylabel(r"Normal Contact Force $F_n$ (N)", fontsize=9, fontweight="bold")
    ax.set_title("(A) Benchmark B: Non-Planar Stiff Contour ($N=20$)", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(168.0, float(np.max(traces_b["wrist_ft_admittance"])) * 1.28))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=6.9, framealpha=0.94, edgecolor="#CBD5E1")

    # Panel B: Benchmark C Mean Cumulative Slip Trajectory across mu in [0.22, 0.58]
    ax = axes[1]
    t_c = np.arange(len(traces_c["tactai_ours_slip"]))
    ax.plot(t_c, traces_c["unconstrained_farm_slip"], color="#DC2626", linewidth=2.0, label=f"Unconstrained Policy ({c_sum['unconstrained_farm']['mean_slip_mm']:.1f} mm)")
    ax.plot(t_c, traces_c["wrist_ft_admittance_slip"], color="#2563EB", linewidth=1.8, linestyle="-.", label=f"Wrist F/T ({c_sum['wrist_ft_admittance']['mean_slip_mm']:.1f} mm)")
    ax.plot(t_c, traces_c["tactai_no_slip_cbf_slip"], color="#D97706", linewidth=1.8, linestyle="--", label=f"w/o Slip CBF ({c_sum['tactai_no_slip_cbf']['mean_slip_mm']:.1f} mm)")
    ax.plot(t_c, traces_c["tactai_ours_slip"], color="#059669", linewidth=2.3, label=f"TactAI Slip-CBF (Ours: {c_sum['tactai_ours']['mean_slip_mm']:.2f} mm)")
    ax.axhline(2.0, color="#DC2626", linestyle=":", linewidth=1.3, label="Gross Slip / Drop Limit (2.0 mm)")
    ax.set_xlabel("Simulation sample index (dt set by environment)", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Mean Cumulative Tool Slip (mm)", fontsize=9, fontweight="bold")
    ax.set_title("(B) Benchmark C: Extrinsic Shear Slip ($N=20$)", fontsize=9.5, fontweight="bold")
    ax.set_ylim(-0.2, max(6.5, float(np.max(traces_c["unconstrained_farm_slip"])) * 1.15))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=6.9, framealpha=0.94, edgecolor="#CBD5E1")

    # Panel C: Component Ablation Study Bar Chart
    ax = axes[2]
    ab_keys = ["full_tactai", "no_tactile_rot", "no_unwedge_cbf", "no_energy_tank"]
    ab_labels = [
        "Full TactAI\n(Ours)",
        "w/o Diff. CoP\n& Tensor C",
        "w/o Stage 2\nCBF $h_{\\mathrm{jam}}$",
        "w/o Energy\nTank $T(t)$",
    ]
    succs = [ablation[k]["insertion_success_pct"] for k in ab_keys]
    pfs = [ablation[k]["peak_force_n"] for k in ab_keys]
    x = np.arange(len(ab_keys))
    width = 0.34
    b1 = ax.bar(x - width / 2, succs, width, color="#10B981", edgecolor="#065F46", label="Insertion Success (%)")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + width / 2, pfs, width, color="#3B82F6", edgecolor="#1D4ED8", label="Peak Force (N)")
    ax2.axhline(15.0, color="#DC2626", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ab_labels, fontsize=7.4, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=9, fontweight="bold", color="#065F46")
    ax2.set_ylabel("Peak Force (N)", fontsize=9, fontweight="bold", color="#1D4ED8")
    ax.set_ylim(0, 118)
    ax2.set_ylim(0, 42)
    ax.set_title("(C) Component Ablation Study ($N=20$ OOD Trials)", fontsize=9.5, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", fontsize=7.2, fontweight="bold", color="#065F46")
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}N", ha="center", fontsize=7.2, fontweight="bold", color="#1D4ED8")

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(fig_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f">>> Saved Benchmarks B/C & Ablation figure to: {fig_path} and {pdf_path}")


def main():
    t_start = time.perf_counter()
    out_dir = "data/phase3_validation"
    fig_dir = "docs/figures"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if "--fast-update" in sys.argv:
        print("WARNING: --fast-update no longer performs a partial update. Running the full suite so controller results cannot be mixed across scenario sets.")

    # Always recompute every controller and benchmark in one run. This avoids the
    # previous failure mode where only TactAI was rerun and stale baseline JSON remained.
    bench_a = run_benchmark_a(out_dir)
    bc_ablation = run_benchmarks_bc_and_ablations(out_dir)
    table2 = run_table2_robustness(out_dir, n_seeds=5)

    fig1_path = os.path.join(fig_dir, "phase3_benchmark_comparison.png")
    fig2_path = os.path.join(fig_dir, "phase3_benchmarks_bc_ablation.png")
    generate_phase3_figure(bench_a, table2, fig1_path)
    generate_bc_ablation_figure(bc_ablation, fig2_path)

    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "benchmark_script": os.path.basename(__file__),
        "benchmark_a_scenario_count": len(OOD_SCENARIOS),
        "robustness_scenario_count": len(ROBUSTNESS_SCENARIOS),
        "robustness_seeds_per_scenario": 5,
        "notes": [
            "Full suite recomputed; no partial-update mixing.",
            "95% Wilson intervals are reported for binary success and force-safety rates.",
            "Sensor RNG reproducibility must be verified in GelSightSensor implementation.",
            "Simulation results are not hardware-validation results.",
        ],
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    elapsed = time.perf_counter() - t_start
    print(f"\n=== FULL PHASE 3 BENCHMARK & ABLATION SUITE COMPLETED IN {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
