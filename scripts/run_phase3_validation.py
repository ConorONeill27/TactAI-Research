"""
Structured insertion, contour, slip, ablation and environmental robustness benchmarks.

Insertion uses the archived learned checkpoint and frozen development settings.
This driver consumes the archived primary held-out results only when drawing figures.
The contour and slip tasks use independent task-specific references.

Outputs (anchored to PROJECT_ROOT):
- data/phase3_validation/benchmark_a_summary.json
- data/phase3_validation/benchmarks_bc_ablation_summary.json
- data/phase3_validation/table2_robustness_summary.json
- data/phase3_validation/phase3_trajectories.npz
- data/phase3_validation/run_metadata.json
- docs/figures/phase3_benchmark_comparison.png (.pdf)
- docs/figures/phase3_benchmarks_bc_ablation.png (.pdf)
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

# Ensure canonical project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.envs.franka_slot_env import FrankaSlotEnv
from src.controllers.insertion_protocol import insertion_controllers, tactai
from src.envs.tactile_benchmarks_env import NonPlanarContourEnv, IncipientSlipManipulationEnv
from src.controllers.baseline_controllers import UnconstrainedPolicyBaseline


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

# Fixed scenario panel crossed with multiple seeds for paired robustness evaluation
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
    """Keep the numerator, denominator, observed rate, and Wilson 95% CI together."""
    n = len(values)
    successes = int(sum(bool(v) for v in values))
    return {
        "successes": successes,
        "num_trials": n,
        "rate_pct": round(100.0 * successes / n, 2) if n else None,
        "wilson_95_ci_pct": wilson_interval_pct(successes, n),
    }


def summarize_episode_set(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Common, explicit summaries separating failed episode durations from successful cycle times."""
    n = len(episodes)
    successes = [e for e in episodes if e["is_success"]]
    peak = np.asarray([e["peak_force_n"] for e in episodes], dtype=float)
    duration = np.asarray([e.get("episode_duration_s", e.get("cycle_time_s", 0.0)) for e in episodes], dtype=float)
    success_duration = np.asarray([e.get("episode_duration_s", e.get("cycle_time_s", 0.0)) for e in successes], dtype=float)
    success = binary_summary([e["is_success"] for e in episodes])
    jam = binary_summary([e["is_jammed"] for e in episodes])
    safe = binary_summary([e["is_safe_force"] for e in episodes])
    solve_times = [e["mean_solve_time_ms"] for e in episodes if e.get("mean_solve_time_ms") is not None]
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
        "mean_peak_force_n": round(float(np.mean(peak)), 2) if n else None,
        "std_peak_force_n": round(float(np.std(peak, ddof=1)), 2) if n > 1 else 0.0,
        "max_peak_force_n": round(float(np.max(peak)), 2) if n else None,
        "mean_episode_duration_s": round(float(np.mean(duration)), 4) if n else None,
        "std_episode_duration_s": round(float(np.std(duration, ddof=1)), 4) if n > 1 else 0.0,
        "mean_cycle_time_s": round(float(np.mean(success_duration)), 2) if len(success_duration) else (round(float(np.mean(duration)), 2) if n else 0.0),
        "std_cycle_time_s": round(float(np.std(success_duration, ddof=1)), 2) if len(success_duration) > 1 else 0.0,
        "mean_successful_cycle_time_s": round(float(np.mean(success_duration)), 4) if len(success_duration) else None,
        "std_successful_cycle_time_s": round(float(np.std(success_duration, ddof=1)), 4) if len(success_duration) > 1 else None,
        "num_successful_cycle_times": int(len(success_duration)),
        "mean_solve_time_ms": round(float(np.mean(solve_times)), 4) if solve_times else None,
    }


def run_single_episode(
    env: FrankaSlotEnv,
    controller: Any,
    tilt_deg: Tuple[float, float],
    offset_mm: Tuple[float, float],
    max_steps: int = 550,
    use_action_chunk_policy: bool = True,
    trial_seed: int = 42,
    tactile_transform=None,
    policy_factory=None,
) -> Dict[str, Any]:
    """Runs a single deterministic insertion episode wrapping the 20 Hz nominal policy."""
    obs, info = env.reset(seed=trial_seed, ood_tilt_deg=tilt_deg, ood_offset_mm=offset_mm)
    controller.reset()
    if use_action_chunk_policy:
        if policy_factory is None:
            from src.controllers.learned_policy import LearnedChunkPolicy
            policy_factory = LearnedChunkPolicy
        nom_policy = policy_factory()
        nom_policy.reset(seed=trial_seed)
    u_const = np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0])

    forces: List[float] = []
    depths: List[float] = []
    tilts: List[float] = []
    solve_times: List[float] = []
    stick_ratios: List[float] = []
    tank_energies: List[float] = []
    stages = []
    velocities = []
    first_contact = None
    ever_jammed = False

    for step in range(max_steps):
        u_nominal = nom_policy.get_action(obs) if use_action_chunk_policy else u_const
        controller_obs = tactile_transform(obs, step) if tactile_transform else obs
        u_cmd, diag = controller.compute_safe_action(u_nominal, controller_obs)
        approach_speed = float(-obs["ee_linvel"][2])
        obs, reward, term, trunc, info = env.step(u_cmd)

        forces.append(float(info["current_force_n"]))
        depths.append(float(info["insertion_depth_m"] * 1e3))
        tilts.append(float(info["tilt_deg"]))
        solve_times.append(float(diag.get("solve_time_ms", 0.0)))
        stick_ratios.append(float(obs["tactile"]["stick_ratio"]))
        tank_energies.append(float(diag.get("tank_energy_j", 1.0)))
        stages.append(diag.get("stage"))
        velocities.append(approach_speed)
        if first_contact is None and forces[-1] > 0.1:
            first_contact = step + 1

        if info["is_jammed"]:
            ever_jammed = True

        if info["is_success"] or info["is_jammed"] or term or trunc:
            break

    peak_force = float(max(forces)) if forces else 0.0
    final_depth = float(depths[-1]) if depths else 0.0
    is_success = bool(info["is_success"])
    peak_index = int(np.argmax(forces))
    is_jammed_or_wedged = bool(ever_jammed or (not is_success and peak_force > 22.0 and final_depth < 10.0))

    return {
        "trial_seed": int(trial_seed),
        "final_stage": diag.get("stage"),
        "stage_at_peak": stages[peak_index],
        "first_contact_step": first_contact,
        "approach_speed_at_first_contact_m_s": velocities[first_contact - 1] if first_contact else None,
        "peak_force_step": peak_index + 1,
        "rim_arrest_latched": diag.get("rim_arrest_latched"),
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
        "cycle_time_s": float((step + 1) * env.dt),
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
    """Run the five insertion controllers on the structured development panel."""
    print("=" * 84)
    print(f"  BENCHMARK A (TABLE I): {len(OOD_SCENARIOS)}-TRIAL OOD MISALIGNMENT SWEEP (+-3° to +-15°)")
    print("=" * 84)

    env = FrankaSlotEnv(seed=42)
    controllers = insertion_controllers(env.dt)

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
            f"  [SUMMARY] Success: {stats['success_rate_pct']:5.1f}% | Safe (<=15N): {stats['safe_force_rate_pct']:5.1f}% | "
            f"Peak Force: {stats['mean_peak_force_n']:5.2f} ± {stats['std_peak_force_n']:4.2f} N (Max {stats['max_peak_force_n']:5.2f} N) | "
            f"Jamming: {stats['jamming_rate_pct']:5.1f}% | Cycle Time: {stats['mean_cycle_time_s']:4.2f}s"
        )

    json_path = os.path.join(out_dir, "benchmark_a_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "phase3_trajectories.npz"), **npz_payload)

    return summary


def run_benchmarks_bc_and_ablations(out_dir: str) -> Dict[str, Any]:
    """Executes Benchmark B (Contour), Benchmark C (Slip), and Component Ablations."""
    print("\n" + "=" * 84)
    print("  BENCHMARKS B & C + COMPONENT ABLATIONS (TABLES II & III, N=20 TRIALS EACH)")
    print("=" * 84)

    # -------------------------------------------------------------
    # 1. Benchmark B: Non-Planar Stiff Contour Tracking (N=20)
    # -------------------------------------------------------------
    print("\n>>> [Benchmark B] Non-Planar Stiff Contour Tracking (K_env = 10^4 N/m, 25 ms delay)...")
    contour_env = NonPlanarContourEnv(stiffness_env=1.0e4, seed=42)
    bench_b_ctrls = [
        ("unconstrained_farm", "Unconstrained contour reference"),
        ("fixed_impedance_k200", "Fixed Impedance (K=200)"),
        ("wrist_ft_admittance", "Wrist F/T Admittance"),
        ("tactai_ours", "TactAI Safety Filter (Ours)"),
        ("tactai_no_tank", "TactAI without Energy Tank"),
    ]
    bench_b_summary: Dict[str, Any] = {}
    bench_b_traces: Dict[str, np.ndarray] = {}

    phase_shifts = np.linspace(0.0, np.pi, 20)
    for mode_key, label in bench_b_ctrls:
        trials = [
            contour_env.run_trial(controller_mode=mode_key, phase_shift=p, latency_ms=25.0)
            for p in phase_shifts
        ]
        safe_tracking = float(np.mean([t.get("safe_tracking", t.get("is_safe", False)) for t in trials])) * 100.0
        force_rmse = float(np.mean([t.get("force_rmse", t.get("rmse_force_n", 0.0)) for t in trials]))
        peak_force = float(np.max([t.get("peak_force", t.get("peak_force_n", 0.0)) for t in trials]))
        chatter = bool(np.any([t.get("chatter_detected", t.get("chatter_reversals", 0) > 5) for t in trials]))

        bench_b_summary[mode_key] = {
            "label": label,
            "num_trials": len(trials),
            "safe_tracking_pct": round(safe_tracking, 1),
            "force_rmse_n": round(force_rmse, 2),
            "peak_force_n": round(peak_force, 2),
            "mean_peak_force_n": round(float(np.mean([t["peak_force_n"] for t in trials])), 2),
            "std_peak_force_n": round(float(np.std([t["peak_force_n"] for t in trials], ddof=1)), 2),
            "chatter_detected": chatter,
            "trials": [
                {"phase_rad": float(phase), "seed": contour_env.seed,
                 **{key: value for key, value in trial.items() if not isinstance(value, np.ndarray)}}
                for phase, trial in zip(phase_shifts, trials)
            ],
        }
        bench_b_traces[mode_key] = trials[0]["forces_trace"]
        print(f"  {label:35s} -> Safe: {safe_tracking:5.1f}% | RMSE: {force_rmse:5.2f} N | Peak: {peak_force:6.2f} N | Chatter: {chatter}")

    # -------------------------------------------------------------
    # 2. Benchmark C: Dynamic Tool Manipulation under Incipient Slip (N=20)
    # -------------------------------------------------------------
    print("\n>>> [Benchmark C] Incipient Slip Dynamic Tool Manipulation (mu in [0.22, 0.58])...")
    slip_env = IncipientSlipManipulationEnv(seed=42)
    bench_c_ctrls = [
        ("unconstrained_farm", "Unconstrained manipulation reference"),
        ("fixed_impedance_k200", "Fixed Impedance (K=200)"),
        ("wrist_ft_admittance", "Wrist F/T Admittance"),
        ("tactai_ours", "TactAI Slip-CBF (Ours)"),
    ]
    bench_c_summary: Dict[str, Any] = {}
    bench_c_traces: Dict[str, np.ndarray] = {}
    mu_list = np.linspace(0.22, 0.58, 20)

    for mode_key, label in bench_c_ctrls:
        trials = [
            slip_env.run_trial(controller_mode=mode_key, mu_friction=float(mu), peak_disturbance_n=3.1)
            for mu in mu_list
        ]
        drop_rate = float(np.mean([t.get("drop_occurred", t.get("is_dropped", False)) for t in trials])) * 100.0
        mean_slip = float(np.mean([t.get("final_slip_mm", t.get("total_slip_mm", 0.0)) for t in trials]))
        max_slip = float(np.max([t.get("final_slip_mm", t.get("total_slip_mm", 0.0)) for t in trials]))
        mean_grip = float(np.mean([t.get("mean_grip_force_n", t.get("peak_grip_force_n", 0.0)) for t in trials]))

        bench_c_summary[mode_key] = {
            "label": label,
            "num_trials": len(trials),
            "drop_rate_pct": round(drop_rate, 1),
            "success_rate_pct": round(100.0 - drop_rate, 1),
            "mean_slip_mm": round(mean_slip, 2),
            "max_slip_mm": round(max_slip, 2),
            "mean_grip_force_n": round(mean_grip, 2),
            "trials": [
                {"friction": float(mu), "seed": slip_env.seed,
                 **{key: value for key, value in trial.items() if not isinstance(value, np.ndarray)}}
                for mu, trial in zip(mu_list, trials)
            ],
        }
        bench_c_traces[f"{mode_key}_slip"] = np.mean([t["slip_trace_mm"] for t in trials], axis=0)
        bench_c_traces[f"{mode_key}_grip"] = np.mean([t["grip_trace_n"] for t in trials], axis=0)
        print(f"  {label:35s} -> Drop Rate: {drop_rate:5.1f}% | Slip: {mean_slip:6.2f} mm | Grip: {mean_grip:5.2f} N")

    # -------------------------------------------------------------
    # 3. Component Ablation Study (Table III, N=20 trials each)
    # -------------------------------------------------------------
    print("\n>>> [Ablations] 4 Component Ablations on OOD Benchmark A Subset (N=20)...")
    ablation_scenarios = OOD_SCENARIOS[15:35]  # Moderate to severe OOD (+-7 deg to +-11 deg)
    env = FrankaSlotEnv(seed=42)

    ablations = [
        ("tactai_full", "Full TactAI", tactai(env.dt)),
        ("tactai_no_cop_tensor", "Without proportional rotational alignment",
         tactai(env.dt, enable_tactile_rot=False)),
        ("tactai_no_jam_cbf", "Without Stage 2", tactai(env.dt, enable_unwedge_cbf=False)),
        ("tactai_no_tank", "Without energy tank", tactai(env.dt, enable_energy_tank=False)),
    ]

    ablation_summary: Dict[str, Any] = {}
    for ab_key, ab_label, ctrl in ablations:
        episodes = [
            run_single_episode(env, ctrl, tilt, offset, max_steps=550, use_action_chunk_policy=True, trial_seed=100 + i)
            for i, (tilt, offset) in enumerate(ablation_scenarios)
        ]
        stats = summarize_episode_set(episodes)
        ablation_summary[ab_key] = {
            "label": ab_label,
            **stats,
            "trials": [{k: v for k, v in trial.items() if not k.endswith("_trace")}
                       for trial in episodes],
        }
        print(f"  {ab_label:45s} -> Succ: {stats['success_rate_pct']:5.1f}% | Peak: {stats['mean_peak_force_n']:5.2f} ± {stats['std_peak_force_n']:4.2f} N | Jam: {stats['jamming_rate_pct']:5.1f}%")

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
    np.savez_compressed(os.path.join(out_dir, "benchmark_bc_plot_traces.npz"),
                        **{f"contour_{key}": trace for key, trace in bench_b_traces.items()},
                        **bench_c_traces)

    return payload


ROBUSTNESS_POINTS = {
    "stiffness_n_per_m": (1.0e2, 1.0e3, 1.0e4, 1.0e5),
    "friction_mu": (0.15, 0.30, 0.50, 0.85),
    "latency_ms": (10.0, 20.0, 35.0, 50.0),
}


def run_robustness_point(task):
    field, value, repeats = task
    base_seed = 42000
    env_args = {
        "stiffness_n_per_m": {"stiffness_env": value},
        "friction_mu": {"friction_env": value},
        "latency_ms": {},
    }
    env = FrankaSlotEnv(seed=base_seed, **env_args[field])
    # Delay only the supervisory sensor path. The learned policy uses live proprioception.
    latency = value if field == "latency_ms" else 0.0
    controllers = {
        "baseline": UnconstrainedPolicyBaseline(dt=env.dt),
        "tactai": tactai(env.dt, latency_ms=latency),
    }
    if field != "friction_mu":
        controllers["wrist_ft"] = tactai(env.dt, latency_ms=latency, wrist_ft_only=True)
    row = {field: value, "num_scenarios": len(ROBUSTNESS_SCENARIOS),
           "seeds_per_scenario": repeats, "episode_results": {}}
    fields = {
        "num_trials": "num_trials", "success_count": "success_count",
        "success_rate_pct": "success_rate_pct",
        "success_rate_wilson_95_ci_pct": "success_rate_wilson_95_ci_pct",
        "peak_force_n": "mean_peak_force_n", "std_force_n": "std_peak_force_n",
        "max_force_n": "max_peak_force_n", "safe_force_rate_pct": "safe_force_rate_pct",
        "safe_force_wilson_95_ci_pct": "safe_force_rate_wilson_95_ci_pct",
        "jamming_rate_pct": "jamming_rate_pct", "episode_duration_s": "mean_episode_duration_s",
        "successful_cycle_time_s": "mean_successful_cycle_time_s",
    }
    for key, controller in controllers.items():
        trials = []
        for i, (tilt, offset) in enumerate(ROBUSTNESS_SCENARIOS):
            for j in range(repeats):
                trial = run_single_episode(env, controller, tilt, offset,
                                           trial_seed=base_seed + i * 1000 + j)
                trials.append(dict(scenario_id=i, repeat_index=j,
                                   **{k: v for k, v in trial.items() if not k.endswith("_trace")}))
        row["episode_results"][key] = trials
        summary = summarize_episode_set(trials)
        row.update({f"{key}_{name}": summary[metric] for name, metric in fields.items()})
    return field, row


def run_table2_robustness(out_dir: str, n_seeds: int = 5) -> Dict[str, Any]:
    """Run the paired stress cases across independent MuJoCo processes."""
    from concurrent.futures import ProcessPoolExecutor

    output_keys = {"stiffness_n_per_m": "stiffness_sweep", "friction_mu": "friction_sweep",
                   "latency_ms": "latency_sweep"}
    summary = {key: [] for key in output_keys.values()}
    tasks = [(field, value, n_seeds) for field, values in ROBUSTNESS_POINTS.items()
             for value in values]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for field, row in pool.map(run_robustness_point, tasks):
            summary[output_keys[field]].append(row)
            print(field, row[field], row["tactai_success_count"], row["tactai_max_force_n"], flush=True)
    summary["protocol"] = {
        "n_seeds_per_scenario": n_seeds,
        "scenario_count": len(ROBUSTNESS_SCENARIOS),
        "episodes_per_controller_per_parameter_point": n_seeds * len(ROBUSTNESS_SCENARIOS),
    }
    with open(os.path.join(out_dir, "table2_robustness_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def generate_phase3_figure(
    bench_a: Dict[str, Any],
    table2: Dict[str, Any],
    fig_path: str
) -> None:
    """Generates Figure 1: Benchmark A Insertion & Multi-Seed Robustness with Held-Out support."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=220)
    plt.subplots_adjust(top=0.82, hspace=0.81, wspace=0.26)

    from textwrap import fill

    heldout_file = os.path.join(PROJECT_ROOT, "data", "phase3_validation", "learned_heldout_summary.json")
    with open(heldout_file, encoding="utf-8") as stream:
        heldout_data = json.load(stream)
    runs = heldout_data["controllers"]
    keys = list(runs)
    short_labels = [fill(runs[key]["label"], width=11) for key in keys]

    ax = axes[0, 0]
    x = np.arange(len(keys))
    width = 0.36
    succ_vals = [100 * runs[k]["all"]["success_rate"] for k in keys]
    jam_vals = [100 * runs[k]["all"]["jam_rate"] for k in keys]
    title_a = f"(A) Held-out insertion ($N={heldout_data['n']}$)"
    bars1 = ax.bar(x - width / 2, succ_vals, width, label="Insertion Success (%)", color="#10B981", edgecolor="#065F46")
    bars2 = ax.bar(x + width / 2, jam_vals, width, label="Jamming / Wedging (%)", color="#EF4444", edgecolor="#991B1B")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7.8, fontweight="bold")
    ax.set_ylabel("Rate (%)", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_title(title_a, fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, fontsize=8)
    for idx_b, b in enumerate(bars1):
        k = keys[idx_b]
        if k == "tactile_safety_filter_ours":
            basin = runs[k]["in_basin"]
            txt = f"{b.get_height():.1f}%\n({100 * basin['success'] / basin['n']:.1f}% Basin)"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, txt, ha="center", fontsize=7.2, fontweight="bold", color="#065F46")
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=7.8, fontweight="bold", color="#065F46")
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=7.8, fontweight="bold", color="#991B1B")

    # Panel B: Peak Contact Force across OOD Tilt Angles
    ax = axes[0, 1]
    colors_map = {
        "unconstrained_policy": ("#DC2626", "o-"),
        "fixed_impedance_k200": ("#D97706", "s--"),
        "ft_cbf_qp": ("#2563EB", "^-."),
        "impedance_spiral": ("#7C3AED", "v:"),
        "tactile_safety_filter_ours": ("#059669", "D-"),
    }
    for k in keys:
        trials = bench_a[k]["trials"]
        tilt_bins = sorted(list({int(round(t["initial_tilt_norm_deg"])) for t in trials}))
        mean_f = np.array([
            np.mean([t["peak_force_n"] for t in trials if int(round(t["initial_tilt_norm_deg"])) == tb])
            for tb in tilt_bins
        ])
        std_f = np.array([
            np.std([t["peak_force_n"] for t in trials if int(round(t["initial_tilt_norm_deg"])) == tb])
            for tb in tilt_bins
        ])
        col, style = colors_map[k]
        lbl = bench_a[k]["label"]
        ax.plot(tilt_bins, mean_f, style, color=col, linewidth=2.0, markersize=5.0, label=lbl)
        ax.fill_between(tilt_bins, mean_f - std_f, mean_f + std_f, color=col, alpha=0.14)
        ax.scatter(
            [round(t["initial_tilt_norm_deg"]) for t in trials],
            [t["peak_force_n"] for t in trials],
            color=col, s=9, alpha=0.45, zorder=3,
        )

    ax.axhline(15.0, color="#059669", linestyle="--", linewidth=1.2, label="Force target (15 N)")
    ax.axhline(40.0, color="#DC2626", linestyle=":", linewidth=1.2, label="Simulator force stop (40 N)")
    ax.set_xlabel(r"Initial 2-Axis OOD Tilt Norm $\|\theta_0\|_2$ (deg)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Peak Contact Force (N)", fontsize=9.5, fontweight="bold")
    ax.set_title(r"(B) Peak Contact Force vs. 2-Axis OOD Tilt Norm ($\mu \pm 1\sigma$)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 54)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=7.5)

    # Panel C: Environmental Stiffness Robustness
    ax = axes[1, 0]
    stiff_rows = table2["stiffness_sweep"]
    k_vals = np.array([r["stiffness_n_per_m"] for r in stiff_rows])
    b_k = np.array([r["baseline_peak_force_n"] for r in stiff_rows])
    b_k_std = np.array([r["baseline_std_force_n"] for r in stiff_rows])
    w_k = np.array([r["wrist_ft_peak_force_n"] for r in stiff_rows])
    w_k_std = np.array([r["wrist_ft_std_force_n"] for r in stiff_rows])
    t_k = np.array([r["tactai_peak_force_n"] for r in stiff_rows])
    t_k_std = np.array([r["tactai_std_force_n"] for r in stiff_rows])

    ax.set_xscale("log")
    ax.plot(k_vals, b_k, "o-", color="#DC2626", linewidth=2.0, label=runs["unconstrained_policy"]["label"])
    ax.fill_between(k_vals, b_k - b_k_std, b_k + b_k_std, color="#DC2626", alpha=0.15)
    ax.plot(k_vals, w_k, "^-.", color="#2563EB", linewidth=2.0, label=runs["ft_cbf_qp"]["label"])
    ax.fill_between(k_vals, w_k - w_k_std, w_k + w_k_std, color="#2563EB", alpha=0.15)
    ax.plot(k_vals, t_k, "D-", color="#059669", linewidth=2.2, label="TactAI Filter (Ours)")
    ax.fill_between(k_vals, t_k - t_k_std, t_k + t_k_std, color="#059669", alpha=0.18)
    ax.axhline(15.0, color="#059669", linestyle="--", linewidth=1.2, label="CBF Force Limit (15 N)")
    ax.set_xlabel(r"Environment Stiffness $K_{\mathrm{env}}$ (N/m)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Peak Contact Force (N)", fontsize=9.5, fontweight="bold")
    ax.set_title(f"(C) Stiffness sweep ({len(stiff_rows[0]['episode_results']['tactai'])} episodes per point)",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, 54)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="center right", fontsize=8)

    # Panel D: Perception Latency Robustness
    ax = axes[1, 1]
    lat_rows = table2["latency_sweep"]
    lat_vals = np.array([r["latency_ms"] for r in lat_rows])
    b_l = np.array([r["baseline_peak_force_n"] for r in lat_rows])
    b_l_std = np.array([r["baseline_std_force_n"] for r in lat_rows])
    w_l = np.array([r["wrist_ft_peak_force_n"] for r in lat_rows])
    w_l_std = np.array([r["wrist_ft_std_force_n"] for r in lat_rows])
    t_l = np.array([r["tactai_peak_force_n"] for r in lat_rows])
    t_l_std = np.array([r["tactai_std_force_n"] for r in lat_rows])

    ax.plot(lat_vals, b_l, "o-", color="#DC2626", linewidth=2.0, label=runs["unconstrained_policy"]["label"])
    ax.fill_between(lat_vals, b_l - b_l_std, b_l + b_l_std, color="#DC2626", alpha=0.15)
    ax.plot(lat_vals, w_l, "^-.", color="#2563EB", linewidth=2.0, label=runs["ft_cbf_qp"]["label"])
    ax.fill_between(lat_vals, w_l - w_l_std, w_l + w_l_std, color="#2563EB", alpha=0.15)
    ax.plot(lat_vals, t_l, "D-", color="#059669", linewidth=2.2, label="TactAI Filter (Ours)")
    ax.fill_between(lat_vals, t_l - t_l_std, t_l + t_l_std, color="#059669", alpha=0.18)
    ax.axhline(15.0, color="#059669", linestyle="--", linewidth=1.2, label="Force target (15 N)")
    ax.set_xlabel(r"Perception Latency $\tau_{\mathrm{delay}}$ (ms)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Peak Contact Force (N)", fontsize=9.5, fontweight="bold")
    ax.set_title(f"(D) Latency sweep ({len(lat_rows[0]['episode_results']['tactai'])} episodes per point)",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, 58)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center right", fontsize=8)

    fig.suptitle(
        "Learned-policy insertion and robustness",
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
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), dpi=300)
    plt.subplots_adjust(wspace=0.40)

    # Panel A: Benchmark B (Contour Tracking) Force Profiles
    ax = axes[0]
    traces_b = bc_ablation.get("_traces_b", {})
    for key, label, color, style in [
        ("unconstrained_farm", "Unconstrained", "#DC2626", "-"),
        ("wrist_ft_admittance", "Wrist F/T", "#2563EB", "--"),
        ("tactai_no_tank", "TactAI without tank", "#D97706", "-."),
        ("tactai_ours", "TactAI with tank", "#059669", "-"),
    ]:
        if key in traces_b:
            trace = traces_b[key][:600]
            ax.plot(np.arange(len(trace)), trace, color=color, linestyle=style,
                    label=f"{label} ({bc_ablation['benchmark_b_contour'][key]['trials'][0]['peak_force_n']:.1f} N)")

    ax.axhline(15.0, color="#059669", linestyle=":", linewidth=1.2, label="CBF Safety Bound (15.0 N)")
    ax.axhline(8.0, color="#475569", linestyle="--", linewidth=1.0, alpha=0.8, label="Target Force (8.0 N)")
    ax.set_xlabel("Time (ms)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel(r"Normal Contact Force $F_n$ (N)", fontsize=9.5, fontweight="bold")
    ax.set_title("(A) Benchmark B: phase-zero force traces", fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=-2)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.5)

    # Panel B: Benchmark C (Dynamic In-Hand Slip vs Time)
    ax = axes[1]
    traces_c = bc_ablation.get("_traces_c", {})
    t_c = np.arange(400)

    for key, label, color, style in [
        ("unconstrained_farm_slip", "Unconstrained", "#DC2626", "-"),
        ("wrist_ft_admittance_slip", "Wrist F/T", "#2563EB", "--"),
        ("tactai_ours_slip", "TactAI", "#059669", "-"),
    ]:
        if key in traces_c:
            trace = traces_c[key][:400]
            ax.plot(t_c[:len(trace)], trace, color=color, linestyle=style,
                    label=f"{label} ({bc_ablation['benchmark_c_slip'][key.removesuffix('_slip')]['mean_slip_mm']:.2f} mm)")

    ax.axhline(2.0, color="#DC2626", linestyle=":", linewidth=1.2, label="Gross Slip / Drop Limit (2.0 mm)")
    ax.set_xlabel("Time (ms)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Mean Cumulative Tool Slip (mm)", fontsize=9.5, fontweight="bold")
    ax.set_title(r"(B) Benchmark C: Extrinsic Shear Slip ($N=20$)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 165)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.5)

    # Panel C: Component Ablations
    ax = axes[2]
    ab_dict = bc_ablation.get("ablation_study", {})
    ab_keys = ["tactai_full", "tactai_no_cop_tensor", "tactai_no_jam_cbf", "tactai_no_tank"]
    ab_labels = ["Full\nTactAI", "No prop.\nalign.", "No Stage 2", "No energy\ntank"]

    succ_rates = [ab_dict.get(k, {}).get("success_rate_pct", 0.0) for k in ab_keys]
    peak_f = [ab_dict.get(k, {}).get("mean_peak_force_n", 0.0) for k in ab_keys]
    std_f = [ab_dict.get(k, {}).get("std_peak_force_n", 0.0) for k in ab_keys]

    x_ab = np.arange(len(ab_keys))
    w_ab = 0.38
    b1 = ax.bar(x_ab - w_ab / 2, succ_rates, w_ab, label="Insertion Success (%)", color="#10B981", edgecolor="#065F46")
    ax2 = ax.twinx()
    b2 = ax2.bar(
        x_ab + w_ab / 2,
        peak_f,
        w_ab,
        yerr=std_f,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "#1E3A8A"},
        label="Mean Peak Force (N)",
        color="#3B82F6",
        edgecolor="#1D4ED8",
    )

    ax.set_xticks(x_ab)
    ax.set_xticklabels(ab_labels, fontsize=8, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=9, fontweight="bold", color="#065F46")
    ax2.set_ylabel("Mean Peak Force (N)", fontsize=9, fontweight="bold", color="#1D4ED8")
    ax.set_ylim(0, 118)
    ax2.set_ylim(0, 42)
    ax.set_title(r"(C) Component Ablation Study ($N=20$ OOD Trials)", fontsize=9.5, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", fontsize=7.2, fontweight="bold", color="#065F46")
    for b, err in zip(b2, std_f):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + err + 0.8, f"{b.get_height():.1f}N", ha="center", fontsize=7.2, fontweight="bold", color="#1D4ED8")

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(fig_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f">>> Saved Benchmarks B/C & Ablation figure to: {fig_path} and {pdf_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("insertion", "contour-slip", "robustness", "plots"))
    args = parser.parse_args()
    t_start = time.perf_counter()
    out_dir = os.path.join(PROJECT_ROOT, "data", "phase3_validation")
    fig_dir = os.path.join(PROJECT_ROOT, "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if args.only in (None, "insertion"):
        run_benchmark_a(out_dir)
    if args.only in (None, "contour-slip"):
        run_benchmarks_bc_and_ablations(out_dir)
    if args.only in (None, "robustness"):
        run_table2_robustness(out_dir)
    if args.only not in (None, "plots"):
        return

    def load(filename):
        with open(os.path.join(out_dir, filename), encoding="utf-8") as stream:
            return json.load(stream)

    bench_a = load("benchmark_a_summary.json")
    bc_ablation = load("benchmarks_bc_ablation_summary.json")
    table2 = load("table2_robustness_summary.json")
    with np.load(os.path.join(out_dir, "benchmark_bc_plot_traces.npz")) as traces:
        bc_ablation["_traces_b"] = {k.removeprefix("contour_"): traces[k]
                                  for k in traces if k.startswith("contour_")}
        bc_ablation["_traces_c"] = {k: traces[k] for k in traces if not k.startswith("contour_")}

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
            "Figures generated from saved simulation logs.",
            "95% Wilson intervals are reported for binary success and force-safety rates.",
            "Held-out randomized evaluation loaded from learned_heldout_summary.json.",
            "Simulation results are not hardware-validation results.",
        ],
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    elapsed = time.perf_counter() - t_start
    print(f"\n=== FULL PHASE 3 BENCHMARK & ABLATION SUITE COMPLETED IN {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
