"""Run the fixed A2 development panel and the A1 compatibility/ablation check."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_phase3_validation import OOD_SCENARIOS, run_single_episode, summarize_episode_set
from src.controllers.insertion_protocol import insertion_controllers, tactai
from src.controllers.learned_policy import LearnedChunkPolicy
from src.envs.franka_slot_env import FrankaSlotEnv
from src.envs.unknown_grasp_env import UnknownGraspEnv

OUT = Path(__file__).resolve().parents[1] / "data" / "grasp_development"
DEV_SEED = 81901
A2_STEPS = 850
A2_DEV_QP = {"rim_damping": 185.0}


def conditions(seed=DEV_SEED, n=18, representative=True):
    rng = np.random.default_rng(seed)
    result = [((3.0, 0.0), (0.0, 0.0), (7.0, 0.0))] if representative else []
    for _ in range(n - len(result)):
        direction = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(3, 10)
        grasp = radius * np.array([np.cos(direction), np.sin(direction)])
        result.append((rng.uniform(-4, 4, 2), rng.uniform(-.35, .35, 2), grasp))
    return result


def a2_episode(env, controller, condition, seed, capture_trace=False, tactile_transform=None):
    wrist, offset, grasp = condition
    obs, _ = env.reset(seed=seed, ood_tilt_deg=wrist, ood_offset_mm=offset, grasp_tilt_deg=grasp)
    policy = LearnedChunkPolicy()
    controller.reset()
    trace = []
    solve_times = []
    peak = 0.0
    for _ in range(A2_STEPS):
        sensed = tactile_transform(obs, obs["step"]) if tactile_transform else obs
        command, diag = controller.compute_safe_action(policy.get_action(obs), sensed)
        if "solve_time_ms" in diag:
            solve_times.append(diag["solve_time_ms"])
        step, _, term, trunc, info = env.step(command)
        force = float(info["current_force_n"])
        peak = max(force, peak)
        if capture_trace:
            trace.append(dict(step=step["step"], force_n=force,
                              wrist_deg=np.rad2deg(step["ee_rpy"][:2]).tolist(),
                              peg_deg=np.rad2deg(env.peg_tilt_rad()).tolist(),
                              residual_deg=np.rad2deg(diag.get("grasp_residual_rad", [0, 0])).tolist(),
                              stage=diag.get("stage")))
        obs = step
        if term or trunc:
            break
    depth = float(info["insertion_depth_m"] * 1000)
    success = bool(info["is_success"])
    return dict(trial_seed=seed, initial_wrist_deg=list(wrist), initial_offset_mm=list(offset),
                grasp_rest_deg=list(grasp), final_wrist_deg=np.rad2deg(obs["ee_rpy"][:2]).tolist(),
                final_peg_deg=np.rad2deg(env.peg_tilt_rad()).tolist(),
                final_residual_deg=np.rad2deg(diag.get("grasp_residual_rad", [0, 0])).tolist(),
                final_depth_mm=depth, peak_force_n=peak, is_success=success,
                is_jammed=bool(info["is_jammed"] or (not success and peak > 22 and depth < 10)),
                is_safe_force=peak <= 15, episode_duration_s=obs["time"],
                final_stage=diag.get("stage"), rim_arrest_latched=diag.get("rim_arrest_latched"),
                mean_solve_time_ms=float(np.mean(solve_times)) if solve_times else None,
                **({"trace": trace} if capture_trace else {}))


def run_a2(overrides, qp_only=False):
    env = UnknownGraspEnv()
    controllers = insertion_controllers(env.dt)
    controllers.pop("fixed_impedance_k200")
    for name in ("ft_cbf_qp", "tactile_safety_filter_ours"):
        label, _ = controllers[name]
        controllers[name] = (label, tactai(env.dt, wrist_ft_only=name == "ft_cbf_qp",
                                           unknown_grasp_mode=True, **overrides))
    controllers["cop_only"] = ("TactAI CoP only", tactai(env.dt, unknown_grasp_mode=True,
                                                        enable_tensor_modulation=False,
                                                        **overrides))
    if qp_only:
        controllers = {"tactile_safety_filter_ours": controllers["tactile_safety_filter_ours"]}
    runs = {}
    for name, (label, controller) in controllers.items():
        trials = [a2_episode(env, controller, c, DEV_SEED + i,
                             capture_trace=i == 0 and name in ("ft_cbf_qp", "tactile_safety_filter_ours"))
                  for i, c in enumerate(conditions())]
        runs[name] = dict(label=label, summary=summarize_episode_set(trials), trials=trials)
        print(label, runs[name]["summary"], flush=True)
    return dict(seed=DEV_SEED, horizon=A2_STEPS, qp_overrides=overrides, controllers=runs)


def run_a1():
    env = FrankaSlotEnv()
    runs = {}
    for name, options in (("ft_cbf_qp", {"wrist_ft_only": True}),
                          ("cop_only", {"enable_tensor_modulation": False}),
                          ("tactile_safety_filter_ours", {})):
        controller = tactai(env.dt, **options)
        trials = []
        for i, (tilt, offset) in enumerate(OOD_SCENARIOS):
            trial = run_single_episode(env, controller, tilt, offset, trial_seed=42+i)
            trials.append({k: v for k, v in trial.items() if not k.endswith("_trace")})
        runs[name] = dict(summary=summarize_episode_set(trials), trials=trials)
        print(name, runs[name]["summary"], flush=True)
    return dict(controllers=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=("a1", "a2"))
    parser.add_argument("--rim-damping", type=float)
    parser.add_argument("--settle-ticks", type=int)
    parser.add_argument("--qp-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    overrides = A2_DEV_QP | {k: getattr(args, k) for k in ("rim_damping", "settle_ticks")
                            if getattr(args, k) is not None}
    result = run_a1() if args.panel == "a1" else run_a2(overrides, args.qp_only)
    path = args.output or OUT / f"{args.panel}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
