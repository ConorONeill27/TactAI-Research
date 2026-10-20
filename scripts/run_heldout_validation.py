"""
Held-out generalization check for Benchmark A.

The Stage 3 rim-arrest thresholds were tuned while observing the 50 fixed
OOD_SCENARIOS. To report an unbiased estimate, this script evaluates the
frozen controller (and baselines) on N freshly sampled initial conditions
drawn from the same stated ranges, with a seed that was never used during
development.

Sampling (per trial):
  theta_x, theta_y ~ U(-15, 15) deg, |theta|_2 clipped to <= 19.2 deg
  dx, dy           ~ U(-0.92, 0.92) mm
In-basin is defined a priori as |theta|_2 <= 15 deg and max(|dx|,|dy|) <= 0.80 mm.

Output: data/phase3_validation/heldout_summary.json
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv
from src.controllers.tactile_alignment_controller import TactileAlignmentController
from src.controllers.baseline_controllers import (
    UnconstrainedPolicyBaseline,
    WristFTAdmittanceController,
    FixedIsotropicImpedanceController,
)
from scripts.run_phase3_validation import run_single_episode


def sample_conditions(n: int, seed: int):
    rng = np.random.default_rng(seed)
    conds = []
    while len(conds) < n:
        tx, ty = rng.uniform(-15.0, 15.0, size=2)
        if np.hypot(tx, ty) > 19.2:
            continue
        dx, dy = rng.uniform(-0.92, 0.92, size=2)
        conds.append(((float(tx), float(ty)), (float(dx), float(dy))))
    return conds


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(trials):
    n = len(trials)
    succ = sum(t["is_success"] for t in trials)
    safe = sum(t["is_safe_force"] for t in trials)
    jam = sum(t["is_jammed"] for t in trials)
    peaks = np.array([t["peak_force_n"] for t in trials]) if trials else np.zeros(1)
    return {
        "n": n,
        "success": succ,
        "success_rate": succ / n if n else 0.0,
        "success_ci95": wilson_ci(succ, n),
        "force_safe": safe,
        "force_safe_rate": safe / n if n else 0.0,
        "jammed": jam,
        "jam_rate": jam / n if n else 0.0,
        "peak_force_mean": float(peaks.mean()),
        "peak_force_std": float(peaks.std()),
        "peak_force_max": float(peaks.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--ours-only", action="store_true")
    args = ap.parse_args()

    conds = sample_conditions(args.n, args.seed)
    env = FrankaSlotEnv()
    controllers = {"tactile_safety_filter_ours": TactileAlignmentController()}
    if not args.ours_only:
        controllers["unconstrained_policy"] = UnconstrainedPolicyBaseline()
        controllers["wrist_ft_admittance"] = WristFTAdmittanceController()
        controllers["fixed_impedance_k200"] = FixedIsotropicImpedanceController()

    out = {"seed": args.seed, "n": args.n, "controllers": {}}
    for name, ctrl in controllers.items():
        trials = []
        for i, (tilt, off) in enumerate(conds):
            r = run_single_episode(env, ctrl, tilt, off, trial_seed=args.seed + i)
            r.pop("forces_trace", None)
            r.pop("depths_trace", None)
            r["in_basin"] = bool(r["initial_tilt_norm_deg"] <= 15.0 and max(abs(off[0]), abs(off[1])) <= 0.80)
            trials.append(r)
        in_b = [t for t in trials if t["in_basin"]]
        out_b = [t for t in trials if not t["in_basin"]]
        out["controllers"][name] = {
            "all": summarize(trials),
            "in_basin": summarize(in_b),
            "out_of_basin": summarize(out_b),
            "trials": trials,
        }
        s = out["controllers"][name]
        print(f"[{name}] all: {s['all']['success']}/{s['all']['n']} succ "
              f"(95% CI {s['all']['success_ci95'][0]:.2f}-{s['all']['success_ci95'][1]:.2f}), "
              f"safe {s['all']['force_safe']}/{s['all']['n']}, jam {s['all']['jammed']}, "
              f"peak {s['all']['peak_force_mean']:.2f}+-{s['all']['peak_force_std']:.2f} (max {s['all']['peak_force_max']:.2f}) | "
              f"in-basin {s['in_basin']['success']}/{s['in_basin']['n']}, "
              f"out-of-basin {s['out_of_basin']['success']}/{s['out_of_basin']['n']}")

    os.makedirs("data/phase3_validation", exist_ok=True)
    with open("data/phase3_validation/heldout_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved data/phase3_validation/heldout_summary.json")


if __name__ == "__main__":
    main()
