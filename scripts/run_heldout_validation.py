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

Output: data/phase3_validation/learned_heldout_summary.json
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.franka_slot_env import FrankaSlotEnv
from src.controllers.insertion_protocol import DATA, frozen_parameters, insertion_controllers
from src.controllers.learned_policy import CHECKPOINT
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
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("data/phase3_validation/learned_heldout_summary.json"))
    args = ap.parse_args()

    if args.output.exists():
        ap.error(f"Refusing to overwrite {args.output}")
    if args.seed == 20261020:
        ap.error("The historical held-out seed is reserved.")
    tuning_path = DATA / "development_tuning.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "freeze_manifest.json").read_text(encoding="utf-8"))
    if args.seed != manifest["heldout_seed"]:
        ap.error("The seed must match the frozen evaluation manifest.")
    if hashlib.sha256(tuning_path.read_bytes()).hexdigest() != manifest["tuning_sha256"]:
        ap.error("Development tuning data differs from the frozen manifest.")
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if tuning["checkpoint_sha256"] != digest:
        ap.error("The checkpoint differs from the one used during structured development.")
    conds = sample_conditions(50, args.seed)
    env = FrankaSlotEnv()
    controllers = insertion_controllers(env.dt)

    out = {"seed": args.seed, "n": len(conds), "checkpoint_sha256": digest,
           "controller_parameters": frozen_parameters(), "controllers": {}}
    for name, (label, ctrl) in controllers.items():
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
            "label": label,
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
