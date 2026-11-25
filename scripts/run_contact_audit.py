"""Paired, fixed-seed Benchmark A perturbations and held-out failure audit."""

import argparse
import json
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_heldout_validation import sample_conditions
from scripts.run_phase3_validation import run_single_episode, summarize_episode_set
from src.controllers.tactile_alignment_controller import TactileAlignmentController
from src.envs.franka_slot_env import FrankaSlotEnv


HELDOUT_SEED = 20261020
STRESS_SEEDS = (73101, 73102)
ROOT = Path(__file__).resolve().parents[1]


class TactilePerturbation:
    """Apply errors at 50 Hz and hold the last accepted tactile frame."""

    def __init__(self, seed, *, cop_sigma_mm=0, stick_sigma=0, bias_deg=0,
                 scale_error=0, dropout=0):
        self.rng = np.random.default_rng([seed, 531])
        self.cop_sigma_mm = cop_sigma_mm
        self.stick_sigma = stick_sigma
        self.bias_deg = bias_deg
        self.scale_error = scale_error
        self.dropout = dropout
        self.frame = None
        self.dropped = 0

    def __call__(self, obs, step):
        if step % 20 == 0:
            drop = self.rng.random() < self.dropout and self.frame is not None
            if drop:
                self.dropped += 1
            else:
                frame = obs["tactile"]
                left = frame["left_pad"]
                right = frame["right_pad"]
                left_cop = left["center_of_pressure"]
                right_cop = right["center_of_pressure"]
                center = (left_cop + right_cop) / 2
                differential = (left_cop - right_cop) / 2
                angle = np.arcsin(np.clip(differential / 0.006, -1, 1))
                angle = angle * (1 + self.scale_error) + np.deg2rad(self.bias_deg)
                delta = 0.006 * np.sin(angle) + self.rng.normal(0, self.cop_sigma_mm * 1e-3, 2)
                ratio = np.clip(frame["stick_ratio"] + self.rng.normal(0, self.stick_sigma), 0, 1)
                # Jamming index is derived from the simulated wrist wrench.
                self.frame = dict(frame, stick_ratio=float(ratio),
                                  left_pad=dict(left, center_of_pressure=center + delta),
                                  right_pad=dict(right, center_of_pressure=center - delta))
        tactile = dict(self.frame, jamming_index=obs["tactile"]["jamming_index"])
        return dict(obs, tactile=tactile)


def in_basin(tilt, offset):
    return bool(np.linalg.norm(tilt) <= 15 and max(map(abs, offset)) <= 0.8)


def cases():
    stress = [("clean_50hz", {})]
    stress += [(f"noise_{s:g}mm_{r:g}", dict(cop_sigma_mm=s, stick_sigma=r))
               for s, r in [(0.025, 0.02), (0.1, 0.05), (0.25, 0.10)]]
    stress += [(f"bias_{b:+d}deg", dict(bias_deg=b)) for b in (-3, -2, -1, 1, 2, 3)]
    stress += [(f"scale_{s:+.0%}", dict(scale_error=s)) for s in (-0.2, -0.1, 0.1, 0.2)]
    stress += [(f"drop_{d:.0%}", dict(dropout=d)) for d in (0.1, 0.2, 0.3)]
    for label, errors in stress:
        yield "tactile", label, {}, errors
    for label, gains in (
        ("without_cop_tensor", dict(enable_tactile_rot=False)),
        ("cop_only", dict(enable_tensor_modulation=False)),
        ("full", {}),
    ):
        yield "tensor", label, gains, None
    yield "guard", "default", {}, None
    for name, nominal in (("rim_brake_force", 4.2), ("rim_latch_force", 6.5), ("rim_damping", 52)):
        for factor in (0.8, 1.2):
            yield "guard", f"{name}_{factor - 1:+.0%}", {name: nominal * factor}, None


def run_case(case):
    group, label, gains, errors = case
    if group == "guard":
        conditions = [(HELDOUT_SEED + i, tilt, off)
                      for i, (tilt, off) in enumerate(sample_conditions(50, HELDOUT_SEED))]
    else:
        conditions = []
        for seed in STRESS_SEEDS:
            eligible = [(tilt, off) for tilt, off in sample_conditions(50, seed) if in_basin(tilt, off)]
            conditions.extend((seed * 100 + i, tilt, off) for i, (tilt, off) in enumerate(eligible[:6]))
    env = FrankaSlotEnv()
    ctrl = TactileAlignmentController(dt=env.dt, **gains)
    trials = []
    for seed, tilt, offset in conditions:
        transform = TactilePerturbation(seed, **errors) if errors is not None else None
        trial = run_single_episode(env, ctrl, tilt, offset, trial_seed=seed,
                                   tactile_transform=transform)
        for key in ("forces_trace", "depths_trace"):
            trial.pop(key, None)
        trial["in_basin"] = in_basin(tilt, offset)
        if transform is not None:
            trial["dropped_frames"] = transform.dropped
        trials.append(trial)
    return group, label, dict(summary=summarize_episode_set(trials), trials=trials,
                              controller_parameters=gains, tactile_errors=errors)


def plot(results):
    fig = plt.figure(figsize=(13, 9), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(17, 3, 7), width_ratios=(1.25, 1))
    for i, (group, runs) in enumerate(results.items()):
        ax = fig.add_subplot(grid[i, 0])
        force_ax = fig.add_subplot(grid[i, 1], sharey=ax)
        names = list(runs)
        y = np.arange(len(names))
        ax.barh(y - 0.17, [runs[n]["summary"]["success_rate_pct"] for n in names],
                height=0.32, label="Insertion")
        ax.barh(y + 0.17, [runs[n]["summary"]["jamming_rate_pct"] for n in names],
                height=0.32, label="Jam")
        ax.set(yticks=y, yticklabels=[n.replace("_", " ") for n in names], xlim=(0, 100),
               xlabel="Episodes (%)", title=group.capitalize(), ylim=(len(names) - 0.5, -0.5))
        if i == 0:
            ax.legend(loc="lower right")
        for row, name in enumerate(names):
            force_ax.scatter([t["peak_force_n"] for t in runs[name]["trials"]],
                             np.full(len(runs[name]["trials"]), row), s=10, alpha=0.4)
        force_ax.axvline(15, color="black", linestyle="--", linewidth=1)
        force_ax.tick_params(labelleft=False, left=False)
        force_ax.set_xlabel("Peak contact force (N)")
    fig.savefig(ROOT / "docs" / "figures" / "benchmark_a_contact_audit.pdf")
    plt.close(fig)


def main():
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    results = {"tactile": {}, "tensor": {}, "guard": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for group, label, result in pool.map(run_case, cases()):
            results[group][label] = result
            print(group, label, result["summary"]["success_count"], flush=True)
    failures = [t for t in results["guard"]["default"]["trials"] if not t["is_success"]]
    path = ROOT / "data" / "phase3_validation" / "contact_audit.json"
    path.write_text(json.dumps(dict(heldout_seed=HELDOUT_SEED, stress_seeds=STRESS_SEEDS,
                                    results=results, heldout_failures=failures), indent=2),
                    encoding="utf-8")
    plot(results)


if __name__ == "__main__":
    main()
