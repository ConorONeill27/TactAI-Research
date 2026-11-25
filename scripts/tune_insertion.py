"""Select an insertion controller exclusively on the structured development panel."""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.run_phase3_validation import OOD_SCENARIOS, run_single_episode, summarize_episode_set
from src.controllers.learned_policy import CHECKPOINT
from src.controllers.tactile_alignment_controller import TactileAlignmentController
from src.envs.franka_slot_env import FrankaSlotEnv

DATA = CHECKPOINT.parent
CANDIDATES = [
    {},
    dict(rim_damping=62.4),
    dict(approach_damping=55, retraction_damping=26, retraction_floor=4,
         force_trigger_fraction=.6, rim_damping=74, rim_band_lower=-.008),
    dict(approach_damping=70, retraction_damping=30, retraction_floor=4,
         force_trigger_fraction=.60, rim_damping=90, rim_band_lower=-.008),
    dict(approach_damping=85, retraction_damping=30, retraction_floor=4,
         force_trigger_fraction=.60, rim_damping=105, rim_band_lower=-.008,
         rim_brake_force=3.8, rim_latch_force=6.8),
    dict(approach_damping=95, retraction_damping=28, retraction_floor=4,
         force_trigger_fraction=.55, rim_damping=115, rim_band_lower=-.008,
         rim_brake_force=3.8, rim_latch_force=7.5),
]
FOLLOWUP = [
    dict(approach_damping=45, retraction_damping=26, retraction_floor=4,
         force_trigger_fraction=.65, rim_damping=damping, rim_band_lower=band,
         rim_brake_force=brake, rim_latch_force=latch, settle_ticks=settle)
    for damping, band, brake, latch, settle in [
        (105, -.0055, 7.5, 9.8, 100),
        (105, -.007, 7.5, 9.8, 100),
        (115, -.007, 7.5, 9.8, 90),
        (115, -.008, 6.5, 9.5, 85),
        (120, -.008, 8.0, 11.0, 85),
        (135, -.007, 8.0, 11.0, 80),
    ]
]
SCREEN = (0, 5, 12, 24, 26, 30, 36, 42, 44, 47, 48, 49)
RIM_RECONTACT = [
    dict(learned_approach=True, rim_damping=damping, rim_band_lower=-.0063,
         settle_ticks=settle, force_trigger_fraction=.75, retraction_floor=4,
         kp_lat=kp, centering_damping=center_damping, deep_insertion_push=16)
    for damping, settle, kp, center_damping in [(125, 120, 1100, 55),
                                                (125, 115, 2000, 75),
                                                (135, 110, 1100, 55)]
]


def evaluate(config, indices=range(len(OOD_SCENARIOS))):
    env = FrankaSlotEnv()
    controller = TactileAlignmentController(**config)
    trials = []
    for i in indices:
        tilt, offset = OOD_SCENARIOS[i]
        trial = run_single_episode(env, controller, tilt, offset, trial_seed=42 + i)
        trial["scenario_index"] = i
        trials.append({k: v for k, v in trial.items() if not k.endswith("_trace")})
    return dict(parameters=config, summary=summarize_episode_set(trials), trials=trials)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rim-recontact", action="store_true",
                        help="Evaluate the narrowed re-contact design after the initial search.")
    args = parser.parse_args()
    if (DATA / "freeze_manifest.json").exists():
        parser.error("This experiment is frozen. Use a separate experiment directory for new tuning.")
    path = DATA / "development_tuning.json"
    with ProcessPoolExecutor(max_workers=4) as pool:
        from itertools import repeat

        if args.rim_recontact:
            existing = json.loads(path.read_text(encoding="utf-8"))
            initial = existing["candidates"]
            screened = existing["screened_candidates"]
            followup = list(pool.map(evaluate, RIM_RECONTACT))
        elif path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            initial = existing["candidates"][:len(CANDIDATES)]
        else:
            initial = list(pool.map(evaluate, [dict(learned_approach=True, **c) for c in CANDIDATES]))
        if not args.rim_recontact:
            screened = list(pool.map(evaluate, [dict(learned_approach=True, **c) for c in FOLLOWUP],
                                     repeat(SCREEN)))
            for row in screened:
                print("Screen:", row["parameters"], row["summary"]["success_count"],
                      row["summary"]["safe_force_count"], flush=True)
            ranked = sorted(screened, key=lambda c: (
                c["summary"]["success_count"] >= 8, c["summary"]["safe_force_count"],
                c["summary"]["success_count"], -c["summary"]["max_peak_force_n"]), reverse=True)
            followup = list(pool.map(evaluate, [r["parameters"] for r in ranked[:3]]))
    candidates = initial + followup
    for candidate in candidates:
        s = candidate["summary"]
        print(candidate["parameters"], [s[k] for k in
              ("success_count", "safe_force_count", "jamming_count", "max_peak_force_n")])
    eligible = [c for c in candidates if c["summary"]["success_count"] >= 42]
    if not eligible:
        raise RuntimeError("No development configuration met the 42/50 insertion floor.")
    selected = max(eligible, key=lambda c: (c["summary"]["safe_force_count"],
                   -c["summary"]["jamming_count"], c["summary"]["success_count"],
                   -c["summary"]["max_peak_force_n"]))
    report = dict(protocol="structured development only", candidates=candidates,
                  screened_candidates=screened,
                  selected_parameters=selected["parameters"],
                  checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Selected:", selected["parameters"])


if __name__ == "__main__":
    main()
