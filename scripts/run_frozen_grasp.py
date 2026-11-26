"""Evaluate the frozen known- and unknown-grasp protocols."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import secrets

from scripts.run_contact_audit import TactilePerturbation, cases
from scripts.run_grasp_development import A2_DEV_QP, A2_STEPS, DEV_SEED, a2_episode, conditions
from scripts.run_heldout_validation import sample_conditions, summarize
from scripts.run_phase3_validation import OOD_SCENARIOS, run_single_episode, summarize_episode_set
from src.controllers.insertion_protocol import frozen_parameters, insertion_controllers, tactai
from src.envs.franka_slot_env import FrankaSlotEnv
from src.envs.unknown_grasp_env import UnknownGraspEnv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/phase3_validation"
MANIFEST = DATA / "unified_grasp_manifest.json"
FILES = ("src/controllers/tactile_alignment_controller.py", "src/controllers/insertion_protocol.py",
         "src/envs/franka_slot_env.py", "src/envs/unknown_grasp_env.py",
         "src/sensors/gelsight_sensor.py", "src/controllers/baseline_controllers.py",
         "src/controllers/learned_policy.py", "data/learned_policy/weights.pt")
LABELS = {"cop_only": "TactAI CoP only",
          "no_alignment": "Without proportional rotational alignment"}


def _normalized_bytes(path: Path) -> bytes:
    """Normalize CRLF line endings on Python sources for cross-platform SHA-256 verification."""
    raw = path.read_bytes()
    return raw.replace(b"\r\n", b"\n") if path.suffix == ".py" else raw


def fingerprint():
    return {path: hashlib.sha256(_normalized_bytes(ROOT / path)).hexdigest() for path in FILES}


def freeze():
    payload = dict(files=fingerprint(), a1_seed=1524526683, n=50, a1=frozen_parameters(),
                   a2=frozen_parameters() | A2_DEV_QP | {"unknown_grasp_mode": True},
                   a2_horizon=A2_STEPS)
    with MANIFEST.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    # Draw only after the controller and protocol have been recorded.
    payload["a2_seed"] = 2_000_000_000 + secrets.randbelow(100_000_000)
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_case(task):
    panel, key, label, errors, manifest = task
    a2 = panel != "a1" and panel != "a1_ablation"
    env = UnknownGraspEnv() if a2 else FrankaSlotEnv()
    params = manifest["a2" if a2 else "a1"]
    if key in ("ft_cbf_qp", "tactile_safety_filter_ours", *LABELS):
        controller = tactai(env.dt, **(params | dict(wrist_ft_only=key == "ft_cbf_qp",
                            enable_tensor_modulation=key not in ("ft_cbf_qp", "cop_only"),
                            enable_tactile_rot=key != "no_alignment")))
    else:
        _, controller = insertion_controllers(env.dt)[key]
    if panel in ("a2_ablation", "a2_stress"):
        seed, draws = DEV_SEED, conditions()
    elif panel == "a1_ablation":
        seed, draws = 42, OOD_SCENARIOS
    else:
        seed = manifest[f"{panel}_seed"]
        draws = conditions(seed, manifest["n"], False) if a2 else sample_conditions(manifest["n"], seed)
    rows = []
    for i, draw in enumerate(draws):
        if a2:
            transform = TactilePerturbation(seed+i, **errors) if errors is not None else None
            row = a2_episode(env, controller, draw, seed+i,
                             capture_trace=panel == "a2_ablation" and i == 0,
                             tactile_transform=transform)
            if transform is not None:
                row["dropped_frames"] = transform.dropped
        else:
            row = run_single_episode(env, controller, *draw, trial_seed=seed+i)
            row = {k:v for k,v in row.items() if not k.endswith("_trace")}
            row["in_basin"] = row["initial_tilt_norm_deg"] <= 15 and max(map(abs, draw[1])) <= .8
        rows.append(row)
    result = dict(label=label, all=summarize(rows), trials=rows)
    if not a2:
        for group, inside in (("in_basin", True), ("out_of_basin", False)):
            result[group] = summarize([r for r in rows if r["in_basin"] == inside])
    return label, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=("freeze", "a1", "a2", "a1_ablation", "a2_ablation", "a2_stress"))
    args = parser.parse_args()
    panel = args.panel
    if panel == "freeze":
        freeze()
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if fingerprint() != manifest["files"]:
        parser.error("Frozen controller, sensor, environment or checkpoint changed.")
    output = DATA / f"unified_{panel}.json"
    if output.exists():
        parser.error(f"{output} already exists")
    labels = {k:label for k,(label,_) in insertion_controllers(.001).items()} | LABELS
    if panel == "a2_stress":
        tasks = [(panel, "tactile_safety_filter_ours", label, errors, manifest)
                 for group,label,_,errors in cases() if group == "tactile"]
        keys = [task[2] for task in tasks]
    else:
        keys = list(labels)[:5] if panel in ("a1", "a2") else [
            "ft_cbf_qp", "cop_only", "tactile_safety_filter_ours", "no_alignment"]
        tasks = [(panel, key, labels[key], None, manifest) for key in keys]
    with ProcessPoolExecutor(max_workers=4) as pool:
        runs = {}
        for key, (_, result) in zip(keys, pool.map(run_case, tasks), strict=True):
            runs[key] = result
            print(panel, key, result["all"], flush=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(dict(panel=panel, n=manifest["n"] if panel in ("a1","a2") else len(next(iter(runs.values()))["trials"]),
                       seed=manifest.get(f"{panel}_seed"), controllers=runs,
                       failures={k:[r for r in v["trials"] if not r["is_success"]] for k,v in runs.items()}
                       if panel in ("a1","a2") else {}), stream, indent=2)


if __name__ == "__main__":
    main()
