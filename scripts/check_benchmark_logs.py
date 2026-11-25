"""Verify learned policy, frozen split, and aggregate results against raw logs."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_heldout_validation import sample_conditions, summarize
from scripts.run_phase3_validation import summarize_episode_set

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "phase3_validation"
POLICY = ROOT / "data" / "learned_policy"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_logs():
    training = read(POLICY / "training.json")
    freeze = read(POLICY / "freeze_manifest.json")
    tuning = read(POLICY / "development_tuning.json")
    heldout = read(DATA / "learned_heldout_summary.json")
    digest = hashlib.sha256((POLICY / "weights.pt").read_bytes()).hexdigest()
    assert digest == training["checkpoint_sha256"] == freeze["checkpoint_sha256"] == heldout["checkpoint_sha256"]
    assert hashlib.sha256((POLICY / "development_tuning.json").read_bytes()).hexdigest() == freeze["tuning_sha256"]
    assert tuning["selected_parameters"] == freeze["controller_parameters"] == heldout["controller_parameters"]
    assert heldout["seed"] == freeze["heldout_seed"] != 20261020
    assert heldout["n"] == 50
    conditions = sample_conditions(heldout["n"], heldout["seed"])
    for run in heldout["controllers"].values():
        trials = run["trials"]
        for i, (trial, condition) in enumerate(zip(trials, conditions, strict=True)):
            tilt, offset = condition
            assert trial["trial_seed"] == heldout["seed"] + i
            assert trial["tilt_deg"] == list(tilt) and trial["offset_mm"] == list(offset)
            assert trial["in_basin"] == (np.linalg.norm(tilt) <= 15 and max(map(abs, offset)) <= .8)
        assert run["all"] == json.loads(json.dumps(summarize(trials)))
        for group, basin in (("in_basin", True), ("out_of_basin", False)):
            expected = summarize([t for t in trials if t["in_basin"] == basin])
            assert run[group] == json.loads(json.dumps(expected))

    for run in read(DATA / "benchmark_a_summary.json").values():
        assert all(run[k] == v for k, v in summarize_episode_set(run["trials"]).items())
    audit = read(DATA / "contact_audit.json")
    assert audit["guard_audit_seed"] != heldout["seed"]
    for group in audit["results"].values():
        for run in group.values():
            assert run["summary"] == summarize_episode_set(run["trials"])

    robust = read(DATA / "table2_robustness_summary.json")
    for key, points in robust.items():
        if key == "protocol":
            continue
        for point in points:
            for controller, trials in point["episode_results"].items():
                summary = summarize_episode_set(trials)
                assert point[f"{controller}_success_count"] == summary["success_count"]
                assert point[f"{controller}_peak_force_n"] == summary["mean_peak_force_n"]
                assert point[f"{controller}_max_force_n"] == summary["max_peak_force_n"]

    bc = read(DATA / "benchmarks_bc_ablation_summary.json")
    for run in bc["ablation_study"].values():
        assert all(run[k] == v for k, v in summarize_episode_set(run["trials"]).items())
    with np.load(DATA / "benchmark_bc_plot_traces.npz") as traces:
        for key, run in bc["benchmark_b_contour"].items():
            trials = run["trials"]
            assert run["peak_force_n"] == max(t["peak_force_n"] for t in trials)
            assert run["force_rmse_n"] == round(float(np.mean([t["rmse_force_n"] for t in trials])), 2)
            assert round(float(traces["contour_" + key].max()), 2) == trials[0]["peak_force_n"]
        for key, run in bc["benchmark_c_slip"].items():
            trials = run["trials"]
            assert run["drop_rate_pct"] == round(100 * np.mean([t["is_dropped"] for t in trials]), 1)
            assert abs(run["mean_slip_mm"] - float(traces[key + "_slip"][-1])) < .01
    print("Verified frozen split, checkpoint, raw aggregates, and contour/slip plot traces.")


if __name__ == "__main__":
    check_logs()
