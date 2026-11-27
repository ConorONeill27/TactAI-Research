"""Populate manuscript tables and headline statistics from benchmark JSON."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "phase3_validation"
END_ROW = r"\\"


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def render():
    heldout_doc = load("unified_a1")
    heldout = heldout_doc["controllers"]
    audit = load("contact_audit")
    bc = load("benchmarks_bc_ablation_summary")
    robustness = load("table2_robustness_summary")
    ours = heldout["tactile_safety_filter_ours"]
    overall = ours["all"]
    lo, hi = overall["success_ci95"]
    failures = [t for t in ours["trials"] if not t["is_success"]]
    training = json.loads((ROOT / "data/learned_policy/training.json").read_text(encoding="utf-8"))
    telemetry = json.loads((ROOT / "data/phase2_validation/case2_tactai_success_summary.json").read_text(encoding="utf-8"))
    baseline_telemetry = json.loads((ROOT / "data/phase2_validation/case1_baseline_jamming_summary.json").read_text(encoding="utf-8"))
    guard = audit["results"]["guard"]
    tactile = audit["results"]["tactile"]
    tensor = audit["results"]["tensor"]
    params = load("unified_grasp_manifest")["a1"]
    d = (15 - .1) / 10000
    speed = math.sqrt(2 / 2.5 * (params["retraction_floor"] * d + (15**2 - .1**2) / 20000))
    def rate(n, d):
        return f"{100*n/d:.0f}" + r"\%"

    def inserts(run):
        s = run["summary"]
        return f"{s['success_count']}/{s['num_trials']}"

    b = bc["benchmark_b_contour"]
    c = bc["benchmark_c_slip"]
    validation = training["nominal_validation"]
    values = {
        "HeldSuccess": f"{overall['success_rate']:.0%}".replace("%", r"\%"),
        "HeldCI": f"{lo * 100:.0f}--{hi * 100:.0f}",
        "HeldInBasin": f"{ours['in_basin']['success']}/{ours['in_basin']['n']}",
        "HeldOutBasin": f"{ours['out_of_basin']['success']}/{ours['out_of_basin']['n']}",
        "HeldSafe": f"{overall['force_safe']}/{overall['n']}",
        "HeldJam": str(overall["jammed"]),
        "HeldMax": f"{overall['peak_force_max']:.2f}",
        "HeldN": str(heldout_doc["n"]),
        "HeldSeed": str(heldout_doc["seed"]),
        "FTSuccess": rate(heldout["ft_cbf_qp"]["all"]["success"], heldout_doc["n"]),
        "SpiralSuccess": rate(heldout["impedance_spiral"]["all"]["success"], heldout_doc["n"]),
        "UnconstrainedSuccess": rate(heldout["unconstrained_policy"]["all"]["success"], heldout_doc["n"]),
        "PairedDisagreements": str(sum(x["is_success"] != y["is_success"] for x, y in
                                      zip(ours["trials"], heldout["ft_cbf_qp"]["trials"], strict=True))),
        "FailureCount": str(len(failures)),
        "FailureLatched": str(sum(t["rim_arrest_latched"] for t in failures)),
        "FailureOut": str(sum(not t["in_basin"] for t in failures)),
        "FailureSafe": str(sum(t["is_safe_force"] for t in failures)),
        "FailureZeroDepth": str(sum(t["final_depth_mm"] == 0 for t in failures)),
        "ContourRMSE": f"{b['tactai_ours']['force_rmse_n']:.2f}",
        "ContourSafe": f"{b['tactai_ours']['safe_tracking_pct']:.0f}" + r"\%",
        "ContourPeak": f"{b['tactai_ours']['peak_force_n']:.2f}",
        "ContourNoTankPeak": f"{b['tactai_no_tank']['peak_force_n']:.2f}",
        "SlipDrop": f"{c['tactai_ours']['drop_rate_pct']:.0f}" + r"\%",
        "SlipBaselineDrop": f"{min(v['drop_rate_pct'] for k,v in c.items() if k != 'tactai_ours'):.0f}--"
                            f"{max(v['drop_rate_pct'] for k,v in c.items() if k != 'tactai_ours'):.0f}" + r"\%",
        "TaskTrialCount": str(b["tactai_ours"]["num_trials"]),
        "DemoCount": str(len(training["demonstrations"])),
        "DemoSuccesses": str(sum(t["success"] for t in training["demonstrations"])),
        "NominalSuccess": f"{sum(t['success'] for t in validation)}/{len(validation)}",
        "TrainEpochs": str(training["epochs"]),
        "TrainBatch": str(training["batch_size"]),
        "TrainLR": f"{training['learning_rate']:g}",
        "TrainSeed": str(training["training_seed"]),
        "TrainExamples": str(training["examples"]),
        "ChunkHorizon": str(training["horizon"]),
        "RetractionFloor": f"{params['retraction_floor']:g}",
        "ArrestSpeed": f"{speed:.3f}",
        "SamplingDistance": f"{speed:.3f}",
        "SamplingForce": f"{10000 * speed * .001:.2f}",
        "RimDamping": f"{params['rim_damping']:g}",
        "RimBandLower": f"{params['rim_band_lower'] * 1000:g}",
        "ForceTrigger": f"{params['force_trigger_fraction'] * 15:g}",
        "CenterK": f"{params['kp_lat']:g}",
        "CenterD": f"{params['centering_damping']:g}",
        "SettleTicks": str(params["settle_ticks"]),
        "DeepPush": f"{params['deep_insertion_push']:g}",
        "TelemetryTilt": f"{telemetry['metadata']['env_params']['initial_tilt_deg']:g}",
        "TelemetryPeak": f"{telemetry['kpis']['peak_force_n']:.2f}",
        "TelemetryBaselinePeak": f"{baseline_telemetry['kpis']['peak_force_n']:.2f}",
        "TelemetryBaselineEnd": ("jam stop" if baseline_telemetry["kpis"]["is_jammed"] else "timeout"),
        "TactilePanelN": str(tactile["clean_50hz"]["summary"]["num_trials"]),
        "StressMinSuccess": str(min(r["summary"]["success_count"] for r in tactile.values())),
        "StressPeak": f"{max(r['summary']['max_peak_force_n'] for r in tactile.values()):.2f}",
        "BiasNeg": inserts(tactile["bias_-3deg"]),
        "BiasPos": inserts(tactile["bias_+3deg"]),
        "TensorCop": inserts(tensor["cop_only"]),
        "TensorFull": inserts(tensor["full"]),
        "TensorNone": inserts(tensor["without_cop_tensor"]),
        "GuardDefault": inserts(guard["default"]),
        "GuardMoreDamping": inserts(guard["rim_damping_+20%"]),
        "GuardLowDampingMax": f"{guard['rim_damping_-20%']['summary']['max_peak_force_n']:.2f}",
        "GuardLowDampingSafe": str(guard["rim_damping_-20%"]["summary"]["safe_force_count"]),
        "GuardN": str(guard["default"]["summary"]["num_trials"]),
        "RobustPerPoint": str(robustness["protocol"]["episodes_per_controller_per_parameter_point"]),
        "AblationN": str(bc["ablation_study"]["tactai_full"]["num_trials"]),
        "ATankSuccess": str(bc["ablation_study"]["tactai_full"]["success_count"]),
        "ANoTankSuccess": str(bc["ablation_study"]["tactai_no_tank"]["success_count"]),
    }
    lines = ["% Generated by scripts/generate_manuscript_results.py"]
    lines.extend(fr"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items())
    lines.append(r"\newcommand{\AblationRows}{%")
    for case in bc["ablation_study"].values():
        lines.append(f"{case['label']} (A) & {case['success_count']} & {case['jamming_count']} & "
                     fr"\${case['mean_peak_force_n']:.2f}\pm{case['std_peak_force_n']:.2f}\$ & "
                     f"{case['max_peak_force_n']:.2f} {END_ROW}")
    for key, label in (("tactai_ours", "Full TactAI (B)"), ("tactai_no_tank", "Without energy tank (B)"),
                       ("wrist_ft_admittance", "Wrist F/T (B)")):
        row = b[key]
        lines.append(fr"{label} & -- & -- & \${row['mean_peak_force_n']:.2f}\pm{row['std_peak_force_n']:.2f}\$ & "
                     f"{row['peak_force_n']:.2f} {END_ROW}")
    lines.append("}")
    lines.append(r"\newcommand{\BenchmarkBCRows}{%")
    for key, label in [
        ("unconstrained_farm", "Unconstrained"),
        ("fixed_impedance_k200", "Fixed impedance"),
        ("wrist_ft_admittance", "Wrist F/T"),
        ("tactai_ours", "TactAI"),
    ]:
        b = bc["benchmark_b_contour"][key]
        c = bc["benchmark_c_slip"][key]
        lines.append(fr"{label} & {b['safe_tracking_pct']:.0f}\% & {b['force_rmse_n']:.2f} & "
                     fr"{c['drop_rate_pct']:.0f}\% & {c['mean_slip_mm']:.2f} {END_ROW}")
    lines.append("}")
    lines.append(r"\newcommand{\RobustnessRows}{%")
    for sweep, parameter, label in (
        ("stiffness_sweep", "stiffness_n_per_m", r"$K$ (N/m)"),
        ("friction_sweep", "friction_mu", r"$\mu$"),
        ("latency_sweep", "latency_ms", r"$\tau_{\mathrm{delay}}$ (ms)"),
    ):
        for row in robustness[sweep]:
            lines.append(f"{label} & {row[parameter]:g} & {row['tactai_success_count']} & "
                         f"{row['tactai_peak_force_n']:.2f} {END_ROW}")
    lines.append("}")
    lines.append(r"\newcommand{\HeldFailureRows}{%")
    for trial in failures:
        basin = "In" if trial["in_basin"] else "Out"
        wedge = trial["termination_reason"].replace("_", " ").capitalize()
        latch = "Yes" if trial["rim_arrest_latched"] else "No"
        lines.append(f"{trial['trial_seed']} & {basin} & {wedge} & {latch} & {trial['peak_force_n']:.2f} {END_ROW}")
    lines.append("}")
    for group, command in (("tensor", "TensorRows"), ("guard", "GuardRows")):
        lines.append(fr"\newcommand{{\{command}}}{{%")
        for name, case in audit["results"][group].items():
            stats = case["summary"]
            label = name.replace("_", " ").replace("%", r"\%")
            lines.append(
                f"{label} & {stats['success_count']} & {stats['jamming_count']} & "
                f"{stats['safe_force_count']} & {stats['max_peak_force_n']:.2f} {END_ROW}"
            )
        lines.append("}")
    from scripts.grasp_manuscript import grasp_macros
    return "\n".join(lines + grasp_macros()).replace(r"\$", "$") + "\n"


def main():
    (ROOT / "docs" / "benchmark_results.tex").write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
