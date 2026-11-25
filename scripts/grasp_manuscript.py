"""Generate the paired grasp comparison, trace, and ablation tables."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/phase3_validation"
END = r"\\"


def load(panel):
    return json.loads((DATA / f"unified_{panel}.json").read_text(encoding="utf-8"))


def grasp_macros():
    a1, a2, stress, dev1, dev2 = map(load, ("a1", "a2", "a2_stress", "a1_ablation", "a2_ablation"))
    full = "tactile_safety_filter_ours"
    ft = "ft_cbf_qp"
    s2 = a2["controllers"][full]["all"]
    f2 = a2["controllers"][ft]["all"]
    example = dev2["controllers"][full]["trials"][0]
    stress_rows = [r["all"] for r in stress["controllers"].values()]
    failures = a2["failures"][full]
    stress_runs = stress["controllers"]
    values = dict(UnknownSeed=a2["seed"], UnknownSuccess=f"{s2['success_rate']:.0%}",
                  UnknownFT=f"{f2['success_rate']:.0%}", UnknownSafe=f"{s2['force_safe']}/{s2['n']}",
                  UnknownFTSafe=f"{f2['force_safe']}/{f2['n']}", UnknownMax=f"{s2['peak_force_max']:.2f}",
                  UnknownJam=s2["jammed"], UnknownFailures=len(failures),
                  UnknownStressN=stress_rows[0]["n"],
                  UnknownStressLow=min(s["success"] for s in stress_rows),
                  UnknownStressHigh=max(s["success"] for s in stress_rows),
                  UnknownStressSafe=min(s["force_safe"] for s in stress_rows),
                  UnknownStressPeak=f"{max(s['peak_force_max'] for s in stress_rows):.2f}",
                  UnknownClean=stress_runs["clean_50hz"]["all"]["success"],
                  UnknownHighNoise=stress_runs["noise_0.25mm_0.1"]["all"]["success"],
                  UnknownBiasNegOne=stress_runs["bias_-1deg"]["all"]["success"],
                  UnknownBiasPosOne=stress_runs["bias_+1deg"]["all"]["success"],
                  UnknownDropHigh=stress_runs["drop_30%"]["all"]["success"],
                  ExampleGrasp=f"{example['grasp_rest_deg'][0]:+g}",
                  ExampleWrist=f"{example['initial_wrist_deg'][0]:+g}",
                  UnknownRimDamping=f"{load('grasp_manifest')['a2']['rim_damping']:g}")
    for macro, panel in (("KnownCopAgreement", dev1), ("UnknownCopAgreement", dev2)):
        full_stats = panel["controllers"][full]["all"]
        cop_stats = panel["controllers"]["cop_only"]["all"]
        if (full_stats["success"], full_stats["n"]) != (cop_stats["success"], cop_stats["n"]):
            raise ValueError("The two development ablations no longer match.")
        values[macro] = f"{full_stats['success']}/{full_stats['n']}"
    lines = [fr"\newcommand{{\{k}}}{{{str(v).replace('%', chr(92) + '%')}}}" for k,v in values.items()]
    lines.append(r"\newcommand{\GraspComparisonRows}{%")
    for key, run in a1["controllers"].items():
        cells = [run["label"]]
        for document in (a1, a2):
            s = document["controllers"][key]["all"]
            cells.extend([f"{s['success']}/{s['n']}", str(s["jammed"]), str(s["force_safe"]),
                          f"{s['peak_force_mean']:.2f}", f"{s['peak_force_max']:.2f}"])
        lines.append(" & ".join(cells) + " " + END)
    lines.append("}")
    lines.append(r"\newcommand{\SharedAblationRows}{%")
    for panel, data in (("A1", dev1), ("A2", dev2)):
        for run in data["controllers"].values():
            s = run["all"]
            lines.append(f"{panel} & {run['label']} & {s['success']}/{s['n']} & "
                         f"{s['jammed']} & {s['force_safe']} & {s['peak_force_max']:.2f} {END}")
    lines.append("}")
    lines.append(r"\newcommand{\CounterTiltRows}{%")
    for key in (ft, full):
        run = dev2["controllers"][key]
        t = run["trials"][0]
        lines.append(f"{run['label']} & {t['final_wrist_deg'][0]:+.2f} & "
                     f"{t['final_peg_deg'][0]:+.2f} & {t['final_residual_deg'][0]:+.2f} & "
                     f"{t['peak_force_n']:.2f} & {t['final_depth_mm']:.2f} {END}")
    lines.append("}")
    lines.append(r"\newcommand{\UnknownFailureRows}{%")
    for t in failures:
        lines.append(f"{t['trial_seed']} & {t['final_depth_mm']:.2f} & {t['peak_force_n']:.2f} & "
                     f"{'Yes' if t['rim_arrest_latched'] else 'No'} {END}")
    if not failures:
        lines.append(r"\multicolumn{4}{l}{No failed insertions} " + END)
    lines.append("}")
    return lines
