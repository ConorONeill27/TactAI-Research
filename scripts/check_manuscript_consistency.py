"""Check the generated manuscript and figure annotations against benchmark logs."""

import re

import pymupdf

from scripts.check_benchmark_logs import DATA, ROOT, read
from scripts.generate_manuscript_results import render
from scripts.grasp_manuscript import load
from scripts.run_heldout_validation import summarize
from scripts.run_frozen_grasp import fingerprint


def pdf_text(path, *, check_bounds=False):
    with pymupdf.open(path) as pdf:
        text = []
        for page in pdf:
            if check_bounds:
                frame = page.rect + (-.5, -.5, .5, .5)
                for block in page.get_text("blocks"):
                    assert pymupdf.Rect(block[:4]) in frame, (path, page.number, block[:4])
                for image in page.get_image_info():
                    assert pymupdf.Rect(image["bbox"]) in frame, (path, page.number, image["bbox"])
            text.append(page.get_text())
    return " ".join(" ".join(text).split())


def require(text, expected, artifact):
    for item in expected:
        if item not in text:
            raise ValueError(f"{artifact} is missing logged value or label {item!r}")


def main():
    import json
    manifest = load("grasp_manifest")
    assert fingerprint() == manifest["files"]
    for panel in ("a1", "a2", "a1_ablation", "a2_ablation", "a2_stress"):
        data = load(panel)
        if panel in ("a1", "a2"):
            assert data["seed"] == manifest[f"{panel}_seed"]
        for k, run in data["controllers"].items():
            assert len(run["trials"]) == data["n"]
            assert run["all"] == json.loads(json.dumps(summarize(run["trials"])))
            if panel in ("a1", "a2"):
                assert data["failures"][k] == [t for t in run["trials"] if not t["is_success"]]
    docs = ROOT / "docs"
    expected = render()
    assert (docs / "benchmark_results.tex").read_text(encoding="utf-8") == expected
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}\n]*)\}", expected))
    source = (docs / "main.tex").read_text(encoding="utf-8")
    row_macros = ("GraspComparisonRows", "BenchmarkBCRows", "AblationRows",
                  "SharedAblationRows", "CounterTiltRows", "UnknownFailureRows", "HeldFailureRows",
                  "RobustnessRows", "GuardRows")
    tables = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", source, re.S)
    assert len(tables) == len(row_macros)
    for table, name in zip(tables, row_macros, strict=True):
        assert f"\\{name}" in table
        assert re.search(r"\\midrule\s*\\[A-Za-z]+Rows\s*\\bottomrule", table)

    paper = pdf_text(docs / "research_paper.pdf", check_bounds=True)
    require(paper, [macros[key].replace(r"\%", "%") for key in (
        "HeldSuccess", "FTSuccess", "HeldSafe", "HeldInBasin", "HeldSeed",
        "UnknownSeed", "UnknownSuccess", "UnknownFT", "UnknownMax", "UnknownSafe",
        "UnknownStressPeak", "ArrestSpeed", "TelemetryPeak", "TelemetryBaselinePeak")],
        "compiled manuscript")
    with pymupdf.open(docs / "research_paper.pdf") as compiled:
        lines = [line.strip() for page in compiled for line in page.get_text().splitlines()]
    references = next(i for i, line in enumerate(lines) if line.casefold() == "references")
    assert not any(re.fullmatch(r"TABLE\s+[IVXL]+", line, re.I) for line in lines[references+1:])
    bench_fig = pdf_text(docs / "figures/unified_grasp_comparison.pdf")
    for panel in ("a1", "a2"):
        for run in load(panel)["controllers"].values():
            require(bench_fig, [f"{100 * run['all'][key]:.0f}%"
                                for key in ("success_rate", "force_safe_rate")], "comparison figure")

    bc = read(DATA / "benchmarks_bc_ablation_summary.json")
    bc_fig = pdf_text(docs / "figures/phase3_benchmarks_bc_ablation.pdf")
    for key in ("unconstrained_farm", "wrist_ft_admittance", "tactai_no_tank", "tactai_ours"):
        require(bc_fig, [f"({bc['benchmark_b_contour'][key]['trials'][0]['peak_force_n']:.1f} N)"],
                "contour figure")
    for key in ("unconstrained_farm", "wrist_ft_admittance", "tactai_ours"):
        require(bc_fig, [f"({bc['benchmark_c_slip'][key]['mean_slip_mm']:.2f} mm)"], "slip figure")
    for row in bc["ablation_study"].values():
        require(bc_fig, [f"{row['success_rate_pct']:.0f}%", f"{row['mean_peak_force_n']:.1f}N"],
                "ablation figure")

    telemetry = ROOT / "data/phase2_validation"
    cases = [read(telemetry / name)["kpis"] for name in
             ("case1_baseline_jamming_summary.json", "case2_tactai_success_summary.json")]
    trace_fig = pdf_text(docs / "figures/phase2_validation_benchmark.pdf")
    require(trace_fig, [f"({case['final_insertion_depth_mm']:.2f} mm)" for case in cases],
            "illustrative insertion figure")
    require(trace_fig, ["Jam termination" if cases[0]["is_jammed"] else "Baseline timeout"],
            "illustrative insertion figure")
    workbench = read(telemetry / "workbench_snapshots.json")
    workbench_fig = pdf_text(docs / "figures/simulation_3d_overview.pdf")
    require(workbench_fig, [f"{workbench[key]['peak_force_n']:.1f} N"
                            for key in ("aligned", "wedged")], "workbench render")
    assert workbench["wedged"]["peak_force_n"] == cases[0]["peak_force_n"]
    assert round(workbench["wedged"]["insertion_depth_m"] * 1000, 2) == round(cases[0]["final_insertion_depth_mm"], 2)
    print("Verified the frozen runs, all nine generated tables, figure annotations, and bibliography order.")


if __name__ == "__main__":
    main()
