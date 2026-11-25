"""Render the frozen grasp comparison and the independent A2 perturbation panel."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.grasp_manuscript import ROOT, load
from scripts.generate_manuscript_results import load as load_legacy
from scripts.run_phase3_validation import generate_bc_ablation_figure


def main():
    folder = ROOT / "docs/figures"
    bc = load_legacy("benchmarks_bc_ablation_summary")
    with np.load(ROOT / "data/phase3_validation/benchmark_bc_plot_traces.npz") as trace:
        bc["_traces_b"] = {k.removeprefix("contour_"): trace[k] for k in trace if k.startswith("contour_")}
        bc["_traces_c"] = {k: trace[k] for k in trace if not k.startswith("contour_")}
    generate_bc_ablation_figure(bc, str(folder / "phase3_benchmarks_bc_ablation.png"))
    plot_comparison(folder)
    plot_stress(folder)


def plot_comparison(folder):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.25), layout="constrained", sharey=True)
    labels = ("Unconstrained\npolicy", "Fixed\nimpedance", "Wrist F/T-\nonly QP",
              "Impedance\n+ spiral", "TactAI")
    x = np.arange(len(labels))
    for ax, panel, title in zip(axes, ("a1", "a2"), ("A1: Known grasp", "A2: Unknown in-hand tilt")):
        data = load(panel)
        rows = [r["all"] for r in data["controllers"].values()]
        for shift, key, label, color in ((-.18, "success_rate", "Insertion", "#177d70"),
                                         (.18, "force_safe_rate", "Peak ≤15 N", "#4b70b5")):
            bars = ax.bar(x + shift, [r[key] * 100 for r in rows], .34, color=color, label=label)
            ax.bar_label(bars, fmt="%.0f%%", padding=3 if shift < 0 else 8, fontsize=8)
        ax.set(title=f"{title} (N={data['n']})", xticks=x, xticklabels=labels, ylim=(0, 117))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Trials (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center",
               frameon=False, ncol=2, fontsize=8)
    fig.savefig(folder / "unified_grasp_comparison.pdf")
    plt.close(fig)


def plot_stress(folder):
    rows = load("a2_stress")["controllers"]
    y = np.arange(len(rows))
    fig, (ax, force) = plt.subplots(1, 2, figsize=(9, 5.4), sharey=True, layout="constrained")
    ax.barh(y-.18, [r["all"]["success_rate"]*100 for r in rows.values()], .35,
            color="#177d70", label="Insertion")
    ax.barh(y+.18, [r["all"]["jam_rate"]*100 for r in rows.values()], .35,
            color="#c15442", label="Benchmark jam")
    for i, run in enumerate(rows.values()):
        force.scatter([t["peak_force_n"] for t in run["trials"]],
                      [i]*len(run["trials"]), color="#4b70b5", s=11, alpha=.55)
    ax.set(yticks=y, yticklabels=[k.replace("_", " ") for k in rows], xlim=(0, 105),
           xlabel="Trials (%)")
    ax.invert_yaxis()
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2)
    force.axvline(15, color="#c15442", linestyle="--")
    force.set_xlabel("Episode peak force (N)")
    fig.savefig(folder / "a2_tactile_stress.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
