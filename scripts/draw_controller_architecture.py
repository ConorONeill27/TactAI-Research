"""Draw the learned-policy supervisor and hybrid automaton as a vector PDF."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath


def main():
    fig, ax = plt.subplots(figsize=(11.8, 4.25), layout="constrained")
    ax.set(xlim=(0, 15.1), ylim=(0, 6.5))
    ax.axis("off")
    ink = "#25374d"
    inputs, compute, stages = "#e8f2fc", "#dff3e9", "#ffefce"

    def box(x, y, width, height, label, color):
        ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                  boxstyle="round,pad=0.02,rounding_size=0.11",
                                  facecolor=color, edgecolor=ink, linewidth=1))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=ink)

    def route(*points):
        if len(points) > 2:
            ax.add_patch(PathPatch(PlotPath(points[:-1]), fill=False, color=ink, linewidth=1.15))
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     mutation_scale=10, linewidth=1.15, color=ink,
                                     shrinkA=0, shrinkB=1))

    box(1.55, 5.8, 2.65, .84, "20 Hz learned policy\n(MLP action chunks)", inputs)
    box(1.55, 4.35, 2.65, .84, "25–50 Hz target\nTactile invariants", inputs)
    box(1.55, 3.15, 2.65, .84, "1 kHz wrist pose\nand wrench", inputs)
    box(5.0, 4.25, 2.6, .88, "Complementary\nobserver + latch", compute)
    box(8.25, 5.8, 2.6, .84, "7-variable OSQP\nwrench filter", compute)
    box(11.65, 5.8, 2.75, .84, "Net-wrench energy tank\nand actuator limits", compute)
    box(1.40, 1.55, 2.50, .74, "Grip barrier\n(separate channel)", compute)
    route((4.20, 3.79), (4.20, 2.72), (1.40, 2.72), (1.40, 1.94))
    ax.text(2.30, 2.48, r"$\hat{\rho}_{\mathrm{stick}},\,\hat{\mathbf{F}}_{xy}$",
            va="center", ha="center", fontsize=9, color=ink)
    route((2.89, 5.8), (6.93, 5.8))
    route((2.89, 4.35), (3.67, 4.35))
    route((2.89, 3.15), (3.22, 3.15), (3.22, 4.04), (3.67, 4.04))
    route((6.32, 4.42), (7.1, 4.42), (7.1, 5.37))
    route((9.58, 5.8), (10.25, 5.8))
    route((13.05, 5.8), (13.57, 5.8))
    route((1.40, 1.16), (1.40, .76))
    ax.text(13.65, 5.8, "Arm wrench", va="center", color=ink, fontsize=9.5)
    ax.text(1.40, .56, "Grip force", va="center", ha="center", color=ink, fontsize=9.5)

    ax.add_patch(FancyBboxPatch((2.8, .25), 10.90, 2.35,
                               boxstyle="round,pad=0.01,rounding_size=0.13",
                               fill=False, edgecolor=ink, linestyle=(0, (5, 4)), linewidth=1.1))
    ax.text(8.25, 2.25, "3-Stage Hybrid Automaton", ha="center",
            va="center", fontsize=10.5, weight="bold", color=ink)
    route((8.25, 2.62), (8.25, 5.37))
    route((5.0, 3.79), (5.0, 2.62))
    ax.text(5.18, 3.26, r"$\hat{\mathbf{F}},\,\hat{\boldsymbol{\theta}}_{\rm est},\,h_{\rm jam}$",
            ha="left", va="center", fontsize=9, color=ink)

    for x, label in [(4.0, "Stage 1\nApproach"), (7.93, "Stage 2\nUn-wedge"),
                     (11.91, "Stage 3\nAlign & insert")]:
        box(x, .94, 1.78, .75, label, stages)
    route((4.92, .94), (7.0, .94))
    route((8.85, .94), (10.98, .94))
    ax.text(5.98, 1.1, r"$G_{1\to2}$: force $>0.1$ N and" "\n"
            r"($h_{\rm jam}<0$ or tilt $\geq0.65^\circ$)",
            ha="center", va="bottom", fontsize=7.6, color=ink)
    ax.text(9.91, 1.1, r"$G_{2\to3}$: $z>-3.2$ mm" "\n" "and dwell elapsed",
            ha="center", va="bottom", fontsize=7.6, color=ink)
    fig.savefig(Path(__file__).resolve().parents[1] / "docs/figures/controller_architecture.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
