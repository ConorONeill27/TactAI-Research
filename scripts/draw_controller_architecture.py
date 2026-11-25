"""Draw the supervisory controller and its contact-state guards as a vector PDF."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    ax.set(xlim=(-0.6, 10.6), ylim=(-0.3, 3.6))
    ax.axis("off")

    def box(x, y, label, color):
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="#334155"))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2))

    inputs, compute, stages = "#e0f2fe", "#dcfce7", "#fef3c7"
    box(0.55, 3.1, "20 Hz policy\nproxy", inputs)
    box(0.55, 2.25, "25–50 Hz target\ntactile invariants", inputs)
    box(0.55, 1.4, "1 kHz wrist pose\nand wrench", inputs)
    box(3.15, 2.0, "Complementary\nobserver + latch", compute)
    box(5.55, 2.6, "7-variable OSQP\nwrench filter", compute)
    box(5.55, 1.5, "Grip barrier\nseparate channel", compute)
    box(8.2, 2.6, "Net-wrench energy tank\nand actuator limits", compute)
    arrow(1.15, 3.1, 4.85, 2.78)
    arrow(1.42, 2.25, 2.38, 2.1)
    arrow(1.47, 1.4, 2.38, 1.86)
    arrow(3.94, 2.11, 4.85, 2.5)
    arrow(3.94, 1.89, 4.78, 1.56)
    arrow(6.28, 2.6, 7.17, 2.6)
    arrow(9.16, 2.6, 9.65, 2.6)
    ax.text(9.72, 2.6, "Arm\nwrench", va="center", fontsize=9)

    box(0.55, 0.37, "Stage 1\nApproach", stages)
    box(5.55, 0.37, "Stage 2\nUn-wedge", stages)
    box(9.45, 0.37, "Stage 3\nAlign and insert", stages)
    arrow(1.1, 0.37, 4.97, 0.37)
    arrow(6.12, 0.37, 8.63, 0.37)
    arrow(6.26, 0.71, 6.26, 2.17)
    ax.text(3.02, 0.49, r"$G_{1\to2}$: force $>0.1$ N and" "\n"
            r"($h_{\rm jam}<0$ or tilt $\geq0.65^\circ$)",
            ha="center", va="bottom", fontsize=9)
    ax.text(7.47, 0.49, r"$G_{2\to3}$: $z>-3.2$ mm" "\n"
            "and dwell elapsed", ha="center", va="bottom", fontsize=9)
    fig.savefig(Path(__file__).resolve().parents[1] / "docs/figures/controller_architecture.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
