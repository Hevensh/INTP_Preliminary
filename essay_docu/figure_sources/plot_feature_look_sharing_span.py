from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_CANDIDATE = ROOT / "candidate_paper_figures"
OUT_PAPER = ROOT.parent / "figs"

G = np.array([1, 2, 3, 4, 6, 12])
X = np.array([0.00, 0.95, 1.90, 2.85, 4.05, 6.20])
GROUPS = np.array([11, 6, 4, 3, 2, 1])

# Schedule-aligned ImageNet-100 runs from runs/README.md.
PE_CENTER = np.array([54.64, 54.64, 55.00, 55.18, 54.46, 54.48])
DUAL_LOOK = np.array([54.76, 55.02, 55.54, 54.96, 54.92, 54.70])

INK = "#263248"
MUTED = "#667289"
GRID = "#DCE2EB"
AXIS = "#A9B4C5"
PURPLE = "#7C67C7"
PURPLE_LIGHT = "#D9D1F0"
BLUE = "#377FC0"
BLUE_LIGHT = "#C9DFF1"
GOLD = "#E99A3A"


def add_value_labels(ax, xs, ys, color, offsets, weight="semibold"):
    for x, y, dy in zip(xs, ys, offsets):
        ax.text(
            x,
            y + dy,
            f"{y:.2f}",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=10.5,
            fontweight=weight,
            color=color,
            zorder=8,
        )


def build_figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 24,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(12.4, 7.6), facecolor="white")
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.55, 1.15],
        left=0.085,
        right=0.975,
        top=0.86,
        bottom=0.14,
        hspace=0.08,
    )
    ax = fig.add_subplot(gs[0])
    ax_groups = fig.add_subplot(gs[1], sharex=ax)

    fig.text(
        0.085,
        0.965,
        r"Hex $\mathcal{G}_{6,3}$, ImageNet-100, 20 epochs; G denotes consecutive blocks sharing one pose-probe evaluation.",
        fontsize=12.5,
        color=MUTED,
        ha="left",
        va="top",
    )

    ax.set_ylim(54.33, 55.82)
    ax.set_xlim(-0.42, 6.62)
    ax.set_yticks(np.arange(54.4, 55.9, 0.2))
    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", colors=INK, width=1.0, length=5)
    ax.set_ylabel("Best Top-1 accuracy (%)", color=INK, labelpad=8)

    ax.axhline(54.54, color="#929CAD", linewidth=1.5, linestyle=(0, (4, 4)), zorder=1)
    ax.axhline(55.04, color=GOLD, linewidth=1.4, linestyle=(0, (2, 4)), alpha=0.9, zorder=1)
    ax.text(6.50, 54.555, "PE only 54.54", ha="right", va="bottom", fontsize=10.5, color="#707B8E")
    ax.text(6.50, 55.055, "PE + Image Look 55.04", ha="right", va="bottom", fontsize=10.5, color="#B67122")

    ax.plot(
        X,
        PE_CENTER,
        color=PURPLE,
        linewidth=3.0,
        marker="o",
        markersize=8.5,
        markerfacecolor="white",
        markeredgecolor=PURPLE,
        markeredgewidth=2.4,
        label="PE + Feature Look",
        zorder=5,
    )
    ax.plot(
        X,
        DUAL_LOOK,
        color=BLUE,
        linewidth=3.0,
        marker="o",
        markersize=8.5,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markeredgewidth=2.4,
        label="PE + Image Look + Feature Look",
        zorder=6,
    )

    # Fill each series' best observed point.
    ax.scatter([X[3]], [PE_CENTER[3]], s=92, color=PURPLE, edgecolor="white", linewidth=1.5, zorder=7)
    ax.scatter([X[2]], [DUAL_LOOK[2]], s=92, color=BLUE, edgecolor="white", linewidth=1.5, zorder=7)

    add_value_labels(ax, X, PE_CENTER, PURPLE, [-0.07, -0.07, -0.07, 0.09, -0.08, -0.08])
    add_value_labels(ax, X, DUAL_LOOK, BLUE, [0.07, 0.07, 0.08, 0.07, 0.07, 0.07])

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.115),
        frameon=False,
        ncol=2,
        handlelength=2.6,
        columnspacing=2.0,
    )

    bar_colors = [PURPLE_LIGHT] * len(G)
    bar_edges = [PURPLE] * len(G)
    bars = ax_groups.bar(X, GROUPS, width=0.60, color=bar_colors, edgecolor=bar_edges, linewidth=1.4)
    for bar, value in zip(bars, GROUPS):
        ax_groups.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
            color=INK,
        )

    ax_groups.set_ylim(0, 12.6)
    ax_groups.set_yticks([])
    ax_groups.set_xticks(X, [str(v) for v in G])
    ax_groups.tick_params(axis="x", colors=MUTED, width=1.0, length=4)
    ax_groups.set_xlabel("Sharing span G (blocks)", fontsize=13, labelpad=8)
    ax_groups.set_ylabel("Probe\ngroups", rotation=0, ha="right", va="center", labelpad=18, fontsize=11.5)
    for spine in ("top", "right", "left"):
        ax_groups.spines[spine].set_visible(False)
    ax_groups.spines["bottom"].set_color(AXIS)

    fig.text(
        0.085,
        0.045,
        "Source: runs/README.md. All values are single-seed, 20-epoch ImageNet-100 results.",
        fontsize=10.2,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    return fig


def main():
    OUT_CANDIDATE.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    for path in (
        OUT_CANDIDATE / "05_feature_look_sharing_span.pdf",
        OUT_PAPER / "05_feature_look_sharing_span.pdf",
    ):
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT_CANDIDATE / "05_feature_look_sharing_span.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
