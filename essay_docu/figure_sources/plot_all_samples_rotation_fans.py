from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs" / "rotation_consistency" / "all5000_six_models_local5.json"
OUTPUT = ROOT / "essay_docu" / "figs" / "08_all5000_six_models_rotation_fans"

SERIES = (
    ("Standard DeiT-Tiny", "Standard", "#64748B", "--", "o", 1.9),
    (r"SHARE $\mathcal{G}_{4,4}$", "SHARE-4d4r", "#E58A3A", "-", "s", 2.25),
    (r"SHARE $\mathcal{G}_{6,3}$ + dual Look", "SHARE-6d3r", "#6857C7", "-", "o", 2.45),
    ("Equi-ViT / GMR", "Equi-GMR", "#3B82B9", ":", "^", 1.9),
    ("ARC Adaptive", "ARC-Adaptive", "#C75B8A", "-.", "v", 1.9),
    ("GE-ViT p4 local", "GE-ViT-p4", "#149A8A", "-", "D", 2.2),
)


def signed_angle(angle: float) -> float:
    return angle - 360.0 if angle > 180.0 else angle


def style_fan(
    ax: plt.Axes,
    *,
    radial_limits: tuple[float, float],
    radial_ticks: list[float],
    radial_suffix: str,
    title: str,
) -> None:
    ax.set_facecolor("#FBFCFE")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-62)
    ax.set_thetamax(62)
    tick_angles = list(range(-30, 31, 5))
    ax.set_thetagrids(
        [2 * angle for angle in tick_angles],
        labels=[f"{angle}°" for angle in tick_angles],
        fontsize=8,
        color="#475569",
    )
    ax.set_ylim(*radial_limits)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(
        [f"{value:g}{radial_suffix}" for value in radial_ticks],
        fontsize=7.6,
        color="#64748B",
    )
    ax.set_rlabel_position(67)
    ax.grid(color="#CBD5E1", linewidth=0.7, alpha=0.9)
    ax.spines["polar"].set_color("#94A3B8")
    ax.spines["polar"].set_linewidth(0.9)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#172033", pad=12)


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.55),
        subplot_kw={"projection": "polar"},
    )
    style_fan(
        axes[0],
        radial_limits=(38, 58),
        radial_ticks=[40, 44, 48, 52, 56],
        radial_suffix="%",
        title="Absolute Top-1 accuracy",
    )
    style_fan(
        axes[1],
        radial_limits=(68, 84),
        radial_ticks=[70, 74, 78, 82],
        radial_suffix="%",
        title="Absolute Top-5 accuracy",
    )
    style_fan(
        axes[2],
        radial_limits=(0, 8.5),
        radial_ticks=[2, 4, 6, 8],
        radial_suffix="",
        title="Prediction drift from 0° (100×JSD, lower is better)",
    )

    for label, key, color, linestyle, marker, width in SERIES:
        rows = sorted(
            payload["results"][key]["angles"],
            key=lambda row: signed_angle(row["angle_degrees"]),
        )
        input_angles = np.asarray(
            [signed_angle(row["angle_degrees"]) for row in rows], dtype=float
        )
        display_angles = np.deg2rad(2.0 * input_angles)
        top1 = np.asarray([row["top1"] for row in rows], dtype=float)
        top5 = np.asarray([row["top5"] for row in rows], dtype=float)
        jsd = 100.0 * np.asarray(
            [row["js_divergence"] for row in rows], dtype=float
        )
        # JSD is identically zero at its 0-degree reference. Leaving that
        # tautological point blank keeps the positive and negative rotation
        # branches separate instead of drawing two visually dominant spokes
        # into the fan center.
        jsd[input_angles == 0.0] = np.nan
        for ax, values in zip(axes, (top1, top5, jsd), strict=True):
            ax.plot(
                display_angles,
                values,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=width,
                marker=marker,
                markersize=3.7,
                markerfacecolor="white",
                markeredgewidth=1.1,
                zorder=4,
            )
        axes[0].fill_between(
            display_angles,
            38,
            top1,
            color=color,
            alpha=0.035 if key == "SHARE-6d3r" else 0.012,
            zorder=1,
        )
        axes[1].fill_between(
            display_angles,
            68,
            top5,
            color=color,
            alpha=0.035 if key == "SHARE-6d3r" else 0.012,
            zorder=1,
        )
        axes[2].fill_between(
            display_angles,
            0,
            jsd,
            color=color,
            alpha=0.035 if key == "SHARE-6d3r" else 0.012,
            zorder=1,
        )

    fig.text(
        0.5,
        0.155,
        "Measured ±30° interval expanded to a 120° display fan; angle labels show true input rotation",
        ha="center",
        fontsize=8.8,
        color="#64748B",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.4,
        bbox_to_anchor=(0.5, 0.050),
        handlelength=2.5,
        columnspacing=1.55,
    )
    fig.subplots_adjust(left=0.012, right=0.988, top=0.92, bottom=0.225, wspace=-0.105)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
