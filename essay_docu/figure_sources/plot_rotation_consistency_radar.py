from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "rotation_consistency"
OUTPUT = ROOT / "essay_docu" / "figs" / "06_rotation_consistency_radar"

SERIES = (
    (
        "Standard DeiT-Tiny",
        "standard_balanced1000_dense15.json",
        "#64748B",
        "--",
    ),
    (
        "SHARE half4d4r",
        "share_half4d4r_pe_look_balanced1000_dense15.json",
        "#E58A3A",
        "-",
    ),
    (
        "SHARE half6d3r + dual Look",
        "share_half6d3r_pe_image_feature_g3_balanced1000_dense15.json",
        "#6857C7",
        "-",
    ),
    (
        "GE-ViT p4 local",
        "gevit_p4_local_balanced1000_dense15.json",
        "#149A8A",
        "-",
    ),
)


def load_rows(filename: str) -> list[dict[str, float]]:
    path = RUNS / filename
    return json.loads(path.read_text(encoding="utf-8"))["angles"]


def close(values: list[float]) -> np.ndarray:
    return np.asarray([*values, values[0]], dtype=float)


def style_axis(ax: plt.Axes, title: str, radial_ticks: list[float]) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        np.arange(0, 360, 45),
        labels=[f"{angle}°" for angle in range(0, 360, 45)],
        fontsize=9,
        color="#334155",
    )
    ax.set_rlabel_position(22.5)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{value:.0f}" for value in radial_ticks], fontsize=8)
    ax.grid(color="#CBD5E1", linewidth=0.7, alpha=0.8)
    ax.spines["polar"].set_color("#94A3B8")
    ax.spines["polar"].set_linewidth(0.9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#172033")


def main() -> None:
    loaded = [(label, load_rows(filename), color, linestyle) for label, filename, color, linestyle in SERIES]
    angles = np.deg2rad([row["angle_degrees"] for row in loaded[0][1]])
    angles = np.concatenate([angles, angles[:1]])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 6.2),
        subplot_kw={"projection": "polar"},
    )
    style_axis(axes[0], "Classification accuracy by rotation", [30, 40, 50, 60])
    style_axis(axes[1], "Prediction agreement with 0°", [40, 55, 70, 85, 100])
    axes[0].set_ylim(25, 62)
    axes[1].set_ylim(35, 101)

    for label, rows, color, linestyle in loaded:
        top1 = close([row["top1"] for row in rows])
        agreement = close([row["agreement"] for row in rows])
        for ax, values in zip(axes, (top1, agreement), strict=True):
            ax.plot(
                angles,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                marker="o",
                markersize=2.8,
                markeredgewidth=0,
                label=label,
                zorder=3,
            )
        if label in {"SHARE half6d3r + dual Look", "GE-ViT p4 local"}:
            axes[0].fill(angles, top1, color=color, alpha=0.045, zorder=1)
            axes[1].fill(angles, agreement, color=color, alpha=0.045, zorder=1)

    fig.suptitle(
        "Rotation-consistency probe on ImageNet-100",
        fontsize=15,
        fontweight="bold",
        color="#172033",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "1,000 class-balanced validation images · 15° increments · no test-time adaptation",
        ha="center",
        fontsize=9.5,
        color="#64748B",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9.2,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=2.5,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.035, right=0.965, top=0.79, bottom=0.14, wspace=0.18)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
