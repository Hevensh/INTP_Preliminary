from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs" / "rotation_consistency" / "common_correct_fullval_local5.json"
OUTPUT = ROOT / "essay_docu" / "figs" / "07b_local_rotation_common_correct_fan"

SERIES = (
    ("Standard DeiT-Tiny", "Standard", "#64748B", "--", "o"),
    ("SHARE half4d4r", "SHARE-4d4r", "#E58A3A", "-", "s"),
    ("SHARE half6d3r + dual Look", "SHARE-6d3r", "#6857C7", "-", "o"),
    ("GE-ViT p4 local", "GE-ViT-p4", "#149A8A", "-", "D"),
)


def signed_angle(angle: float) -> float:
    return angle - 360.0 if angle > 180.0 else angle


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    sample_count = payload["protocol"]["common_correct_images"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#FBFCFE")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-62)
    ax.set_thetamax(62)

    radial_floor = 86.0
    for label, key, color, linestyle, marker in SERIES:
        rows = sorted(
            payload["results"][key]["angles"],
            key=lambda row: signed_angle(row["angle_degrees"]),
        )
        input_angles = np.asarray(
            [signed_angle(row["angle_degrees"]) for row in rows], dtype=float
        )
        # Expand the measured 60-degree interval to a readable 120-degree fan.
        display_angles = np.deg2rad(2.0 * input_angles)
        retention = np.asarray([row["top1"] for row in rows], dtype=float)
        ax.plot(
            display_angles,
            retention,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.35,
            marker=marker,
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.25,
            zorder=4,
        )
        ax.fill_between(
            display_angles,
            radial_floor,
            retention,
            color=color,
            alpha=0.025 if key != "SHARE-6d3r" else 0.055,
            zorder=1,
        )

    tick_angles = list(range(-30, 31, 5))
    ax.set_thetagrids(
        [2 * angle for angle in tick_angles],
        labels=[f"{angle}°" for angle in tick_angles],
        fontsize=8.5,
        color="#475569",
    )
    ax.set_ylim(radial_floor, 100.5)
    ax.set_yticks([88, 90, 92, 94, 96, 98, 100])
    ax.set_yticklabels(
        ["88%", "90%", "92%", "94%", "96%", "98%", "100%"],
        fontsize=8.2,
        color="#64748B",
    )
    ax.set_rlabel_position(67)
    ax.grid(color="#CBD5E1", linewidth=0.75, alpha=0.9)
    ax.spines["polar"].set_color("#94A3B8")
    ax.spines["polar"].set_linewidth(0.9)

    fig.suptitle(
        "Local rotation consistency on jointly recognized images",
        fontsize=15,
        fontweight="bold",
        color="#172033",
        y=0.965,
    )
    fig.text(
        0.5,
        0.91,
        (
            f"Correct-class retention · {sample_count:,} ImageNet-100 images correct at 0° "
            "for all four models · 5° increments"
        ),
        ha="center",
        fontsize=9.3,
        color="#64748B",
    )
    fig.text(
        0.5,
        0.115,
        "Input rotation (measured ±30° interval expanded to a 120° display fan)",
        ha="center",
        fontsize=9.0,
        color="#64748B",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9.2,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=2.8,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.20)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
