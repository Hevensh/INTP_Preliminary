from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs" / "rotation_consistency" / "common_correct_fullval_local5.json"
OUTPUT = ROOT / "essay_docu" / "figs" / "07_local_rotation_common_correct"

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
    protocol = payload["protocol"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))

    for label, key, color, linestyle, marker in SERIES:
        rows = sorted(
            payload["results"][key]["angles"],
            key=lambda row: signed_angle(row["angle_degrees"]),
        )
        angles = [signed_angle(row["angle_degrees"]) for row in rows]
        axes[0].plot(
            angles,
            [row["top1"] for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            marker=marker,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.25,
        )
        axes[1].plot(
            angles,
            [row["js_divergence"] for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            marker=marker,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.25,
        )

    for ax in axes:
        ax.set_xlim(-31, 31)
        ax.set_xticks(range(-30, 31, 5))
        ax.set_xticklabels([f"{angle}°" for angle in range(-30, 31, 5)], fontsize=8)
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.75, alpha=0.8)
        ax.axvline(0, color="#94A3B8", linewidth=0.9, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#94A3B8")
        ax.tick_params(colors="#475569")
        ax.set_xlabel("Input rotation", fontsize=10, color="#334155", labelpad=7)

    axes[0].set_title(
        "Correct-class retention", fontsize=12, fontweight="bold", color="#172033"
    )
    axes[0].set_ylabel("Top-1 accuracy on common-correct set (%)", fontsize=9.5)
    axes[0].set_ylim(86, 101)
    axes[0].set_yticks([88, 90, 92, 94, 96, 98, 100])

    axes[1].set_title(
        "Probability-distribution drift", fontsize=12, fontweight="bold", color="#172033"
    )
    axes[1].set_ylabel("Jensen-Shannon divergence from 0°", fontsize=9.5)
    axes[1].set_ylim(-0.002, 0.068)
    axes[1].set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

    fig.suptitle(
        "Local rotation consistency on jointly recognized ImageNet-100 images",
        fontsize=15,
        fontweight="bold",
        color="#172033",
        y=0.98,
    )
    fig.text(
        0.5,
        0.915,
        (
            f"{protocol['common_correct_images']:,} images correctly classified by all four "
            "models at 0° · 5° increments · no test-time adaptation"
        ),
        ha="center",
        fontsize=9.3,
        color="#64748B",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9.1,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=2.5,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.19, wspace=0.23)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
