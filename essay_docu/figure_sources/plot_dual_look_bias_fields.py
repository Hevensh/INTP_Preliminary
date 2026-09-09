from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps


def _rms(grid: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(grid), axis=(-2, -1)))


def _draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    layer_offset: int,
    vmax: float,
) -> None:
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    ax.set_xticks(range(values.shape[1]), [f"H{i + 1}" for i in range(values.shape[1])])
    ax.set_yticks(
        range(values.shape[0]),
        [f"L{i + layer_offset}" for i in range(values.shape[0])],
    )
    ax.tick_params(labelsize=8, length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "#101828" if value > 0.58 * vmax else "white"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.6,
                color=color,
            )
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#667085")
    ax._look_image = image  # type: ignore[attr-defined]


def _top_directional_fields(grid: np.ndarray, count: int = 3) -> list[tuple[int, int]]:
    """Rank layer-head fields by angular variation after removing ring offsets."""
    centered = grid - grid.mean(axis=-1, keepdims=True)
    contrast = centered.std(axis=-1).mean(axis=-1)
    order = np.argsort(contrast.reshape(-1))[::-1]
    return [tuple(np.unravel_index(int(index), contrast.shape)) for index in order[:count]]


def _draw_polar_field(
    ax: plt.Axes,
    field: np.ndarray,
    *,
    title: str,
    limit: float,
) -> None:
    radial_bins, angular_bins = field.shape
    width = 2.0 * np.pi / angular_bins
    colors = colormaps["RdBu_r"](np.clip(field / (2.0 * limit) + 0.5, 0.0, 1.0))
    for radius in range(radial_bins):
        for angle in range(angular_bins):
            ax.bar(
                angle * width,
                1.0,
                width=width,
                bottom=float(radius),
                align="center",
                color=colors[radius, angle],
                edgecolor="white",
                linewidth=0.65,
            )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, radial_bins)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Image Look and G3 Feature Look parameters from a dual-Look checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    image_grid = state["look_bank.look_grid"].detach().float().cpu().numpy()
    feature_grid = state["center_look.look_grid"].detach().float().cpu().numpy()
    if image_grid.shape != (36, 4, 12):
        raise ValueError(f"expected Image Look grid (36,4,12), got {image_grid.shape}")
    if feature_grid.shape != (11, 3, 4, 12):
        raise ValueError(
            f"expected G3 Feature Look grid (11,3,4,12), got {feature_grid.shape}"
        )
    image_grid = image_grid.reshape(12, 3, 4, 12)
    image_selected = _top_directional_fields(image_grid)
    feature_selected = _top_directional_fields(feature_grid)

    image_rms = _rms(image_grid)
    feature_rms = _rms(feature_grid)
    image_rms_limit = float(image_rms.max())
    feature_rms_limit = float(feature_rms.max())
    image_field_limit = float(
        max(np.max(np.abs(image_grid[layer, head])) for layer, head in image_selected)
    )
    feature_field_limit = float(
        max(np.max(np.abs(feature_grid[layer, head])) for layer, head in feature_selected)
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlecolor": "#172B4D",
            "text.color": "#172B4D",
        }
    )
    figure = plt.figure(figsize=(13.0, 6.5), constrained_layout=False)
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(0.88, 1.55),
        left=0.055,
        right=0.975,
        bottom=0.105,
        top=0.95,
        wspace=0.22,
    )
    left = outer[0].subgridspec(
        2, 2, width_ratios=(1.0, 0.045), hspace=0.34, wspace=0.08
    )
    image_ax = figure.add_subplot(left[0, 0])
    image_cax = figure.add_subplot(left[0, 1])
    feature_ax = figure.add_subplot(left[1, 0])
    feature_cax = figure.add_subplot(left[1, 1])
    _draw_heatmap(
        image_ax,
        image_rms,
        title="Image Look field RMS",
        layer_offset=1,
        vmax=image_rms_limit,
    )
    _draw_heatmap(
        feature_ax,
        feature_rms,
        title="Feature Look field RMS (G3)",
        layer_offset=1,
        vmax=feature_rms_limit,
    )
    image_colorbar = figure.colorbar(
        image_ax._look_image,  # type: ignore[attr-defined]
        cax=image_cax,
    )
    image_colorbar.set_label("Image RMS", fontsize=8.2)
    image_colorbar.ax.tick_params(labelsize=7.3)
    feature_colorbar = figure.colorbar(
        feature_ax._look_image,  # type: ignore[attr-defined]
        cax=feature_cax,
    )
    feature_colorbar.set_label("Feature RMS", fontsize=8.2)
    feature_colorbar.ax.tick_params(labelsize=7.3)

    right = outer[1].subgridspec(2, 3, hspace=0.28, wspace=0.08)
    for panel, (layer, head) in enumerate(image_selected):
        axis = figure.add_subplot(right[0, panel], projection="polar")
        _draw_polar_field(
            axis,
            image_grid[layer, head],
            title=f"Image Look L{layer + 1} · H{head + 1}",
            limit=image_field_limit,
        )
    for panel, (layer, head) in enumerate(feature_selected):
        axis = figure.add_subplot(right[1, panel], projection="polar")
        _draw_polar_field(
            axis,
            feature_grid[layer, head],
            title=f"Feature Look L{layer + 1} · H{head + 1}",
            limit=feature_field_limit,
        )

    figure.text(
        0.735,
        0.055,
        "Three Image Look fields (top) and three Feature Look fields (bottom), ranked by angular contrast · blue/white/red = negative/zero/positive",
        ha="center",
        fontsize=8.5,
        color="#475467",
    )
    figure.text(
        0.735,
        0.025,
        "Each branch uses its own shared contrast scale; every field is one layer-head map with 4 radial bins × 12 directed bins.",
        ha="center",
        fontsize=7.8,
        color="#667085",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


if __name__ == "__main__":
    main()
