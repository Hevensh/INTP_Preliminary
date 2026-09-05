from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from experiments.imagenet100.models import build_imagenet100_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "runs" / "xv2_r1_g3_dual_look" / "best.pt"
DEFAULT_OUTPUT = ROOT / "essay_docu" / "figs" / "all_geometry_prototypes.pdf"
FAMILY_COLORS = {
    "full": "#ef8a47",
    "angular": "#2a9d8f",
    "stripe": "#4e79a7",
    "color": "#bdbdbd",
}


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a dictionary payload")
    if not isinstance(payload.get("config"), dict):
        raise KeyError("checkpoint does not contain a config dictionary")
    if not isinstance(payload.get("model"), dict):
        raise KeyError("checkpoint does not contain a model state dictionary")
    return payload


def _build_model(config: dict[str, Any]) -> torch.nn.Module:
    return build_imagenet100_model(
        variant=config["model_variant"],
        model_name=config.get("model", "deit_tiny_patch16_224"),
        pretrained=False,
        num_classes=int(config.get("num_classes", 100)),
        image_size=int(config.get("image_size", 224)),
        hex_kernel_size=int(config.get("hex_kernel_size", 21)),
        hex_stride=int(config.get("hex_stride", 18)),
        rot_kernel_sizes=tuple(config.get("rot_kernel_sizes", (24, 12))),
        rot_bases=int(config.get("rot_bases", 96)),
        rot_directions=int(config.get("rot_directions", 4)),
        rot_global_directions=int(config.get("rot_global_directions", 8)),
        rot_angular_bins_per_radius=int(config.get("rot_angular_bins_per_radius", 4)),
        look_compact_variable_rings=bool(
            config.get("look_compact_variable_rings", False)
        ),
        center_look_layers_per_probe=int(
            config.get("center_look_layers_per_probe", 1)
        ),
        image_look_probes=int(config.get("image_look_probes", 1)),
        feature_look_probes=int(config.get("feature_look_probes", 1)),
        feature_look_rotating_probes=bool(config.get("feature_look_rotating_probes", False)),
        feature_ring_look=bool(config.get("feature_ring_look", False)),
        feature_ring_start_layer=int(config.get("feature_ring_start_layer", 0)),
        feature_ring_group_size=int(config.get("feature_ring_group_size", 4)),
        feature_ring_frequency=bool(config.get("feature_ring_frequency", False)),
        rot_prototype_chunk_size=int(config.get("rot_prototype_chunk_size", 16)),
        rot_use_null=bool(config.get("rot_use_null", True)),
        rot_null_initial_score=float(config.get("rot_null_initial_score", -1.0)),
        rot_score_normalization=str(config.get("rot_score_normalization", "none")),
        rot_response_gate=str(config.get("rot_response_gate", "exp2")),
        rot_response_gate_location=str(
            config.get("rot_response_gate_location", "pose")
        ),
        rot_score_clamp=float(config.get("rot_score_clamp", 4.0)),
        rot_progressive_differentiation=bool(
            config.get("rot_progressive_differentiation", False)
        ),
        rot_stripe_longitudinal_bins=int(
            config.get("rot_stripe_longitudinal_bins", 3)
        ),
        rot_stripe_offset_subdivisions=int(
            config.get("rot_stripe_offset_subdivisions", 4)
        ),
        gmr_hidden_channels=int(config.get("gmr_hidden_channels", 24)),
        arc_kernel_number=int(config.get("arc_kernel_number", 4)),
        arc_max_angle_degrees=float(config.get("arc_max_angle_degrees", 40.0)),
        arc_batch_chunk_size=int(config.get("arc_batch_chunk_size", 32)),
    )


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def _unwrap_variable_rings(
    prototype: torch.Tensor,
    ring_counts: torch.Tensor,
    ring_offsets: torch.Tensor,
) -> torch.Tensor:
    """Expand variable-length polar rings to a rectangular nearest-neighbour view."""
    maximum = int(ring_counts.max())
    rows = []
    for ring_index, count_tensor in enumerate(ring_counts):
        count = int(count_tensor)
        start = int(ring_offsets[ring_index])
        ring = prototype[..., start : start + count]
        sample = torch.div(
            torch.arange(maximum) * count,
            maximum,
            rounding_mode="floor",
        ).clamp_max(count - 1)
        rows.append(ring[..., sample])
    return torch.stack(rows, dim=-2)


def _render_to_square(
    rendered: torch.Tensor,
    geometry: torch.nn.Module,
) -> torch.Tensor:
    size = int(geometry.kernel_size)
    canvas = torch.full((3, size, size), float("nan"), dtype=rendered.dtype)
    canvas[:, geometry.idx_x.cpu(), geometry.idx_y.cpu()] = rendered.cpu()
    return canvas


def _signed_rgb(value: torch.Tensor, scale: float) -> np.ndarray:
    array = value.detach().cpu().float().numpy()
    if array.shape[0] != 3:
        raise ValueError(f"expected RGB-first tensor, got {array.shape}")
    image = np.moveaxis(array, 0, -1)
    invalid = ~np.isfinite(image).all(axis=-1)
    image = np.clip(0.5 + 0.5 * np.nan_to_num(image) / scale, 0.0, 1.0)
    image[invalid] = 1.0
    return image


def _prototype_scale(*views: torch.Tensor) -> float:
    flattened = [view.detach().cpu().float().flatten() for view in views]
    values = torch.cat([value[torch.isfinite(value)].abs() for value in flattened])
    scale = float(torch.quantile(values, 0.99))
    return max(scale, 1e-8)


def _stored_view(patch_embed: torch.nn.Module, base: int, family: str) -> torch.Tensor:
    prototype = patch_embed.prototype_bank[base].detach().cpu()
    if family == "full":
        return _unwrap_variable_rings(
            prototype,
            patch_embed.ring_counts.cpu(),
            patch_embed.ring_offsets.cpu(),
        )
    if family == "angular":
        return prototype.unsqueeze(-2)
    if family == "stripe":
        return prototype
    if family == "color":
        return prototype[:, None, None]
    raise ValueError(f"unsupported prototype family {family}")


def _render_differentiated(
    patch_embed: torch.nn.Module,
    *,
    base: int,
    family: str,
    scale: int,
    pose_index: int,
) -> torch.Tensor:
    prototype = patch_embed.prototype_bank[base].detach().cpu().unsqueeze(0)
    if family == "full":
        return patch_embed.renderers[scale](prototype)[0, pose_index]
    if family == "angular":
        return patch_embed.angular_renderers[scale](prototype)[0, pose_index]
    if family == "stripe":
        offset = patch_embed.stripe_angle_offset[base : base + 1].detach().cpu()
        return patch_embed.stripe_renderers[scale](prototype, offset)[0, pose_index]
    if family == "color":
        return prototype[0, :, None].expand(
            -1, patch_embed.geometries[scale].patch_offsets_xy.shape[0]
        )
    raise ValueError(f"unsupported prototype family {family}")


def _draw_page(
    pdf: PdfPages,
    *,
    indices: range,
    stored: list[torch.Tensor],
    rendered_scales: list[list[torch.Tensor]],
    families: list[str],
    family_counts: dict[str, int],
    kernel_sizes: tuple[int, ...],
    pose_degrees: float,
    columns: int,
    page_number: int,
    page_count: int,
) -> None:
    rows = math.ceil(len(indices) / columns)
    views_per_prototype = 1 + len(rendered_scales)
    figure = plt.figure(figsize=(15.5, 3.65 * rows + 0.65), facecolor="white")
    outer = figure.add_gridspec(
        rows,
        columns,
        left=0.035,
        right=0.985,
        bottom=0.055,
        top=0.91,
        wspace=0.18,
        hspace=0.32,
    )

    for slot, prototype_index in enumerate(indices):
        row, column = divmod(slot, columns)
        inner = outer[row, column].subgridspec(
            1, views_per_prototype, width_ratios=[1.35] + [1.0] * len(rendered_scales), wspace=0.08
        )
        family = families[prototype_index]
        views = [stored[prototype_index]] + [view[prototype_index] for view in rendered_scales]
        scale = _prototype_scale(*views)
        titles = [f"{family} storage"] + [f"K{size}" for size in kernel_sizes]
        for view_index, (view, title) in enumerate(zip(views, titles, strict=True)):
            axis = figure.add_subplot(inner[0, view_index])
            axis.imshow(
                _signed_rgb(view, scale),
                interpolation="nearest",
                origin="lower" if view_index == 0 else "upper",
                aspect="auto" if view_index == 0 else "equal",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(title, fontsize=8.5, pad=3, color="#4d5562")
            for spine in axis.spines.values():
                spine.set_visible(False)
        figure.text(
            (outer[row, column].get_position(figure).x0 + outer[row, column].get_position(figure).x1) / 2,
            outer[row, column].get_position(figure).y1 + 0.018,
            f"Prototype {prototype_index:02d} · {family.capitalize()}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
            color=FAMILY_COLORS[family],
        )

    figure.suptitle(
        f"All learned differentiated geometry prototypes · pose {pose_degrees:g}° · page {page_number}/{page_count}",
        fontsize=15,
        fontweight="semibold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.018,
        "Signed RGB weights: zero is mid-gray; color and brightness show channel sign and magnitude. "
        "Each prototype shares one display scale across storage and rendered views. "
        + " · ".join(
            f"{family.capitalize()} {family_counts.get(family, 0)}"
            for family in ("full", "angular", "stripe", "color")
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#5d6571",
    )
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export every learned polar geometry prototype from an ImageNet-100 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--per-page", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.per_page <= 0 or args.columns <= 0:
        raise ValueError("per-page and columns must be positive")
    if args.per_page % args.columns:
        raise ValueError("per-page must be divisible by columns")

    payload = _checkpoint_payload(checkpoint)
    config = payload["config"]
    model = _build_model(config)
    state = _clean_state_dict(payload["model"])
    prepare = getattr(model.patch_embed, "prepare_for_state_dict", None)
    if prepare is not None:
        prepare(state, prefix="patch_embed.")
    model.load_state_dict(state, strict=True)
    model.eval()
    patch_embed = model.patch_embed
    if not all(hasattr(patch_embed, attribute) for attribute in ("renderers", "geometries")):
        raise TypeError("checkpoint patch embedding does not use supported geometry prototypes")

    directions = int(patch_embed.directions)
    if not 0 <= args.pose_index < directions:
        raise ValueError(f"pose-index must be in [0, {directions - 1}]")

    if hasattr(patch_embed, "prototype_bank"):
        families = [patch_embed.family_name(base) for base in range(patch_embed.bases)]
        stored = [
            _stored_view(patch_embed, base, family)
            for base, family in enumerate(families)
        ]
        rendered_scales = [
            [
                _render_to_square(
                    _render_differentiated(
                        patch_embed,
                        base=base,
                        family=family,
                        scale=scale,
                        pose_index=args.pose_index,
                    ),
                    geometry,
                )
                for base, family in enumerate(families)
            ]
            for scale, geometry in enumerate(patch_embed.geometries)
        ]
        family_counts = patch_embed.family_counts()
        prototype_count = patch_embed.bases
    elif hasattr(patch_embed, "prototype"):
        prototype = patch_embed.prototype.detach().cpu()
        stored_tensor = _unwrap_variable_rings(
            prototype,
            patch_embed.ring_counts.cpu(),
            patch_embed.ring_offsets.cpu(),
        )
        stored = [stored_tensor[index] for index in range(len(prototype))]
        rendered_scales = []
        for renderer, geometry in zip(
            patch_embed.renderers, patch_embed.geometries, strict=True
        ):
            rendered = renderer(prototype)[:, args.pose_index]
            rendered_scales.append(
                [
                    _render_to_square(rendered[index], geometry)
                    for index in range(len(prototype))
                ]
            )
        families = ["full"] * len(prototype)
        family_counts = {"full": len(prototype), "angular": 0, "stripe": 0, "color": 0}
        prototype_count = len(prototype)
    else:
        raise TypeError("checkpoint has neither a Full prototype tensor nor a differentiated bank")

    kernel_sizes = tuple(int(geometry.kernel_size) for geometry in patch_embed.geometries)
    global_directions = int(config.get("rot_global_directions", directions))
    pose_degrees = args.pose_index * 360.0 / global_directions
    output.parent.mkdir(parents=True, exist_ok=True)
    page_count = math.ceil(prototype_count / args.per_page)
    with PdfPages(output) as pdf:
        for page in range(page_count):
            start = page * args.per_page
            stop = min(start + args.per_page, prototype_count)
            _draw_page(
                pdf,
                indices=range(start, stop),
                stored=stored,
                rendered_scales=rendered_scales,
                families=families,
                family_counts=family_counts,
                kernel_sizes=kernel_sizes,
                pose_degrees=pose_degrees,
                columns=args.columns,
                page_number=page + 1,
                page_count=page_count,
            )

    print(
        f"wrote {prototype_count} prototypes across {page_count} pages to {output} "
        f"(pose {args.pose_index}, {pose_degrees:g} degrees)"
    )


if __name__ == "__main__":
    main()
