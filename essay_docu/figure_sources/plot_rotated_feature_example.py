"""Visualize how SHARE-ViT tokenizer features transform under image rotation.

This is a qualitative diagnostic inspired by GE-ViT's feature-transformation
figure.  It separates the tokenizer's undirected feature axis from the two
directed Look fields, then shows residual error after the expected rotation.
"""

from __future__ import annotations

import colorsys
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.interpolate import griddata
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.imagenet100.data import imagenet100_transforms
from experiments.imagenet100.models import build_imagenet100_model
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat


CHECKPOINT = ROOT / "runs" / "xv2_r1_g3_dual_look" / "best.pt"
VAL_ROOT = ROOT / "downloads" / "kaggle" / "imagenet100-val" / "val.X"
OUTPUT = ROOT / "essay_docu" / "figs" / "07_rotated_feature_equivariance_example.pdf"
ANGLE = 60.0


def load_model() -> torch.nn.Module:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = build_imagenet100_model(
        variant=cfg["model_variant"], model_name=cfg["model"], pretrained=False,
        num_classes=cfg["num_classes"], image_size=cfg["image_size"],
        hex_stride=cfg["hex_stride"], rot_kernel_sizes=tuple(cfg["rot_kernel_sizes"]),
        rot_bases=cfg["rot_bases"], rot_directions=cfg["rot_directions"],
        rot_global_directions=cfg["rot_global_directions"],
        rot_angular_bins_per_radius=cfg["rot_angular_bins_per_radius"],
        look_compact_variable_rings=cfg["look_compact_variable_rings"],
        center_look_layers_per_probe=cfg["center_look_layers_per_probe"],
        rot_prototype_chunk_size=cfg["rot_prototype_chunk_size"],
        rot_null_initial_score=cfg["rot_null_initial_score"],
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def choose_image() -> Path:
    # Sample across classes.  Taking the first global paths over-represents a
    # single class and previously selected an almost uniformly blue image.
    candidates: list[Path] = []
    for class_dir in sorted(path for path in VAL_ROOT.iterdir() if path.is_dir()):
        candidates.extend(sorted(path for path in class_dir.iterdir() if path.is_file())[:10])
    best: tuple[float, Path] | None = None
    for path in candidates:
        try:
            image = Image.open(path).convert("RGB").resize((96, 96))
            rgb = np.asarray(image, dtype=np.float32) / 255
        except OSError:
            continue
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        chromatic = (saturation > .18) & (value > .12)
        if chromatic.any():
            histogram, _ = np.histogram(hue[chromatic], bins=12, range=(0, 1))
            probability = histogram / max(histogram.sum(), 1)
            nonzero = probability[probability > 0]
            hue_entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(12))
            hue_dominance = float(probability.max())
        else:
            hue_entropy, hue_dominance = 0.0, 1.0
        gray = rgb.mean(axis=2)
        edge = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
        chromatic_fraction = float(chromatic.mean())
        # Prefer multi-hue, spatially structured scenes and explicitly punish
        # a single dominant color family.
        score = (
            2.2 * hue_entropy + 1.3 * chromatic_fraction + 3.0 * edge
            + .25 * float(rgb.std()) - 1.6 * hue_dominance
        )
        if best is None or score > best[0]:
            best = (score, path)
    if best is None:
        raise FileNotFoundError(f"no readable images below {VAL_ROOT}")
    return best[1]


def feature_rgb(z: np.ndarray, activation: np.ndarray) -> np.ndarray:
    phase = (np.arctan2(z[:, 1], z[:, 0]) / (2 * np.pi)) % 1.0
    rgb = np.asarray([colorsys.hsv_to_rgb(float(h), 0.84, 0.88) for h in phase])
    alpha = 0.05 + 0.95 * np.clip(activation, 0, 1) ** 0.72
    return np.c_[rgb, alpha]


def transform_feature(z: np.ndarray, xy: np.ndarray, angle_deg: float) -> np.ndarray:
    center = xy.mean(0)
    angle = math.radians(angle_deg)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    source_xy = (xy - center) @ rot + center  # inverse sampling location
    spatial = np.stack([
        griddata(xy, z[:, channel], source_xy, method="linear", fill_value=0.0)
        for channel in range(2)
    ], axis=1)
    # A shifted pose distribution rotates its first circular moment by +angle.
    return spatial @ rot.T


def scatter(ax, xy: np.ndarray, z: np.ndarray, activation: np.ndarray,
            colors: np.ndarray, title: str) -> None:
    ax.scatter(xy[:, 0], -xy[:, 1], c=colors, s=31, marker="h", linewidths=0)
    # The half-turn pose domain represents axes rather than directed vectors.
    # Draw a centered two-ended segment (no arrowhead) for each cos/sin pair.
    phase = np.arctan2(z[:, 1], z[:, 0])
    magnitude = np.linalg.norm(z, axis=1)
    concentration = magnitude / np.maximum(activation, 1e-8)
    strength = np.clip(activation, 0, 1) * np.sqrt(np.clip(concentration, 0, 1))
    half_length = 6.0
    delta = np.stack((np.cos(phase), -np.sin(phase)), axis=1) * half_length
    centers = np.stack((xy[:, 0], -xy[:, 1]), axis=1)
    segments = np.stack((centers - delta, centers + delta), axis=1)
    halo_colors = np.ones((len(z), 4))
    halo_colors[:, 3] = 0.18 + 0.78 * strength
    ax.add_collection(LineCollection(segments, colors=halo_colors,
                                     linewidths=2.8 + 1.2 * strength,
                                     capstyle="round"))
    line_colors = np.zeros((len(z), 4))
    line_colors[:, :3] = np.array([0.035, 0.05, 0.09])
    line_colors[:, 3] = 0.35 + 0.65 * strength
    ax.add_collection(LineCollection(segments, colors=line_colors,
                                     linewidths=1.15 + 1.15 * strength,
                                     capstyle="round"))
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_aspect("equal")
    ax.axis("off")


def _tokenizer_activation_mass(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Return exact real-pose mass 1-p_null for every token/prototype."""
    with torch.inference_mode():
        patches = [geometry(batch.float()) for geometry in model.patch_embed.geometries]
        patches = [
            weighted_patch_flat(
                patch, getattr(model.patch_embed, f"scale_cover_{scale_index}")
            )
            for scale_index, patch in enumerate(patches)
        ]
        chunks = []
        for start in range(0, model.patch_embed.bases, model.patch_embed.prototype_chunk_size):
            stop = min(start + model.patch_embed.prototype_chunk_size, model.patch_embed.bases)
            prototype = model.patch_embed.prototype[start:stop]
            score = None
            for patch, renderer in zip(patches, model.patch_embed.renderers):
                scale_score = rotating_dot_score(patch, renderer(prototype))
                score = scale_score if score is None else score + scale_score
            null = model.patch_embed.null_score[start:stop][None, None, :, None].expand(
                score.shape[0], score.shape[1], -1, -1
            )
            real_probability = torch.cat((score, null), dim=-1).float().softmax(-1)[..., :-1]
            chunks.append(real_probability.sum(-1))
    return torch.cat(chunks, dim=-1)


def _look_vector(bias: np.ndarray, coordinates: np.ndarray, radius: float = 4.0) -> np.ndarray:
    """Compress each query's local directed bias pattern to its first moment.

    Per-query standardization intentionally removes absolute bias scale here:
    this panel visualizes preferred direction, not the bias amplitude already
    reported by the separate Look-field figure.
    """
    relative = coordinates[None, :, :] - coordinates[:, None, :]
    distance = np.linalg.norm(relative, axis=-1)
    valid = (distance > 1e-6) & (distance <= radius + 1e-6)
    unit = relative / np.maximum(distance[..., None], 1e-6)
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = (bias * valid).sum(axis=1, keepdims=True) / count
    variance = (((bias - mean) * valid) ** 2).sum(axis=1, keepdims=True) / count
    normalized = (bias - mean) / np.sqrt(variance + 1e-8)
    masked = np.where(valid, normalized, -1e9)
    masked -= masked.max(axis=1, keepdims=True)
    probability = np.exp(masked) * valid
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-8)
    return np.einsum("qk,qkc->qc", probability, unit)


def _look_biases(model: torch.nn.Module, batch: torch.Tensor, flat_tokens: torch.Tensor):
    """Materialize Image Look and Feature Look separately for visualization."""
    with torch.inference_mode():
        rings, coverage = model.look_bank.extract_rings(batch.float(), track_input_grad=False)
        image_pose = model.look_bank.pose_weights(rings, coverage)
        image_bias = model.look_bank.look_bias(image_pose, include_cls=False)
        image_bias = image_bias.reshape(batch.shape[0], 12, 3, model.patch_embed.num_patches,
                                        model.patch_embed.num_patches)

        shared_pose = model.center_look.pose_weights(
            flat_tokens - model.patch_embed.output_bias
        )
        feature_bias = []
        for layer in range(model.center_look.depth):
            pose = model.center_look.pose_for_layer(shared_pose, layer)
            fields = model.center_look.fields(layer, dtype=flat_tokens.dtype)
            feature_bias.append(torch.einsum("bqha,haqk->bhqk", pose, fields))
        feature_bias = torch.stack(feature_bias, dim=1)
    return image_bias.cpu().numpy(), feature_bias.cpu().numpy()


def draw_look(ax, xy: np.ndarray, vector: np.ndarray, *, title: str,
              cmap: str, arrow_color: str) -> None:
    magnitude = np.linalg.norm(vector, axis=1)
    scale = float(np.quantile(magnitude, .94)) + 1e-8
    strength = np.clip(magnitude / scale, 0, 1)
    ax.scatter(xy[:, 0], -xy[:, 1], c=strength, cmap=cmap, vmin=0, vmax=1,
               s=38, marker="h", linewidths=0, alpha=.88)
    direction = vector / np.maximum(magnitude[:, None], 1e-8)
    length = 6.0 + 4.2 * strength
    ax.quiver(
        xy[:, 0], -xy[:, 1], direction[:, 0] * length, -direction[:, 1] * length,
        angles="xy", scale_units="xy", scale=1, color=arrow_color,
        width=.0048, headwidth=3.5, headlength=4.0, headaxislength=3.6,
        alpha=.58 + .40 * strength,
    )
    ax.set_title(title, fontsize=10.5, pad=5)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    torch.manual_seed(0)
    model = load_model()
    image_path = choose_image()
    pil = Image.open(image_path).convert("RGB")
    rotated = TF.rotate(pil, ANGLE, interpolation=InterpolationMode.BILINEAR, fill=0)
    _, val_transform = imagenet100_transforms(224)
    batch = torch.stack([val_transform(pil), val_transform(rotated)])
    with torch.inference_mode():
        flat_tokens = model.patch_embed(batch)
    activation_mass = _tokenizer_activation_mass(model, batch).cpu().numpy()
    tokens = flat_tokens.reshape(2, -1, 96, 2).cpu().numpy()
    xy = model.patch_embed.patch_centers_xy.cpu().numpy()
    graph_xy = model.look_bank.patch_coordinates_xy.cpu().numpy()
    image_bias, feature_bias = _look_biases(model, batch, flat_tokens)
    image_vector = np.empty((2, 11, 3, len(xy), 2), dtype=np.float32)
    feature_vector = np.empty_like(image_vector)
    for sample in range(2):
        for layer in range(11):
            for head in range(3):
                image_vector[sample, layer, head] = _look_vector(
                    image_bias[sample, layer, head], graph_xy
                )
                feature_vector[sample, layer, head] = _look_vector(
                    feature_bias[sample, layer, head], graph_xy
                )
    combined_vector = np.empty_like(image_vector)
    for sample in range(2):
        for layer in range(11):
            for head in range(3):
                combined_vector[sample, layer, head] = _look_vector(
                    image_bias[sample, layer, head]
                    + feature_bias[sample, layer, head],
                    graph_xy,
                )
    direction_quality = np.linalg.norm(combined_vector, axis=-1).mean((0, 3))
    layer, head = np.unravel_index(np.argmax(direction_quality), direction_quality.shape)

    # Pick the most structured tokenizer pair belonging to the selected head.
    pair_slice = slice(head * 32, (head + 1) * 32)
    magnitude = np.linalg.norm(tokens[:, :, pair_slice], axis=-1)
    local_mass = activation_mass[:, :, pair_slice]
    quality = local_mass.mean((0, 1)) * (magnitude.std((0, 1)) + 1e-6)
    prototype = head * 32 + int(np.argmax(quality))

    fig = plt.figure(figsize=(13.4, 5.65), constrained_layout=False)
    grid = fig.add_gridspec(
        2, 6, width_ratios=(1.05, 1, 1, 1, 1, 1.03),
        left=.018, right=.965, bottom=.13, top=.90, wspace=.12, hspace=.28,
    )
    expected_combined = transform_feature(combined_vector[0, layer, head], xy, ANGLE)
    for row, (sample_image, row_name) in enumerate(((pil, "Original"), (rotated, f"Rotated {ANGLE:.0f}°"))):
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(sample_image.resize((224, 224)))
        ax.set_title(f"{row_name} input", fontsize=10.5)
        ax.axis("off")

        feature = tokens[row, :, prototype]
        feature_mass = activation_mass[row, :, prototype]
        scatter(fig.add_subplot(grid[row, 1]), xy, feature, feature_mass,
                feature_rgb(feature, feature_mass),
                f"Tokenizer axis\nH{head + 1} pair {prototype}")
        draw_look(fig.add_subplot(grid[row, 2]), xy, image_vector[row, layer, head],
                  title=f"Image Look\nL{layer + 1} H{head + 1}", cmap="Blues", arrow_color="#075985")
        draw_look(fig.add_subplot(grid[row, 3]), xy, feature_vector[row, layer, head],
                  title=f"Feature Look\nL{layer + 1} H{head + 1}", cmap="Purples", arrow_color="#6b21a8")
        draw_look(fig.add_subplot(grid[row, 4]), xy, combined_vector[row, layer, head],
                  title="Combined directed Look", cmap="YlOrRd", arrow_color="#9a3412")

        ax = fig.add_subplot(grid[row, 5])
        if row == 0:
            draw_look(ax, xy, expected_combined, title=f"Expected Look after {ANGLE:.0f}°",
                      cmap="Greens", arrow_color="#166534")
        else:
            error = np.linalg.norm(combined_vector[1, layer, head] - expected_combined, axis=1)
            points = ax.scatter(xy[:, 0], -xy[:, 1], c=error, s=38, marker="h",
                                linewidths=0, cmap="magma", vmin=0,
                                vmax=float(np.quantile(error, .96)) + 1e-8)
            ax.set_title("Look alignment residual", fontsize=10.5, pad=5)
            ax.set_aspect("equal"); ax.axis("off")
            fig.colorbar(points, ax=ax, fraction=.045, pad=.02)

    fig.text(.515, .012,
             "Tokenizer color/line opacity = real pose mass 1-p_null; two-ended line = undirected axis; arrows = directed Look preference.",
             ha="center", fontsize=9.2, color="#4b5563")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=.10)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=190, bbox_inches="tight", pad_inches=.10)
    print(f"sample={image_path}")
    print(f"selected_layer_head_pair=L{layer + 1},H{head + 1},P{prototype}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
