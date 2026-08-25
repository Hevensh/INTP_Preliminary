from __future__ import annotations

import torch.nn as nn
import timm

from layers.hex_linear_patch_embed import HexLinearPatchEmbed
from layers.hex_rotating_polar_patch_embed import HexRotatingPolarPatchEmbed
from layers.hex_rotating_dot_patch_embed import HexRotatingDotPatchEmbed
from layers.hex_rotating_grouped_dot_patch_embed import HexRotatingGroupedDotPatchEmbed
from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed


MODEL_VARIANTS = {
    "deit_tiny", "hex_patch", "rot_hex_pe", "rot_hex_dot_simple_pe",
    "rot_hex_dot_grouped_pe",
    "rot_hex_harmonic_pe",
}


def build_imagenet100_model(
    *,
    variant: str,
    model_name: str,
    pretrained: bool,
    num_classes: int,
    image_size: int,
    hex_kernel_size: int = 21,
    hex_stride: int = 18,
    rot_kernel_sizes: tuple[int, ...] = (24, 12),
    rot_bases: int = 96,
    rot_directions: int = 4,
    rot_global_directions: int = 8,
    rot_prototype_chunk_size: int = 16,
    rot_use_null: bool = True,
    rot_null_initial_score: float = -1.0,
    rot_score_normalization: str = "none",
    rot_response_gate: str = "exp2",
    rot_response_gate_location: str = "pose",
    rot_score_clamp: float = 4.0,
) -> nn.Module:
    """Build matched DeiT-Tiny models that differ only in patch embedding."""

    if variant not in MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(MODEL_VARIANTS)}")
    if variant != "deit_tiny" and pretrained:
        raise ValueError(
            f"{variant} ImageNet-100 comparison is a from-scratch experiment; "
            "set pretrained=false"
        )
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=image_size,
    )
    if variant == "deit_tiny":
        return model

    embed_dim = int(model.embed_dim)
    if variant == "hex_patch":
        patch_embed = HexLinearPatchEmbed(
            img_size=image_size,
            in_chans=3,
            embed_dim=embed_dim,
            kernel_size=hex_kernel_size,
            lattice_stride=hex_stride,
        )
    elif variant == "rot_hex_pe":
        patch_embed = HexRotatingPolarPatchEmbed(
            img_size=image_size,
            in_chans=3,
            embed_dim=embed_dim,
            lattice_stride=hex_stride,
            kernel_sizes=rot_kernel_sizes,
            bases=rot_bases,
            directions=rot_directions,
            global_directions=rot_global_directions,
            prototype_chunk_size=rot_prototype_chunk_size,
            use_null=rot_use_null,
            null_initial_score=rot_null_initial_score,
            score_normalization=rot_score_normalization,
            response_gate=rot_response_gate,
            response_gate_location=rot_response_gate_location,
            score_clamp=rot_score_clamp,
        )
    elif variant == "rot_hex_dot_simple_pe":
        patch_embed = HexRotatingDotPatchEmbed(
            img_size=image_size,
            in_chans=3,
            embed_dim=embed_dim,
            lattice_stride=hex_stride,
            kernel_sizes=rot_kernel_sizes,
            bases=rot_bases,
            directions=rot_directions,
            global_directions=rot_global_directions,
            prototype_chunk_size=rot_prototype_chunk_size,
            use_null=rot_use_null,
            null_initial_score=rot_null_initial_score,
        )
    elif variant == "rot_hex_dot_grouped_pe":
        patch_embed = HexRotatingGroupedDotPatchEmbed(
            img_size=image_size,
            in_chans=3,
            embed_dim=embed_dim,
            lattice_stride=hex_stride,
            kernel_sizes=rot_kernel_sizes,
            bases=rot_bases,
            directions=rot_directions,
            global_directions=rot_global_directions,
            prototype_chunk_size=rot_prototype_chunk_size,
            use_null=rot_use_null,
            null_initial_score=rot_null_initial_score,
        )
    else:
        patch_embed = HexRotatingHarmonicPatchEmbed(
            img_size=image_size,
            in_chans=3,
            embed_dim=embed_dim,
            lattice_stride=hex_stride,
            kernel_sizes=rot_kernel_sizes,
            bases=rot_bases,
            directions=rot_directions,
            global_directions=rot_global_directions,
            prototype_chunk_size=rot_prototype_chunk_size,
        )
    model.patch_embed = patch_embed
    model.pos_embed = nn.Parameter(
        model.pos_embed.new_empty(1, model.num_prefix_tokens + patch_embed.num_patches, embed_dim)
    )
    nn.init.trunc_normal_(model.pos_embed, std=0.02)
    return model
