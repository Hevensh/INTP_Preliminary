from __future__ import annotations

import torch.nn as nn
import timm
from torchvision.models import resnet18

from layers.hex_linear_patch_embed import HexLinearPatchEmbed
from layers.hex_rotating_polar_patch_embed import HexRotatingPolarPatchEmbed
from layers.hex_rotating_dot_patch_embed import HexRotatingDotPatchEmbed
from layers.hex_rotating_grouped_dot_patch_embed import HexRotatingGroupedDotPatchEmbed
from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed
from model.deit_tiny_rot_hex_look import DeiTTinyRotHexLook
from model.resnet_geometric_baselines import (
    build_resnet18_arc4bank,
    build_resnet18_fixed_rotinterp8,
    build_resnet18_mixconv4,
    build_resnet18_multiscale,
    build_resnet18_multiscale_rotconv,
    build_resnet18_rotconv,
)
from model.resnet_mams import (
    build_resnet18_mams,
    build_resnet18_mams_fourv_paired,
    build_resnet18_stage_mams,
    build_resnet18_stage_mams_additive,
)


MODEL_VARIANTS = {
    "resnet18", "resnet18_multiscale", "resnet18_rotconv4",
    "resnet18_multiscale_rotconv4", "resnet18_mams",
    "resnet18_mixconv4", "resnet18_fixed_rotinterp8", "resnet18_arc4bank",
    "resnet18_mams_fourv_paired",
    "resnet18_stage_mams",
    "resnet18_stage_mams_additive",
    "deit_tiny", "hex_patch", "rot_hex_pe", "rot_hex_dot_simple_pe",
    "rot_hex_dot_grouped_pe",
    "rot_hex_harmonic_pe",
    "rot_hex_harmonic_softmax_pe",
    "rot_hex_harmonic_l1_softmax_pe",
    "rot_hex_dot_grouped_compensated_pe",
    "rot_hex_harmonic_look", "rot_hex_harmonic_pe_look",
    "rot_hex_harmonic_look_ring", "rot_hex_harmonic_pe_look_ring",
    "rot_hex_harmonic_center_look", "rot_hex_harmonic_pe_center_look",
    "rot_hex_harmonic_pe_look_center_look",
    "rot_hex_harmonic_pe_center_grid_look",
    "rot_hex_harmonic_center_grid_look",
    "rot_hex_harmonic_pe_look_center_grid_look",
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
    rot_angular_bins_per_radius: int = 4,
    look_compact_variable_rings: bool = False,
    center_look_layers_per_probe: int = 1,
    feature_ring_look: bool = False,
    feature_ring_start_layer: int = 0,
    feature_ring_group_size: int = 4,
    feature_ring_frequency: bool = False,
    rot_prototype_chunk_size: int = 16,
    rot_use_null: bool = True,
    rot_null_initial_score: float = -1.0,
    rot_score_normalization: str = "none",
    rot_response_gate: str = "exp2",
    rot_response_gate_location: str = "pose",
    rot_score_clamp: float = 4.0,
) -> nn.Module:
    """Build the aligned ImageNet-100 comparison models."""

    if variant not in MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(MODEL_VARIANTS)}")
    if variant == "resnet18":
        if pretrained:
            raise ValueError("the ResNet comparison is trained from scratch")
        return resnet18(weights=None, num_classes=num_classes)
    if variant in {
        "resnet18_mixconv4",
        "resnet18_fixed_rotinterp8",
        "resnet18_arc4bank",
        "resnet18_multiscale",
        "resnet18_rotconv4",
        "resnet18_multiscale_rotconv4",
    }:
        if pretrained:
            raise ValueError("the ResNet comparison is trained from scratch")
        if variant == "resnet18_mixconv4":
            return build_resnet18_mixconv4(num_classes=num_classes)
        if variant == "resnet18_fixed_rotinterp8":
            return build_resnet18_fixed_rotinterp8(
                num_classes=num_classes,
                directions=8,
            )
        if variant == "resnet18_arc4bank":
            return build_resnet18_arc4bank(num_classes=num_classes, kernel_number=4)
        if variant == "resnet18_multiscale":
            return build_resnet18_multiscale(num_classes=num_classes)
        if variant == "resnet18_rotconv4":
            return build_resnet18_rotconv(
                num_classes=num_classes,
                kernel_size=5,
                directions=4,
            )
        return build_resnet18_multiscale_rotconv(
            num_classes=num_classes,
            kernel_sizes=(5, 3),
            directions=4,
        )
    if variant == "resnet18_mams":
        if pretrained:
            raise ValueError("the MAMS ResNet comparison is trained from scratch")
        return build_resnet18_mams(
            num_classes=num_classes,
            diameters=rot_kernel_sizes,
            directions=rot_directions,
            global_directions=rot_global_directions,
            angular_bins_per_radius=rot_angular_bins_per_radius,
            prototype_chunk_size=rot_prototype_chunk_size,
            use_null=rot_use_null,
            null_initial_score=rot_null_initial_score,
        )
    if variant == "resnet18_mams_fourv_paired":
        if pretrained:
            raise ValueError("the paired MAMS ResNet comparison is trained from scratch")
        if len(rot_kernel_sizes) != 2:
            raise ValueError("paired four-value MAMS requires exactly two scales")
        return build_resnet18_mams_fourv_paired(
            num_classes=num_classes,
            diameters=(int(rot_kernel_sizes[0]), int(rot_kernel_sizes[1])),
            directions=rot_directions,
            global_directions=rot_global_directions,
            angular_bins_per_radius=rot_angular_bins_per_radius,
            prototype_chunk_size=rot_prototype_chunk_size,
            null_initial_score=rot_null_initial_score,
        )
    if variant in {"resnet18_stage_mams", "resnet18_stage_mams_additive"}:
        if pretrained:
            raise ValueError("the stage-routed MAMS comparison is trained from scratch")
        builder = (
            build_resnet18_stage_mams_additive
            if variant == "resnet18_stage_mams_additive"
            else build_resnet18_stage_mams
        )
        return builder(
            num_classes=num_classes,
            directions=rot_directions,
            global_directions=rot_global_directions,
            angular_bins_per_radius=rot_angular_bins_per_radius,
            prototype_chunk_size=rot_prototype_chunk_size,
            null_initial_score=rot_null_initial_score,
        )
    if variant != "deit_tiny" and pretrained:
        raise ValueError(
            f"{variant} ImageNet-100 comparison is a from-scratch experiment; "
            "set pretrained=false"
        )
    if variant in {
        "rot_hex_harmonic_look",
        "rot_hex_harmonic_pe_look",
        "rot_hex_harmonic_look_ring",
        "rot_hex_harmonic_pe_look_ring",
        "rot_hex_harmonic_center_look",
        "rot_hex_harmonic_pe_center_look",
        "rot_hex_harmonic_pe_look_center_look",
        "rot_hex_harmonic_pe_center_grid_look",
        "rot_hex_harmonic_center_grid_look",
        "rot_hex_harmonic_pe_look_center_grid_look",
    }:
        return DeiTTinyRotHexLook(
            num_classes=num_classes,
            image_size=image_size,
            use_pos_embed=variant in {
                "rot_hex_harmonic_pe_look",
                "rot_hex_harmonic_pe_look_ring",
                "rot_hex_harmonic_pe_center_look",
                "rot_hex_harmonic_pe_look_center_look",
                "rot_hex_harmonic_pe_center_grid_look",
                "rot_hex_harmonic_pe_look_center_grid_look",
            },
            lattice_stride=hex_stride,
            kernel_sizes=rot_kernel_sizes,
            bases=rot_bases,
            directions=rot_directions,
            global_directions=rot_global_directions,
            angular_bins_per_radius=rot_angular_bins_per_radius,
            look_compact_variable_rings=look_compact_variable_rings,
            image_look=variant not in {
                "rot_hex_harmonic_center_look",
                "rot_hex_harmonic_pe_center_look",
                "rot_hex_harmonic_pe_center_grid_look",
                "rot_hex_harmonic_center_grid_look",
            },
            center_pose_look=variant in {
                "rot_hex_harmonic_center_look",
                "rot_hex_harmonic_pe_center_look",
                "rot_hex_harmonic_pe_look_center_look",
            },
            center_pose_grid_look=variant in {
                "rot_hex_harmonic_pe_center_grid_look",
                "rot_hex_harmonic_center_grid_look",
                "rot_hex_harmonic_pe_look_center_grid_look",
            },
            center_look_layers_per_probe=center_look_layers_per_probe,
            feature_ring_look=(
                feature_ring_look
                or variant in {
                    "rot_hex_harmonic_look_ring",
                    "rot_hex_harmonic_pe_look_ring",
                }
            ),
            feature_ring_start_layer=feature_ring_start_layer,
            feature_ring_group_size=feature_ring_group_size,
            feature_ring_frequency=feature_ring_frequency,
            prototype_chunk_size=rot_prototype_chunk_size,
            tokenizer_null_initial_score=rot_null_initial_score,
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
            angular_bins_per_radius=rot_angular_bins_per_radius,
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
    elif variant in {"rot_hex_dot_grouped_pe", "rot_hex_dot_grouped_compensated_pe"}:
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
            compensate_small_scales=variant == "rot_hex_dot_grouped_compensated_pe",
        )
    else:
        harmonic_softmax = variant in {
            "rot_hex_harmonic_softmax_pe",
            "rot_hex_harmonic_l1_softmax_pe",
        }
        patch_embed = HexRotatingHarmonicPatchEmbed(
            img_size=image_size,
            in_chans=3,
            embed_dim=embed_dim,
            lattice_stride=hex_stride,
            kernel_sizes=rot_kernel_sizes,
            bases=rot_bases,
            directions=rot_directions,
            global_directions=rot_global_directions,
            angular_bins_per_radius=rot_angular_bins_per_radius,
            prototype_chunk_size=rot_prototype_chunk_size,
            pose_softmax=harmonic_softmax,
            use_null=harmonic_softmax,
            null_initial_score=rot_null_initial_score,
            match_metric=(
                "relative_l1"
                if variant == "rot_hex_harmonic_l1_softmax_pe"
                else "dot"
            ),
        )
    model.patch_embed = patch_embed
    model.pos_embed = nn.Parameter(
        model.pos_embed.new_empty(1, model.num_prefix_tokens + patch_embed.num_patches, embed_dim)
    )
    nn.init.trunc_normal_(model.pos_embed, std=0.02)
    return model
