from __future__ import annotations

import torch.nn as nn
import timm

from layers.hex_linear_patch_embed import HexLinearPatchEmbed


MODEL_VARIANTS = {"deit_tiny", "hex_patch"}


def build_imagenet100_model(
    *,
    variant: str,
    model_name: str,
    pretrained: bool,
    num_classes: int,
    image_size: int,
    hex_kernel_size: int = 21,
    hex_stride: int = 18,
) -> nn.Module:
    """Build matched DeiT-Tiny models that differ only in patch embedding."""

    if variant not in MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(MODEL_VARIANTS)}")
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=image_size,
    )
    if variant == "deit_tiny":
        return model

    if pretrained:
        raise ValueError(
            "hex_patch ImageNet-100 comparison is a from-scratch experiment; "
            "set pretrained=false"
        )
    embed_dim = int(model.embed_dim)
    patch_embed = HexLinearPatchEmbed(
        img_size=image_size,
        in_chans=3,
        embed_dim=embed_dim,
        kernel_size=hex_kernel_size,
        lattice_stride=hex_stride,
    )
    model.patch_embed = patch_embed
    model.pos_embed = nn.Parameter(
        model.pos_embed.new_empty(1, model.num_prefix_tokens + patch_embed.num_patches, embed_dim)
    )
    nn.init.trunc_normal_(model.pos_embed, std=0.02)
    return model
