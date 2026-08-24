from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from layers.hex_linear_patch_embed import HexLinearPatchEmbed


@dataclass(frozen=True)
class DeiTLoadReport:
    copied_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]


def interpolate_deit_pos_embed(
    pos_embed: torch.Tensor,
    patch_centers_xy: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Sample a square DeiT absolute PE field at original-image Hex centers."""
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError("pos_embed must have shape (1, 1+N, D)")
    cls_pos, patch_pos = pos_embed[:, :1], pos_embed[:, 1:]
    grid_side = round(patch_pos.shape[1] ** 0.5)
    if grid_side * grid_side != patch_pos.shape[1]:
        raise ValueError("source patch position count must form a square grid")
    if patch_centers_xy.ndim != 2 or patch_centers_xy.shape[-1] != 2:
        raise ValueError("patch_centers_xy must have shape (N, 2)")

    height, width = image_size
    xy = patch_centers_xy.to(device=pos_embed.device, dtype=pos_embed.dtype)
    grid_x = (xy[:, 0] * (2.0 / max(width - 1, 1)) - 1.0).clamp(-1.0, 1.0)
    grid_y = (xy[:, 1] * (2.0 / max(height - 1, 1)) - 1.0).clamp(-1.0, 1.0)
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, -1, 2)

    field = patch_pos.reshape(1, grid_side, grid_side, -1).permute(0, 3, 1, 2)
    sampled = F.grid_sample(
        field,
        grid,
        mode="bicubic",
        padding_mode="border",
        align_corners=True,
    ).squeeze(2).transpose(1, 2)
    return torch.cat((cls_pos, sampled), dim=1)


@torch.no_grad()
def load_deit_tiny_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> DeiTLoadReport:
    """Load a plain timm DeiT-Tiny backbone into a compatible HexViTClassifier."""
    if not isinstance(model.patch_embed, HexLinearPatchEmbed):
        raise TypeError("DeiT patch projection transfer requires HexLinearPatchEmbed")
    if model.embed_dim != 192 or len(model.blocks) != 12 or model.blocks[0].attn.num_heads != 3:
        raise ValueError("target must use DeiT-Tiny dimensions: dim=192, depth=12, heads=3")

    copied: list[str] = []
    model.cls_token.copy_(state_dict["cls_token"].to(model.cls_token))
    copied.append("cls_token")

    model.patch_embed.load_vit_patch_projection(
        state_dict["patch_embed.proj.weight"],
        state_dict.get("patch_embed.proj.bias"),
    )
    copied.extend(("patch_embed.proj.weight", "patch_embed.proj.bias"))

    if model.pos_embed is None:
        raise ValueError("target must enable learned absolute position embeddings")
    target_pos = interpolate_deit_pos_embed(
        state_dict["pos_embed"],
        model.patch_embed.patch_centers_xy,
        (model.img_size, model.img_size),
    )
    if target_pos.shape != model.pos_embed.shape:
        raise ValueError(f"interpolated PE shape {tuple(target_pos.shape)} != {tuple(model.pos_embed.shape)}")
    model.pos_embed.copy_(target_pos.to(model.pos_embed))
    copied.append("pos_embed")

    target_state = model.state_dict()
    transferable_prefixes = ("blocks.", "norm.")
    for key, value in state_dict.items():
        if not key.startswith(transferable_prefixes):
            continue
        if key not in target_state or target_state[key].shape != value.shape:
            continue
        target_state[key].copy_(value.to(target_state[key]))
        copied.append(key)

    skipped = tuple(sorted(set(state_dict) - set(copied)))
    return DeiTLoadReport(tuple(copied), skipped)
