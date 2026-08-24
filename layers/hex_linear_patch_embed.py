from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hex_patch_geometry import HexPatchGeometry


class HexLinearPatchEmbed(nn.Module):
    """Linear projection of raw circular Hex patches into ViT tokens."""

    def __init__(
        self,
        img_size: int,
        in_chans: int,
        embed_dim: int,
        kernel_size: int,
        lattice_stride: int,
    ) -> None:
        super().__init__()
        self.geometry = HexPatchGeometry(
            img_size=img_size,
            in_chans=in_chans,
            kernel_size=kernel_size,
            lattice_stride=lattice_stride,
        )
        self.weight = nn.Parameter(torch.empty(embed_dim, in_chans, self.geometry.num_samples))
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.reset_parameters()

    @property
    def num_patches(self) -> int:
        return self.geometry.num_patches

    @property
    def patch_centers_xy(self) -> torch.Tensor:
        return self.geometry.patch_centers_xy

    @property
    def coo_patchs(self) -> torch.Tensor:
        return self.geometry.coo_patchs

    @property
    def patch_offsets_xy(self) -> torch.Tensor:
        return self.geometry.patch_offsets_xy

    def reset_parameters(self) -> None:
        fan_in = self.weight.shape[1] * self.weight.shape[2]
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def load_vit_patch_projection(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float = 1e-12,
    ) -> None:
        """Resample a square ViT Conv2d projection onto the circular patch offsets."""
        if weight.ndim != 4:
            raise ValueError("ViT patch projection weight must have shape (D, C, H, W)")
        out_dim, in_chans, src_h, src_w = weight.shape
        if (out_dim, in_chans) != tuple(self.weight.shape[:2]):
            raise ValueError(
                f"projection channels {(out_dim, in_chans)} do not match {tuple(self.weight.shape[:2])}"
            )

        target = self.geometry
        grid_x = target.idx_y.to(weight.dtype) * (2.0 / max(target.kernel_size - 1, 1)) - 1.0
        grid_y = target.idx_x.to(weight.dtype) * (2.0 / max(target.kernel_size - 1, 1)) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, -1, 2).to(weight.device)
        source = weight.reshape(1, out_dim * in_chans, src_h, src_w)
        sampled = F.grid_sample(
            source,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        ).reshape(out_dim, in_chans, -1)

        source_norm = weight.flatten(1).norm(dim=1, keepdim=True)
        sampled_norm = sampled.flatten(1).norm(dim=1, keepdim=True)
        sampled = sampled * (source_norm / sampled_norm.clamp_min(eps)).unsqueeze(-1)
        self.weight.copy_(sampled.to(device=self.weight.device, dtype=self.weight.dtype))
        if bias is None:
            self.bias.zero_()
        else:
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.geometry(image)
        return torch.einsum("bncl,dcl->bnd", patches, self.weight) + self.bias
