from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarRotationalDistanceProjection(nn.Module):
    """High-resolution polar prototypes with exact circular angular rotation."""

    def __init__(
        self,
        out_channels: int = 192,
        bases: int = 48,
        directions: int = 16,
        kernel_size: int = 16,
        radial_bins: int = 8,
        angular_bins: int = 64,
        stride: int = 16,
        prototype_std: float = 0.02,
        radial_weight_mode: str = "none",
    ) -> None:
        super().__init__()
        if angular_bins % directions:
            raise ValueError("angular_bins must be divisible by directions")
        if kernel_size < stride or (kernel_size - stride) % 2:
            raise ValueError("kernel_size - stride must be a nonnegative even number")
        self.out_channels, self.bases, self.directions = out_channels, bases, directions
        self.kernel_size, self.radial_bins, self.angular_bins = kernel_size, radial_bins, angular_bins
        self.stride = stride; self.input_padding = (kernel_size - stride) // 2
        self.prototype = nn.Parameter(
            torch.randn(bases, 3, radial_bins, angular_bins) * prototype_std
        )
        self.log2_scale = nn.Parameter(torch.zeros(bases))
        self.value = nn.Parameter(torch.empty(bases, directions, out_channels))
        nn.init.trunc_normal_(self.value, std=0.02)

        # Radius starts at half a pixel: the singular center is never sampled.
        radius = torch.linspace(0.5, kernel_size / 2 - 0.5, radial_bins)
        angle = torch.arange(angular_bins) * (2 * math.pi / angular_bins)
        center = (kernel_size - 1) / 2
        x = center + radius[:, None] * angle.cos()[None]
        y = center + radius[:, None] * angle.sin()[None]
        # align_corners=False pixel-center coordinates.
        grid = torch.stack((2 * (x + 0.5) / kernel_size - 1, 2 * (y + 0.5) / kernel_size - 1), -1)
        self.register_buffer("polar_grid", grid)

        cover = torch.cos(radius * math.pi / kernel_size).clamp_min(0)
        if radial_weight_mode == "none":
            # Uniform polar samples already give every ring equal total mass,
            # implicitly compensating for Cartesian circumference growth.
            radial_weight = torch.ones_like(radius)
        elif radial_weight_mode == "cosine":
            radial_weight = cover
        elif radial_weight_mode == "rho_cos2":
            radial_weight = radius * cover.square()
        else:
            raise ValueError("radial_weight_mode must be none, cosine, or rho_cos2")
        self.radial_weight_mode = radial_weight_mode
        self.register_buffer("radial_weight", radial_weight)
        n_in = float(3 * radial_weight.sum() * angular_bins)
        self.multi = 1.0 / ((n_in / 6.0) - 0.5 * (n_in * 7.0 / 180.0) ** 0.5)
        self.register_buffer(
            "direction_index",
            torch.arange(directions) * (angular_bins // directions),
        )

    def polar_patches(self, image: torch.Tensor) -> torch.Tensor:
        if self.input_padding:
            image = F.pad(image, (self.input_padding,) * 4, mode="reflect")
        square = F.unfold(image, self.kernel_size, stride=self.stride)
        batch, _, tokens = square.shape
        square = square.transpose(1, 2).reshape(batch * tokens, 3, self.kernel_size, self.kernel_size)
        grid = self.polar_grid[None].expand(batch * tokens, -1, -1, -1)
        polar = F.grid_sample(square, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return polar.reshape(batch, tokens, 3, self.radial_bins, self.angular_bins)

    def pose_scores(self, image: torch.Tensor) -> torch.Tensor:
        """B x base x direction x H x W negative distances."""
        with torch.autocast(device_type=image.device.type, enabled=False):
            polar = self.polar_patches(image.float())
            weight = self.radial_weight[None, None, None, :, None]
            patch_energy = (polar.square() * weight).sum((2, 3, 4))
            proto_energy = (
                self.prototype.float().square() * self.radial_weight[None, None, :, None]
            ).sum((1, 2, 3))
            # Only the requested discrete poses are formed. Rotation in polar
            # coordinates is an exact circular shift, without interpolation.
            rotated = torch.stack(
                [torch.roll(self.prototype.float(), shifts=int(shift), dims=-1)
                 for shift in self.direction_index],
                dim=1,
            )
            corr = torch.einsum("qtcrf,ndcrf->qtnd", polar * weight, rotated)
            distance = (
                patch_energy[:, :, None, None]
                + proto_energy[None, None, :, None]
                - 2 * corr
            ).clamp_min(0)
            scale = torch.exp2(self.log2_scale)[None, None, :, None]
            score = -distance * scale * self.multi
            side = int(polar.shape[1] ** 0.5)
            return score.permute(0, 2, 3, 1).reshape(image.shape[0], self.bases, self.directions, side, side)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        scores = self.pose_scores(image)
        direction_weight = scores.softmax(dim=2)
        base_value = torch.einsum("qbrhw,brc->qbhwc", direction_weight, self.value)
        amplitude = scores.amax(dim=2) - scores.mean(dim=2)
        output = (base_value * amplitude[..., None]).sum(dim=1)
        return output.permute(0, 3, 1, 2).contiguous()
