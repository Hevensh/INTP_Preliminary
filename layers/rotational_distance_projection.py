from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationalDistanceProjection(nn.Module):
    """Distance tokenizer with shared prototypes and explicit rotated poses."""

    def __init__(
        self,
        out_channels: int = 192,
        bases: int = 48,
        directions: int = 16,
        kernel_size: int = 24,
        stride: int = 16,
        prototype_std: float = 0.02,
        base_aggregation: str = "softmax",
    ) -> None:
        super().__init__()
        if kernel_size < stride or (kernel_size - stride) % 2:
            raise ValueError("kernel_size - stride must be a nonnegative even number")
        self.out_channels = out_channels
        self.bases = bases
        self.directions = directions
        self.kernel_size = kernel_size
        self.stride = stride
        if base_aggregation not in {"softmax", "amplitude_sum"}:
            raise ValueError("base_aggregation must be softmax or amplitude_sum")
        self.base_aggregation = base_aggregation
        self.input_padding = (kernel_size - stride) // 2

        self.prototype = nn.Parameter(
            torch.randn(bases, 3, kernel_size, kernel_size) * prototype_std
        )
        self.log2_scale = nn.Parameter(torch.zeros(bases))
        self.value = nn.Parameter(torch.empty(bases, directions, out_channels))
        nn.init.trunc_normal_(self.value, std=0.02)

        yy, xx = torch.meshgrid(
            torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij"
        )
        center = (kernel_size - 1) / 2
        radius = torch.sqrt((xx - center).square() + (yy - center).square())
        cover = torch.cos(radius * torch.pi / kernel_size).clamp_min(0)
        self.register_buffer("cover", cover)
        n_in = float(3 * cover.sum())
        self.multi = 1.0 / ((n_in / 6.0) - 0.5 * (n_in * 7.0 / 180.0) ** 0.5)

        angles = torch.arange(directions, dtype=torch.float32) * (2 * math.pi / directions)
        theta = torch.zeros(directions, 2, 3)
        theta[:, 0, 0] = angles.cos()
        theta[:, 0, 1] = -angles.sin()
        theta[:, 1, 0] = angles.sin()
        theta[:, 1, 1] = angles.cos()
        self.register_buffer("rotation_theta", theta)

    def rotated_prototypes(self) -> torch.Tensor:
        # Treat every (base, direction) pair as an image for differentiable
        # bilinear rotation. Circular cover makes the support rotation-invariant.
        source = self.prototype[:, None].expand(-1, self.directions, -1, -1, -1)
        source = source.reshape(self.bases * self.directions, 3, self.kernel_size, self.kernel_size)
        theta = self.rotation_theta[None].expand(self.bases, -1, -1, -1)
        theta = theta.reshape(self.bases * self.directions, 2, 3)
        grid = F.affine_grid(theta, source.shape, align_corners=False)
        return F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def pose_scores(self, image: torch.Tensor) -> torch.Tensor:
        """Return negative scaled distances as B x base x direction x H x W."""
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            if self.input_padding:
                image = F.pad(image, (self.input_padding,) * 4, mode="reflect")
            rotated = self.rotated_prototypes().float()
            cross = F.conv2d(image, rotated * self.cover, stride=self.stride)
            patch_energy = F.conv2d(
                image.square(), self.cover.expand(1, 3, -1, -1), stride=self.stride
            )
            prototype_energy = (rotated.square() * self.cover).sum((1, 2, 3))
            distance = (patch_energy + prototype_energy[None, :, None, None] - 2 * cross).clamp_min(0)
            h, w = distance.shape[-2:]
            distance = distance.view(image.shape[0], self.bases, self.directions, h, w)
            scale = torch.exp2(self.log2_scale)[None, :, None, None, None]
            return -distance * scale * self.multi

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        scores = self.pose_scores(image)
        direction_weight = scores.softmax(dim=2)
        # B, base, H, W, C: orientation-conditioned value for each base.
        base_value = torch.einsum("qbrhw,brc->qbhwc", direction_weight, self.value)
        amplitude = scores.amax(dim=2) - scores.mean(dim=2)
        base_weight = amplitude.softmax(dim=1) if self.base_aggregation == "softmax" else amplitude
        output = (base_value * base_weight[..., None]).sum(dim=1)
        return output.permute(0, 3, 1, 2).contiguous()
