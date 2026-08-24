from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceNormLinearProjection(nn.Module):
    """Cosine-covered squared distances, basis LayerNorm, then 1x1 Linear."""

    def __init__(self, teacher: nn.Conv2d, kernel_size: int = 24) -> None:
        super().__init__()
        if kernel_size < 16 or (kernel_size - 16) % 2:
            raise ValueError("kernel_size must be an even integer no smaller than 16")
        self.kernel_size = int(kernel_size)
        self.input_padding = (self.kernel_size - 16) // 2
        centers = teacher.weight.detach().clone()
        if self.input_padding:
            centers = F.pad(centers, (self.input_padding,) * 4, mode="reflect")
        self.prototype = nn.Parameter(centers)
        self.log2_scale = nn.Parameter(torch.zeros(centers.shape[0]))
        self.norm = nn.LayerNorm(centers.shape[0], elementwise_affine=False)
        self.linear = nn.Conv2d(centers.shape[0], teacher.out_channels, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        yy, xx = torch.meshgrid(
            torch.arange(self.kernel_size), torch.arange(self.kernel_size), indexing="ij"
        )
        center = (self.kernel_size - 1) / 2
        radius = torch.sqrt((xx - center).square() + (yy - center).square())
        self.register_buffer(
            "cover", torch.cos(radius * torch.pi / self.kernel_size).clamp_min(0)
        )

    def distances(self, image: torch.Tensor) -> torch.Tensor:
        if self.input_padding:
            image = F.pad(image, (self.input_padding,) * 4, mode="reflect")
        cross = F.conv2d(image, self.prototype * self.cover, stride=16)
        patch_energy = F.conv2d(
            image.square(), self.cover.expand(1, 3, -1, -1), stride=16
        )
        prototype_energy = (self.prototype.square() * self.cover).sum((1, 2, 3))
        return (
            patch_energy + prototype_energy[None, :, None, None] - 2 * cross
        ).clamp_min(0)

    def basis(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            distance = self.distances(image.float())
            score = -distance * torch.exp2(self.log2_scale)[None, :, None, None]
            # Normalize the 192 competing bases independently at each token.
            return self.norm(score.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.linear(self.basis(image))

