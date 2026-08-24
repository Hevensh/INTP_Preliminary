from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_distance import triton_distance_similarity


class IndexedL1DistanceProjection(nn.Module):
    """Circular indexed L1 prototypes followed by a learned 1x1 projection."""

    def __init__(self, teacher: nn.Conv2d, kernel_size: int = 24) -> None:
        super().__init__()
        if kernel_size < 16 or (kernel_size - 16) % 2:
            raise ValueError("kernel_size must be an even integer no smaller than 16")
        self.kernel_size = int(kernel_size)
        self.input_padding = (self.kernel_size - 16) // 2
        centers = teacher.weight.detach().clone()
        if self.input_padding:
            centers = F.pad(centers, (self.input_padding,) * 4, mode="reflect")
        yy, xx = torch.meshgrid(
            torch.arange(self.kernel_size), torch.arange(self.kernel_size), indexing="ij"
        )
        center = (self.kernel_size - 1) / 2
        radius = torch.sqrt((xx - center).square() + (yy - center).square())
        mask = radius < self.kernel_size / 2
        cover = torch.cos(radius[mask] * torch.pi / self.kernel_size)
        self.prototype = nn.Parameter(centers.flatten(2)[..., mask.flatten()].clone())
        self.log2_scale = nn.Parameter(torch.zeros(self.prototype.shape[0]))
        self.linear = nn.Conv2d(self.prototype.shape[0], teacher.out_channels, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("cover", cover)
        n_in = float(3 * cover.sum())
        self.multi = 1.0 / ((n_in / 6.0) - 0.5 * (n_in * 7.0 / 180.0) ** 0.5)

    def extract(self, image: torch.Tensor) -> torch.Tensor:
        if self.input_padding:
            image = F.pad(image, (self.input_padding,) * 4, mode="reflect")
        square = F.unfold(image, self.kernel_size, stride=16)
        b, _, n = square.shape
        square = square.view(b, 3, self.kernel_size**2, n)
        yy, xx = torch.meshgrid(
            torch.arange(self.kernel_size, device=image.device),
            torch.arange(self.kernel_size, device=image.device), indexing="ij"
        )
        center = (self.kernel_size - 1) / 2
        mask = torch.sqrt((xx - center).square() + (yy - center).square()).flatten() < self.kernel_size / 2
        return square[:, :, mask, :].permute(0, 3, 1, 2).contiguous()

    def basis(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            patches = self.extract(image.float())
            flat = patches.reshape(-1, patches.shape[-2] * patches.shape[-1])
            prototype = self.prototype.reshape(self.prototype.shape[0], -1)
            similarity = triton_distance_similarity(
                flat, prototype, self.log2_scale, self.cover.repeat(3), self.multi, "l1"
            )
            basis = similarity.view(image.shape[0], patches.shape[1], -1).permute(0, 2, 1)
            side = int(patches.shape[1] ** 0.5)
            return basis.reshape(image.shape[0], -1, side, side)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.linear(self.basis(image))

    @torch.no_grad()
    def calibrate_l1_scale_from_l2(self, image: torch.Tensor, max_patches: int = 1024) -> None:
        patches = self.extract(image.float()).reshape(-1, 3, self.cover.numel())[:max_patches]
        difference = patches[:, None] - self.prototype[None]
        l1 = (difference.abs() * self.cover).sum((-1, -2)).mean(0)
        l2 = (difference.square() * self.cover).sum((-1, -2)).mean(0)
        self.log2_scale.copy_((l2 / l1.clamp_min(1e-8)).clamp_min(1e-8).log2())

