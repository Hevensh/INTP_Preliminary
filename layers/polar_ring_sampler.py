from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarRingSampler(nn.Module):
    """Differentiably sample scale-indexed polar rings around image locations.

    Centers use ordinary image pixel coordinates ``(x=column, y=row)``.  The
    returned tensor has shape ``(B, N, C, S, R, A)``.  The unresolved disk
    ``rho < rho_min`` is never sampled or stored.
    """

    def __init__(
        self,
        *,
        radial_bins: int = 8,
        angular_bins: int = 24,
        rotation_samples: int = 12,
        scales: Sequence[float] = (
            1.0,
            2.0 ** 0.5,
            3.0 ** 0.5,
            2.0,
        ),
        rho_min: float | None = None,
        padding_mode: str = "reflection",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(radial_bins, angular_bins, rotation_samples) <= 0:
            raise ValueError("sample counts must be positive")
        if angular_bins % rotation_samples:
            raise ValueError("angular_bins must be divisible by rotation_samples")
        if not scales or any(float(scale) <= 0.0 for scale in scales):
            raise ValueError("scales must contain positive values")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("unsupported padding_mode")

        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self.rotation_samples = int(rotation_samples)
        self.rho_min = float(1.0 / radial_bins if rho_min is None else rho_min)
        if not 0.0 < self.rho_min < 1.0:
            raise ValueError("rho_min must be in (0, 1)")
        self.padding_mode = padding_mode
        self.eps = float(eps)
        self.register_buffer(
            "scales", torch.tensor(tuple(float(scale) for scale in scales)), persistent=True
        )

    @property
    def num_scales(self) -> int:
        return int(self.scales.numel())

    def forward(
        self,
        image: torch.Tensor,
        centers_xy: torch.Tensor,
        *,
        base_radius: float,
        return_coverage: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError("image must have shape (B, C, H, W)")
        if centers_xy.ndim != 2 or centers_xy.shape[-1] != 2:
            raise ValueError("centers_xy must have shape (N, 2)")
        if base_radius <= 0.0:
            raise ValueError("base_radius must be positive")

        batch, channels, height, width = image.shape
        centers = centers_xy.to(device=image.device, dtype=image.dtype)
        max_radius = float(base_radius) * float(self.scales.max())
        extent = int(math.ceil(max_radius))
        axis = torch.arange(-extent, extent + 1, device=image.device, dtype=image.dtype)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2)
        distances = offsets.norm(dim=-1)
        angles = torch.remainder(torch.atan2(offsets[:, 1], offsets[:, 0]), 2.0 * math.pi)

        # Extract the largest Cartesian support once.  Usually Hex centers are
        # integral; bilinear sampling also keeps fractional learned centers smooth.
        points = centers[:, None, :] + offsets[None]
        grid = points.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
        grid = grid.reshape(1, centers.shape[0] * offsets.shape[0], 1, 2).expand(batch, -1, -1, -1)
        cartesian = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=True,
        )
        cartesian = cartesian.reshape(batch, channels, centers.shape[0], offsets.shape[0])
        cartesian = cartesian.permute(0, 2, 1, 3)  # (B,N,C,K)

        sampled_scales = []
        coverage_scales = []
        for scale in self.scales.to(device=image.device, dtype=image.dtype):
            rho = distances / (float(base_radius) * scale)
            valid = (rho >= self.rho_min) & (rho <= 1.0)
            radial_position = (rho - self.rho_min) / (1.0 - self.rho_min) * (self.radial_bins - 1)
            angular_position = angles / (2.0 * math.pi) * self.angular_bins
            r0 = radial_position.floor().clamp(0, self.radial_bins - 1).long()
            r1 = (r0 + 1).clamp(max=self.radial_bins - 1)
            a0 = angular_position.floor().long() % self.angular_bins
            a1 = (a0 + 1) % self.angular_bins
            rw = (radial_position - radial_position.floor()).clamp(0.0, 1.0)
            aw = angular_position - angular_position.floor()

            values_flat = image.new_zeros(batch, centers.shape[0], channels, self.radial_bins * self.angular_bins)
            coverage_flat = image.new_zeros(self.radial_bins * self.angular_bins)
            for radial, angular, weight in (
                (r0, a0, (1.0 - rw) * (1.0 - aw)),
                (r0, a1, (1.0 - rw) * aw),
                (r1, a0, rw * (1.0 - aw)),
                (r1, a1, rw * aw),
            ):
                weight = weight * valid.to(weight.dtype)
                index = radial * self.angular_bins + angular
                values_flat.scatter_add_(
                    -1,
                    index.view(1, 1, 1, -1).expand(batch, centers.shape[0], channels, -1),
                    cartesian * weight.view(1, 1, 1, -1),
                )
                coverage_flat.scatter_add_(0, index, weight)
            values_flat = values_flat / coverage_flat.clamp_min(self.eps).view(1, 1, 1, -1)
            sampled_scales.append(
                values_flat.reshape(batch, centers.shape[0], channels, self.radial_bins, self.angular_bins)
            )
            coverage_scales.append(coverage_flat.reshape(self.radial_bins, self.angular_bins))
        rings = torch.stack(sampled_scales, dim=3)
        coverage = torch.stack(coverage_scales, dim=0)
        return (rings, coverage) if return_coverage else rings

    def circular_match(
        self,
        rings: torch.Tensor,
        prototypes: torch.Tensor,
        coverage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-sided normalized circular correlation, ``(B,N,H,S,T)``.

        The valid polar area is averaged so larger rings do not gain merely by
        containing more samples.  Only the prototype RMS is normalized: the
        input ring norm is deliberately retained as signal strength.
        """
        if rings.ndim != 6:
            raise ValueError("rings must have shape (B, N, C, S, R, A)")
        if prototypes.ndim != 4:
            raise ValueError("prototypes must have shape (H, C, R, A)")
        if rings.shape[2] != prototypes.shape[1]:
            raise ValueError("ring and prototype channel counts differ")
        if rings.shape[3:] != (
            self.num_scales,
            self.radial_bins,
            self.angular_bins,
        ):
            raise ValueError("rings do not match sampler dimensions")
        if prototypes.shape[-2:] != (self.radial_bins, self.angular_bins):
            raise ValueError("prototypes do not match sampler dimensions")

        rho = torch.linspace(
            self.rho_min, 1.0, self.radial_bins,
            device=rings.device, dtype=rings.dtype,
        )
        radial_weight = rho / rho.sum()
        if coverage is None:
            mask = rings.new_ones(self.num_scales, self.radial_bins, self.angular_bins)
        else:
            if coverage.shape != (self.num_scales, self.radial_bins, self.angular_bins):
                raise ValueError("coverage must have shape (S, R, A)")
            mask = (coverage > self.eps).to(device=rings.device, dtype=rings.dtype)

        doubled = torch.cat((rings, rings), dim=-1)
        windows = doubled.unfold(-1, self.angular_bins, 1)[..., : self.angular_bins, :]
        doubled_mask = torch.cat((mask, mask), dim=-1)
        mask_windows = doubled_mask.unfold(-1, self.angular_bins, 1)[..., : self.angular_bins, :]
        stride = self.angular_bins // self.rotation_samples
        shifts = torch.arange(
            0, self.angular_bins, stride, device=rings.device, dtype=torch.long
        )
        windows = windows.index_select(-2, shifts)
        mask_windows = mask_windows.index_select(-2, shifts)
        weight = mask_windows * radial_weight.view(1, self.radial_bins, 1, 1)
        weight = weight / weight.sum(dim=(1, 3), keepdim=True).clamp_min(self.eps)
        numerator = torch.einsum("bncsrtu,hcru,srtu->bnhst", windows, prototypes, weight)
        prototype_rms = torch.einsum("hcru,srtu->hst", prototypes.square(), weight).sqrt()
        return numerator / prototype_rms[None, None].clamp_min(self.eps)
