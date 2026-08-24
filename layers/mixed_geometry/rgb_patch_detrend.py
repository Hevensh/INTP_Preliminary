from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from layers.mixed_geometry.geometries import DiskGeometry


@dataclass(frozen=True)
class RGBTrendState:
    """Lossless low-order state accompanying a canonical RGB residual."""

    mean: torch.Tensor
    gradient_x: torch.Tensor
    gradient_y: torch.Tensor
    sigma_safe: torch.Tensor


class RGBPatchDetrender(nn.Module):
    """Separate RGB DC/trend state from the residual used for matching.

    Geometry matching consumes only the zero-mean, zero-linear-trend residual.
    A fixed stride-sized circular patch independently supplies output-side RGB
    mean/scale modulation and six signed RGB-by-xy trend V vectors.
    """

    def __init__(
        self,
        out_channels: int,
        stride: int,
        variance_epsilon: float = 1e-4,
        imagenet_mean=(0.485, 0.456, 0.406),
        imagenet_std=(0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        if variance_epsilon <= 0:
            raise ValueError("variance_epsilon must be positive")
        self.out_channels = int(out_channels)
        self.variance_epsilon = float(variance_epsilon)
        self.reference_geometry = DiskGeometry(stride, stride)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(imagenet_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(imagenet_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # R_x, R_y, G_x, G_y, B_x, and B_y: exactly six new V vectors.
        self.trend_value = nn.Parameter(torch.empty(3, 2, self.out_channels))
        nn.init.trunc_normal_(self.trend_value, std=0.02)

    def unit_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """Invert the ImageNet input normalization before RGB decomposition."""
        return (image * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)

    def decompose(
        self,
        samples: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, RGBTrendState]:
        """Fit an RGB plane over the final sample axis.

        ``samples`` may have arbitrary leading axes but must end in
        ``[..., RGB, support]``.  One joint RGB residual variance preserves
        the relative amplitude between channels.
        """
        if samples.shape[-2] != 3:
            raise ValueError("RGB samples must have three channels")
        if (
            samples.shape[-1] != support_x.numel()
            or support_x.shape != support_y.shape
            or support_weight.shape != support_x.shape
        ):
            raise ValueError("sample support does not match its coordinates")

        dtype = samples.dtype
        x = support_x.to(device=samples.device, dtype=dtype)
        y = support_y.to(device=samples.device, dtype=dtype)
        weight = support_weight.to(device=samples.device, dtype=dtype)
        weight_sum = weight.sum().clamp_min(torch.finfo(dtype).eps)
        radius = max(float(x.abs().max()), float(y.abs().max()), 1.0)
        x = x / radius
        y = y / radius

        # One shared RGB DC term preserves channel differences in the residual.
        mean = (samples * weight).sum(dim=(-2, -1)) / (3.0 * weight_sum)
        centered = samples - mean[..., None, None]
        xx = (weight * x.square()).sum().clamp_min(torch.finfo(dtype).eps)
        yy = (weight * y.square()).sum().clamp_min(torch.finfo(dtype).eps)
        xy = (weight * x * y).sum()
        determinant = (xx * yy - xy.square()).clamp_min(torch.finfo(dtype).eps)
        projection_x = (centered * weight * x).sum(dim=-1)
        projection_y = (centered * weight * y).sum(dim=-1)
        gradient_x = (yy * projection_x - xy * projection_y) / determinant
        gradient_y = (xx * projection_y - xy * projection_x) / determinant
        residual = (
            centered
            - gradient_x[..., None] * x
            - gradient_y[..., None] * y
        )
        variance = (
            residual.square() * weight
        ).sum(dim=(-2, -1)) / (3.0 * weight_sum)
        sigma_safe = torch.sqrt(variance + self.variance_epsilon)
        normalized = residual / sigma_safe[..., None, None]
        return normalized, RGBTrendState(
            mean=mean,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            sigma_safe=sigma_safe,
        )

    def canonicalize(
        self,
        samples: torch.Tensor,
        geometry,
    ) -> torch.Tensor:
        normalized, _ = self.decompose(
            samples,
            geometry.support_x,
            geometry.support_y,
            geometry.support_cover,
        )
        return normalized

    def output_terms(
        self,
        unit_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return feature-wise multiplicative gate and additive trend V."""
        samples = self.reference_geometry.indexed_patches(unit_image)
        _, state = self.decompose(
            samples,
            self.reference_geometry.support_x,
            self.reference_geometry.support_y,
            self.reference_geometry.support_cover,
        )
        gradient = torch.stack(
            (state.gradient_x, state.gradient_y), dim=-1
        )
        trend = torch.einsum(
            "qtck,ckd->qtd", gradient, self.trend_value
        )
        side = math.isqrt(samples.shape[1])
        if side * side != samples.shape[1]:
            raise RuntimeError("reference patch grid must be square")
        modulation = state.mean.reshape(
            unit_image.shape[0], side, side, 1
        )
        trend = trend.reshape(
            unit_image.shape[0], side, side, self.out_channels
        )
        return modulation, trend
