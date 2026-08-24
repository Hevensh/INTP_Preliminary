from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from layers.mixed_geometry.geometries import DiskGeometry


@dataclass(frozen=True)
class RGBMomentState:
    """Shared RGB moments removed from one circular patch."""

    mean: torch.Tensor
    sigma_safe: torch.Tensor


class RGBPatchStandardizer(nn.Module):
    """Standardize circular RGB patches without removing spatial trends.

    Mean and variance are jointly estimated over RGB and spatial support with
    the geometry's cosine cover.  A single pair of scalars therefore preserves
    chromatic differences inside the normalized residual.  The removed mean
    and standard deviation are exposed separately for value-side conditioning.
    """

    def __init__(
        self,
        stride: int,
        variance_epsilon: float = 1e-4,
        imagenet_mean=(0.485, 0.456, 0.406),
        imagenet_std=(0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        if variance_epsilon <= 0:
            raise ValueError("variance_epsilon must be positive")
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

    def unit_rgb(self, image: torch.Tensor) -> torch.Tensor:
        return (image * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)

    def decompose(
        self,
        samples: torch.Tensor,
        support_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, RGBMomentState]:
        if samples.shape[-2] != 3:
            raise ValueError("RGB samples must have three channels")
        if samples.shape[-1] != support_weight.numel():
            raise ValueError("sample support does not match its weights")
        weight = support_weight.to(device=samples.device, dtype=samples.dtype)
        weight_sum = weight.sum().clamp_min(torch.finfo(samples.dtype).eps)
        mean = (samples * weight).sum(dim=(-2, -1)) / (3.0 * weight_sum)
        centered = samples - mean[..., None, None]
        variance = (
            centered.square() * weight
        ).sum(dim=(-2, -1)) / (3.0 * weight_sum)
        sigma_safe = torch.sqrt(variance + self.variance_epsilon)
        return centered / sigma_safe[..., None, None], RGBMomentState(
            mean=mean,
            sigma_safe=sigma_safe,
        )

    def canonicalize(self, samples: torch.Tensor, geometry) -> torch.Tensor:
        normalized, _ = self.decompose(samples, geometry.support_cover)
        return normalized

    def output_stats(self, unit_image: torch.Tensor) -> torch.Tensor:
        """Return ``[mean, sigma_safe]`` maps aligned to tokenizer patches."""
        samples = self.reference_geometry.indexed_patches(unit_image)
        _, state = self.decompose(
            samples, self.reference_geometry.support_cover
        )
        side = int(samples.shape[1] ** 0.5)
        if side * side != samples.shape[1]:
            raise RuntimeError("reference patch grid must be square")
        return torch.stack((state.mean, state.sigma_safe), dim=-1).reshape(
            unit_image.shape[0], side, side, 2
        )
