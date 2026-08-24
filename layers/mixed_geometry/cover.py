from __future__ import annotations

import math

import torch
import torch.nn as nn


def _logit(values: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.logit(values.clamp(eps, 1 - eps))


def cosine_mask_initializers(
    *,
    family_counts,
    angular_bins,
    radial_bins,
    stripe_bins,
    stripe_longitudinal_bins,
    ring_counts,
    radius_scale,
    eps,
):
    """Cosine-compatible logits in each prototype's native coordinates."""
    sample = torch.linspace(-1, 1, 513)
    angular_frequency = math.pi / (2 * radius_scale)
    disk_mean = torch.tensor(2 * (
        math.sin(angular_frequency) / angular_frequency
        + (math.cos(angular_frequency) - 1) / angular_frequency**2
    ))

    angular = _logit(disk_mean, eps).expand(angular_bins).clone()
    radial_position = (
        torch.arange(radial_bins, dtype=torch.float32) + .5
    ) / radial_bins
    radial = _logit(
        torch.cos(
            radial_position * math.pi / (2 * radius_scale)
        ).clamp_min(0),
        eps,
    )
    color = _logit(disk_mean, eps)

    across = torch.linspace(-1, 1, stripe_bins)
    if stripe_longitudinal_bins == 1:
        stripe_values = []
        for coordinate in across:
            valid = sample.square() + coordinate.square() < 1
            if valid.any():
                value = torch.cos(
                    torch.sqrt(sample[valid].square() + coordinate.square())
                    * math.pi / (2 * radius_scale)
                ).clamp_min(0).mean()
            else:
                value = torch.tensor(0.0)
            stripe_values.append(value)
        stripe = _logit(torch.stack(stripe_values), eps)
    else:
        along = torch.linspace(-1, 1, stripe_longitudinal_bins)
        stripe_radius = torch.sqrt(
            across[:, None].square() + along[None].square()
        )
        stripe = _logit(
            torch.cos(
                stripe_radius * math.pi / (2 * radius_scale)
            ).clamp_min(0),
            eps,
        )

    full_parts = []
    for ring, count in enumerate(ring_counts.tolist()):
        position = (ring + .5) / radial_bins
        value = math.cos(position * math.pi / (2 * radius_scale))
        full_parts.append(torch.full((int(count),), max(value, 0)))
    full = _logit(torch.cat(full_parts), eps)

    templates = {
        "angular": angular,
        "radial": radial,
        "color": color,
        "stripe": stripe,
        "full": full,
    }
    return {
        family: template.expand(
            int(family_counts[family]), 3, *template.shape
        ).clone()
        for family, template in templates.items()
    }


class CoverMaskBank(nn.ParameterDict):
    """Per-prototype mask logits stored in native prototype coordinates.

    The same geometry renderer transforms the RGB prototype and its scalar mask
    logits. Sigmoid is applied after interpolation, so scale and rotation cannot
    take a different sampling path from the distance kernel.
    """

    def __init__(self, initializers, eps, *, enabled=False):
        super().__init__()
        self.eps = float(eps)
        self.enabled = bool(enabled)
        if not self.enabled:
            return
        for family, initial in initializers.items():
            self[family] = nn.Parameter(initial)

    def render(self, family, geometry, renderer, *, normalize=True):
        if not self.enabled:
            return None
        rendered_logits = renderer(self[family])
        mask = rendered_logits.sigmoid().clamp_min(self.eps)
        if not normalize:
            return mask
        # Keep the cosine-cover mass so the model cannot reduce every distance
        # simply by shrinking the complete mask.
        target_mass = 3 * geometry.support_cover.sum()
        return mask * (
            target_mass
            / mask.sum((-2, -1), keepdim=True).clamp_min(self.eps)
        )

    @torch.no_grad()
    def summary(self):
        if not self.enabled:
            return None
        result = {}
        for family, logits in self.items():
            values = logits.sigmoid().flatten(1)
            if values.shape[0] == 0:
                result[family] = {
                    "minimum": [], "mean": [], "maximum": [],
                    "effective_points": [], "shape": list(logits.shape[1:]),
                }
                continue
            effective = (
                values.sum(1).square()
                / values.square().sum(1).clamp_min(1e-12)
            )
            result[family] = {
                "minimum": values.min(1).values.tolist(),
                "mean": values.mean(1).tolist(),
                "maximum": values.max(1).values.tolist(),
                "effective_points": effective.tolist(),
                "shape": list(logits.shape[1:]),
            }
        return result
