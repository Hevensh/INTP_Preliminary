from __future__ import annotations

import torch
import torch.nn as nn

from layers.mixed_geometry_distance_projection import _DiskGeometry, _StripeGeometry
from layers.multiscale_variable_ring_polar_distance_projection import _PolarScaleGeometry


class SingleScaleGeometricProjection(nn.Module):
    """Single-scale tokenizer with four explicit geometric kernel families.

    Families:
      * cartesian: a full learned circular-support kernel, one fixed pose;
      * rotating_stripe: a compact stripe kernel evaluated at every direction;
      * fixed_stripe: the same compact stripe kernel at one stored direction;
      * rotating_polar: a compact polar kernel evaluated at every direction.

    Every distance is kept as a negative score.  Pose softmax is local to one
    prototype, and exp2(score) retains the distance-response amplitude used by
    the smooth polar baseline.
    """

    def __init__(
        self,
        out_channels: int = 192,
        cartesian_bases: int = 32,
        rotating_stripe_bases: int = 16,
        fixed_stripe_bases: int = 24,
        rotating_polar_bases: int = 24,
        directions: int = 8,
        kernel_size: int = 24,
        stripe_bins: int = 24,
        stripe_longitudinal_bins: int = 4,
        radial_bins: int = 12,
        ring_counts=None,
        stride: int = 16,
        cover_radius_scale: float = 1.0,
        prototype_std: float = .02,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.directions = int(directions)
        self.kernel_size = int(kernel_size)
        self.stripe_bins = int(stripe_bins)
        self.stripe_longitudinal_bins = int(stripe_longitudinal_bins)
        self.radial_bins = int(radial_bins)
        self.cover_radius_scale = float(cover_radius_scale)
        self.family_counts = {
            "cartesian": int(cartesian_bases),
            "rotating_stripe": int(rotating_stripe_bases),
            "fixed_stripe": int(fixed_stripe_bases),
            "rotating_polar": int(rotating_polar_bases),
        }
        if ring_counts is None:
            ring_counts = [4 * (radius + 1) for radius in range(radial_bins)]
        counts = torch.as_tensor(ring_counts, dtype=torch.long)
        if counts.numel() != radial_bins:
            raise ValueError("ring_counts must contain radial_bins entries")
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.register_buffer("ring_counts", counts)
        self.register_buffer("ring_offsets", offsets)

        self.cartesian_geometry = _DiskGeometry(
            kernel_size, stride, cover_radius_scale
        )
        self.rotating_stripe_geometry = _StripeGeometry(
            kernel_size, directions, stripe_bins, stripe_longitudinal_bins,
            stride, cover_radius_scale,
        )
        self.fixed_stripe_geometry = _StripeGeometry(
            kernel_size, directions, stripe_bins, stripe_longitudinal_bins,
            stride, cover_radius_scale,
        )
        self.rotating_polar_geometry = _PolarScaleGeometry(
            kernel_size, stride, radial_bins, counts, offsets, directions,
            cover_radius_scale,
        )
        support_points = int(self.cartesian_geometry.support_mask.sum())

        self.cartesian_prototype = nn.Parameter(
            torch.randn(cartesian_bases, 3, support_points) * prototype_std
        )
        stripe_shape = (3, stripe_bins, stripe_longitudinal_bins)
        self.rotating_stripe_prototype = nn.Parameter(
            torch.randn(rotating_stripe_bases, *stripe_shape) * prototype_std
        )
        self.fixed_stripe_prototype = nn.Parameter(
            torch.randn(fixed_stripe_bases, *stripe_shape) * prototype_std
        )
        self.rotating_polar_prototype = nn.Parameter(
            torch.randn(rotating_polar_bases, 3, int(offsets[-1])) * prototype_std
        )
        # A fixed stripe may still be horizontal, vertical, or diagonal.  This
        # buffer records its one allowed orientation without creating poses.
        self.register_buffer(
            "fixed_stripe_direction",
            torch.zeros(fixed_stripe_bases, dtype=torch.long),
        )

        self.cartesian_log2_scale = nn.Parameter(torch.zeros(cartesian_bases))
        self.rotating_stripe_log2_scale = nn.Parameter(
            torch.zeros(rotating_stripe_bases)
        )
        self.fixed_stripe_log2_scale = nn.Parameter(torch.zeros(fixed_stripe_bases))
        self.rotating_polar_log2_scale = nn.Parameter(
            torch.zeros(rotating_polar_bases)
        )

        self.cartesian_value = nn.Parameter(
            torch.empty(cartesian_bases, 1, out_channels)
        )
        self.rotating_stripe_value = nn.Parameter(
            torch.empty(rotating_stripe_bases, directions, out_channels)
        )
        self.fixed_stripe_value = nn.Parameter(
            torch.empty(fixed_stripe_bases, 1, out_channels)
        )
        self.rotating_polar_value = nn.Parameter(
            torch.empty(rotating_polar_bases, directions, out_channels)
        )
        for value in (
            self.cartesian_value,
            self.rotating_stripe_value,
            self.fixed_stripe_value,
            self.rotating_polar_value,
        ):
            nn.init.trunc_normal_(value, std=.02)
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        # A null pose has score but no V.  It joins each base's pose softmax,
        # allowing a patch to reject every real direction/scale candidate.
        self.null_scores = nn.ParameterDict({
            family: nn.Parameter(torch.full((number,), -20.0))
            for family, number in self.family_counts.items()
        })
        self.pose_temperature = 1.0
        self.hard_pose = False
        self.use_null = False
        # Diagnostics and auxiliary routing losses can opt in to retaining the
        # probabilities already computed by forward.  It is disabled by
        # default so ordinary inference does not keep autograd graphs alive.
        self.capture_routing_probabilities = False
        self.routing_probability_cache = {}

    @property
    def pose_value_count(self) -> int:
        counts = self.family_counts
        return (
            counts["cartesian"]
            + counts["fixed_stripe"]
            + self.directions
            * (counts["rotating_stripe"] + counts["rotating_polar"])
        )

    def set_pose_selection(
        self, temperature: float = 1.0, hard: bool = False,
        use_null: bool | None = None,
    ):
        if temperature <= 0:
            raise ValueError("pose temperature must be positive")
        self.pose_temperature = float(temperature)
        self.hard_pose = bool(hard)
        if use_null is not None:
            self.use_null = bool(use_null)

    def set_routing_probability_capture(self, enabled: bool = True):
        self.capture_routing_probabilities = bool(enabled)
        self.routing_probability_cache = {}

    def _cache_routing_probability(self, family, probability, pose_shape):
        if self.capture_routing_probabilities:
            self.routing_probability_cache[family] = {
                "probability": probability,
                "pose_shape": tuple(int(size) for size in pose_shape),
                "has_null": bool(self.use_null),
            }

    @staticmethod
    def _render_cartesian(prototype: torch.Tensor) -> torch.Tensor:
        return prototype[:, None]

    def _render_fixed_stripe(self) -> torch.Tensor:
        rendered = self.fixed_stripe_geometry.render(self.fixed_stripe_prototype)
        base = torch.arange(rendered.shape[0], device=rendered.device)
        return rendered[base, self.fixed_stripe_direction][:, None]

    @staticmethod
    def _scores(image, geometry, rendered, log2_scale):
        patch = geometry.indexed_patches(image)
        weight = geometry.support_cover[None, None, None]
        cross = torch.einsum("qtcm,bpcm->qtbp", patch * weight, rendered)
        patch_energy = (patch.square() * weight).sum((2, 3))
        prototype_energy = (
            rendered.square() * geometry.support_cover[None, None, None]
        ).sum((2, 3))
        distance = (
            patch_energy[:, :, None, None]
            + prototype_energy[None, None]
            - 2 * cross
        ).clamp_min(0)
        side = int(patch.shape[1] ** .5)
        distance = distance.permute(0, 2, 3, 1).reshape(
            image.shape[0], rendered.shape[0], rendered.shape[1], side, side
        )
        rate = torch.exp2(log2_scale)[None, :, None, None, None]
        return -distance * rate * geometry.distance_multiplier

    def family_scores(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            cartesian = self._scores(
                image,
                self.cartesian_geometry,
                self._render_cartesian(self.cartesian_prototype),
                self.cartesian_log2_scale,
            )
            rotating_stripe = self._scores(
                image,
                self.rotating_stripe_geometry,
                self.rotating_stripe_geometry.render(
                    self.rotating_stripe_prototype
                ),
                self.rotating_stripe_log2_scale,
            )
            fixed_stripe = self._scores(
                image,
                self.fixed_stripe_geometry,
                self._render_fixed_stripe(),
                self.fixed_stripe_log2_scale,
            )
            rotating_polar = self._scores(
                image,
                self.rotating_polar_geometry,
                self.rotating_polar_geometry.rendered_prototypes(
                    self.rotating_polar_prototype
                ),
                self.rotating_polar_log2_scale,
            )
        return {
            "cartesian": cartesian,
            "rotating_stripe": rotating_stripe,
            "fixed_stripe": fixed_stripe,
            "rotating_polar": rotating_polar,
        }

    def _aggregate(self, scores, value, family):
        pose_shape = scores.shape[2:-2]
        if self.use_null:
            null = self.null_scores[family][None, :, None, None, None].expand(
                scores.shape[0], -1, 1, scores.shape[-2], scores.shape[-1]
            )
            routing_scores = torch.cat((scores, null), dim=2)
        else:
            routing_scores = scores
        if self.hard_pose:
            winner = routing_scores.argmax(2)
            real = winner < scores.shape[2]
            pose = winner.clamp_max(scores.shape[2] - 1)
            selected_score = scores.gather(2, pose[:, :, None]).squeeze(2)
            base = torch.arange(value.shape[0], device=value.device)[None, :, None, None]
            selected_value = value[base, pose]
            return (
                real[..., None]
                * torch.exp2(selected_score)[..., None]
                * selected_value
            ).sum(1)
        full_probability = (
            routing_scores / self.pose_temperature
        ).softmax(2)
        self._cache_routing_probability(
            family, full_probability, pose_shape
        )
        probability = full_probability[:, :, :scores.shape[2]]
        weight = probability * torch.exp2(scores)
        return torch.einsum("qbphw,bpc->qhwc", weight, value)

    def forward(self, image):
        if self.capture_routing_probabilities:
            self.routing_probability_cache = {}
        scores = self.family_scores(image)
        output = self._aggregate(
            scores["cartesian"], self.cartesian_value, "cartesian"
        )
        output = output + self._aggregate(
            scores["rotating_stripe"], self.rotating_stripe_value,
            "rotating_stripe",
        )
        output = output + self._aggregate(
            scores["fixed_stripe"], self.fixed_stripe_value, "fixed_stripe"
        )
        output = output + self._aggregate(
            scores["rotating_polar"], self.rotating_polar_value,
            "rotating_polar",
        )
        return (output + self.output_bias).permute(0, 3, 1, 2).contiguous()
