from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mixed_geometry_distance_projection import _DiskGeometry, _StripeGeometry
from layers.multiscale_variable_ring_polar_distance_projection import _PolarScaleGeometry
from layers.single_scale_geometric_projection import SingleScaleGeometricProjection


class DualScaleGeometricProjection(SingleScaleGeometricProjection):
    """Four geometric families sharing prototypes across two kernel scales."""

    def __init__(self, *args, kernel_sizes=(24, 12), stride=16, **kwargs):
        kernel_sizes = tuple(int(size) for size in kernel_sizes)
        if len(kernel_sizes) != 2:
            raise ValueError("DualScaleGeometricProjection requires exactly two scales")
        if kernel_sizes[0] != max(kernel_sizes):
            raise ValueError("the first kernel size must be the canonical largest scale")
        super().__init__(
            *args, kernel_size=kernel_sizes[0], stride=stride, **kwargs
        )
        self.kernel_sizes = kernel_sizes
        self.scales = len(kernel_sizes)
        self.active_scale = None

        self.cartesian_geometries = nn.ModuleList(
            _DiskGeometry(size, stride, self.cover_radius_scale)
            for size in kernel_sizes
        )
        self.rotating_stripe_geometries = nn.ModuleList(
            _StripeGeometry(
                size, self.directions, self.stripe_bins,
                self.stripe_longitudinal_bins, stride,
                self.cover_radius_scale,
            )
            for size in kernel_sizes
        )
        self.fixed_stripe_geometries = nn.ModuleList(
            _StripeGeometry(
                size, self.directions, self.stripe_bins,
                self.stripe_longitudinal_bins, stride,
                self.cover_radius_scale,
            )
            for size in kernel_sizes
        )
        self.rotating_polar_geometries = nn.ModuleList(
            _PolarScaleGeometry(
                size, stride, self.radial_bins,
                self.ring_counts, self.ring_offsets, self.directions,
                self.cover_radius_scale,
            )
            for size in kernel_sizes
        )

        def repeated_scale(parameter):
            return nn.Parameter(parameter.detach()[:, None].repeat(1, self.scales))

        self.cartesian_log2_scale = repeated_scale(self.cartesian_log2_scale)
        self.rotating_stripe_log2_scale = repeated_scale(
            self.rotating_stripe_log2_scale
        )
        self.fixed_stripe_log2_scale = repeated_scale(
            self.fixed_stripe_log2_scale
        )
        self.rotating_polar_log2_scale = repeated_scale(
            self.rotating_polar_log2_scale
        )

        def repeated_value(value):
            return nn.Parameter(value.detach()[:, None].repeat(1, self.scales, 1, 1))

        self.cartesian_value = repeated_value(self.cartesian_value)
        self.rotating_stripe_value = repeated_value(self.rotating_stripe_value)
        self.fixed_stripe_value = repeated_value(self.fixed_stripe_value)
        self.rotating_polar_value = repeated_value(self.rotating_polar_value)

    @property
    def pose_value_count(self) -> int:
        counts = self.family_counts
        return self.scales * (
            counts["cartesian"]
            + counts["fixed_stripe"]
            + self.directions
            * (counts["rotating_stripe"] + counts["rotating_polar"])
        )

    def set_active_scale(self, scale=None):
        """Optionally restrict inference to one scale for ablation."""
        if scale is not None and not 0 <= int(scale) < self.scales:
            raise ValueError("active scale is outside the configured range")
        self.active_scale = None if scale is None else int(scale)

    def _render_cartesian_at(self, geometry):
        canonical = self.cartesian_prototype.new_zeros(
            self.cartesian_prototype.shape[0], 3, self.kernel_size**2
        )
        canonical[:, :, self.cartesian_geometry.support_mask] = (
            self.cartesian_prototype
        )
        canonical = canonical.view(
            self.cartesian_prototype.shape[0], 3,
            self.kernel_size, self.kernel_size,
        )
        if geometry.kernel_size != self.kernel_size:
            canonical = F.interpolate(
                canonical,
                size=(geometry.kernel_size, geometry.kernel_size),
                mode="bilinear",
                align_corners=False,
            )
        rendered = canonical.flatten(2)[:, :, geometry.support_mask]
        return rendered[:, None]

    def _render_fixed_stripe_at(self, geometry):
        rendered = geometry.render(self.fixed_stripe_prototype)
        base = torch.arange(rendered.shape[0], device=rendered.device)
        return rendered[base, self.fixed_stripe_direction][:, None]

    def _family_scale_scores(self, image, geometries, render, log2_scale):
        return torch.stack([
            self._scores(
                image, geometry, render(geometry), log2_scale[:, scale]
            )
            for scale, geometry in enumerate(geometries)
        ], dim=2)

    def family_scores(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            cartesian = self._family_scale_scores(
                image,
                self.cartesian_geometries,
                self._render_cartesian_at,
                self.cartesian_log2_scale,
            )
            rotating_stripe = self._family_scale_scores(
                image,
                self.rotating_stripe_geometries,
                lambda geometry: geometry.render(
                    self.rotating_stripe_prototype
                ),
                self.rotating_stripe_log2_scale,
            )
            fixed_stripe = self._family_scale_scores(
                image,
                self.fixed_stripe_geometries,
                self._render_fixed_stripe_at,
                self.fixed_stripe_log2_scale,
            )
            rotating_polar = self._family_scale_scores(
                image,
                self.rotating_polar_geometries,
                lambda geometry: geometry.rendered_prototypes(
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
        if self.active_scale is not None:
            index = self.active_scale
            scores = scores[:, :, index:index + 1]
            value = value[:, index:index + 1]
            pose_shape = scores.shape[2:-2]
        scores = scores.flatten(2, 3)
        value = value.flatten(1, 2)
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
