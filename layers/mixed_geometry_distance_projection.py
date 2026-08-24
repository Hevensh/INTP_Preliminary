from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.mixed_geometry.cover import (
    CoverMaskBank,
    cosine_mask_initializers,
)
from layers.mixed_geometry.geometries import (
    AngularGeometry as _AngularGeometry,
    ColorGeometry as _ColorGeometry,
    DiskGeometry as _DiskGeometry,
    RadialGeometry as _RadialGeometry,
    StripeGeometry as _StripeGeometry,
)
from layers.mixed_geometry.state import PrototypeState
from layers.mixed_geometry.rgb_patch_detrend import RGBPatchDetrender
from layers.mixed_geometry.rgb_patch_standardize import RGBPatchStandardizer
from layers.multiscale_variable_ring_polar_distance_projection import _PolarScaleGeometry
from layers.triton_harmonic_moments import triton_harmonic_moments


class MixedGeometryDistanceProjection(nn.Module):
    """Angular, radial, uniform-color, stripe, and full-polar prototype bank."""

    def __init__(
        self,
        in_channels=3,
        out_channels=192,
        angular_bases=16,
        radial_bases=8,
        color_bases=8,
        stripe_bases=32,
        full_bases=32,
        directions=8,
        angular_directions=None,
        angular_kernel_sizes=None,
        stripe_directions=None,
        full_directions=None,
        kernel_sizes=(24, 20, 16, 12),
        color_kernel_sizes=(24,),
        angular_bins=48,
        radial_bins=12,
        stripe_bins=24,
        stripe_longitudinal_bins=1,
        ring_counts=None,
        stride=16,
        cover_radius_scale=1.0,
        learnable_cover=False,
        cover_mask_eps=1e-4,
        prototype_std=.02,
        value_mode="independent",
        angular_value_frequency=1,
        stripe_value_frequency=2,
        full_value_frequency=1,
        use_triton_harmonic=True,
        rgb_patch_detrend=False,
        rgb_patch_standardize=False,
        detrend_variance_epsilon=1e-4,
        standardize_variance_epsilon=1e-4,
        match_mode="negative_l2",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.out_channels = out_channels
        self.directions = directions
        self.angular_directions = (
            directions if angular_directions is None else int(angular_directions)
        )
        self.stripe_directions = (
            directions if stripe_directions is None else int(stripe_directions)
        )
        self.full_directions = (
            directions if full_directions is None else int(full_directions)
        )
        for name in ("angular_directions", "stripe_directions", "full_directions"):
            value = getattr(self, name)
            if not 1 <= value <= self.directions:
                raise ValueError(f"{name} must be in [1, directions]")
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.scales = len(self.kernel_sizes)
        self.angular_kernel_sizes = tuple(
            int(k) for k in (
                (24,) if angular_kernel_sizes is None else angular_kernel_sizes
            )
        )
        if not self.angular_kernel_sizes:
            raise ValueError("angular_kernel_sizes must contain at least one size")
        self.angular_scales = len(self.angular_kernel_sizes)
        self.color_kernel_sizes = tuple(int(k) for k in color_kernel_sizes)
        if not self.color_kernel_sizes:
            raise ValueError("color_kernel_sizes must contain at least one size")
        self.color_scales = len(self.color_kernel_sizes)
        self.stripe_longitudinal_bins = int(stripe_longitudinal_bins)
        self.cover_radius_scale = float(cover_radius_scale)
        self.learnable_cover = bool(learnable_cover)
        self.cover_mask_eps = float(cover_mask_eps)
        if value_mode not in {
            "independent", "single_harmonic", "scale_harmonic",
            "scale_affine_harmonic", "shared_scale_affine_harmonic",
            "shared_scale_affine_harmonic_stats",
            "rope_shared_scale_affine_harmonic",
            "log_scale_shared_affine_harmonic",
        }:
            raise ValueError(
                "value_mode must be independent, single_harmonic, or "
                "scale_harmonic, scale_affine_harmonic, or "
                "shared_scale_affine_harmonic, or "
                "shared_scale_affine_harmonic_stats, or "
                "rope_shared_scale_affine_harmonic, or "
                "log_scale_shared_affine_harmonic"
            )
        self.value_mode = value_mode
        self.value_frequencies = {
            "angular": int(angular_value_frequency),
            "stripe": int(stripe_value_frequency),
            "full": int(full_value_frequency),
        }
        for family, direction_count in (
            ("angular", self.angular_directions),
            ("stripe", self.stripe_directions),
            ("full", self.full_directions),
        ):
            theta = torch.arange(direction_count) * (2 * math.pi / directions)
            frequency = self.value_frequencies[family]
            self.register_buffer(
                f"{family}_rope_direction_pair",
                torch.stack((
                    torch.cos(frequency * theta),
                    torch.sin(frequency * theta),
                ), -1),
                persistent=False,
            )
        self.use_triton_harmonic = bool(use_triton_harmonic)
        self.rgb_patch_detrend = bool(rgb_patch_detrend)
        self.rgb_patch_standardize = bool(rgb_patch_standardize)
        self.detrend_variance_epsilon = float(detrend_variance_epsilon)
        self.standardize_variance_epsilon = float(
            standardize_variance_epsilon
        )
        if match_mode not in {"negative_l2", "dot_product"}:
            raise ValueError(
                "match_mode must be negative_l2 or dot_product"
            )
        self.match_mode = match_mode
        if self.rgb_patch_detrend and self.rgb_patch_standardize:
            raise ValueError(
                "RGB detrending and mean/std-only standardization are exclusive"
            )
        if (
            self.value_mode == "shared_scale_affine_harmonic_stats"
            and not self.rgb_patch_standardize
        ):
            raise ValueError(
                "shared_scale_affine_harmonic_stats requires "
                "rgb_patch_standardize"
            )
        if any(frequency <= 0 for frequency in self.value_frequencies.values()):
            raise ValueError("value frequencies must be positive")
        if not 0 < self.cover_mask_eps < .5:
            raise ValueError("cover_mask_eps must be in (0, .5)")
        self.pose_temperature = 1.0
        self.hard_pose = False
        self.pose_topk = None
        self.straight_through_topk = False
        self.tail_keep = None
        self.use_null = False
        self.compression_topk = None
        self.last_tail_mass = None
        self.last_null_mass = None
        self.family_counts = {
            "angular": angular_bases,
            "radial": radial_bases,
            "color": color_bases,
            "stripe": stripe_bases,
            "full": full_bases,
        }
        if ring_counts is None:
            ring_counts = [4 * (r + 1) for r in range(radial_bins)]
        counts = torch.tensor(ring_counts, dtype=torch.long)
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.register_buffer("ring_counts", counts)
        self.register_buffer("ring_offsets", offsets)
        # Per-base pose-axis degradation.  The prototype remains intact; a
        # tied base uses its direction-mean score and direction-mean V.
        self.register_buffer(
            "angular_direction_tied", torch.zeros(angular_bases, dtype=torch.bool)
        )
        self.register_buffer(
            "stripe_direction_tied", torch.zeros(stripe_bases, dtype=torch.bool)
        )
        self.register_buffer(
            "full_direction_tied", torch.zeros(full_bases, dtype=torch.bool)
        )
        for family, number in (
            ("angular", angular_bases),
            ("stripe", stripe_bases),
            ("full", full_bases),
        ):
            family_directions = (
                self.angular_directions if family == "angular"
                else self.stripe_directions if family == "stripe"
                else self.full_directions
            )
            self.register_buffer(
                f"{family}_direction_fixed",
                torch.full((number,), -1, dtype=torch.long),
            )
            self.register_buffer(
                f"{family}_direction_subset",
                torch.ones(number, family_directions, dtype=torch.bool),
            )
        self.register_buffer(
            "stripe_scale_direction_subset",
            torch.ones(
                stripe_bases, self.scales, self.stripe_directions,
                dtype=torch.bool,
            ),
        )
        # Each Stripe base owns a fixed canonical orientation.  Its direction
        # slots are relative offsets around this angle, rather than one global
        # angle grid shared by every base.
        self.register_buffer(
            "stripe_angle_offset", torch.zeros(stripe_bases, dtype=torch.float32)
        )
        self.register_buffer(
            "angular_scale_direction_subset",
            torch.ones(
                angular_bases, self.angular_scales, self.angular_directions,
                dtype=torch.bool,
            ),
        )
        self.register_buffer(
            "full_scale_direction_subset",
            torch.ones(
                full_bases, self.scales, self.full_directions,
                dtype=torch.bool,
            ),
        )
        for family, number in (
            ("radial", radial_bases),
            ("stripe", stripe_bases),
            ("full", full_bases),
        ):
            self.register_buffer(
                f"{family}_scale_tied", torch.zeros(number, dtype=torch.bool)
            )
            self.register_buffer(
                f"{family}_scale_fixed",
                torch.full((number,), -1, dtype=torch.long),
            )

        self.angular_prototype = nn.Parameter(torch.randn(angular_bases, self.in_channels, angular_bins) * prototype_std)
        self.radial_prototype = nn.Parameter(torch.randn(radial_bases, self.in_channels, radial_bins) * prototype_std)
        self.color_prototype = nn.Parameter(torch.randn(color_bases, self.in_channels) * prototype_std)
        stripe_shape = (
            (stripe_bases, self.in_channels, stripe_bins)
            if self.stripe_longitudinal_bins == 1
            else (stripe_bases, self.in_channels, stripe_bins, self.stripe_longitudinal_bins)
        )
        self.stripe_prototype = nn.Parameter(
            torch.randn(*stripe_shape) * prototype_std
        )
        self.full_prototype = nn.Parameter(torch.randn(full_bases, self.in_channels, int(offsets[-1])) * prototype_std)

        self.angular_log2_scale = nn.Parameter(
            torch.zeros(angular_bases)
            if self.angular_scales == 1
            else torch.zeros(angular_bases, self.angular_scales)
        )
        self.radial_log2_scale = nn.Parameter(torch.zeros(radial_bases, self.scales))
        self.color_log2_scale = nn.Parameter(
            torch.zeros(color_bases)
            if self.color_scales == 1
            else torch.zeros(color_bases, self.color_scales)
        )
        self.stripe_log2_scale = nn.Parameter(torch.zeros(stripe_bases, self.scales))
        self.full_log2_scale = nn.Parameter(torch.zeros(full_bases, self.scales))

        if self.value_mode in {
            "single_harmonic", "scale_harmonic", "scale_affine_harmonic",
            "shared_scale_affine_harmonic",
            "shared_scale_affine_harmonic_stats",
            "rope_shared_scale_affine_harmonic",
            "log_scale_shared_affine_harmonic",
        }:
            # Directional families store one cosine/sine pair globally or per
            # scale. Non-directional families follow the same scale-sharing
            # choice. All pose mixing happens before the 192D multiplication.
            angular_pairs = 1 if self.value_mode == "single_harmonic" else self.angular_scales
            scale_pairs = 1 if self.value_mode == "single_harmonic" else self.scales
            color_pairs = 1 if self.value_mode == "single_harmonic" else self.color_scales
            radial_pairs = scale_pairs
            if self.value_mode == "log_scale_shared_affine_harmonic":
                # Four learned vectors per directional prototype regardless
                # of scale count: a log-scale cosine/sine pair and a direction
                # cosine/sine pair. Non-directional families only need the
                # two scale vectors.
                angular_width = 4
                scale_width = 4
                radial_pairs = 2
                color_pairs = 2
            elif self.value_mode == "rope_shared_scale_affine_harmonic":
                if out_channels % 2:
                    raise ValueError(
                        "rope_shared_scale_affine_harmonic requires an even "
                        "out_channels"
                    )
                # Directional families learn one scale vector and one
                # directional vector. Their quarter-turn partners are derived
                # exactly as in RoPE, so they are never stored as parameters.
                angular_width = 2
                scale_width = 2
                radial_pairs = 1
                color_pairs = 1
                self.scale_value_mix = nn.ParameterDict()
                for family, number, scale_count in (
                    ("angular", angular_bases, self.angular_scales),
                    ("radial", radial_bases, self.scales),
                    ("color", color_bases, self.color_scales),
                    ("stripe", stripe_bases, self.scales),
                    ("full", full_bases, self.scales),
                ):
                    phase = torch.linspace(0, math.pi / 2, scale_count)
                    initial = torch.stack((phase.cos(), phase.sin()), -1)
                    self.scale_value_mix[family] = nn.Parameter(
                        initial[None].expand(number, -1, -1).clone()
                    )
            elif self.value_mode in {
                "shared_scale_affine_harmonic",
                "shared_scale_affine_harmonic_stats",
            }:
                stats_width = (
                    2
                    if self.value_mode == "shared_scale_affine_harmonic_stats"
                    else 0
                )
                angular_width = self.angular_scales + 2 + stats_width
                scale_width = self.scales + 2 + stats_width
                radial_pairs = self.scales + stats_width
                color_pairs = self.color_scales + stats_width
            else:
                directional_width = (
                    3 if self.value_mode == "scale_affine_harmonic" else 2
                )
                angular_width = directional_width * angular_pairs
                scale_width = directional_width * scale_pairs
            self.angular_value = nn.Parameter(
                torch.empty(
                    angular_bases, angular_width,
                    out_channels,
                )
            )
            self.radial_value = nn.Parameter(
                torch.empty(radial_bases, radial_pairs, out_channels)
            )
            self.color_value = nn.Parameter(
                torch.empty(color_bases, color_pairs, out_channels)
            )
            self.stripe_value = nn.Parameter(
                torch.empty(
                    stripe_bases, scale_width,
                    out_channels,
                )
            )
            self.full_value = nn.Parameter(
                torch.empty(
                    full_bases, scale_width,
                    out_channels,
                )
            )
        else:
            self.angular_value = nn.Parameter(
                torch.empty(angular_bases, self.angular_directions, out_channels)
                if self.angular_scales == 1
                else torch.empty(
                    angular_bases, self.angular_scales,
                    self.angular_directions, out_channels,
                )
            )
            self.radial_value = nn.Parameter(torch.empty(radial_bases, self.scales, out_channels))
            self.color_value = nn.Parameter(
                torch.empty(color_bases, 1, out_channels)
                if self.color_scales == 1
                else torch.empty(color_bases, self.color_scales, out_channels)
            )
            self.stripe_value = nn.Parameter(
                torch.empty(
                    stripe_bases, self.scales, self.stripe_directions, out_channels
                )
            )
            self.full_value = nn.Parameter(
                torch.empty(
                    full_bases, self.scales, self.full_directions, out_channels
                )
            )
        for value in (self.angular_value, self.radial_value, self.color_value, self.stripe_value, self.full_value):
            nn.init.trunc_normal_(value, std=.02)
        if self.value_mode == "shared_scale_affine_harmonic_stats":
            self.orthogonalize_value_banks_()
        # In the affine harmonic mode every prototype additionally owns one
        # value used only by the null route.  It is deliberately separate from
        # the per-scale zero-frequency values embedded in ``*_value``.
        if self.value_mode == "scale_affine_harmonic":
            self.null_values = nn.ParameterDict({
                family: nn.Parameter(torch.zeros(number, out_channels))
                for family, number in self.family_counts.items()
            })
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.rgb_detrender = (
            RGBPatchDetrender(
                out_channels=out_channels,
                stride=stride,
                variance_epsilon=self.detrend_variance_epsilon,
            )
            if self.rgb_patch_detrend else None
        )
        self.rgb_standardizer = (
            RGBPatchStandardizer(
                stride=stride,
                variance_epsilon=self.standardize_variance_epsilon,
            )
            if self.rgb_patch_standardize else None
        )
        self._input_stats = None
        self.null_scores = nn.ParameterDict({
            family: nn.Parameter(torch.full((number,), -20.0))
            for family, number in self.family_counts.items()
        })

        self.angular_geometries = nn.ModuleList(
            _AngularGeometry(
                k, self.angular_directions, angular_bins, stride,
                cover_radius_scale,
                direction_step=2 * math.pi / self.directions,
                in_channels=self.in_channels,
            )
            for k in self.angular_kernel_sizes
        )
        self.angular_geometry = self.angular_geometries[0]
        self.radial_geometries = nn.ModuleList(
            _RadialGeometry(
                k, radial_bins, stride, cover_radius_scale,
                in_channels=self.in_channels,
            )
            for k in self.kernel_sizes
        )
        self.color_geometries = nn.ModuleList(
            _ColorGeometry(
                k, stride, cover_radius_scale,
                in_channels=self.in_channels,
            )
            for k in self.color_kernel_sizes
        )
        # Preserve the old public attribute and checkpoint behavior when color
        # has a single pose.
        self.color_geometry = self.color_geometries[0]
        self.stripe_geometries = nn.ModuleList(
            _StripeGeometry(
                k, self.stripe_directions, stripe_bins,
                self.stripe_longitudinal_bins, stride, cover_radius_scale,
                # A reduced Stripe bank keeps the original angular spacing and
                # drops the second half-circle.  For directions=8 and
                # stripe_directions=4 this is 0, 45, 90, and 135 degrees.
                direction_step=2 * math.pi / self.directions,
                in_channels=self.in_channels,
            )
            for k in self.kernel_sizes
        )
        self.full_geometries = nn.ModuleList(
            _PolarScaleGeometry(
                k, stride, radial_bins, counts, offsets, self.full_directions,
                cover_radius_scale,
                direction_step=2 * math.pi / self.directions,
                in_channels=self.in_channels,
            )
            for k in self.kernel_sizes
        )

        mask_initializers = cosine_mask_initializers(
            family_counts=self.family_counts,
            angular_bins=angular_bins,
            radial_bins=radial_bins,
            stripe_bins=stripe_bins,
            stripe_longitudinal_bins=self.stripe_longitudinal_bins,
            ring_counts=self.ring_counts,
            radius_scale=self.cover_radius_scale,
            eps=self.cover_mask_eps,
        )
        self.cover_mask_logits = CoverMaskBank(
            mask_initializers,
            self.cover_mask_eps,
            enabled=self.learnable_cover,
        )
        self.cover_mask_shapes = {
            family: tuple(initializer.shape[1:])
            for family, initializer in mask_initializers.items()
        }

    def _cover_weight(self, family, geometry, renderer):
        """Render native mask logits through the prototype's own sampler.

        Sigmoid is evaluated after interpolation. The resulting mask is then
        mass-normalized to the original cosine cover, so learning can only
        redistribute spatial importance and cannot reduce a distance by
        shrinking the whole mask.
        """
        return self.cover_mask_logits.render(
            family, geometry, renderer
        )

    @torch.no_grad()
    def cover_mask_summary(self):
        return self.cover_mask_logits.summary()

    @torch.no_grad()
    def orthogonalize_value_banks_(self):
        """Retract each prototype's compact V bank to an orthogonal frame.

        Row norms are preserved, so this changes directions without silently
        changing the initial output scale.  This is intentionally an
        initialization/reinitialization operation, not a per-step constraint.
        """
        for family in self.family_counts:
            value = getattr(self, f"{family}_value")
            if value.shape[0] == 0:
                continue
            for base in range(value.shape[0]):
                rows = value[base]
                norms = rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
                frame, _ = torch.linalg.qr(rows.T, mode="reduced")
                value[base].copy_(frame.T * norms)

    def _scores(
        self, image, prototype, geometry, rendered, log2_scale,
        learned_weight=None, canonicalize=True,
    ):
        patch = geometry.indexed_patches(image)
        canonicalizer = self.rgb_detrender or self.rgb_standardizer
        if canonicalizer is not None and canonicalize:
            patch = canonicalizer.canonicalize(patch, geometry)
            rendered = canonicalizer.canonicalize(rendered, geometry)
        if learned_weight is None:
            weight = geometry.support_cover[None, None, None, :]
            cross = torch.einsum("qtcm,bpcm->qtbp", patch * weight, rendered)
            if self.match_mode == "dot_product":
                response = cross
            else:
                patch_energy = (patch.square() * weight).sum((2, 3))
                proto_energy = (rendered.square() * weight).sum((2, 3))
                response = -(
                    patch_energy[:, :, None, None]
                    + proto_energy[None, None] - 2 * cross
                ).clamp_min(0) * geometry.distance_multiplier
        else:
            cross = torch.einsum(
                "qtcm,bpcm,bpcm->qtbp", patch, rendered, learned_weight
            )
            if self.match_mode == "dot_product":
                response = cross
            else:
                patch_energy = torch.einsum(
                    "qtcm,bpcm->qtbp", patch.square(), learned_weight
                )
                proto_energy = (
                    rendered.square() * learned_weight
                ).sum((2, 3))
                response = -(
                    patch_energy + proto_energy[None, None] - 2 * cross
                ).clamp_min(0) * geometry.distance_multiplier
        side = int(patch.shape[1] ** .5)
        response = response.permute(0, 2, 3, 1).reshape(
            image.shape[0], prototype.shape[0], rendered.shape[1], side, side
        )
        return response * torch.exp2(log2_scale)[None, :, :, None, None]

    def _angular_scores(self, image):
        if self.angular_scales == 1:
            rendered = self.angular_geometry.render(self.angular_prototype)
            scale = self.angular_log2_scale[:, None].expand(
                -1, self.angular_directions
            )
            weight = self._cover_weight(
                "angular", self.angular_geometry, self.angular_geometry.render
            )
            return self._scores(
                image, self.angular_prototype, self.angular_geometry,
                rendered, scale, weight,
            )
        return self._multiscale_scores(
            image, self.angular_prototype, self.angular_geometries,
            self.angular_log2_scale,
            lambda geometry, prototype: geometry.render(prototype),
            "angular",
        )

    def _color_scores(self, image):
        if self.color_scales == 1:
            rendered = self.color_geometry.render(self.color_prototype)
            scale = self.color_log2_scale[:, None]
            return self._scores(
                image, self.color_prototype, self.color_geometry,
                rendered, scale,
                self._cover_weight(
                    "color", self.color_geometry, self.color_geometry.render
                ),
            )
        return self._multiscale_scores(
            image, self.color_prototype, self.color_geometries,
            self.color_log2_scale, lambda geometry, prototype: geometry.render(prototype),
            "color",
        ).squeeze(3)

    def _multiscale_scores(
        self, image, prototype, geometries, log2_scale, render, family,
    ):
        scores = []
        for si, geometry in enumerate(geometries):
            rendered = render(geometry, prototype)
            scale = log2_scale[:, si, None].expand(-1, rendered.shape[1])
            if family == "full":
                renderer = geometry.rendered_prototypes
            else:
                renderer = geometry.render
            weight = self._cover_weight(family, geometry, renderer)
            scores.append(self._scores(
                image, prototype, geometry, rendered, scale, weight,
            ))
        return torch.stack(scores, dim=2)

    def family_scores(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            if self.rgb_detrender is not None:
                image = self.rgb_detrender.unit_rgb(image)
            elif self.rgb_standardizer is not None:
                image = self.rgb_standardizer.unit_rgb(image)
            angular = self._angular_scores(image)
            radial = self._multiscale_scores(image, self.radial_prototype, self.radial_geometries, self.radial_log2_scale, lambda g, p: g.render(p), "radial").squeeze(3)
            color = self._color_scores(image)
            stripe = self._multiscale_scores(
                image, self.stripe_prototype, self.stripe_geometries,
                self.stripe_log2_scale,
                lambda g, p: g.render(p, self.stripe_angle_offset), "stripe",
            )
            full = self._multiscale_scores(image, self.full_prototype, self.full_geometries, self.full_log2_scale, lambda g, p: g.rendered_prototypes(p), "full")
            return {"angular": angular, "radial": radial, "color": color, "stripe": stripe, "full": full}

    def set_pose_selection(
        self, temperature=1.0, hard=False, topk=None, straight_through=False,
        use_null=False, tail_keep=None,
    ):
        if temperature <= 0:
            raise ValueError("pose temperature must be positive")
        self.pose_temperature = float(temperature)
        self.hard_pose = bool(hard)
        self.pose_topk = dict(topk) if topk is not None else None
        self.straight_through_topk = bool(straight_through)
        self.use_null = bool(use_null)
        if tail_keep is not None and not 0.0 <= tail_keep <= 1.0:
            raise ValueError("tail_keep must be in [0, 1]")
        self.tail_keep = None if tail_keep is None else float(tail_keep)

    def set_compression_targets(self, topk=None):
        self.compression_topk = dict(topk) if topk is not None else None

    def set_direction_tying(self, tied_by_family=None):
        """Tie redundant rotation poses while retaining the learned kernel.

        Values can be boolean masks, integer index tensors, or index lists.
        Directional V is synchronized non-destructively by averaging in forward.
        """
        tied_by_family = tied_by_family or {}
        for family in ("angular", "stripe", "full"):
            target = getattr(self, f"{family}_direction_tied")
            specification = tied_by_family.get(family, [])
            if torch.is_tensor(specification) and specification.dtype == torch.bool:
                if specification.numel() != target.numel():
                    raise ValueError(f"{family} direction mask has the wrong size")
                mask = specification.to(device=target.device)
            else:
                mask = torch.zeros_like(target)
                if torch.is_tensor(specification):
                    specification = specification.flatten().tolist()
                if specification:
                    mask[torch.as_tensor(specification, device=target.device)] = True
            target.copy_(mask)
            getattr(self, f"{family}_direction_fixed").fill_(-1)
            getattr(self, f"{family}_direction_subset").fill_(True)
            getattr(self, f"{family}_scale_direction_subset").fill_(True)

    def set_pose_degradation(
        self, *, direction_mean=None, direction_fixed=None,
        scale_mean=None, scale_fixed=None, scale_direction_fixed=None,
    ):
        """Configure free, mean-tied, or fixed pose axes per base."""
        self.set_direction_tying(direction_mean)
        direction_fixed = direction_fixed or {}
        for family in ("angular", "stripe", "full"):
            fixed = getattr(self, f"{family}_direction_fixed")
            subset = getattr(self, f"{family}_direction_subset")
            family_directions = (
                self.angular_directions if family == "angular"
                else self.stripe_directions if family == "stripe"
                else self.full_directions
            )
            for base, poses in direction_fixed.get(family, {}).items():
                base = int(base)
                if torch.is_tensor(poses):
                    poses = poses.flatten().tolist()
                if isinstance(poses, (list, tuple, set)):
                    poses = [int(pose) for pose in poses]
                    if not poses or len(set(poses)) != len(poses):
                        raise ValueError(
                            f"{family} direction subset must be non-empty and unique"
                        )
                    if any(not 0 <= pose < family_directions for pose in poses):
                        raise ValueError(f"invalid {family} direction subset {poses}")
                    subset[base].zero_()
                    subset[base, poses] = True
                else:
                    pose = int(poses)
                    if not 0 <= pose < family_directions:
                        raise ValueError(f"invalid {family} direction {pose}")
                    fixed[base] = pose
                getattr(self, f"{family}_direction_tied")[base] = False

        self.set_scale_direction_subsets(
            scale_direction_fixed or {}, reset=False
        )

        scale_mean = scale_mean or {}
        scale_fixed = scale_fixed or {}
        for family in ("radial", "stripe", "full"):
            tied = getattr(self, f"{family}_scale_tied")
            tied.zero_()
            specification = scale_mean.get(family, [])
            if torch.is_tensor(specification) and specification.dtype == torch.bool:
                if specification.numel() != tied.numel():
                    raise ValueError(f"{family} scale mask has the wrong size")
                tied.copy_(specification.to(tied.device))
            elif len(specification):
                tied[torch.as_tensor(specification, device=tied.device)] = True
            fixed = getattr(self, f"{family}_scale_fixed")
            fixed.fill_(-1)
            for base, pose in scale_fixed.get(family, {}).items():
                base, pose = int(base), int(pose)
                if not 0 <= pose < self.scales:
                    raise ValueError(f"invalid {family} scale {pose}")
                fixed[base] = pose
                tied[base] = False

    def set_scale_direction_subsets(self, specification=None, *, reset=True):
        """Restrict directions independently for every base and scale."""
        specification = specification or {}
        if reset:
            for family in ("angular", "stripe", "full"):
                getattr(self, f"{family}_scale_direction_subset").fill_(True)
        for family in ("angular", "stripe", "full"):
            scale_count = (
                self.angular_scales if family == "angular" else self.scales
            )
            direction_count = getattr(self, f"{family}_directions")
            subset = getattr(self, f"{family}_scale_direction_subset")
            for base, by_scale in specification.get(family, {}).items():
                base = int(base)
                for scale, poses in by_scale.items():
                    scale = int(scale)
                    if not 0 <= scale < scale_count:
                        raise ValueError(f"invalid {family} scale {scale}")
                    if torch.is_tensor(poses):
                        poses = poses.flatten().tolist()
                    poses = [int(pose) for pose in poses]
                    if not poses or len(set(poses)) != len(poses):
                        raise ValueError(
                            f"{family} per-scale directions must be non-empty and unique"
                        )
                    if any(not 0 <= pose < direction_count for pose in poses):
                        raise ValueError(
                            f"invalid {family} direction subset {poses}"
                        )
                    subset[base, scale].zero_()
                    subset[base, scale, poses] = True

    @torch.no_grad()
    def synchronize_tied_direction_values(self):
        """Make every tied directional V an explicit copy of its mean."""
        for family in ("angular", "stripe", "full"):
            tied = getattr(self, f"{family}_direction_tied")
            if not tied.any():
                continue
            value = getattr(self, f"{family}_value")
            direction_dim = value.ndim - 2
            selected = value[tied]
            mean = selected.mean(direction_dim, keepdim=True)
            value[tied] = mean.expand_as(selected)

    @torch.no_grad()
    def synchronize_tied_pose_values(self):
        self.synchronize_tied_direction_values()
        for family in ("radial", "stripe", "full"):
            tied = getattr(self, f"{family}_scale_tied")
            if not tied.any():
                continue
            value = getattr(self, f"{family}_value")
            selected = value[tied]
            mean = selected.mean(1, keepdim=True)
            value[tied] = mean.expand_as(selected)

    def direction_pose_inventory(self):
        def axis_counts(family, axis, full_count):
            tied = getattr(self, f"{family}_{axis}_tied")
            fixed = getattr(self, f"{family}_{axis}_fixed") >= 0
            if axis == "direction":
                available = getattr(
                    self, f"{family}_direction_subset"
                ).sum(1)
            else:
                available = torch.full_like(tied, full_count, dtype=torch.long)
            return torch.where(tied | fixed, 1, available)

        angular_direction = axis_counts(
            "angular", "direction", self.angular_directions
        )
        radial_scale = axis_counts("radial", "scale", self.scales)
        stripe_direction = axis_counts(
            "stripe", "direction", self.stripe_directions
        )
        stripe_scale = axis_counts("stripe", "scale", self.scales)
        full_direction = axis_counts("full", "direction", self.full_directions)
        full_scale = axis_counts("full", "scale", self.scales)
        inventory = {
            "angular": int(
                self.angular_scale_direction_subset.sum()
                if (~self.angular_scale_direction_subset).any()
                else (angular_direction * self.angular_scales).sum()
            ),
            "radial": int(radial_scale.sum()),
            "color": self.family_counts["color"] * self.color_scales,
            "stripe": int(
                self.stripe_scale_direction_subset.sum()
                if (~self.stripe_scale_direction_subset).any()
                else (stripe_direction * stripe_scale).sum()
            ),
            "full": int(
                self.full_scale_direction_subset.sum()
                if (~self.full_scale_direction_subset).any()
                else (full_direction * full_scale).sum()
            ),
        }
        inventory["total"] = sum(inventory.values())
        return inventory

    def value_vector_inventory(self):
        """Count learned high-dimensional V vectors, separately from poses."""
        inventory = {
            family: int(getattr(self, f"{family}_value").numel() // self.out_channels)
            for family in self.family_counts
        }
        if hasattr(self, "null_values"):
            for family in self.family_counts:
                inventory[family] += int(self.null_values[family].shape[0])
        inventory["total"] = sum(inventory.values())
        return inventory

    def _scale_factors(self, family, *, device, dtype):
        sizes = (
            self.angular_kernel_sizes if family == "angular"
            else self.color_kernel_sizes if family == "color"
            else self.kernel_sizes
        )
        largest = float(sizes[0])
        return torch.tensor(
            [size / largest for size in sizes], device=device, dtype=dtype
        )

    def _log_scale_pair(self, family, count, *, device, dtype):
        """Smooth endpoint-preserving sin/cos coordinates in log scale."""
        sizes = (
            self.angular_kernel_sizes if family == "angular"
            else self.color_kernel_sizes if family == "color"
            else self.kernel_sizes
        )[:count]
        if len(sizes) <= 1:
            return torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        log_sizes = torch.tensor(sizes, device=device, dtype=dtype).log()
        denominator = (log_sizes[0] - log_sizes[-1]).clamp_min(
            torch.finfo(dtype).eps
        )
        position = (log_sizes[0] - log_sizes) / denominator
        phase = position * (math.pi / 2)
        return torch.stack((phase.cos(), phase.sin()), -1)

    def _harmonic_pose_coefficients(self, family, pose_shape, *, device, dtype):
        scale = self._scale_factors(family, device=device, dtype=dtype)
        if family not in self.value_frequencies:
            if self.value_mode == "log_scale_shared_affine_harmonic":
                scale_count = len(pose_shape) and pose_shape[0] or 1
                return self._log_scale_pair(
                    family, scale_count, device=device, dtype=dtype
                )
            if self.value_mode in {
                "scale_harmonic", "scale_affine_harmonic",
                "shared_scale_affine_harmonic",
                "shared_scale_affine_harmonic_stats",
            }:
                return torch.eye(
                    len(pose_shape) and pose_shape[0] or 1,
                    device=device, dtype=dtype,
                )
            return scale.reshape(-1, 1)
        if len(pose_shape) == 1:
            scale = scale[:1]
            direction_count = pose_shape[0]
        else:
            direction_count = pose_shape[1]
            scale = scale[:pose_shape[0]]
        theta = (
            torch.arange(direction_count, device=device, dtype=dtype)
            * (2 * math.pi / self.directions)
        )
        frequency = self.value_frequencies[family]
        pair = torch.stack(
            (torch.cos(frequency * theta), torch.sin(frequency * theta)), 1
        )
        if self.value_mode == "log_scale_shared_affine_harmonic":
            scale_count = 1 if len(pose_shape) == 1 else pose_shape[0]
            scale_pair = self._log_scale_pair(
                family, scale_count, device=device, dtype=dtype
            )
            coefficients = torch.empty(
                scale_count, direction_count, 4,
                device=device, dtype=dtype,
            )
            coefficients[..., :2] = scale_pair[:, None]
            coefficients[..., 2:] = pair[None]
            return coefficients.reshape(-1, 4)
        if self.value_mode in {
            "shared_scale_affine_harmonic",
            "shared_scale_affine_harmonic_stats",
        }:
            scale_count = 1 if len(pose_shape) == 1 else pose_shape[0]
            coefficients = torch.zeros(
                scale_count, direction_count, scale_count + 2,
                device=device, dtype=dtype,
            )
            for scale_index in range(scale_count):
                coefficients[scale_index, :, scale_index] = 1
                coefficients[scale_index, :, -2:] = pair
            return coefficients.reshape(-1, scale_count + 2)
        if self.value_mode in {"scale_harmonic", "scale_affine_harmonic"}:
            scale_count = 1 if len(pose_shape) == 1 else pose_shape[0]
            width = 3 if self.value_mode == "scale_affine_harmonic" else 2
            coefficients = torch.zeros(
                scale_count, direction_count, width * scale_count,
                device=device, dtype=dtype,
            )
            for scale_index in range(scale_count):
                start = width * scale_index
                if width == 3:
                    coefficients[scale_index, :, start] = 1
                    start += 1
                coefficients[scale_index, :, start:start + 2] = pair
            return coefficients.reshape(-1, width * scale_count)
        return (scale[:, None, None] * pair[None]).reshape(-1, 2)

    @staticmethod
    def _rope_quarter_turn(value):
        """Apply the fixed +90 degree RoPE operator to feature pairs."""
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _rope_value_components(self, family, value):
        """Expand stored vectors into derived scale/direction RoPE pairs."""
        scale_value = value[:, 0]
        components = [
            scale_value,
            self._rope_quarter_turn(scale_value),
        ]
        if family in self.value_frequencies:
            direction_value = value[:, 1]
            components.extend((
                direction_value,
                self._rope_quarter_turn(direction_value),
            ))
        return torch.stack(components, 1)

    def _rope_pose_coefficients(
        self, family, pose_shape, *, device, dtype, base_indices=None,
    ):
        """Coefficients for scale mixing plus the shared direction phase."""
        if family in self.value_frequencies:
            scale_count = 1 if len(pose_shape) == 1 else pose_shape[0]
            direction_count = (
                pose_shape[0] if len(pose_shape) == 1 else pose_shape[1]
            )
        else:
            # Color/radial have no direction axis; their single pose axis is
            # scale, unlike single-scale Angular whose axis is direction.
            scale_count = pose_shape[0] if pose_shape else 1
            direction_count = 1
        mix = self.scale_value_mix[family]
        if base_indices is not None:
            mix = mix[base_indices]
        mix = mix[:, :scale_count].to(
            device=device, dtype=dtype
        )
        scale_coefficients = mix[:, :, None, :].expand(
            -1, -1, direction_count, -1
        )
        if family not in self.value_frequencies:
            return scale_coefficients.reshape(
                mix.shape[0], scale_count * direction_count, 2
            )
        theta = (
            torch.arange(direction_count, device=device, dtype=dtype)
            * (2 * math.pi / self.directions)
        )
        frequency = self.value_frequencies[family]
        direction = torch.stack(
            (torch.cos(frequency * theta), torch.sin(frequency * theta)), -1
        )[None, None].expand(mix.shape[0], scale_count, -1, -1)
        return torch.cat((scale_coefficients, direction), -1).reshape(
            mix.shape[0], scale_count * direction_count, 4
        )

    def expand_rope_pose_values(self, family, value, pose_shape):
        """Materialize pose values for conversion, audits, and tests only."""
        coefficients = self._rope_pose_coefficients(
            family, pose_shape, device=value.device, dtype=value.dtype
        )
        components = self._rope_value_components(family, value)
        return torch.einsum("bpk,bkc->bpc", coefficients, components)

    def project_pose_values_to_rope(self, family, source_value):
        """Least-squares projection of dense pose values into the tied bank.

        The fixed quarter turn is multiplication by ``i`` after pairing the
        first and second feature halves.  The constrained real problem is
        therefore a tiny complex least-squares problem with one scale column
        and, for directional families, one direction column.
        """
        if source_value.shape[-1] != self.out_channels:
            raise ValueError("source pose values have the wrong channel width")
        pose_shape = source_value.shape[1:-1]
        coefficients = self._rope_pose_coefficients(
            family, pose_shape, device=source_value.device,
            dtype=source_value.dtype,
        )
        target = source_value.flatten(1, source_value.ndim - 2)
        target_real, target_imag = target.chunk(2, dim=-1)
        target_complex = torch.complex(target_real, target_imag)
        columns = [torch.complex(coefficients[..., 0], coefficients[..., 1])]
        if family in self.value_frequencies:
            columns.append(
                torch.complex(coefficients[..., 2], coefficients[..., 3])
            )
        design = torch.stack(columns, -1)
        solution = torch.linalg.pinv(design) @ target_complex
        return torch.cat((solution.real, solution.imag), dim=-1).to(
            dtype=source_value.dtype
        )

    def _rope_moments(self, weight, family, pose_shape, base_indices=None):
        """Factor scale and direction reductions before the 192D GEMM."""
        queries, bases, _, height, width = weight.shape
        mix = self.scale_value_mix[family]
        if base_indices is not None:
            mix = mix[base_indices]
        mix = mix.to(device=weight.device, dtype=weight.dtype)
        if family not in self.value_frequencies:
            scale_count = pose_shape[0] if pose_shape else 1
            scale_mass = weight.reshape(
                queries, bases, scale_count, height, width
            )
            return torch.einsum(
                "qbshw,bsk->qbkhw", scale_mass, mix[:, :scale_count]
            )

        scale_count = 1 if len(pose_shape) == 1 else pose_shape[0]
        direction_count = (
            pose_shape[0] if len(pose_shape) == 1 else pose_shape[1]
        )
        pose_weight = weight.reshape(
            queries, bases, scale_count, direction_count, height, width
        )
        scale_mass = pose_weight.sum(3)
        scale_moments = torch.einsum(
            "qbshw,bsk->qbkhw", scale_mass, mix[:, :scale_count]
        )
        # The direction table is common to every scale/base/patch. Summing
        # scales first avoids repeating the same cosine/sine multiplications.
        direction_mass = pose_weight.sum(2)
        pair = getattr(self, f"{family}_rope_direction_pair")
        if pair.shape[0] != direction_count:
            theta = (
                torch.arange(direction_count, device=weight.device,
                             dtype=weight.dtype)
                * (2 * math.pi / self.directions)
            )
            frequency = self.value_frequencies[family]
            pair = torch.stack((
                torch.cos(frequency * theta),
                torch.sin(frequency * theta),
            ), -1)
        else:
            pair = pair.to(device=weight.device, dtype=weight.dtype)
        direction_moments = torch.einsum(
            "qbdhw,dk->qbkhw", direction_mass, pair
        )
        return torch.cat((scale_moments, direction_moments), dim=2)

    def _add_null_value(self, output, gate, family, base_indices=None):
        """Add the value owned exclusively by the null softmax branch."""
        if gate is None or not hasattr(self, "null_values"):
            return output
        value = self.null_values[family]
        if base_indices is not None:
            value = value[base_indices]
        null_probability = 1 - gate
        return output + torch.einsum(
            "qbhw,bc->qhwc", null_probability, value
        )

    def _weighted_value_sum(
        self, weight, value, family, pose_shape, *,
        null_probability=None, base_indices=None,
    ):
        if self.value_mode == "independent":
            value_flat = value.flatten(1, value.ndim - 2)
            return torch.einsum("qbphw,bpc->qhwc", weight, value_flat)
        if self.value_mode == "rope_shared_scale_affine_harmonic":
            moments = self._rope_moments(
                weight, family, pose_shape, base_indices
            )
            value_for_gemm = self._rope_value_components(family, value)
            queries, bases, harmonics, height, width = moments.shape
            output = moments.permute(0, 3, 4, 1, 2).reshape(
                queries * height * width, bases * harmonics
            ) @ value_for_gemm.reshape(bases * harmonics, self.out_channels)
            return output.reshape(queries, height, width, self.out_channels)
        coefficients = self._harmonic_pose_coefficients(
            family, pose_shape, device=weight.device, dtype=weight.dtype
        )
        if coefficients.shape[0] != weight.shape[2]:
            raise RuntimeError(
                f"{family} pose coefficients have {coefficients.shape[0]} "
                f"entries for {weight.shape[2]} route weights"
            )
        if self.use_triton_harmonic:
            moments = triton_harmonic_moments(weight, coefficients).permute(
                0, 1, 4, 2, 3
            )
        else:
            moments = torch.einsum("qbphw,pk->qbkhw", weight, coefficients)
        if self.value_mode == "shared_scale_affine_harmonic_stats":
            if self._input_stats is None:
                raise RuntimeError("mean/std input statistics are unavailable")
            route_mass = weight.sum(dim=2, keepdim=True)
            stats = self._input_stats.permute(0, 3, 1, 2)[:, None]
            moments = torch.cat(
                (moments, route_mass * stats), dim=2
            )
        value_for_gemm = value
        if (
            self.value_mode == "scale_affine_harmonic"
            and null_probability is not None
        ):
            null_value = self.null_values[family]
            if base_indices is not None:
                null_value = null_value[base_indices]
            # [scale mass, cosine moment, sine moment, ..., null probability]
            # is contracted with all corresponding V vectors in one GEMM.
            moments = torch.cat(
                (moments, null_probability[:, :, None]), dim=2
            )
            value_for_gemm = torch.cat(
                (value, null_value[:, None]), dim=1
            )
        # The second contraction is a regular GEMM.  Expressing it explicitly
        # avoids the extra einsum planning/layout overhead and lets cuBLAS use
        # the low-rank [base * harmonic, channel] matrix directly.
        queries, bases, harmonics, height, width = moments.shape
        moment_matrix = moments.permute(0, 3, 4, 1, 2).reshape(
            queries * height * width, bases * harmonics
        )
        output = moment_matrix @ value_for_gemm.reshape(
            bases * harmonics, self.out_channels
        )
        return output.reshape(queries, height, width, self.out_channels)

    @torch.no_grad()
    def prototype_states(self):
        """Describe every base through one common state interface.

        The method is deliberately read-only: a future differentiator can rank
        these states and emit component assignments without changing the source
        bank or its checkpoint.  This keeps all candidate components at the
        same trained starting point.
        """

        def direction_state(family, base, count):
            if family not in {"angular", "stripe", "full"}:
                return "none", 1
            tied = getattr(self, f"{family}_direction_tied")[base]
            fixed = getattr(self, f"{family}_direction_fixed")[base]
            subset = getattr(self, f"{family}_direction_subset")[base]
            if tied:
                return "tied", 1
            if fixed >= 0:
                return "fixed", 1
            active = int(subset.sum())
            return ("free" if active == count else "sparse"), active

        def scale_state(family, base, count):
            if family == "color":
                return ("none", 1) if count == 1 else ("free", count)
            if family not in {"radial", "stripe", "full"}:
                return "none", 1
            tied = getattr(self, f"{family}_scale_tied")[base]
            fixed = getattr(self, f"{family}_scale_fixed")[base]
            if tied:
                return "tied", 1
            if fixed >= 0:
                return "fixed", 1
            return "free", count

        layout = {
            "angular": (self.angular_scales, self.angular_directions),
            "radial": (self.scales, 1),
            "color": (self.color_scales, 1),
            "stripe": (self.scales, self.stripe_directions),
            "full": (self.scales, self.full_directions),
        }
        states = []
        for family, (scale_count, direction_count) in layout.items():
            for base in range(self.family_counts[family]):
                direction_mode, active_directions = direction_state(
                    family, base, direction_count
                )
                scale_mode, active_scales = scale_state(
                    family, base, scale_count
                )
                per_scale_subset = (
                    getattr(self, f"{family}_scale_direction_subset", None)
                    if family in {"angular", "stripe", "full"} else None
                )
                if per_scale_subset is not None and (
                    ~per_scale_subset[base]
                ).any():
                    active_poses = int(
                        per_scale_subset[base].sum()
                    )
                    direction_mode = "per_scale_sparse"
                else:
                    active_poses = active_directions * active_scales
                states.append(PrototypeState(
                    family=family,
                    base=base,
                    scale_count=scale_count,
                    direction_count=direction_count,
                    active_pose_count=active_poses,
                    scale_mode=scale_mode,
                    direction_mode=direction_mode,
                    null_score=float(self.null_scores[family][base]),
                ))
        return tuple(states)

    @staticmethod
    def _selected_pose_indices(scores, flat, pose_shape, selection):
        if isinstance(selection, str) and selection.startswith("per_scale_"):
            if len(pose_shape) != 2:
                raise ValueError("per-scale routing requires scale and direction poses")
            per_scale = int(selection.rsplit("_", 1)[1])
            per_scale = min(per_scale, pose_shape[1])
            direction_index = scores.topk(per_scale, dim=3).indices
            scale_offset = (
                torch.arange(pose_shape[0], device=scores.device)
                * pose_shape[1]
            )[None, None, :, None, None, None]
            return (direction_index + scale_offset).flatten(2, 3)
        selected = min(int(selection), flat.shape[2])
        return flat.topk(selected, dim=2).indices

    def _aggregate(self, scores, value, family, base_indices=None):
        pose_shape = scores.shape[2:-2]
        flat = scores.flatten(2, 2 + len(pose_shape) - 1)
        value_flat = value.flatten(1, value.ndim - 2)
        top_k = None if self.pose_topk is None else self.pose_topk.get(family)

        def null_gate(active_scores):
            if not self.use_null:
                return None
            real_logsumexp = torch.logsumexp(
                active_scores / self.pose_temperature, dim=2
            )
            null_score = self.null_scores[family]
            if base_indices is not None:
                null_score = null_score[base_indices]
            null_logit = null_score[None, :, None, None] / self.pose_temperature
            gate = torch.sigmoid(real_logsumexp - null_logit)
            self._null_terms.append((1 - gate).mean())
            return gate

        if self.hard_pose:
            if top_k is not None:
                pose_index = self._selected_pose_indices(
                    scores, flat, pose_shape, top_k
                )
                selected_score = flat.gather(2, pose_index)
                # Truncation should remove tail contributions, not redistribute
                # their probability mass over the retained values.  Keeping the
                # dense softmax denominator also preserves the output amplitude.
                log_denominator = torch.logsumexp(
                    flat / self.pose_temperature, dim=2, keepdim=True
                )
                selected_weight = torch.exp(
                    selected_score / self.pose_temperature - log_denominator
                )
                selected_weight = selected_weight * torch.exp2(selected_score)
                gate = null_gate(selected_score)
                if gate is not None:
                    selected_weight = selected_weight * gate[:, :, None]
                base_index = torch.arange(
                    value_flat.shape[0], device=value_flat.device
                )[None, :, None, None, None]
                selected_value = value_flat[base_index, pose_index]
                output = (selected_weight[..., None] * selected_value).sum((1, 2))
                return self._add_null_value(
                    output, gate, family, base_indices
                )
            pose_index = flat.argmax(2)
            selected_score = flat.gather(2, pose_index[:, :, None]).squeeze(2)
            gate = null_gate(selected_score[:, :, None])
            base_index = torch.arange(
                value_flat.shape[0], device=value_flat.device
            )[None, :, None, None]
            selected_value = value_flat[base_index, pose_index]
            selected_weight = torch.exp2(selected_score)
            if gate is not None:
                selected_weight = selected_weight * gate
            output = (selected_weight[..., None] * selected_value).sum(1)
            return self._add_null_value(
                output, gate, family, base_indices
            )
        soft_weight = (flat / self.pose_temperature).softmax(2)
        compression_target = (
            None if self.compression_topk is None
            else self.compression_topk.get(family)
        )
        compression_index = None
        if compression_target is not None:
            compression_index = self._selected_pose_indices(
                scores, flat, pose_shape, compression_target
            )
            retained_mass = soft_weight.gather(2, compression_index).sum(2)
            self._tail_terms.append((1 - retained_mass).mean())
        if top_k is not None:
            pose_index = self._selected_pose_indices(scores, flat, pose_shape, top_k)
            mask = torch.zeros_like(soft_weight).scatter_(2, pose_index, 1)
            if self.tail_keep is not None:
                # Anneal tail V contributions without changing the dense
                # softmax denominator or amplifying the retained values.
                route_weight = soft_weight * (
                    mask + self.tail_keep * (1 - mask)
                )
            else:
                sparse_weight = soft_weight * mask
                sparse_weight = sparse_weight / sparse_weight.sum(2, keepdim=True).clamp_min(1e-12)
                if self.straight_through_topk:
                    route_weight = sparse_weight.detach() + soft_weight - soft_weight.detach()
                else:
                    route_weight = sparse_weight
        else:
            route_weight = soft_weight
            pose_index = None
        if pose_index is not None:
            active_scores = flat.gather(2, pose_index)
        elif compression_index is not None:
            active_scores = flat.gather(2, compression_index)
        else:
            active_scores = flat
        gate = null_gate(active_scores)
        weight = route_weight * torch.exp2(flat)
        if gate is not None:
            weight = weight * gate[:, :, None]
        return self._weighted_value_sum(
            weight, value, family, pose_shape,
            null_probability=None if gate is None else 1 - gate,
            base_indices=base_indices,
        )

    def _aggregate_with_pose_degradation(self, scores, value, family):
        bases = scores.shape[1]
        scale_direction_subset = (
            getattr(self, f"{family}_scale_direction_subset", None)
            if family in {"angular", "stripe", "full"} else None
        )
        if scale_direction_subset is not None and (
            ~scale_direction_subset
        ).any():
            if scores.ndim == 5:
                scores = scores.unsqueeze(2)
                value = value.unsqueeze(1)
            active_counts = scale_direction_subset.sum(2)
            # Uniform per-scale Top-k can be gathered and aggregated in one
            # vectorized operation.  Besides removing the Python base loop,
            # this performs only the retained V multiplications.
            if active_counts.numel() and torch.all(
                active_counts == active_counts.flatten()[0]
            ):
                retained = int(active_counts.flatten()[0])
                direction_indices = scale_direction_subset.to(
                    dtype=torch.int8
                ).topk(retained, dim=2).indices
                score_indices = direction_indices[
                    None, :, :, :, None, None
                ].expand(
                    scores.shape[0], -1, -1, -1,
                    scores.shape[-2], scores.shape[-1],
                )
                selected_scores = scores.gather(3, score_indices)
                value_indices = direction_indices[..., None].expand(
                    -1, -1, -1, value.shape[-1]
                )
                selected_value = value.gather(2, value_indices)
                return self._aggregate_selected_dense(
                    scores, selected_scores, selected_value, family
                )
            outputs = []
            for base in range(bases):
                pose_mask = scale_direction_subset[base].flatten()
                poses = torch.nonzero(pose_mask, as_tuple=False).flatten()
                selected_scores = scores[:, base:base + 1].flatten(
                    2, 3
                ).index_select(2, poses)
                selected_value = value[base:base + 1].flatten(
                    1, 2
                ).index_select(1, poses)
                outputs.append(
                    self._aggregate_selected_dense(
                        scores[:, base:base + 1],
                        selected_scores, selected_value, family,
                        base_indices=torch.tensor([base], device=scores.device),
                    )
                )
            return torch.stack(outputs).sum(0)
        direction_tied = (
            getattr(self, f"{family}_direction_tied")
            if family in {"angular", "stripe", "full"}
            else torch.zeros(bases, dtype=torch.bool, device=scores.device)
        )
        direction_fixed = (
            getattr(self, f"{family}_direction_fixed")
            if family in {"angular", "stripe", "full"}
            else torch.full((bases,), -1, dtype=torch.long, device=scores.device)
        )
        direction_subset = (
            getattr(self, f"{family}_direction_subset")
            if family in {"angular", "stripe", "full"}
            else torch.ones(bases, 1, dtype=torch.bool, device=scores.device)
        )
        scale_tied = (
            getattr(self, f"{family}_scale_tied")
            if family in {"radial", "stripe", "full"}
            else torch.zeros(bases, dtype=torch.bool, device=scores.device)
        )
        scale_fixed = (
            getattr(self, f"{family}_scale_fixed")
            if family in {"radial", "stripe", "full"}
            else torch.full((bases,), -1, dtype=torch.long, device=scores.device)
        )
        if not (
            direction_tied.any() or (direction_fixed >= 0).any()
            or (~direction_subset).any()
            or scale_tied.any() or (scale_fixed >= 0).any()
        ):
            return self._aggregate(scores, value, family)

        groups = {}
        for base in range(bases):
            if direction_tied[base]:
                direction_key = ("mean",)
            elif direction_fixed[base] >= 0:
                direction_key = ("index", int(direction_fixed[base]))
            elif not direction_subset[base].all():
                direction_key = (
                    "subset",
                    *torch.nonzero(
                        direction_subset[base], as_tuple=False
                    ).flatten().tolist(),
                )
            else:
                direction_key = ("all",)
            if scale_tied[base]:
                scale_key = ("mean",)
            elif scale_fixed[base] >= 0:
                scale_key = ("index", int(scale_fixed[base]))
            else:
                scale_key = ("all",)
            groups.setdefault((direction_key, scale_key), []).append(base)

        outputs = []
        for (direction_key, scale_key), base_group in groups.items():
            indices = torch.as_tensor(base_group, device=scores.device)
            selected_scores = scores[:, indices]
            selected_value = value[indices]
            if scale_key[0] != "all":
                if scale_key[0] == "mean":
                    selected_scores = selected_scores.mean(2, keepdim=True)
                    selected_value = selected_value.mean(1, keepdim=True)
                else:
                    scale_index = scale_key[1]
                    selected_scores = selected_scores[:, :, scale_index:scale_index + 1]
                    selected_value = selected_value[:, scale_index:scale_index + 1]
            if direction_key[0] != "all":
                score_direction_dim = 2 if family == "angular" else 3
                value_direction_dim = 1 if family == "angular" else 2
                if direction_key[0] == "mean":
                    if family == "angular":
                        selected_scores = selected_scores.mean(2, keepdim=True)
                        selected_value = selected_value.mean(1, keepdim=True)
                    else:
                        selected_scores = selected_scores.mean(3, keepdim=True)
                        selected_value = selected_value.mean(2, keepdim=True)
                else:
                    selected_directions = torch.as_tensor(
                        direction_key[1:], device=scores.device
                    )
                    selected_scores = selected_scores.index_select(
                        score_direction_dim, selected_directions
                    )
                    selected_value = selected_value.index_select(
                        value_direction_dim, selected_directions
                    )
            outputs.append(
                self._aggregate(
                    selected_scores, selected_value, family, base_indices=indices
                )
            )
        return torch.stack(outputs).sum(0)

    def _aggregate_selected_dense(
        self, full_scores, selected_scores, selected_value, family,
        base_indices=None,
    ):
        """Drop tail V terms while preserving the original routing amplitude."""
        full_flat = full_scores.flatten(2, full_scores.ndim - 3)
        selected_flat = selected_scores.flatten(2, selected_scores.ndim - 3)
        value_flat = selected_value.flatten(1, selected_value.ndim - 2)
        log_denominator = torch.logsumexp(
            full_flat / self.pose_temperature, dim=2, keepdim=True
        )
        weight = torch.exp(
            selected_flat / self.pose_temperature - log_denominator
        ) * torch.exp2(selected_flat)
        gate = None
        if self.use_null:
            real_logsumexp = torch.logsumexp(
                full_flat / self.pose_temperature, dim=2
            )
            null_score = self.null_scores[family]
            if base_indices is not None:
                null_score = null_score[base_indices]
            gate = torch.sigmoid(
                real_logsumexp
                - null_score[None, :, None, None] / self.pose_temperature
            )
            weight = weight * gate[:, :, None]
            self._null_terms.append((1 - gate).mean())
        output = torch.einsum("qbphw,bpc->qhwc", weight, value_flat)
        return self._add_null_value(output, gate, family, base_indices)

    def forward(self, image):
        self._tail_terms = []
        self._null_terms = []
        if self.rgb_standardizer is not None:
            unit_image = self.rgb_standardizer.unit_rgb(image.float())
            self._input_stats = self.rgb_standardizer.output_stats(unit_image)
        else:
            self._input_stats = None
        scores = self.family_scores(image)
        output = self._aggregate_with_pose_degradation(
            scores["angular"], self.angular_value, "angular"
        )
        output = output + self._aggregate_with_pose_degradation(
            scores["radial"], self.radial_value, "radial"
        )
        output = output + self._aggregate(scores["color"], self.color_value, "color")
        output = output + self._aggregate_with_pose_degradation(
            scores["stripe"], self.stripe_value, "stripe"
        )
        output = output + self._aggregate_with_pose_degradation(
            scores["full"], self.full_value, "full"
        )
        if self.rgb_detrender is not None:
            unit_image = self.rgb_detrender.unit_rgb(image.float())
            modulation, trend = self.rgb_detrender.output_terms(unit_image)
            output = modulation * output + trend
        self._input_stats = None
        self.last_tail_mass = (
            torch.stack(self._tail_terms).mean()
            if self._tail_terms else output.new_zeros(())
        )
        self.last_null_mass = (
            torch.stack(self._null_terms).mean()
            if self._null_terms else output.new_zeros(())
        )
        return (output + self.output_bias).permute(0, 3, 1, 2).contiguous()
