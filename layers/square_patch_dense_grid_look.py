from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from .hex_patch_geometry import HexPatchGeometry
from .hex_rotating_polar_patch_embed import _PolarRenderer
from .polar_ring_sampler import PolarRingSampler
from .rotating_dot_product import rotating_dot_score, weighted_patch_flat
from .square_patch_low_rank_look import build_square_patch_centers


class SquarePatchDenseGridLook(nn.Module):
    """Per-head polar matchers with a shared rotating dense Look grid.

    Every attention head owns one image prototype and one canonical
    ``look_radial_bins x look_direction_bins`` table.  A pose match rotates
    and scales that same table; it does not select an independently learned
    table.  The null route contributes zero and the remaining pose weights are
    intentionally not renormalized.
    """

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_heads: int = 36,
        prototype_radial_bins: int = 8,
        prototype_angular_bins: int = 16,
        source_directions: int = 4,
        source_direction_period: int = 8,
        scales: Sequence[float] = (1.0, 0.5),
        prototype_radius: float = 12.0,
        look_direction_bins: int = 8,
        look_radial_bins: int = 4,
        look_radius: float = 4.0,
        patch_centers_xy: torch.Tensor | None = None,
        patch_coordinates_xy: torch.Tensor | None = None,
        compact_angular_bins_per_radius: int | None = None,
        compact_kernel_sizes: Sequence[int] | None = None,
        compact_lattice_stride: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(
            image_size, patch_size, in_channels, num_heads,
            prototype_radial_bins, prototype_angular_bins,
            source_directions, source_direction_period,
            look_direction_bins, look_radial_bins,
        ) <= 0:
            raise ValueError("dimensions and sample counts must be positive")
        if (patch_centers_xy is None) != (patch_coordinates_xy is None):
            raise ValueError(
                "patch_centers_xy and patch_coordinates_xy must be supplied together"
            )
        if source_directions > source_direction_period:
            raise ValueError("source_directions cannot exceed its direction period")
        if prototype_angular_bins % source_direction_period:
            raise ValueError(
                "prototype_angular_bins must be divisible by source_direction_period"
            )
        if not scales or any(float(scale) <= 0 for scale in scales):
            raise ValueError("scales must contain positive values")
        if min(prototype_radius, look_radius) <= 0:
            raise ValueError("prototype_radius and look_radius must be positive")

        if patch_centers_xy is None:
            if image_size % patch_size:
                raise ValueError("image_size must be divisible by patch_size")
            pixel_xy, grid_xy = build_square_patch_centers(image_size, patch_size)
        else:
            pixel_xy = torch.as_tensor(patch_centers_xy, dtype=torch.float32)
            grid_xy = torch.as_tensor(patch_coordinates_xy, dtype=torch.float32)
            if pixel_xy.ndim != 2 or pixel_xy.shape[-1] != 2:
                raise ValueError("patch_centers_xy must have shape (N, 2)")
            if grid_xy.shape != pixel_xy.shape:
                raise ValueError("patch_coordinates_xy must match patch_centers_xy")
        self.register_buffer("patch_centers_xy", pixel_xy, persistent=True)
        self.register_buffer("patch_coordinates_xy", grid_xy, persistent=True)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.num_heads = int(num_heads)
        self.source_directions = int(source_directions)
        self.source_direction_period = int(source_direction_period)
        self.look_direction_bins = int(look_direction_bins)
        self.look_radial_bins = int(look_radial_bins)
        self.prototype_radius = float(prototype_radius)
        self.look_radius = float(look_radius)
        self.eps = float(eps)
        self.compact_variable_rings = compact_angular_bins_per_radius is not None
        self._num_scales = len(scales)

        if self.compact_variable_rings:
            angular_step = int(compact_angular_bins_per_radius)
            if angular_step <= 0:
                raise ValueError("compact_angular_bins_per_radius must be positive")
            if compact_kernel_sizes is None or compact_lattice_stride is None:
                raise ValueError(
                    "compact variable rings require kernel sizes and lattice stride"
                )
            kernel_sizes = tuple(int(size) for size in compact_kernel_sizes)
            if len(kernel_sizes) != self._num_scales:
                raise ValueError("compact kernel sizes must match Look scales")
            self.compact_geometries = nn.ModuleList(
                HexPatchGeometry(
                    self.image_size, self.in_channels, kernel_size,
                    int(compact_lattice_stride),
                )
                for kernel_size in kernel_sizes
            )
            for geometry in self.compact_geometries:
                if not torch.equal(geometry.patch_centers_xy, self.patch_centers_xy):
                    raise ValueError(
                        "compact Look geometry must share tokenizer patch centers"
                    )
            counts = torch.tensor(
                [angular_step * (radius + 1) for radius in range(prototype_radial_bins)],
                dtype=torch.long,
            )
            offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
            self.register_buffer("ring_counts", counts, persistent=False)
            self.register_buffer("ring_offsets", offsets, persistent=False)
            self.compact_renderers = nn.ModuleList(
                _PolarRenderer(
                    geometry,
                    radial_bins=prototype_radial_bins,
                    ring_counts=counts,
                    ring_offsets=offsets,
                    directions=self.source_directions,
                    direction_step=self.direction_step_radians,
                )
                for geometry in self.compact_geometries
            )
            self.match_prototype = nn.Parameter(torch.randn(
                self.num_heads, self.in_channels, int(offsets[-1])
            ) * 0.02)
            reference_cover_mass = self.compact_renderers[0].support_cover.sum()
            for index, renderer in enumerate(self.compact_renderers):
                raw_cover = renderer.support_cover
                cover = raw_cover * (reference_cover_mass / raw_cover.sum())
                self.register_buffer(
                    f"compact_scale_cover_{index}", cover, persistent=False
                )
        else:
            self.ring_sampler = PolarRingSampler(
                radial_bins=prototype_radial_bins,
                angular_bins=prototype_angular_bins,
                rotation_samples=source_direction_period,
                scales=scales,
            )
            self.match_prototype = nn.Parameter(torch.empty(
                self.num_heads,
                self.in_channels,
                prototype_radial_bins,
                prototype_angular_bins,
            ))
            nn.init.normal_(
                self.match_prototype,
                std=1.0 / math.sqrt(in_channels * prototype_radial_bins),
            )
        self.null_score = nn.Parameter(torch.zeros(self.num_heads))
        # Zero makes insertion exactly equivalent to the original attention.
        # The table itself receives gradients immediately; prototype gradients
        # begin once the table moves away from zero.
        self.look_grid = nn.Parameter(torch.zeros(
            self.num_heads, self.look_radial_bins, self.look_direction_bins
        ))

        self._register_look_sampling_buffers(tuple(float(v) for v in scales))

    @property
    def num_patches(self) -> int:
        return int(self.patch_coordinates_xy.shape[0])

    @property
    def num_scales(self) -> int:
        return self._num_scales

    @property
    def direction_step_radians(self) -> float:
        return 2.0 * math.pi / self.source_direction_period

    def _register_look_sampling_buffers(self, scales: tuple[float, ...]) -> None:
        coordinates = self.patch_coordinates_xy
        relative = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
        distance = relative.norm(dim=-1)
        angle = torch.remainder(
            torch.atan2(relative[..., 1], relative[..., 0]), 2.0 * math.pi
        )
        radial0, radial1, angular0, angular1 = [], [], [], []
        radial_fraction, angular_fraction, valid = [], [], []
        for scale in scales:
            effective_radius = self.look_radius * scale
            # Four canonical radial cells are centered at 1/4, 2/4, 3/4,
            # and 4/4 of the transformed support.  Contracting the source
            # scale therefore contracts the Look support with no new weights.
            radial_position = (
                distance / effective_radius * self.look_radial_bins - 1.0
            )
            r0 = radial_position.floor().clamp(0, self.look_radial_bins - 1).long()
            r1 = (r0 + 1).clamp_max(self.look_radial_bins - 1)
            rw = (radial_position - radial_position.floor()).clamp(0.0, 1.0)
            scale_valid = (distance > 0) & (distance <= effective_radius)
            for direction in range(self.source_directions):
                rotation = direction * self.direction_step_radians
                relative_angle = torch.remainder(angle - rotation, 2.0 * math.pi)
                angular_position = (
                    relative_angle / (2.0 * math.pi) * self.look_direction_bins
                )
                a0 = angular_position.floor().long() % self.look_direction_bins
                a1 = (a0 + 1) % self.look_direction_bins
                aw = angular_position - angular_position.floor()
                radial0.append(r0)
                radial1.append(r1)
                angular0.append(a0)
                angular1.append(a1)
                radial_fraction.append(rw)
                angular_fraction.append(aw)
                valid.append(scale_valid)
        shape = (len(scales), self.source_directions, self.num_patches, self.num_patches)
        for name, values in (
            ("look_radial0", radial0),
            ("look_radial1", radial1),
            ("look_angular0", angular0),
            ("look_angular1", angular1),
            ("look_radial_fraction", radial_fraction),
            ("look_angular_fraction", angular_fraction),
            ("look_valid", valid),
        ):
            self.register_buffer(name, torch.stack(values).reshape(shape), persistent=False)

    def extract_rings(
        self,
        image: torch.Tensor,
        *,
        track_input_grad: bool = False,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]:
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected image spatial size {(self.image_size, self.image_size)}, "
                f"got {tuple(image.shape[-2:])}"
            )
        context = torch.enable_grad if track_input_grad else torch.no_grad
        with context():
            if self.compact_variable_rings:
                patches = [
                    weighted_patch_flat(
                        geometry(image),
                        getattr(self, f"compact_scale_cover_{scale_index}"),
                    )
                    for scale_index, geometry in enumerate(self.compact_geometries)
                ]
                return patches, image.new_empty(0)
            return self.ring_sampler(
                image,
                self.patch_centers_xy,
                base_radius=self.prototype_radius,
                return_coverage=True,
            )

    def raw_pose_response(
        self,
        rings: torch.Tensor | list[torch.Tensor],
        coverage: torch.Tensor,
    ) -> torch.Tensor:
        if self.compact_variable_rings:
            if not isinstance(rings, list) or len(rings) != self.num_scales:
                raise ValueError("compact rings must be one flattened patch tensor per scale")
            scores = [
                rotating_dot_score(patch, renderer(self.match_prototype))
                for patch, renderer in zip(rings, self.compact_renderers)
            ]
            return torch.stack(scores, dim=3)
        if not isinstance(rings, torch.Tensor):
            raise ValueError("dense rings must be a tensor")
        response = self.ring_sampler.circular_match(
            rings,
            self.match_prototype,
            coverage,
            rotation_count=self.source_directions,
        )
        return response

    def pose_weights(
        self,
        rings: torch.Tensor | list[torch.Tensor],
        coverage: torch.Tensor,
    ) -> torch.Tensor:
        """Return real-pose mass ``(B,N,H,S,T)`` after dropping null."""
        response = self.raw_pose_response(rings, coverage)
        response_shape = response.shape
        # Compact Look matching is allowed to use AMP/Tensor Cores, but the
        # routing distribution remains float32 so small pose/null differences
        # are not lost in the softmax.
        flat = response.float().flatten(-2)
        null = self.null_score.float().view(1, 1, -1, 1).expand(
            flat.shape[0], flat.shape[1], -1, 1
        )
        weights = torch.softmax(torch.cat((flat, null), dim=-1), dim=-1)
        return weights[..., :-1].reshape(response_shape)

    def transformed_look_grids(self) -> torch.Tensor:
        """Render the shared grid at every scale and direction: ``(H,S,T,N,N)``."""
        grid = self.look_grid

        def gather(radial: torch.Tensor, angular: torch.Tensor) -> torch.Tensor:
            index = radial * self.look_direction_bins + angular
            flat = grid.flatten(1)
            return flat[:, index]

        rw = self.look_radial_fraction
        aw = self.look_angular_fraction
        sampled = (
            gather(self.look_radial0, self.look_angular0) * (1.0 - rw) * (1.0 - aw)
            + gather(self.look_radial0, self.look_angular1) * (1.0 - rw) * aw
            + gather(self.look_radial1, self.look_angular0) * rw * (1.0 - aw)
            + gather(self.look_radial1, self.look_angular1) * rw * aw
        )
        return sampled * self.look_valid.to(sampled.dtype).unsqueeze(0)

    def look_bias(
        self,
        pose_weights: torch.Tensor,
        *,
        include_cls: bool = True,
    ) -> torch.Tensor:
        expected = (
            self.num_patches, self.num_heads,
            self.num_scales, self.source_directions,
        )
        if pose_weights.ndim != 5 or tuple(pose_weights.shape[1:]) != expected:
            raise ValueError(f"pose_weights must have shape (B,{','.join(map(str, expected))})")
        fields = self.transformed_look_grids().to(pose_weights)
        bias = torch.einsum("bqhst,hstqk->bhqk", pose_weights, fields)
        if not include_cls:
            return bias
        result = bias.new_zeros(
            bias.shape[0], self.num_heads,
            self.num_patches + 1, self.num_patches + 1,
        )
        result[:, :, 1:, 1:] = bias
        return result

    def forward_rings(
        self,
        rings: torch.Tensor | list[torch.Tensor],
        coverage: torch.Tensor,
        *,
        include_cls: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.pose_weights(rings, coverage)
        return self.look_bias(weights, include_cls=include_cls), weights

    def forward(
        self,
        image: torch.Tensor,
        *,
        include_cls: bool = True,
        track_input_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rings, coverage = self.extract_rings(
            image, track_input_grad=track_input_grad
        )
        return self.forward_rings(rings, coverage, include_cls=include_cls)
