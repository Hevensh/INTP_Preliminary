from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hex_graph import complex_to_xy
from .polar_ring_sampler import PolarRingSampler


class PolarPrototypeLook(nn.Module):
    """Pose-sampled polar prototype matcher with a paired look-region field.

    Both learned fields are stored as ordinary square tensors whose axes are
    ``(rho, phi)``. The rho axis starts above zero, so the patch center is
    deliberately invisible. A match hypothesis at a rotation and scale votes
    for the look-region field transformed by the same pose.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        in_channels: int,
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
        radial_window_power: float = 1.0,
        match_temperature: float = 0.1,
        match_threshold: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(num_heads, in_channels, radial_bins, angular_bins, rotation_samples) <= 0:
            raise ValueError("all dimensions and sample counts must be positive")
        if not scales or any(float(value) <= 0 for value in scales):
            raise ValueError("scales must contain positive values")
        if match_temperature <= 0:
            raise ValueError("match_temperature must be positive")
        if radial_window_power < 0:
            raise ValueError("radial_window_power must be non-negative")

        self.num_heads = int(num_heads)
        self.in_channels = int(in_channels)
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self.rho_min = float(1.0 / radial_bins if rho_min is None else rho_min)
        if not 0.0 < self.rho_min < 1.0:
            raise ValueError("rho_min must be in (0, 1)")
        self.match_temperature = float(match_temperature)
        self.match_threshold = float(match_threshold)
        self.radial_window_power = float(radial_window_power)
        self.eps = float(eps)

        self.match_prototype = nn.Parameter(
            torch.empty(self.num_heads, self.in_channels, self.radial_bins, self.angular_bins)
        )
        nn.init.normal_(self.match_prototype, std=1.0 / math.sqrt(in_channels * radial_bins))
        self.look_prototype_logits = nn.Parameter(
            torch.empty(self.num_heads, 1, self.radial_bins, self.angular_bins)
        )
        self._initialize_directional_look_prototype()
        rotations = torch.arange(rotation_samples, dtype=torch.float32) * (2.0 * math.pi / rotation_samples)
        self.register_buffer("rotations", rotations, persistent=True)
        self.register_buffer("scales", torch.tensor(tuple(float(value) for value in scales)), persistent=True)
        self.ring_sampler = PolarRingSampler(
            radial_bins=self.radial_bins,
            angular_bins=self.angular_bins,
            rotation_samples=rotation_samples,
            scales=scales,
            rho_min=self.rho_min,
        )

    def _initialize_directional_look_prototype(self) -> None:
        """Initialize a smooth canonical look region pointing along positive x."""
        rho = torch.linspace(0.0, 1.0, self.radial_bins)
        phi = torch.arange(self.angular_bins, dtype=torch.float32) * (
            2.0 * math.pi / self.angular_bins
        )
        wrapped_phi = torch.atan2(phi.sin(), phi.cos())
        radial = torch.exp(-0.5 * ((rho - 0.72) / 0.18).square())
        angular = torch.exp(-0.5 * (wrapped_phi / 0.42).square())
        field = radial[:, None] * angular[None, :]
        field = field / field.max().clamp_min(self.eps)
        logits = torch.logit(field.clamp(1e-4, 1.0 - 1e-4))
        with torch.no_grad():
            self.look_prototype_logits.copy_(
                logits[None, None].expand(self.num_heads, 1, -1, -1)
            )

    @property
    def num_rotations(self) -> int:
        return int(self.rotations.numel())

    @property
    def num_scales(self) -> int:
        return int(self.scales.numel())

    def radial_match_weight(self, patch_offsets_xy: torch.Tensor, *, patch_radius: float) -> torch.Tensor:
        """Cosine radial window with the invisible center annulus removed."""
        if patch_radius <= 0:
            raise ValueError("patch_radius must be positive")
        sample_radius = patch_offsets_xy.norm(dim=-1)
        normalized_radius = (sample_radius / patch_radius).clamp(0.0, 1.0)
        visible = sample_radius >= self.rho_min * patch_radius
        weight = torch.cos(normalized_radius * (math.pi / 2.0)).clamp_min(0.0)
        return weight.pow(self.radial_window_power) * visible.to(weight.dtype)

    def _sample_square_polar_field(
        self,
        field: torch.Tensor,
        points_xy: torch.Tensor,
        *,
        base_radius: float,
        rho_min: float | None = None,
        center_mode: str = "zero",
    ) -> torch.Tensor:
        """Sample a ``(H,C,R,A)`` field for every configured scale/rotation.

        Returns ``(H,S,T,C,...)`` where ``...`` is ``points_xy.shape[:-1]``.
        Angular interpolation is circular and radial samples outside the
        annulus ``rho_min <= rho <= 1`` are zero.
        """
        if field.ndim != 4 or field.shape[-2:] != (self.radial_bins, self.angular_bins):
            raise ValueError("field must have shape (H, C, radial_bins, angular_bins)")
        if points_xy.shape[-1] != 2:
            raise ValueError("points_xy must have shape (..., 2)")
        if base_radius <= 0:
            raise ValueError("base_radius must be positive")
        rho_start = self.rho_min if rho_min is None else float(rho_min)
        if not 0.0 <= rho_start < 1.0:
            raise ValueError("rho_min must be in [0, 1)")
        if center_mode not in {"zero", "angular_mean"}:
            raise ValueError("center_mode must be zero or angular_mean")

        point_shape = points_xy.shape[:-1]
        points = points_xy.reshape(-1, 2).to(device=field.device, dtype=field.dtype)
        radii = points.norm(dim=-1)
        angles = torch.atan2(points[:, 1], points[:, 0])
        flat_field = field.flatten(-2)
        sampled_scales = []
        for scale in self.scales.to(device=field.device, dtype=field.dtype):
            rho = radii / (float(base_radius) * scale)
            radial_position = (rho - rho_start) / (1.0 - rho_start) * (self.radial_bins - 1)
            valid_outer = rho <= 1.0
            valid_annulus = (rho >= rho_start) & valid_outer
            r0 = radial_position.floor().clamp(0, self.radial_bins - 1).to(torch.long)
            r1 = (r0 + 1).clamp(max=self.radial_bins - 1)
            rw = (radial_position - r0.to(radial_position.dtype)).clamp(0.0, 1.0)
            sampled_rotations = []
            for rotation in self.rotations.to(device=field.device, dtype=field.dtype):
                phi = torch.remainder(angles - rotation, 2.0 * math.pi)
                angular_position = phi / (2.0 * math.pi) * self.angular_bins
                a0 = angular_position.floor().to(torch.long) % self.angular_bins
                a1 = (a0 + 1) % self.angular_bins
                aw = angular_position - angular_position.floor()

                def gather(radial: torch.Tensor, angular: torch.Tensor) -> torch.Tensor:
                    index = radial * self.angular_bins + angular
                    return torch.gather(
                        flat_field,
                        -1,
                        index[None, None].expand(field.shape[0], field.shape[1], -1),
                    )

                value = (
                    gather(r0, a0) * (1.0 - rw) * (1.0 - aw)
                    + gather(r0, a1) * (1.0 - rw) * aw
                    + gather(r1, a0) * rw * (1.0 - aw)
                    + gather(r1, a1) * rw * aw
                )
                if center_mode == "angular_mean":
                    center_value = field[:, :, 0].mean(dim=-1, keepdim=True)
                    if rho_start > 0.0:
                        raise ValueError("angular_mean is only valid when rho_min is zero")
                    at_origin = radii <= self.eps
                    value = torch.where(at_origin[None, None], center_value, value)
                    value = value * valid_outer.to(value.dtype)
                else:
                    value = value * valid_annulus.to(value.dtype)
                sampled_rotations.append(value.reshape(field.shape[0], field.shape[1], *point_shape))
            sampled_scales.append(torch.stack(sampled_rotations, dim=1))
        return torch.stack(sampled_scales, dim=1)

    def match(
        self,
        patches: torch.Tensor,
        patch_offsets_xy: torch.Tensor,
        *,
        patch_radius: float,
    ) -> torch.Tensor:
        """Return cosine match responses shaped ``(B,N,H,S,T)``."""
        if patches.ndim != 4:
            raise ValueError("patches must have shape (B, N, C, L)")
        if patches.shape[2] != self.in_channels:
            raise ValueError("patch channel count does not match in_channels")
        if patch_offsets_xy.shape != (patches.shape[-1], 2):
            raise ValueError("patch_offsets_xy must have shape (L, 2)")
        templates = self._sample_square_polar_field(
            self.match_prototype,
            patch_offsets_xy,
            base_radius=patch_radius,
        )  # (H,S,T,C,L)
        radial_weight = self.radial_match_weight(
            patch_offsets_xy, patch_radius=patch_radius
        ).to(device=patches.device, dtype=patches.dtype)
        # sqrt(weight) on both operands produces a weighted cosine similarity.
        sqrt_weight = radial_weight.sqrt()
        patch_vectors = (patches * sqrt_weight).flatten(-2)
        template_vectors = (templates * sqrt_weight).flatten(-2)
        patch_vectors = F.normalize(patch_vectors, dim=-1, eps=self.eps)
        template_vectors = F.normalize(template_vectors, dim=-1, eps=self.eps)
        return torch.einsum("bnf,hstf->bnhst", patch_vectors, template_vectors)

    def match_image(
        self,
        image: torch.Tensor,
        patch_centers_xy: torch.Tensor,
        *,
        base_radius: float,
        return_details: bool = False,
        track_input_grad: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match directly from an image through normalized polar soft-splat.

        The per-cell coverage division removes the raw Cartesian point-count
        gain of large scales.  The matcher normalizes only the prototype, so
        the input ring's intrinsic response magnitude remains observable.
        """
        if track_input_grad:
            rings, coverage = self.ring_sampler(
                image,
                patch_centers_xy,
                base_radius=base_radius,
                return_coverage=True,
            )
        else:
            with torch.no_grad():
                rings, coverage = self.ring_sampler(
                    image,
                    patch_centers_xy,
                    base_radius=base_radius,
                    return_coverage=True,
                )
        response = self.match_rings(rings, coverage)
        return (response, rings, coverage) if return_details else response

    def match_rings(
        self,
        rings: torch.Tensor,
        coverage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Match this layer's head prototypes against a reusable P0 ring cache."""
        return self.ring_sampler.circular_match(rings, self.match_prototype, coverage)

    def forward_rings(
        self,
        rings: torch.Tensor,
        coverage: torch.Tensor,
        patch_coordinates: torch.Tensor,
        *,
        look_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce this layer's independent look bias from shared unencoded rings."""
        response = self.match_rings(rings, coverage)
        look_gain = self.project_look(response, patch_coordinates, look_radius=look_radius)
        return look_gain, response

    def forward_image(
        self,
        image: torch.Tensor,
        patch_centers_xy: torch.Tensor,
        patch_coordinates: torch.Tensor,
        *,
        base_radius: float,
        look_radius: float,
        track_input_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Image-to-look path using polar soft-splat and circular matching."""
        response = self.match_image(
            image,
            patch_centers_xy,
            base_radius=base_radius,
            track_input_grad=track_input_grad,
        )
        look_gain = self.project_look(response, patch_coordinates, look_radius=look_radius)
        return look_gain, response

    def project_look(
        self,
        match_response: torch.Tensor,
        patch_coordinates: torch.Tensor,
        *,
        look_radius: float,
    ) -> torch.Tensor:
        """Project pose matches into a directed dynamic look gain ``(B,H,N,N)``."""
        expected_tail = (self.num_heads, self.num_scales, self.num_rotations)
        if match_response.ndim != 5 or tuple(match_response.shape[-3:]) != expected_tail:
            raise ValueError(f"match_response must end with {expected_tail}")
        if patch_coordinates.ndim == 1 and torch.is_complex(patch_coordinates):
            coordinates_xy = complex_to_xy(patch_coordinates)
        else:
            coordinates_xy = patch_coordinates
        if coordinates_xy.ndim != 2 or coordinates_xy.shape != (match_response.shape[1], 2):
            raise ValueError("patch_coordinates must have shape (N, 2) or complex shape (N,)")

        relative_xy = coordinates_xy.unsqueeze(0) - coordinates_xy.unsqueeze(1)
        look_field = torch.sigmoid(self.look_prototype_logits)
        transformed = self._sample_square_polar_field(
            look_field,
            relative_xy,
            base_radius=look_radius,
            rho_min=0.0,
            center_mode="angular_mean",
        ).squeeze(3)  # (H,S,T,N,N)
        # Preserve the signed match response: every pose directly adds to or
        # subtracts from the spatial region produced by that scale/rotation.
        look_gain = torch.einsum("bnhst,hstnj->bhnj", match_response, transformed)
        return look_gain / float(self.num_scales * self.num_rotations)

    def forward(
        self,
        patches: torch.Tensor,
        patch_offsets_xy: torch.Tensor,
        patch_coordinates: torch.Tensor,
        *,
        patch_radius: float,
        look_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response = self.match(patches, patch_offsets_xy, patch_radius=patch_radius)
        look_gain = self.project_look(response, patch_coordinates, look_radius=look_radius)
        return look_gain, response
