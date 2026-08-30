from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.triton_dense_look_grid import dense_look_grid_sample
from utils.hex_graph import hex_relative_bins


def _ring_neighbor_table(
    coordinates: torch.Tensor,
    *,
    ring: int,
    directions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact query-to-neighbor indices for one hex graph ring."""

    if ring not in {1, 2}:
        raise ValueError("the two-ring matcher supports graph rings 1 and 2")
    if directions not in {6, 12}:
        raise ValueError("ring directions must be 6 or 12")
    rings, _ = hex_relative_bins(coordinates)
    relative = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
    angles = torch.remainder(
        torch.atan2(relative[..., 1], relative[..., 0]), 2.0 * math.pi
    )
    step = 2.0 * math.pi / directions
    bins = torch.floor((angles + 0.5 * step) / step).long() % directions

    neighbors = torch.full(
        (coordinates.shape[0], directions), -1, dtype=torch.long
    )
    for query in range(coordinates.shape[0]):
        keys = torch.nonzero(rings[query] == ring, as_tuple=False).flatten()
        for key in keys.tolist():
            direction = int(bins[query, key])
            if neighbors[query, direction] >= 0:
                raise ValueError(
                    f"multiple ring-{ring} neighbors mapped to direction {direction}"
                )
            neighbors[query, direction] = key
    return neighbors, neighbors >= 0


class TwoRingCircularLookMatcher(nn.Module):
    """C6/C12 circular feature correlation for deep Look refinement.

    Each layer and head stores only six inner-ring and twelve outer-ring
    ``head_dim -> 1`` projections. Circular shifts of those banks evaluate
    every candidate pose, producing 6x6 and 12x12 pairwise matches without
    storing independent weights for every pair.
    """

    def __init__(
        self,
        *,
        coordinates: torch.Tensor,
        depth: int,
        num_heads: int,
        head_dim: int,
        start_layer: int = 8,
    ) -> None:
        super().__init__()
        if min(depth, num_heads, head_dim) <= 0:
            raise ValueError("depth, num_heads, and head_dim must be positive")
        if head_dim % 2:
            raise ValueError("head_dim must be even for paired polar W_look")
        if not 0 <= start_layer < depth:
            raise ValueError("start_layer must index a Transformer block")
        coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (N, 2)")

        inner, inner_valid = _ring_neighbor_table(
            coordinates, ring=1, directions=6
        )
        outer, outer_valid = _ring_neighbor_table(
            coordinates, ring=2, directions=12
        )
        self.register_buffer("inner_neighbors", inner, persistent=True)
        self.register_buffer("outer_neighbors", outer, persistent=True)
        self.register_buffer("inner_valid", inner_valid, persistent=False)
        self.register_buffer("outer_valid", outer_valid, persistent=False)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.start_layer = int(start_layer)
        self.active_depth = self.depth - self.start_layer

        self.inner_radius, self.inner_phase = self._make_polar_weight(6)
        self.outer_radius, self.outer_phase = self._make_polar_weight(12)
        # One canonical dense 4x12 Look map per ring. Candidate orientations
        # rotate this shared map; no absolute-pose copies are stored.
        self.inner_look_grid = nn.Parameter(
            torch.randn(self.active_depth, num_heads, 4, 12) * 0.02
        )
        self.outer_look_grid = nn.Parameter(
            torch.randn(self.active_depth, num_heads, 4, 12) * 0.02
        )
        # Zero initialization preserves the existing Look model exactly and
        # lets each layer/head decide how much the two feature rings matter.
        self.gate = nn.Parameter(torch.zeros(self.active_depth, num_heads, 2))

        inner_relative = (
            torch.arange(6)[:, None] - torch.arange(6)[None, :]
        ) % 6
        outer_relative = (
            torch.arange(12)[:, None] - torch.arange(12)[None, :]
        ) % 12
        self.register_buffer("inner_relative", inner_relative, persistent=False)
        self.register_buffer("outer_relative", outer_relative, persistent=False)
        self.register_buffer(
            "inner_candidate_angle",
            torch.arange(6, dtype=torch.float32) * (2.0 * math.pi / 6.0),
            persistent=False,
        )
        self.register_buffer(
            "outer_candidate_angle",
            torch.arange(12, dtype=torch.float32) * (2.0 * math.pi / 12.0),
            persistent=False,
        )
        self._register_look_sampling_buffers(coordinates)

    @property
    def num_patches(self) -> int:
        return int(self.inner_neighbors.shape[0])

    def _make_polar_weight(
        self,
        directions: int,
    ) -> tuple[nn.Parameter, nn.Parameter]:
        cartesian = torch.randn(
            self.active_depth,
            self.num_heads,
            directions,
            self.head_dim // 2,
            2,
        ) / math.sqrt(self.head_dim)
        radius = nn.Parameter(torch.linalg.vector_norm(cartesian, dim=-1))
        phase = nn.Parameter(torch.atan2(cartesian[..., 1], cartesian[..., 0]))
        return radius, phase

    def _register_look_sampling_buffers(self, coordinates: torch.Tensor) -> None:
        relative = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
        distance = relative.norm(dim=-1)
        angle = torch.remainder(
            torch.atan2(relative[..., 1], relative[..., 0]), 2.0 * math.pi
        )
        radial_position = distance / 4.0 * 4.0 - 1.0
        radial0 = radial_position.floor().clamp(0, 3).long()
        radial1 = (radial0 + 1).clamp_max(3)
        radial_fraction = (
            radial_position - radial_position.floor()
        ).clamp(0.0, 1.0)
        angular_position = angle / (2.0 * math.pi) * 12.0
        angular0 = angular_position.floor().long() % 12
        angular1 = (angular0 + 1) % 12
        angular_fraction = angular_position - angular_position.floor()
        valid = (distance > 0) & (distance <= 4.0)
        for name, value in (
            ("look_radial0", radial0),
            ("look_radial1", radial1),
            ("look_radial_fraction", radial_fraction),
            ("look_angular0", angular0),
            ("look_angular1", angular1),
            ("look_angular_fraction", angular_fraction),
            ("look_valid", valid),
        ):
            self.register_buffer(name, value, persistent=False)

    @staticmethod
    def _rotated_look_grids(
        grid: torch.Tensor,
        *,
        directions: int,
        angular_stride: int,
    ) -> torch.Tensor:
        return torch.stack(
            tuple(
                torch.roll(grid, shifts=direction * angular_stride, dims=-1)
                for direction in range(directions)
            ),
            dim=1,
        )

    def _sample_query_grids(self, grid: torch.Tensor) -> torch.Tensor:
        return dense_look_grid_sample(
            grid,
            self.look_radial0,
            self.look_radial1,
            self.look_angular0,
            self.look_angular1,
            self.look_radial_fraction,
            self.look_angular_fraction,
            self.look_valid,
        )

    @staticmethod
    def _render_polar_weight(
        radius: torch.Tensor,
        phase: torch.Tensor,
        relative: torch.Tensor,
        candidate_angle: torch.Tensor,
    ) -> torch.Tensor:
        # Convert every stored pair once, then rotate it with the small table
        # of constant candidate angles.  This is equivalent to evaluating
        # cos/sin(phase + angle) for every position-pose pair, but avoids the
        # repeated transcendental operations.
        base_x = radius * phase.cos()
        base_y = radius * phase.sin()
        selected_x = base_x[:, relative, :]
        selected_y = base_y[:, relative, :]
        cosine = candidate_angle.cos()[None, :, None, None]
        sine = candidate_angle.sin()[None, :, None, None]
        paired = torch.stack(
            (
                selected_x * cosine - selected_y * sine,
                selected_x * sine + selected_y * cosine,
            ),
            dim=-1,
        )
        return paired.flatten(-2)

    def _correlate(
        self,
        features: torch.Tensor,
        neighbors: torch.Tensor,
        valid: torch.Tensor,
        radius: torch.Tensor,
        phase: torch.Tensor,
        relative: torch.Tensor,
        candidate_angle: torch.Tensor,
    ) -> torch.Tensor:
        batch, patches, heads, channels = features.shape
        sentinel = patches
        padded = torch.cat(
            (features, features.new_zeros(batch, 1, heads, channels)), dim=1
        )
        gather_index = neighbors.clamp_min(0)
        gather_index = torch.where(
            valid, gather_index, torch.full_like(gather_index, sentinel)
        )
        edge = padded[:, gather_index]

        # Each stored W_look is a head_dim -> 1 projection.  Its circular
        # relative index is shared by every absolute candidate orientation.
        circular_weight = self._render_polar_weight(
            radius, phase, relative, candidate_angle
        )
        scores = torch.einsum("bqphc,htpc->bqht", edge, circular_weight)
        valid_count = valid.sum(dim=-1).clamp_min(1).to(scores.dtype)
        scores = scores / torch.sqrt(valid_count[None, :, None, None])
        return scores - scores.mean(dim=-1, keepdim=True)

    def forward(
        self,
        patch_features: torch.Tensor,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return centered gated inner C6 and outer C12 coefficients."""

        if patch_features.ndim != 3:
            raise ValueError("patch_features must have shape (B, N, D)")
        if patch_features.shape[1] != self.num_patches:
            raise ValueError("patch count does not match the stored hex graph")
        if patch_features.shape[2] != self.num_heads * self.head_dim:
            raise ValueError("feature width does not match heads x head_dim")
        if not 0 <= layer_index < self.depth:
            raise ValueError("layer_index is out of range")
        if layer_index < self.start_layer:
            raise ValueError("layer_index precedes the configured start layer")
        active_layer = layer_index - self.start_layer

        features = patch_features.reshape(
            patch_features.shape[0], self.num_patches, self.num_heads, self.head_dim
        )
        inner = self._correlate(
            features,
            self.inner_neighbors,
            self.inner_valid,
            self.inner_radius[active_layer],
            self.inner_phase[active_layer],
            self.inner_relative,
            self.inner_candidate_angle,
        )
        outer = self._correlate(
            features,
            self.outer_neighbors,
            self.outer_valid,
            self.outer_radius[active_layer],
            self.outer_phase[active_layer],
            self.outer_relative,
            self.outer_candidate_angle,
        )
        gate = self.gate[active_layer]
        inner = inner * gate[None, None, :, 0, None]
        outer = outer * gate[None, None, :, 1, None]
        return inner, outer

    def dense_look_bias(
        self,
        inner: torch.Tensor,
        outer: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        """Convolve ring responses on 4x12 grids, then sample query-key bias."""
        active_layer = layer_index - self.start_layer
        if not 0 <= active_layer < self.active_depth:
            raise ValueError("layer_index is outside the active matcher range")
        inner_grid = self._rotated_look_grids(
            self.inner_look_grid[active_layer],
            directions=6,
            angular_stride=2,
        )
        outer_grid = self._rotated_look_grids(
            self.outer_look_grid[active_layer],
            directions=12,
            angular_stride=1,
        )
        query_grid = torch.einsum("bqhd,hdrt->bqhrt", inner, inner_grid)
        query_grid = query_grid + torch.einsum(
            "bqhd,hdrt->bqhrt", outer, outer_grid
        )
        return self._sample_query_grids(query_grid)
