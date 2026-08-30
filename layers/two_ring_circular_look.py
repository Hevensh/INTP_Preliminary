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

    if ring != 1 or directions != 6:
        raise ValueError("the local Look probe requires graph ring 1 with C6")
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
    """Use one C6 local probe to steer a full four-ring Look field.

    Each layer and head stores six ``head_dim -> 1`` projections. Circular
    shifts evaluate the six local poses; those responses rotate one canonical
    4x12 field that controls where attention should look farther away.
    """

    def __init__(
        self,
        *,
        coordinates: torch.Tensor,
        depth: int,
        num_heads: int,
        head_dim: int,
        start_layer: int = 0,
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

        neighbors, valid = _ring_neighbor_table(
            coordinates, ring=1, directions=6
        )
        self.register_buffer("neighbors", neighbors, persistent=True)
        self.register_buffer("valid", valid, persistent=False)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.start_layer = int(start_layer)
        self.active_depth = self.depth - self.start_layer

        self.radius, self.phase = self._make_polar_weight(6)
        # The local C6 probe steers one full-support 4x12 Look map. Probe range
        # and Look range are intentionally different: local evidence decides
        # which farther directions attention should emphasize.
        self.look_grid = nn.Parameter(
            torch.randn(self.active_depth, num_heads, 4, 12) * 0.02
        )
        # Zero initialization preserves the existing Look model exactly.
        self.gate = nn.Parameter(torch.zeros(self.active_depth, num_heads))

        relative = (
            torch.arange(6)[:, None] - torch.arange(6)[None, :]
        ) % 6
        self.register_buffer("relative", relative, persistent=False)
        self.register_buffer(
            "candidate_angle",
            torch.arange(6, dtype=torch.float32) * (2.0 * math.pi / 6.0),
            persistent=False,
        )
        self._register_look_sampling_buffers(coordinates)

    @property
    def num_patches(self) -> int:
        return int(self.neighbors.shape[0])

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
        # The C6 detector is local, but its learned Look field covers all four
        # graph radii. Four cells are centered at radii 1, 2, 3, and 4.
        radial_position = distance - 1.0
        radial0 = radial_position.floor().clamp(0, 3).long()
        radial1 = (radial0 + 1).clamp_max(3)
        radial_fraction = (
            radial_position - radial_position.floor()
        ).clamp(0.0, 1.0)
        valid = (distance > 0) & (distance <= 4.0)
        angular_position = angle / (2.0 * math.pi) * 12.0
        angular0 = angular_position.floor().long() % 12
        angular1 = (angular0 + 1) % 12
        angular_fraction = angular_position - angular_position.floor()
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
        radius: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the six local poses directly; FFT is wasteful at C6."""

        batch, patches, heads, channels = features.shape
        sentinel = patches
        padded = torch.cat(
            (features, features.new_zeros(batch, 1, heads, channels)), dim=1
        )
        gather_index = torch.where(
            self.valid,
            self.neighbors.clamp_min(0),
            torch.full_like(self.neighbors, sentinel),
        )
        edge = padded[:, gather_index]
        circular_weight = self._render_polar_weight(
            radius,
            phase,
            self.relative,
            self.candidate_angle,
        )
        scores = torch.einsum("bqphc,htpc->bqht", edge, circular_weight)
        valid_count = self.valid.sum(dim=-1).clamp_min(1).to(scores.dtype)
        scores = scores / torch.sqrt(valid_count[None, :, None, None])
        return scores - scores.mean(dim=-1, keepdim=True)

    def _validate_features(
        self, patch_features: torch.Tensor, layer_index: int
    ) -> tuple[torch.Tensor, int]:
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
        return features, active_layer

    def forward(
        self,
        patch_features: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        """Return centered gated C6 local-pose coefficients."""

        features, active_layer = self._validate_features(
            patch_features, layer_index
        )
        scores = self._correlate(
            features,
            self.radius[active_layer],
            self.phase[active_layer],
        )
        return scores * self.gate[active_layer][None, None, :, None]

    def dense_look_bias_from_features(
        self,
        patch_features: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        """Use local C6 evidence to steer the complete four-ring Look field."""

        scores = self(patch_features, layer_index=layer_index)
        return self.dense_look_bias(scores, layer_index=layer_index)

    def dense_look_bias(
        self,
        scores: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        """Rotate one 4x12 policy by C6 scores, then sample query-key bias."""

        active_layer = layer_index - self.start_layer
        if not 0 <= active_layer < self.active_depth:
            raise ValueError("layer_index is outside the active matcher range")
        rotated_grid = self._rotated_look_grids(
            self.look_grid[active_layer],
            directions=6,
            angular_stride=2,
        )
        query_grid = torch.einsum(
            "bqhd,hdrt->bqhrt", scores, rotated_grid
        )
        return self._sample_query_grids(query_grid)
