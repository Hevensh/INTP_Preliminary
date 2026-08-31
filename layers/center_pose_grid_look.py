from __future__ import annotations

import math

import torch
import torch.nn as nn


class CenterPoseGridLook(nn.Module):
    """Route one shared centre pose into learned radial-angular Look grids.

    One centre detector may serve several consecutive Transformer blocks.
    Every block/head still retains its own learned
    ``radial_bins x direction_bins`` output template.
    """

    def __init__(
        self,
        *,
        coordinates: torch.Tensor,
        embed_dim: int,
        num_heads: int,
        depth: int,
        axes: int,
        layers_per_probe: int = 1,
        radial_bins: int = 4,
        direction_bins: int = 12,
        look_radius: float = 4.0,
        null_initial_score: float = 0.0,
    ) -> None:
        super().__init__()
        if min(embed_dim, num_heads, depth, axes, layers_per_probe) <= 0:
            raise ValueError("dimensions must be positive")
        if min(radial_bins, direction_bins) <= 0 or look_radius <= 0:
            raise ValueError("grid dimensions and look_radius must be positive")
        if embed_dim % 2:
            raise ValueError("embed_dim must contain complete cosine/sine pairs")
        pair_count = embed_dim // 2
        if pair_count % num_heads:
            raise ValueError("cosine/sine pairs must divide evenly across heads")
        coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (N, 2)")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.depth = int(depth)
        self.axes = int(axes)
        self.layers_per_probe = int(layers_per_probe)
        self.probe_groups = math.ceil(depth / layers_per_probe)
        self.radial_bins = int(radial_bins)
        self.direction_bins = int(direction_bins)
        self.look_radius = float(look_radius)
        self.pairs_per_head = pair_count // num_heads

        self.axis_weight = nn.Parameter(
            torch.randn(
                self.probe_groups, num_heads, axes, self.pairs_per_head, 2
            )
            / math.sqrt(2 * self.pairs_per_head)
        )
        self.axis_bias = nn.Parameter(
            torch.zeros(self.probe_groups, num_heads, axes)
        )
        self.null_score = nn.Parameter(
            torch.full(
                (self.probe_groups, num_heads), float(null_initial_score)
            )
        )
        # Zero initialization preserves the PE/tokenizer baseline at step 0,
        # matching the established image Look branch.
        self.look_grid = nn.Parameter(
            torch.zeros(
                depth,
                num_heads,
                radial_bins,
                direction_bins,
            )
        )

        sampling = self._grid_sampling(
            coordinates,
            axes=axes,
            radial_bins=radial_bins,
            direction_bins=direction_bins,
            look_radius=look_radius,
        )
        for name, value in sampling.items():
            self.register_buffer(name, value, persistent=False)

    @staticmethod
    def _grid_sampling(
        coordinates: torch.Tensor,
        *,
        axes: int,
        radial_bins: int,
        direction_bins: int,
        look_radius: float,
    ) -> dict[str, torch.Tensor]:
        relative = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
        distance = relative.norm(dim=-1)
        angle = torch.remainder(
            torch.atan2(relative[..., 1], relative[..., 0]), 2.0 * math.pi
        )
        radial_position = distance / look_radius * radial_bins - 1.0
        radial0 = radial_position.floor().clamp(0, radial_bins - 1).long()
        radial1 = (radial0 + 1).clamp_max(radial_bins - 1)
        radial_fraction = (radial_position - radial_position.floor()).clamp(0.0, 1.0)
        valid = (distance > 0) & (distance <= look_radius)

        angular0, angular1, angular_fraction = [], [], []
        for axis in range(axes):
            rotation = axis * math.pi / axes
            relative_angle = torch.remainder(angle - rotation, 2.0 * math.pi)
            angular_position = relative_angle / (2.0 * math.pi) * direction_bins
            lower = angular_position.floor().long() % direction_bins
            angular0.append(lower)
            angular1.append((lower + 1) % direction_bins)
            angular_fraction.append(angular_position - angular_position.floor())
        return {
            "look_radial0": radial0,
            "look_radial1": radial1,
            "look_radial_fraction": radial_fraction,
            "look_angular0": torch.stack(angular0),
            "look_angular1": torch.stack(angular1),
            "look_angular_fraction": torch.stack(angular_fraction),
            "look_valid": valid,
        }

    def pose_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"tokens must have shape (B,N,{self.embed_dim}), got {tuple(tokens.shape)}"
            )
        grouped = tokens.reshape(
            tokens.shape[0],
            tokens.shape[1],
            self.num_heads,
            self.pairs_per_head,
            2,
        )
        score = torch.einsum("bnhpc,ghapc->bngha", grouped, self.axis_weight)
        score = score + self.axis_bias[None, None]
        null = self.null_score[None, None, :, :, None].expand(
            tokens.shape[0], tokens.shape[1], -1, -1, -1
        )
        return torch.cat((score, null), dim=-1).float().softmax(-1)[..., :-1].to(
            tokens.dtype
        )

    def pose_for_layer(
        self, pose_weights: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        if not 0 <= layer_index < self.depth:
            raise ValueError("layer_index is outside the configured depth")
        group = layer_index // self.layers_per_probe
        return pose_weights[:, :, group]

    def _transformed_fields(self, layer_index: int) -> torch.Tensor:
        flat = self.look_grid[layer_index].flatten(-2)

        def gather(radial: torch.Tensor, angular: torch.Tensor) -> torch.Tensor:
            index = radial[None] * self.direction_bins + angular
            return flat[:, index]

        rw = self.look_radial_fraction[None]
        aw = self.look_angular_fraction
        sampled = (
            gather(self.look_radial0, self.look_angular0) * (1.0 - rw) * (1.0 - aw)
            + gather(self.look_radial0, self.look_angular1) * (1.0 - rw) * aw
            + gather(self.look_radial1, self.look_angular0) * rw * (1.0 - aw)
            + gather(self.look_radial1, self.look_angular1) * rw * aw
        )
        return sampled * self.look_valid[None, None].to(sampled.dtype)

    def fields(self, layer_index: int, *, dtype: torch.dtype) -> torch.Tensor:
        if not 0 <= layer_index < self.depth:
            raise ValueError("layer_index is outside the configured depth")
        return self._transformed_fields(layer_index).to(dtype=dtype).contiguous()

    def diagnostics(self) -> dict[str, object]:
        grid = self.look_grid.detach().float().cpu()
        return {
            "axes": self.axes,
            "directed_angles": 2 * self.axes,
            "pairs_per_head": self.pairs_per_head,
            "layers_per_probe": self.layers_per_probe,
            "probe_groups": self.probe_groups,
            "look_grid_shape": list(grid.shape),
            "layer_mean_abs_grid": grid.abs().mean(dim=(1, 2, 3)).tolist(),
            "null_score": self.null_score.detach().float().cpu().tolist(),
        }
