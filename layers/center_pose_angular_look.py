from __future__ import annotations

import math

import torch
import torch.nn as nn


class CenterPoseAngularLook(nn.Module):
    """Turn tokenizer cosine/sine pairs into one shared angular Look field.

    The tokenizer already emits one cosine/sine pair for every geometric
    prototype.  This module keeps those pairs intact, assigns an equal group
    to each attention head, and predicts six *axes* from the centre token.
    Each axis is rendered to both opposite directed angles, so the field spans
    the full circle without pretending that a centre-only orientation cue can
    distinguish forward from backward.  Radial distance is intentionally not
    represented: every Transformer layer reuses the same pose probabilities
    and only learns a small head/axis gain.
    """

    def __init__(
        self,
        *,
        coordinates: torch.Tensor,
        embed_dim: int,
        num_heads: int,
        depth: int,
        axes: int,
        null_initial_score: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if min(embed_dim, num_heads, depth, axes) <= 0:
            raise ValueError("embed_dim, num_heads, depth, and axes must be positive")
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
        self.pairs_per_head = pair_count // num_heads

        # A tiny grouped router: each head only consumes its own complete
        # cosine/sine pairs.  Independent axis rows permit several axes to be
        # active at once; no 192x192 projection is introduced.
        self.axis_weight = nn.Parameter(
            torch.randn(num_heads, axes, self.pairs_per_head, 2)
            / math.sqrt(2 * self.pairs_per_head)
        )
        self.axis_bias = nn.Parameter(torch.zeros(num_heads, axes))
        self.null_score = nn.Parameter(
            torch.full((num_heads,), float(null_initial_score))
        )
        self.layer_axis_gain = nn.Parameter(
            torch.full((depth, num_heads, axes), float(gate_init))
        )

        self.register_buffer(
            "angular_fields",
            self._angular_fields(coordinates, axes),
            persistent=True,
        )

    @staticmethod
    def _angular_fields(coordinates: torch.Tensor, axes: int) -> torch.Tensor:
        """Render soft pi-periodic angular bins with no radial coordinate."""

        # [query, key] = key - query.  Taking modulo pi ties opposite viewing
        # directions into the same centre-observable orientation axis.
        relative = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
        distance = relative.norm(dim=-1)
        angle = torch.remainder(
            torch.atan2(relative[..., 1], relative[..., 0]), math.pi
        )
        position = angle / math.pi * axes
        lower = position.floor().long() % axes
        upper = (lower + 1) % axes
        fraction = position - position.floor()
        valid = distance > 0

        fields = torch.zeros(
            axes,
            coordinates.shape[0],
            coordinates.shape[0],
            dtype=coordinates.dtype,
        )
        fields.scatter_add_(
            0,
            lower.unsqueeze(0),
            ((1.0 - fraction) * valid).unsqueeze(0),
        )
        fields.scatter_add_(
            0,
            upper.unsqueeze(0),
            (fraction * valid).unsqueeze(0),
        )
        return fields

    def pose_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return real-axis probabilities as ``(B,N,H,A)``.

        The omitted final null probability suppresses total Look mass without
        renormalising the real axes back to one.
        """

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
        score = torch.einsum("bnhpc,hapc->bnha", grouped, self.axis_weight)
        score = score + self.axis_bias[None, None]
        null = self.null_score[None, None, :, None].expand(
            tokens.shape[0], tokens.shape[1], -1, -1
        )
        return torch.cat((score, null), dim=-1).float().softmax(-1)[..., :-1].to(
            tokens.dtype
        )

    def fields(self, layer_index: int, *, dtype: torch.dtype) -> torch.Tensor:
        if not 0 <= layer_index < self.depth:
            raise ValueError("layer_index is outside the configured depth")
        fields = self.angular_fields.to(dtype=dtype)
        return (
            self.layer_axis_gain[layer_index, :, :, None, None].to(dtype)
            * fields[None]
        ).contiguous()

    def diagnostics(self) -> dict[str, object]:
        gain = self.layer_axis_gain.detach().float().cpu()
        return {
            "axes": self.axes,
            "directed_angles": 2 * self.axes,
            "pairs_per_head": self.pairs_per_head,
            "layer_axis_gain": gain.tolist(),
            "layer_mean_abs_gain": gain.abs().mean(dim=(1, 2)).tolist(),
            "null_score": self.null_score.detach().float().cpu().tolist(),
        }
