from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Sim2cooManifoldLayer(nn.Module):
    """
    Linear-like layer (no bias) that embeds HexConv similarities into a manifold space.

    Input:  similarity shaped (..., C)
    Params: embed matrix, shaped (C, D) (initialized from N(0,1) by default)
    Output: manifold embedding, shaped (..., D)

    Stabilized scaling:
      - forward: multiply by (1 / embed_dim)
      - backward: multiply gradients by (1 / in_channels)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)

        self.weight = nn.Parameter(torch.randn(self.in_channels, self.embed_dim))
        # self.inv_embed_dim = 1.0 / self.embed_dim ** 0.5
        # self.inv_in_channels = 1.0 / self.in_channels ** 0.5
        self.inv_embed_dim = 1.0 
        self.inv_in_channels = 1.0

        class Sim2cooManifoldFunc(Function):
            @staticmethod
            def forward(ctx, similarity: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                ctx.save_for_backward(similarity, weight)
                return similarity.matmul(weight) * self.inv_embed_dim

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
                similarity, weight = ctx.saved_tensors
                scale = self.inv_embed_dim * self.inv_in_channels

                grad_similarity = grad_output.matmul(weight.transpose(-1, -2)) * scale

                sim2 = similarity.reshape(-1, similarity.shape[-1])
                go2 = grad_output.reshape(-1, grad_output.shape[-1])
                grad_weight = sim2.transpose(-1, -2).matmul(go2) * scale

                return grad_similarity, grad_weight

        self.calc_func = Sim2cooManifoldFunc

    def forward(self, similarity: torch.Tensor) -> torch.Tensor:
        return self.calc_func.apply(similarity, self.weight)

    @torch.no_grad()
    def reset_filters_(self, idx: torch.Tensor) -> None:
        """
        Reset (re-initialize) selected input-channel embeddings (rows of weight) in-place.

        idx:
          - LongTensor indices shaped (k,) or scalar
          - BoolTensor mask shaped (in_channels,)
        Resets:
          - self.weight[idx] ~ Normal(0, 1)
        """
        if not torch.is_tensor(idx):
            raise TypeError("idx must be a torch.Tensor")

        if idx.dtype == torch.bool:
            if idx.ndim != 1 or idx.numel() != self.in_channels:
                raise ValueError(f"bool mask idx must have shape ({self.in_channels},)")
            idx_ = idx.nonzero(as_tuple=False).flatten()
        else:
            idx_ = idx.to(dtype=torch.long).flatten()

        if idx_.numel() == 0:
            return

        idx_ = idx_.to(device=self.weight.device)
        if int(idx_.min().item()) < 0 or int(idx_.max().item()) >= self.in_channels:
            raise ValueError(f"idx out of range [0, {self.in_channels - 1}]")

        nn.init.normal_(self.weight[idx_], mean=0.0, std=1.0)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, embed_dim={self.embed_dim}"
        )


# Backwards-friendly alias
Sim2cooManifoldLinear = Sim2cooManifoldLayer
