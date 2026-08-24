from __future__ import annotations

import torch
import torch.nn as nn

from utils.hex_graph import hex_relative_bins


class HexLookBias(nn.Module):
    """Learned directed relative bias indexed by graph ring and coarse direction.

    The table is deliberately coarse: each head owns one value for each
    ``(ring, one-of-six-directions)`` pair. Exact angle/distance rules are left
    for later rule extraction rather than encoded into this first experiment.
    """

    def __init__(self, coordinates: torch.Tensor, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        rings, directions = hex_relative_bins(coordinates.to(torch.float32))
        self.register_buffer("rings", rings, persistent=True)
        self.register_buffer("directions", directions, persistent=True)
        self.num_heads = int(num_heads)
        self.num_rings = int(rings.max().item()) + 1
        self.weight = nn.Parameter(torch.zeros(self.num_heads, self.num_rings, 6))

    def patch_bias(self) -> torch.Tensor:
        """Return directed patch-to-patch attention bias shaped ``(H, N, N)``."""
        head = torch.arange(self.num_heads, device=self.weight.device)[:, None, None]
        return self.weight[head, self.rings[None], self.directions[None]]

    def forward(self, *, include_cls: bool = True) -> torch.Tensor:
        patch_bias = self.patch_bias()
        if not include_cls:
            return patch_bias
        num_patches = patch_bias.shape[-1]
        result = patch_bias.new_zeros(self.num_heads, num_patches + 1, num_patches + 1)
        result[:, 1:, 1:] = patch_bias
        return result

