from __future__ import annotations

import torch

from .distance_norm_projection import DistanceNormLinearProjection


class DistanceNormExpLinearProjection(DistanceNormLinearProjection):
    """Squared distance -> per-token basis LayerNorm -> exp2 -> Linear."""

    def basis(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            distance = self.distances(image.float())
            relative_similarity = self.norm(
                (-distance * torch.exp2(self.log2_scale)[None, :, None, None]).permute(0, 2, 3, 1)
            )
            return torch.exp2(relative_similarity).permute(0, 3, 1, 2)

