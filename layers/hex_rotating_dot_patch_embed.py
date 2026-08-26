from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.hex_patch_geometry import HexPatchGeometry
from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat


class HexRotatingDotPatchEmbed(nn.Module):
    """Rotating polar dot-product bank followed by a small channel projection.

    Each prototype is softly pooled over scale and rotation into one scalar.
    A single 96 -> embed_dim projection then plays the role of a 1x1
    convolution.  No pose V, global score normalization, or exponential
    response gate is used.
    """

    def __init__(
        self,
        *,
        img_size: int,
        in_chans: int,
        embed_dim: int,
        lattice_stride: int = 18,
        kernel_sizes: tuple[int, ...] = (24, 12),
        bases: int = 96,
        directions: int = 4,
        global_directions: int = 8,
        radial_bins: int = 12,
        angular_bins_per_radius: int = 4,
        prototype_chunk_size: int = 16,
        prototype_std: float = 0.04,
        use_null: bool = True,
        null_initial_score: float = -1.0,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        if not 1 <= directions <= global_directions:
            raise ValueError("directions must be in [1, global_directions]")
        if prototype_chunk_size <= 0 or prototype_std <= 0:
            raise ValueError("prototype_chunk_size and prototype_std must be positive")
        self.embed_dim = int(embed_dim)
        self.bases = int(bases)
        self.directions = int(directions)
        self.scales = len(kernel_sizes)
        self.prototype_chunk_size = int(prototype_chunk_size)
        self.prototype_std = float(prototype_std)
        self.use_null = bool(use_null)

        self.geometries = nn.ModuleList(
            HexPatchGeometry(img_size, in_chans, int(kernel), lattice_stride)
            for kernel in kernel_sizes
        )
        patch_counts = {geometry.num_patches for geometry in self.geometries}
        if len(patch_counts) != 1:
            raise ValueError("all scales must produce the same Hex patch centers")
        reference_centers = self.geometries[0].patch_centers_xy
        if any(
            not torch.equal(reference_centers, geometry.patch_centers_xy)
            for geometry in self.geometries[1:]
        ):
            raise ValueError("all scales must share identical Hex patch centers")

        counts = torch.tensor(
            [angular_bins_per_radius * (radius + 1) for radius in range(radial_bins)],
            dtype=torch.long,
        )
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.register_buffer("ring_counts", counts, persistent=False)
        self.register_buffer("ring_offsets", offsets, persistent=False)
        direction_step = 2 * math.pi / global_directions
        self.renderers = nn.ModuleList(
            _PolarRenderer(
                geometry,
                radial_bins=radial_bins,
                ring_counts=counts,
                ring_offsets=offsets,
                directions=directions,
                direction_step=direction_step,
            )
            for geometry in self.geometries
        )

        self.prototype = nn.Parameter(
            torch.randn(bases, in_chans, int(offsets[-1])) * prototype_std
        )
        self.logit_gain = nn.Parameter(torch.zeros(bases, self.scales))
        self.null_score = nn.Parameter(
            torch.full((bases,), float(null_initial_score))
        )
        self.projection = nn.Linear(bases, embed_dim)

        # At initialization, weighted dot products at every scale have roughly
        # unit standard deviation.  Prototype norms remain free to learn.
        multipliers = []
        for renderer in self.renderers:
            effective_fan_in = float(in_chans * renderer.support_cover.square().sum())
            multipliers.append(1.0 / (math.sqrt(effective_fan_in) * prototype_std))
        self.register_buffer(
            "dot_multipliers", torch.tensor(multipliers), persistent=False
        )

    @property
    def num_patches(self) -> int:
        return self.geometries[0].num_patches

    @property
    def patch_centers_xy(self) -> torch.Tensor:
        return self.geometries[0].patch_centers_xy

    @property
    def coo_patchs(self) -> torch.Tensor:
        return self.geometries[0].coo_patchs

    def _chunk_features(
        self,
        patches: list[torch.Tensor],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        prototype = self.prototype[start:stop]
        scale_scores = []
        for scale_index, (patch, renderer) in enumerate(zip(patches, self.renderers)):
            rendered = renderer(prototype)
            score = rotating_dot_score(patch, rendered)
            score = score * self.dot_multipliers[scale_index]
            score = score * torch.exp(
                self.logit_gain[start:stop, scale_index]
            )[None, None, :, None]
            scale_scores.append(score)
        scores = torch.stack(scale_scores, dim=3)  # B, N, P, S, D
        flat_scores = scores.flatten(3, 4)
        if self.use_null:
            null = self.null_score[start:stop][None, None, :, None].expand(
                flat_scores.shape[0], flat_scores.shape[1], -1, -1
            )
            probability = torch.cat((flat_scores, null), -1).softmax(-1)[..., :-1]
        else:
            probability = flat_scores.softmax(-1)
        return (probability * flat_scores).sum(-1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            patches = [
                weighted_patch_flat(geometry(image), renderer.support_cover)
                for geometry, renderer in zip(self.geometries, self.renderers)
            ]
            features = [
                self._chunk_features(
                    patches, start, min(start + self.prototype_chunk_size, self.bases)
                )
                for start in range(0, self.bases, self.prototype_chunk_size)
            ]
            return self.projection(torch.cat(features, dim=-1))
