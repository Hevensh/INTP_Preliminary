from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.hex_patch_geometry import HexPatchGeometry
from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat


class HexRotatingGroupedDotPatchEmbed(nn.Module):
    """Three 64-D rotating dot-product groups concatenated into a ViT token."""

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
        groups: int = 3,
        radial_bins: int = 12,
        angular_bins_per_radius: int = 4,
        prototype_chunk_size: int = 16,
        prototype_std: float = 0.02,
        use_null: bool = True,
        null_initial_score: float = 0.0,
        compensate_small_scales: bool = False,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        if bases % groups or embed_dim % groups:
            raise ValueError("bases and embed_dim must be divisible by groups")
        if not 1 <= directions <= global_directions:
            raise ValueError("directions must be in [1, global_directions]")
        if prototype_chunk_size <= 0 or prototype_std <= 0:
            raise ValueError("prototype_chunk_size and prototype_std must be positive")
        self.embed_dim = int(embed_dim)
        self.bases = int(bases)
        self.directions = int(directions)
        self.scales = len(kernel_sizes)
        self.groups = int(groups)
        self.bases_per_group = bases // groups
        self.group_dim = embed_dim // groups
        self.prototype_chunk_size = int(prototype_chunk_size)
        self.use_null = bool(use_null)
        self.compensate_small_scales = bool(compensate_small_scales)

        self.geometries = nn.ModuleList(
            HexPatchGeometry(img_size, in_chans, int(kernel), lattice_stride)
            for kernel in kernel_sizes
        )
        if len({geometry.num_patches for geometry in self.geometries}) != 1:
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
        self.null_score = nn.Parameter(
            torch.full((bases,), float(null_initial_score))
        )
        self.direction_pair = nn.Parameter(
            torch.empty(bases, 2, self.group_dim)
        )
        self.scale_value = nn.Parameter(
            torch.empty(bases, self.scales, self.group_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.direction_pair, std=0.02)
        nn.init.trunc_normal_(self.scale_value, std=0.02)

        scale_covers = []
        reference_cover_mass = self.renderers[0].support_cover.sum()
        for renderer in self.renderers:
            raw_cover = renderer.support_cover
            if self.compensate_small_scales:
                cover = raw_cover * (reference_cover_mass / raw_cover.sum())
            else:
                cover = raw_cover / raw_cover.sum()
            scale_covers.append(cover)
        for index, cover in enumerate(scale_covers):
            # The legacy path uses unit-mass covers.  The compensated ablation
            # preserves K24's convolution-like accumulation and lifts smaller
            # scales to the same cover mass.
            self.register_buffer(f"scale_cover_{index}", cover, persistent=False)

        theta = torch.arange(directions) * direction_step
        self.register_buffer(
            "direction_coefficients",
            torch.stack((theta.cos(), theta.sin()), dim=-1),
            persistent=False,
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

    def _chunk_probabilities(
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
            scale_scores.append(score)
        scores = torch.stack(scale_scores, dim=3)  # B, N, P, S, D
        flat_scores = scores.flatten(3, 4)
        if not self.use_null:
            return flat_scores.softmax(-1).view_as(scores)
        null = self.null_score[start:stop][None, None, :, None].expand(
            flat_scores.shape[0], flat_scores.shape[1], -1, -1
        )
        return torch.cat((flat_scores, null), -1).softmax(-1)[..., :-1].view_as(scores)

    def _chunk_output(
        self,
        probability: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        cosine_mass = torch.einsum(
            "qnpsd,d->qnp", probability, self.direction_coefficients[:, 0]
        )
        sine_mass = torch.einsum(
            "qnpsd,d->qnp", probability, self.direction_coefficients[:, 1]
        )
        scale_mass = probability.sum(-1)
        pair = self.direction_pair[start:stop]
        output = torch.einsum("qnp,pc->qnc", cosine_mass, pair[:, 0])
        output = output + torch.einsum("qnp,pc->qnc", sine_mass, pair[:, 1])
        return output + torch.einsum(
            "qnps,psc->qnc", scale_mass, self.scale_value[start:stop]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            patches = [
                weighted_patch_flat(
                    geometry(image), getattr(self, f"scale_cover_{scale_index}")
                )
                for scale_index, geometry in enumerate(self.geometries)
            ]
            group_outputs = []
            for group in range(self.groups):
                group_start = group * self.bases_per_group
                group_stop = group_start + self.bases_per_group
                output = None
                for start in range(group_start, group_stop, self.prototype_chunk_size):
                    stop = min(start + self.prototype_chunk_size, group_stop)
                    probability = self._chunk_probabilities(patches, start, stop)
                    chunk = self._chunk_output(probability, start, stop)
                    output = chunk if output is None else output + chunk
                group_outputs.append(output)
            return torch.cat(group_outputs, dim=-1) + self.output_bias
