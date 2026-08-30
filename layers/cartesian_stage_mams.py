from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.cartesian_rotating_harmonic_conv import CartesianCircularPatchGeometry
from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat


class CartesianStageMAMSRouting2d(nn.Module):
    """Match one large-support geometry bank once for several stage blocks.

    The expensive pose/scale probabilities are shared. Every consumer keeps
    independent A/B/Vscale values, so sharing the route does not force the
    stage's residual blocks to receive the same geometric correction.
    """

    def __init__(
        self,
        in_channels: int,
        route_channels: int,
        *,
        consumers: int,
        diameters: tuple[int, ...],
        stride: int,
        directions: int = 4,
        global_directions: int = 8,
        angular_bins_per_radius: int = 4,
        prototype_chunk_size: int = 16,
        null_initial_score: float = 0.0,
    ) -> None:
        super().__init__()
        if route_channels <= 0 or route_channels % 2:
            raise ValueError("route_channels must be a positive even number")
        if consumers <= 0 or not diameters:
            raise ValueError("consumers and diameters must be non-empty")
        if not 1 <= directions <= global_directions:
            raise ValueError("directions must be in [1, global_directions]")

        self.in_channels = int(in_channels)
        self.route_channels = int(route_channels)
        self.bases = self.route_channels // 2
        self.consumers = int(consumers)
        self.directions = int(directions)
        self.global_directions = int(global_directions)
        self.scales = len(diameters)
        self.stride = int(stride)
        self.prototype_chunk_size = int(prototype_chunk_size)

        self.geometries = nn.ModuleList(
            CartesianCircularPatchGeometry(in_channels, diameter, stride)
            for diameter in diameters
        )
        radial_bins = max(diameters) // 2
        counts = torch.tensor(
            [angular_bins_per_radius * (radius + 1) for radius in range(radial_bins)],
            dtype=torch.long,
        )
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.register_buffer("ring_counts", counts, persistent=False)
        self.register_buffer("ring_offsets", offsets, persistent=False)
        direction_step = 2.0 * math.pi / global_directions
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

        prototype_std = 1.0 / math.sqrt(in_channels * int(offsets[-1]))
        self.prototype = nn.Parameter(
            torch.randn(self.bases, in_channels, int(offsets[-1])) * prototype_std
        )
        self.null_score = nn.Parameter(
            torch.full((self.bases,), float(null_initial_score))
        )
        self.direction_value = nn.Parameter(
            torch.zeros(self.consumers, self.bases, 2, 2)
        )
        self.scale_value = nn.Parameter(
            torch.empty(self.consumers, self.bases, self.scales, 2)
        )
        with torch.no_grad():
            self.direction_value[:, :, 0, 0] = 1.0
            self.direction_value[:, :, 1, 1] = 1.0
        nn.init.trunc_normal_(self.scale_value, std=0.02)

        reference_mass = self.renderers[0].support_cover.sum()
        for index, renderer in enumerate(self.renderers):
            cover = renderer.support_cover
            self.register_buffer(
                f"scale_cover_{index}",
                cover * (reference_mass / cover.sum()),
                persistent=False,
            )
        theta = torch.arange(directions) * direction_step
        self.register_buffer(
            "direction_coefficients",
            torch.stack((theta.cos(), theta.sin()), dim=-1),
            persistent=False,
        )

    def _chunk_output(
        self,
        patches: list[torch.Tensor],
        rendered: list[torch.Tensor],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        score = torch.stack(
            [
                rotating_dot_score(patch, weight[start:stop])
                for patch, weight in zip(patches, rendered, strict=True)
            ],
            dim=3,
        ).float()  # B,N,P,S,D
        flat = score.flatten(3, 4)
        null = self.null_score[start:stop].float()[None, None, :, None]
        probability = torch.cat(
            (flat, null.expand(flat.shape[0], flat.shape[1], -1, -1)), dim=-1
        ).softmax(-1)[..., :-1].view_as(score)

        coefficients = self.direction_coefficients.float()
        direction_value = self.direction_value[:, start:stop].float()
        scale_value = self.scale_value[:, start:stop].float()
        pose_value = torch.einsum(
            "dk,lpkv->lpdv", coefficients, direction_value
        )[:, :, None] + scale_value[:, :, :, None]
        output = torch.einsum("bnpsd,lpsdv->lbnpv", probability, pose_value)
        return output.to(patches[0].dtype).flatten(-2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        output_size = self.geometries[0].output_size(height, width)
        if any(
            geometry.output_size(height, width) != output_size
            for geometry in self.geometries[1:]
        ):
            raise RuntimeError("all stage-routing scales must share output centers")
        patches = []
        for index, geometry in enumerate(self.geometries):
            patch = geometry(image)
            cover = getattr(self, f"scale_cover_{index}").to(patch.dtype)
            patches.append(weighted_patch_flat(patch, cover))
        rendered = [renderer(self.prototype) for renderer in self.renderers]
        routed = torch.cat(
            [
                self._chunk_output(
                    patches,
                    rendered,
                    start,
                    min(start + self.prototype_chunk_size, self.bases),
                )
                for start in range(0, self.bases, self.prototype_chunk_size)
            ],
            dim=-1,
        )
        return routed.permute(0, 1, 3, 2).reshape(
            self.consumers,
            image.shape[0],
            self.route_channels,
            *output_size,
        ).contiguous()

    def extra_repr(self) -> str:
        diameters = tuple(geometry.diameter for geometry in self.geometries)
        return (
            f"{self.in_channels}, route_channels={self.route_channels}, "
            f"consumers={self.consumers}, diameters={diameters}, "
            f"directions={self.directions}/{self.global_directions}"
        )
