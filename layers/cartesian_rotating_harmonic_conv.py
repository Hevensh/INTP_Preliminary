from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat


class CartesianCircularPatchGeometry(nn.Module):
    """Dense circular patches on an ordinary Cartesian feature map.

    Unlike :class:`HexPatchGeometry`, centers remain on the input feature grid.
    Odd kernels with symmetric reflection padding reproduce the spatial size for
    stride 1 and the ordinary ResNet downsampling size for stride 2.
    """

    def __init__(self, in_chans: int, diameter: int, stride: int) -> None:
        super().__init__()
        if min(in_chans, diameter, stride) <= 0:
            raise ValueError("in_chans, diameter, and stride must be positive")
        self.in_chans = int(in_chans)
        # _PolarRenderer uses kernel_size as the physical diameter. Sampling
        # itself uses an odd, pixel-centred bounding window so even diameters do
        # not introduce the half-pixel shift of an ordinary even Conv2d kernel.
        self.kernel_size = int(diameter)
        self.diameter = int(diameter)
        self.stride = int(stride)
        self.padding = self.diameter // 2
        self.window_size = 2 * self.padding + 1

        coordinate = torch.arange(self.window_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        center = float(self.padding)
        dx = xx - center
        dy = yy - center
        support = (dx.square() + dy.square()).sqrt() < self.diameter / 2
        self.register_buffer("support_mask", support.flatten(), persistent=False)
        self.register_buffer(
            "patch_offsets_xy",
            torch.stack((dx[support], dy[support]), dim=-1),
            persistent=False,
        )

    @property
    def num_samples(self) -> int:
        return int(self.support_mask.sum())

    def output_size(self, height: int, width: int) -> tuple[int, int]:
        return (
            (height + 2 * self.padding - self.window_size) // self.stride + 1,
            (width + 2 * self.padding - self.window_size) // self.stride + 1,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != self.in_chans:
            raise ValueError(
                f"expected image (B,{self.in_chans},H,W), got {tuple(image.shape)}"
            )
        if self.padding:
            mode = (
                "reflect"
                if min(image.shape[-2:]) > self.padding
                else "replicate"
            )
            image = F.pad(image, (self.padding,) * 4, mode=mode)
        square = F.unfold(image, kernel_size=self.window_size, stride=self.stride)
        batch, _, positions = square.shape
        square = square.view(
            batch, self.in_chans, self.window_size * self.window_size, positions
        )
        return square[:, :, self.support_mask].permute(0, 3, 1, 2).contiguous()


class CartesianRotatingHarmonicConv2d(nn.Module):
    """Multi-angle/multi-scale null-softmax cos/sin convolution.

    One shared polar prototype is rendered at every requested scale and pose.
    The scale scores are accumulated before a direction-plus-null softmax, then
    projected to the first circular moment. Two output channels are produced per
    prototype, so ``out_channels`` must be even.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        diameters: tuple[int, ...] = (6, 3),
        stride: int = 1,
        directions: int = 4,
        global_directions: int = 8,
        angular_bins_per_radius: int = 4,
        prototype_chunk_size: int = 16,
        prototype_std: float | None = None,
        use_null: bool = True,
        null_initial_score: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not diameters:
            raise ValueError("diameters must not be empty")
        if out_channels % 2:
            raise ValueError("out_channels must be even for cos/sin projection")
        if not 1 <= directions <= global_directions:
            raise ValueError("directions must be in [1, global_directions]")
        if min(angular_bins_per_radius, prototype_chunk_size) <= 0:
            raise ValueError("angular resolution and chunk size must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.bases = self.out_channels // 2
        self.directions = int(directions)
        self.global_directions = int(global_directions)
        self.stride = int(stride)
        self.scales = len(diameters)
        self.prototype_chunk_size = int(prototype_chunk_size)
        self.use_null = bool(use_null)

        self.geometries = nn.ModuleList(
            CartesianCircularPatchGeometry(in_channels, int(diameter), stride)
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
        if prototype_std is None:
            # Hidden-stage channel counts vary from 64 to 512. Fan-in scaling
            # prevents later-stage pose logits from becoming sharp merely
            # because a prototype reads more channels.
            prototype_std = 1.0 / math.sqrt(in_channels * int(offsets[-1]))
        if prototype_std <= 0:
            raise ValueError("prototype_std must be positive")
        self.prototype = nn.Parameter(
            torch.randn(self.bases, in_channels, int(offsets[-1])) * prototype_std
        )
        if self.use_null:
            self.null_score = nn.Parameter(
                torch.full((self.bases,), float(null_initial_score))
            )
        if bias:
            self.output_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("output_bias", None)

        reference_mass = self.renderers[0].support_cover.sum()
        for index, renderer in enumerate(self.renderers):
            raw = renderer.support_cover
            self.register_buffer(
                f"scale_cover_{index}",
                raw * (reference_mass / raw.sum()),
                persistent=False,
            )
        theta = torch.arange(directions) * direction_step
        self.register_buffer(
            "direction_coefficients",
            torch.stack((theta.cos(), theta.sin()), dim=-1),
            persistent=False,
        )

    def _chunk_response(
        self,
        patches: list[torch.Tensor],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        prototype = self.prototype[start:stop]
        score = None
        for patch, renderer in zip(patches, self.renderers):
            current = rotating_dot_score(patch, renderer(prototype))
            score = current if score is None else score + current
        score = score.float()
        if self.use_null:
            null = self.null_score[start:stop].float()[None, None, :, None]
            null = null.expand(score.shape[0], score.shape[1], -1, -1)
            probability = torch.cat((score, null), dim=-1).softmax(-1)[..., :-1]
        else:
            probability = score.softmax(-1)
        response = torch.matmul(probability, self.direction_coefficients.float())
        return response.to(patches[0].dtype).flatten(2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        output_size = self.geometries[0].output_size(height, width)
        if any(
            geometry.output_size(height, width) != output_size
            for geometry in self.geometries[1:]
        ):
            raise RuntimeError("all Cartesian scales must share output centers")
        patches = []
        for scale_index, geometry in enumerate(self.geometries):
            patch = geometry(image)
            cover = getattr(self, f"scale_cover_{scale_index}").to(patch.dtype)
            patches.append(weighted_patch_flat(patch, cover))
        response = torch.cat(
            [
                self._chunk_response(patches, start, min(start + self.prototype_chunk_size, self.bases))
                for start in range(0, self.bases, self.prototype_chunk_size)
            ],
            dim=-1,
        )
        if self.output_bias is not None:
            response = response + self.output_bias
        return response.transpose(1, 2).reshape(
            image.shape[0], self.out_channels, *output_size
        ).contiguous()

    def extra_repr(self) -> str:
        diameters = tuple(geometry.diameter for geometry in self.geometries)
        return (
            f"{self.in_channels}, {self.out_channels}, diameters={diameters}, "
            f"stride={self.stride}, directions={self.directions}/"
            f"{self.global_directions}, null={self.use_null}"
        )
