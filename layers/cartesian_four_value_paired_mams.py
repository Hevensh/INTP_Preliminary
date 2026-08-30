from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.cartesian_rotating_harmonic_conv import CartesianCircularPatchGeometry
from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat
from layers.triton_polar_renderer import triton_polar_render


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


class PairedRMSNorm2d(nn.Module):
    """Per-location RMS normalization with one gain shared by each C/S pair."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0 or channels % 2:
            raise ValueError("paired RMS normalization requires even channels")
        self.channels = int(channels)
        self.pairs = self.channels // 2
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.pairs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W], got {tuple(x.shape)}")
        pair = x.reshape(x.shape[0], self.pairs, 2, *x.shape[-2:])
        rms = pair.float().square().mean(dim=(1, 2), keepdim=True)
        rms = (rms + self.eps).rsqrt().to(pair.dtype)
        gain = self.weight.to(pair.dtype)[None, :, None, None, None]
        return (pair * rms * gain).reshape_as(x)


class ChannelRMSNorm2d(nn.Module):
    """LayerNorm-like per-location normalization without mean subtraction."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.float().square().mean(dim=1, keepdim=True) + self.eps).rsqrt()
        gain = self.weight.to(x.dtype)[None, :, None, None]
        return x * rms.to(x.dtype) * gain


class ComplexPointwiseConv2d(nn.Module):
    """A 1x1 complex linear map that commutes with rotations of C/S pairs."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if min(in_channels, out_channels, stride) <= 0:
            raise ValueError("channel counts and stride must be positive")
        if in_channels % 2 or out_channels % 2:
            raise ValueError("complex pointwise convolution requires even channels")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.real = nn.Parameter(torch.empty(out_channels // 2, in_channels // 2, 1, 1))
        self.imag = nn.Parameter(torch.empty(out_channels // 2, in_channels // 2, 1, 1))
        nn.init.kaiming_normal_(self.real, mode="fan_out", nonlinearity="linear")
        nn.init.kaiming_normal_(self.imag, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0::2]
        imag = x[:, 1::2]
        grouped_input = torch.cat((real, imag), dim=1)
        # [R -I; I R] is the real block representation of one complex map.
        # Building it is tiny compared with launching four separate convolutions.
        block_weight = torch.cat(
            (
                torch.cat((self.real, -self.imag), dim=1),
                torch.cat((self.imag, self.real), dim=1),
            ),
            dim=0,
        )
        grouped_output = F.conv2d(
            grouped_input,
            block_weight,
            stride=self.stride,
        )
        real_out, imag_out = grouped_output.chunk(2, dim=1)
        out = torch.stack((real_out, imag_out), dim=2)
        return out.flatten(1, 2)


class CartesianFourValuePairedMAMSConv2d(nn.Module):
    """Four-value A/B/Vscale MAMS with optional paired input prototypes.

    Each base produces one output C/S pair through four learned 2-D values:
    direction A, direction B, large-scale value, and small-scale value.
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
        prototype_chunk_size: int = 256,
        paired_input: bool = True,
        prototype_std: float | None = None,
        null_initial_score: float = 0.0,
    ) -> None:
        super().__init__()
        if len(diameters) != 2:
            raise ValueError("the four-value experiment currently requires two scales")
        if out_channels % 2:
            raise ValueError("out_channels must be even")
        if paired_input and in_channels % 2:
            raise ValueError("paired input requires even in_channels")
        if not 1 <= directions <= global_directions:
            raise ValueError("directions must be in [1, global_directions]")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.bases = self.out_channels // 2
        self.directions = int(directions)
        self.global_directions = int(global_directions)
        self.stride = int(stride)
        self.scales = len(diameters)
        self.prototype_chunk_size = int(prototype_chunk_size)
        self.paired_input = bool(paired_input)

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

        stored_channels = in_channels // 2 if paired_input else in_channels
        if prototype_std is None:
            prototype_std = 1.0 / math.sqrt(in_channels * int(offsets[-1]))
        if paired_input:
            real = torch.randn(self.bases, stored_channels, int(offsets[-1])) * prototype_std
            imag = torch.randn_like(real) * prototype_std
            radius = real.square().add(imag.square()).sqrt().clamp_min(1e-6)
            self.prototype_radius_raw = nn.Parameter(_inverse_softplus(radius))
            self.prototype_theta_offset = nn.Parameter(torch.atan2(imag, real))
        else:
            self.prototype = nn.Parameter(
                torch.randn(self.bases, stored_channels, int(offsets[-1])) * prototype_std
            )

        self.null_score = nn.Parameter(
            torch.full((self.bases,), float(null_initial_score))
        )
        self.direction_value = nn.Parameter(torch.zeros(self.bases, 2, 2))
        self.scale_value = nn.Parameter(torch.empty(self.bases, self.scales, 2))
        with torch.no_grad():
            self.direction_value[:, 0, 0] = 1.0
            self.direction_value[:, 1, 1] = 1.0
        nn.init.trunc_normal_(self.scale_value, std=0.02)

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

    def _render_all(self) -> list[torch.Tensor]:
        if not self.paired_input:
            return [renderer(self.prototype) for renderer in self.renderers]

        radius = F.softplus(self.prototype_radius_raw)
        base_real = radius * self.prototype_theta_offset.cos()
        base_imag = radius * self.prototype_theta_offset.sin()
        cosine = self.direction_coefficients[:, 0][None, :, None, None]
        sine = self.direction_coefficients[:, 1][None, :, None, None]
        rendered = []
        for renderer in self.renderers:
            # One renderer invocation handles both members of every channel
            # pair; interpolation is linear, so this is exactly equivalent to
            # two independent calls.
            pair = triton_polar_render(
                torch.cat((base_real, base_imag), dim=1), renderer
            )
            real, imag = pair.split(self.in_channels // 2, dim=2)
            weight_real = real * cosine - imag * sine
            weight_imag = real * sine + imag * cosine
            weight = torch.stack((weight_real, weight_imag), dim=3).flatten(2, 3)
            rendered.append(weight)
        return rendered

    def _chunk_output(
        self,
        patches: list[torch.Tensor],
        rendered: list[torch.Tensor],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        scale_scores = [
            rotating_dot_score(patch, weight[start:stop])
            for patch, weight in zip(patches, rendered, strict=True)
        ]
        score = torch.stack(scale_scores, dim=3).float()  # B,N,P,S,D
        flat = score.flatten(3, 4)
        null = self.null_score[start:stop].float()[None, None, :, None]
        null = null.expand(flat.shape[0], flat.shape[1], -1, -1)
        probability = torch.cat((flat, null), dim=-1).softmax(-1)[..., :-1]
        probability = probability.view_as(score)

        coefficients = self.direction_coefficients.float()
        direction_value = self.direction_value[start:stop].float()
        scale_value = self.scale_value[start:stop].float()
        # Algebraically fold A/B/Vscale into the value attached to each pose:
        # V[p,s,d] = cos(d) A[p] + sin(d) B[p] + Vscale[p,s].
        pose_value = torch.einsum(
            "dk,pkv->pdv", coefficients, direction_value
        )[:, None] + scale_value[:, :, None]
        # The einsum keeps [B,N,P] in its existing order. A P-major bmm would
        # first materialize a large permuted copy of every routing probability.
        output = torch.einsum("bnpsd,psdv->bnpv", probability, pose_value)
        return output.to(patches[0].dtype).flatten(2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        output_size = self.geometries[0].output_size(height, width)
        # F.unfold materializes two compact scale tensors, but its col2im
        # backward is substantially faster on T4 than the advanced-index
        # backward of a shared strided window view.
        raw_patches = [geometry(image) for geometry in self.geometries]
        patches = []
        for index, patch in enumerate(raw_patches):
            cover = getattr(self, f"scale_cover_{index}").to(patch.dtype)
            patches.append(weighted_patch_flat(patch, cover))
        rendered = self._render_all()
        output = torch.cat(
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
        return output.transpose(1, 2).reshape(
            image.shape[0], self.out_channels, *output_size
        ).contiguous()

    def extra_repr(self) -> str:
        diameters = tuple(geometry.diameter for geometry in self.geometries)
        return (
            f"{self.in_channels}, {self.out_channels}, diameters={diameters}, "
            f"stride={self.stride}, directions={self.directions}/"
            f"{self.global_directions}, paired_input={self.paired_input}"
        )
