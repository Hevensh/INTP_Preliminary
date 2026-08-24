from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PolarScaleGeometry(nn.Module):
    def __init__(
        self, kernel_size, stride, radial_bins, counts, offsets, directions,
        cover_radius_scale=1.0, direction_step=None, in_channels=3,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.cover_radius_scale = float(cover_radius_scale)
        self.in_channels = int(in_channels)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.cover_radius_scale < 1.0:
            raise ValueError("cover_radius_scale must be at least 1")

        yy, xx = torch.meshgrid(
            torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij"
        )
        center = (kernel_size - 1) / 2
        dx, dy = xx - center, yy - center
        radius = torch.sqrt(dx.square() + dy.square())
        angle_turn = torch.remainder(torch.atan2(dy, dx), 2 * math.pi) / (2 * math.pi)
        support = radius < kernel_size / 2
        flat_support = support.flatten()
        support_radius = radius.flatten()[flat_support]
        support_angle = angle_turn.flatten()[flat_support]

        # Every rendered diameter spans the complete stored rho axis.
        radial = (
            support_radius * radial_bins / (kernel_size / 2) - .5
        ).clamp(0, radial_bins - 1)
        r0 = radial.floor().long()
        r1 = (r0 + 1).clamp_max(radial_bins - 1)

        def angular_indices(ring):
            count = counts[ring]
            step = (
                2 * math.pi / directions
                if direction_step is None else float(direction_step)
            )
            shifts = torch.arange(directions)[:, None] * step / (2 * math.pi)
            position = torch.remainder(support_angle[None] - shifts, 1.0) * count[None]
            raw_a0 = position.floor()
            fraction = position - raw_a0
            a0 = torch.remainder(raw_a0.long(), count[None])
            a1 = torch.remainder(a0 + 1, count[None])
            offset = offsets[ring][None]
            return offset + a0, offset + a1, fraction

        i00, i01, a0t = angular_indices(r0)
        i10, i11, a1t = angular_indices(r1)
        cover = torch.cos(
            radius * math.pi / (kernel_size * self.cover_radius_scale)
        ).clamp_min(0)
        support_cover = cover.flatten()[flat_support]

        self.register_buffer("support_mask", flat_support, persistent=False)
        self.register_buffer(
            "support_x", dx.flatten()[flat_support], persistent=False,
        )
        self.register_buffer(
            "support_y", dy.flatten()[flat_support], persistent=False,
        )
        self.register_buffer("support_cover", support_cover, persistent=False)
        self.register_buffer("radial_fraction", radial - r0, persistent=False)
        self.register_buffer("index_r0_a0", i00, persistent=False)
        self.register_buffer("index_r0_a1", i01, persistent=False)
        self.register_buffer("index_r1_a0", i10, persistent=False)
        self.register_buffer("index_r1_a1", i11, persistent=False)
        self.register_buffer("angle_fraction_r0", a0t, persistent=False)
        self.register_buffer("angle_fraction_r1", a1t, persistent=False)

        n_in = float(self.in_channels * support_cover.sum())
        self.distance_multiplier = 1.0 / (
            (n_in / 6.0) - .5 * (n_in * 7.0 / 180.0) ** .5
        )

    def rendered_prototypes(self, prototype):
        p00 = prototype[..., self.index_r0_a0]
        p01 = prototype[..., self.index_r0_a1]
        p10 = prototype[..., self.index_r1_a0]
        p11 = prototype[..., self.index_r1_a1]
        a0t = self.angle_fraction_r0[None, None]
        a1t = self.angle_fraction_r1[None, None]
        lo = p00 * (1 - a0t) + p01 * a0t
        hi = p10 * (1 - a1t) + p11 * a1t
        rt = self.radial_fraction[None, None, None]
        return (lo * (1 - rt) + hi * rt).permute(0, 2, 1, 3).contiguous()

    def indexed_patches(self, image):
        border = (self.kernel_size - self.stride) // 2
        if border > 0:
            image = F.pad(image, (border,) * 4, mode="reflect")
        elif border < 0:
            crop = -border
            image = image[..., crop:-crop, crop:-crop]
        square = F.unfold(image, self.kernel_size, stride=self.stride)
        batch, _, tokens = square.shape
        square = square.view(
            batch, self.in_channels, self.kernel_size**2, tokens
        )
        return square[:, :, self.support_mask].permute(0, 3, 1, 2).contiguous()


class MultiScaleVariableRingPolarDistanceProjection(nn.Module):
    """One compact polar prototype bank jointly matched over scale and rotation."""

    def __init__(
        self,
        out_channels=192,
        bases=96,
        directions=8,
        kernel_sizes=(24, 20, 16, 12),
        radial_bins=12,
        angular_bins_per_radius=4,
        stride=16,
        prototype_std=.02,
        ring_counts=None,
        cover_radius_scale=1.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.bases = bases
        self.directions = directions
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.scales = len(self.kernel_sizes)
        self.radial_bins = radial_bins
        self.stride = stride
        self.cover_radius_scale = float(cover_radius_scale)

        if ring_counts is None:
            ring_counts = [angular_bins_per_radius * (r + 1) for r in range(radial_bins)]
        if len(ring_counts) != radial_bins:
            raise ValueError("ring_counts length must equal radial_bins")
        counts = torch.tensor(ring_counts, dtype=torch.long)
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.register_buffer("ring_counts", counts)
        self.register_buffer("ring_offsets", offsets)

        self.prototype = nn.Parameter(
            torch.randn(bases, 3, int(offsets[-1])) * prototype_std
        )
        self.log2_scale = nn.Parameter(torch.zeros(bases, self.scales))
        self.value = nn.Parameter(
            torch.empty(bases, self.scales, directions, out_channels)
        )
        nn.init.trunc_normal_(self.value, std=.02)
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.geometries = nn.ModuleList(
            _PolarScaleGeometry(
                kernel_size=k,
                stride=stride,
                radial_bins=radial_bins,
                counts=counts,
                offsets=offsets,
                directions=directions,
                cover_radius_scale=cover_radius_scale,
            )
            for k in self.kernel_sizes
        )

    def initialize_from_single_scale(self, state):
        with torch.no_grad():
            self.prototype.copy_(state["prototype"])
            self.log2_scale.copy_(state["log2_scale"][:, None])
            self.value.copy_(state["value"][:, None])
            self.output_bias.copy_(state["output_bias"])

    def pose_scores(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            all_scores = []
            for scale_index, geometry in enumerate(self.geometries):
                rendered = geometry.rendered_prototypes(self.prototype).float()
                patch = geometry.indexed_patches(image)
                weight = geometry.support_cover[None, None, None, :]
                cross = torch.einsum("qtcm,bdcm->qtbd", patch * weight, rendered)
                patch_energy = (patch.square() * weight).sum((2, 3))
                proto_energy = (
                    rendered.square() * geometry.support_cover[None, None, None, :]
                ).sum((2, 3))
                distance = (
                    patch_energy[:, :, None, None]
                    + proto_energy[None, None]
                    - 2 * cross
                ).clamp_min(0)
                side = int(patch.shape[1] ** .5)
                score = distance.permute(0, 2, 3, 1).reshape(
                    image.shape[0], self.bases, self.directions, side, side
                )
                score = (
                    -score
                    * torch.exp2(self.log2_scale[:, scale_index])[None, :, None, None, None]
                    * geometry.distance_multiplier
                )
                all_scores.append(score)
            return torch.stack(all_scores, dim=2)

    def forward(self, image):
        scores = self.pose_scores(image)
        batch, bases, scales, directions, height, width = scores.shape
        flat_scores = scores.flatten(2, 3)
        pose_weight = flat_scores.softmax(2).view_as(scores)
        gated_weight = pose_weight * torch.exp2(scores)
        output = torch.einsum(
            "qbsdxy,bsdc->qxyc", gated_weight, self.value
        )
        return (output + self.output_bias).permute(0, 3, 1, 2).contiguous()
