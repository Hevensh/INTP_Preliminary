from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableRingPolarDistanceProjection(nn.Module):
    """Compact polar prototypes with radius-dependent angular resolution."""

    def __init__(self, out_channels=192, bases=48, directions=16, kernel_size=24,
                 radial_bins=12, angular_bins_per_radius=4, stride=16, prototype_std=.02,
                 amplitude_mode="linear", ring_counts=None, harmonic_count=4,
                 fundamental_frequency=1):
        super().__init__()
        self.out_channels = out_channels
        self.bases = bases
        self.directions = directions
        self.kernel_size = kernel_size
        self.radial_bins = radial_bins
        self.angular_bins_per_radius = angular_bins_per_radius
        self.stride = stride
        self.amplitude_mode = amplitude_mode
        if amplitude_mode not in {"linear", "exp_relative", "group_distance_exp", "pose_exp_weighted", "harmonic_v_exp"}:
            raise ValueError(f"unknown amplitude_mode: {amplitude_mode}")
        self.input_padding = (kernel_size - stride) // 2

        if ring_counts is None:
            # Match angular capacity to circumference. With the default four
            # angular bins per radius, 12 rings store 4, 8, ..., 48 bins.
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
        self.log2_scale = nn.Parameter(torch.zeros(bases))
        if amplitude_mode == "group_distance_exp":
            self.similarity_bias = nn.Parameter(torch.zeros(bases))
        if amplitude_mode in {"pose_exp_weighted", "harmonic_v_exp"}:
            self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.harmonic_count = harmonic_count
        self.fundamental_frequency = fundamental_frequency
        if amplitude_mode == "harmonic_v_exp":
            # Harmonic pairs are ordinary learnable V vectors.  Orthogonal
            # initialization imposed an unnecessary geometric constraint and
            # encouraged the prototype matcher to create sharp pose weights.
            value = torch.randn(bases, harmonic_count * 2, out_channels) * .02
            self.value = nn.Parameter(value.view(bases, harmonic_count, 2, out_channels))
            theta = torch.arange(directions) * (2 * math.pi / directions)
            frequency = torch.arange(1, harmonic_count + 1) * fundamental_frequency
            self.register_buffer("harmonic_cos", torch.cos(theta[:, None] * frequency[None]), persistent=False)
            self.register_buffer("harmonic_sin", torch.sin(theta[:, None] * frequency[None]), persistent=False)
        else:
            self.value = nn.Parameter(torch.empty(bases, directions, out_channels))
            nn.init.trunc_normal_(self.value, std=.02)

        yy, xx = torch.meshgrid(
            torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij"
        )
        center = (kernel_size - 1) / 2
        dx, dy = xx - center, yy - center
        radius = torch.sqrt(dx.square() + dy.square())
        angle_turn = torch.remainder(torch.atan2(dy, dx), 2 * math.pi) / (2 * math.pi)
        cover = torch.cos(radius * math.pi / kernel_size).clamp_min(0)
        support = radius < kernel_size / 2
        flat_support = support.flatten()
        support_radius = radius.flatten()[flat_support]
        support_angle = angle_turn.flatten()[flat_support]
        radial = (support_radius - .5).clamp(0, radial_bins - 1)
        r0 = radial.floor().long()
        r1 = (r0 + 1).clamp_max(radial_bins - 1)

        def angular_indices(ring: torch.Tensor):
            count = counts[ring]
            shifts = torch.arange(directions)[:, None] / directions
            position = torch.remainder(support_angle[None] - shifts, 1.0) * count[None]
            raw_a0 = position.floor()
            frac = position - raw_a0
            a0 = torch.remainder(raw_a0.long(), count[None])
            a1 = torch.remainder(a0 + 1, count[None])
            offset = offsets[ring][None]
            return offset + a0, offset + a1, frac

        i00, i01, a0t = angular_indices(r0)
        i10, i11, a1t = angular_indices(r1)
        self.register_buffer("support_mask", flat_support, persistent=False)
        self.register_buffer("support_cover", cover.flatten()[flat_support], persistent=False)
        self.register_buffer("radial_fraction", radial - r0, persistent=False)
        self.register_buffer("index_r0_a0", i00, persistent=False)
        self.register_buffer("index_r0_a1", i01, persistent=False)
        self.register_buffer("index_r1_a0", i10, persistent=False)
        self.register_buffer("index_r1_a1", i11, persistent=False)
        self.register_buffer("angle_fraction_r0", a0t, persistent=False)
        self.register_buffer("angle_fraction_r1", a1t, persistent=False)
        n_in = float(3 * cover.sum())
        self.multi = 1.0 / ((n_in / 6.0) - .5 * (n_in * 7.0 / 180.0) ** .5)

    def rendered_prototypes(self):
        p00 = self.prototype[..., self.index_r0_a0]
        p01 = self.prototype[..., self.index_r0_a1]
        p10 = self.prototype[..., self.index_r1_a0]
        p11 = self.prototype[..., self.index_r1_a1]
        a0t = self.angle_fraction_r0[None, None]
        a1t = self.angle_fraction_r1[None, None]
        lo = p00 * (1 - a0t) + p01 * a0t
        hi = p10 * (1 - a1t) + p11 * a1t
        rt = self.radial_fraction[None, None, None]
        return (lo * (1 - rt) + hi * rt).permute(0, 2, 1, 3).contiguous()

    def indexed_patches(self, image):
        if self.input_padding:
            image = F.pad(image, (self.input_padding,) * 4, mode="reflect")
        square = F.unfold(image, self.kernel_size, stride=self.stride)
        batch, _, tokens = square.shape
        square = square.view(batch, 3, self.kernel_size * self.kernel_size, tokens)
        return square[:, :, self.support_mask].permute(0, 3, 1, 2).contiguous()

    def pose_scores(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            rendered = self.rendered_prototypes().float()
            patch = self.indexed_patches(image)
            weight = self.support_cover[None, None, None, :]
            cross = torch.einsum("qtcm,ndcm->qtnd", patch * weight, rendered)
            patch_energy = (patch.square() * weight).sum((2, 3))
            proto_energy = (
                rendered.square() * self.support_cover[None, None, None, :]
            ).sum((2, 3))
            distance = (
                patch_energy[:, :, None, None]
                + proto_energy[None, None]
                - 2 * cross
            ).clamp_min(0)
            side = int(patch.shape[1] ** .5)
            distance = distance.permute(0, 2, 3, 1).reshape(
                image.shape[0], self.bases, self.directions, side, side
            )
            return (
                -distance
                * torch.exp2(self.log2_scale)[None, :, None, None, None]
                * self.multi
            )

    def forward(self, image):
        scores = self.pose_scores(image)
        direction_weight = scores.softmax(2)
        if self.amplitude_mode == "harmonic_v_exp":
            cosine_moment = torch.einsum("qbdhw,df->qbfhw", direction_weight, self.harmonic_cos)
            sine_moment = torch.einsum("qbdhw,df->qbfhw", direction_weight, self.harmonic_sin)
            base_value = (
                torch.einsum("qbfhw,bfc->qbhwc", cosine_moment, self.value[:, :, 0])
                + torch.einsum("qbfhw,bfc->qbhwc", sine_moment, self.value[:, :, 1])
            )
            gate = (direction_weight * torch.exp2(scores)).sum(2)
            output = (base_value * gate[..., None]).sum(1)
            return (output + self.output_bias).permute(0, 3, 1, 2).contiguous()
        base_value = torch.einsum("qbrhw,brc->qbhwc", direction_weight, self.value)
        if self.amplitude_mode == "pose_exp_weighted":
            gate = (direction_weight * torch.exp2(scores)).sum(2)
            output = (base_value * gate[..., None]).sum(1)
            return (output + self.output_bias).permute(0, 3, 1, 2).contiguous()
        if self.amplitude_mode == "group_distance_exp":
            mean_score = (direction_weight * scores).sum(2)
            gate = torch.exp2(mean_score + self.similarity_bias[None, :, None, None])
            return (base_value * gate[..., None]).sum(1).permute(0, 3, 1, 2).contiguous()
        if self.amplitude_mode == "exp_relative":
            relative = torch.exp(scores - scores.amax(2, keepdim=True))
            amplitude = 1 - relative.mean(2)
        else:
            amplitude = scores.amax(2) - scores.mean(2)
        return (base_value * amplitude[..., None]).sum(1).permute(0, 3, 1, 2).contiguous()
