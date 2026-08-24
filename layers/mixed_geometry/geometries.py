from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiskGeometry(nn.Module):
    """Extract a circular support from a square strided image window."""

    def __init__(
        self, kernel_size: int, stride: int = 16,
        cover_radius_scale: float = 1.0,
        in_channels: int = 3,
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
        support = radius < kernel_size / 2
        flat_support = support.flatten()
        self.register_buffer("support_mask", flat_support, persistent=False)
        self.register_buffer(
            "support_x", dx.flatten()[flat_support], persistent=False
        )
        self.register_buffer(
            "support_y", dy.flatten()[flat_support], persistent=False
        )
        cover = torch.cos(
            radius * math.pi / (kernel_size * self.cover_radius_scale)
        ).clamp_min(0)
        support_cover = cover.flatten()[flat_support]
        self.register_buffer("support_cover", support_cover, persistent=False)
        n_in = float(self.in_channels * support_cover.sum())
        self.distance_multiplier = 1.0 / (
            (n_in / 6.0) - .5 * (n_in * 7.0 / 180.0) ** .5
        )

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


class AngularGeometry(DiskGeometry):
    def __init__(
        self, kernel_size, directions, angular_bins, stride=16,
        cover_radius_scale=1.0, direction_step=None, in_channels=3,
    ):
        super().__init__(kernel_size, stride, cover_radius_scale, in_channels)
        angle = torch.remainder(
            torch.atan2(self.support_y, self.support_x), 2 * math.pi
        )
        if direction_step is None:
            direction_step = 2 * math.pi / directions
        shifts = torch.arange(directions)[:, None] * float(direction_step)
        position = (
            torch.remainder(angle[None] - shifts, 2 * math.pi)
            / (2 * math.pi) * angular_bins
        )
        raw = position.floor()
        self.register_buffer("index0", raw.long() % angular_bins, persistent=False)
        self.register_buffer(
            "index1", (raw.long() + 1) % angular_bins, persistent=False
        )
        self.register_buffer("fraction", position - raw, persistent=False)

    def render(self, prototype):
        p0 = prototype[..., self.index0]
        p1 = prototype[..., self.index1]
        out = p0 * (1 - self.fraction[None, None]) + p1 * self.fraction[None, None]
        return out.permute(0, 2, 1, 3).contiguous()


class RadialGeometry(DiskGeometry):
    def __init__(
        self, kernel_size, radial_bins, stride=16, cover_radius_scale=1.0,
        in_channels=3,
    ):
        super().__init__(kernel_size, stride, cover_radius_scale, in_channels)
        radius = torch.sqrt(self.support_x.square() + self.support_y.square())
        position = (
            radius * radial_bins / (kernel_size / 2) - .5
        ).clamp(0, radial_bins - 1)
        self.register_buffer("index0", position.floor().long(), persistent=False)
        self.register_buffer(
            "index1",
            (position.floor().long() + 1).clamp_max(radial_bins - 1),
            persistent=False,
        )
        self.register_buffer(
            "fraction", position - position.floor(), persistent=False
        )

    def render(self, prototype):
        p0 = prototype[..., self.index0]
        p1 = prototype[..., self.index1]
        return (p0 * (1 - self.fraction) + p1 * self.fraction)[:, None]


class ColorGeometry(DiskGeometry):
    """Render one RGB vector uniformly over the circular support."""

    def render(self, prototype):
        return prototype[:, None, :, None].expand(
            -1, 1, -1, self.support_x.numel()
        )


class StripeGeometry(DiskGeometry):
    def __init__(
        self, kernel_size, directions, profile_bins,
        longitudinal_bins=1, stride=16, cover_radius_scale=1.0,
        direction_step=None, in_channels=3,
    ):
        super().__init__(kernel_size, stride, cover_radius_scale, in_channels)
        if direction_step is None:
            direction_step = 2 * math.pi / directions
        self.directions = int(directions)
        self.profile_bins = int(profile_bins)
        self.direction_step = float(direction_step)
        theta = torch.arange(directions)[:, None] * float(direction_step)
        coordinate = (
            self.support_x[None] * torch.cos(theta)
            + self.support_y[None] * torch.sin(theta)
        ) / (kernel_size / 2)
        position = (
            (coordinate + 1) * .5 * (profile_bins - 1)
        ).clamp(0, profile_bins - 1)
        self.register_buffer("index0", position.floor().long(), persistent=False)
        self.register_buffer(
            "index1",
            (position.floor().long() + 1).clamp_max(profile_bins - 1),
            persistent=False,
        )
        self.register_buffer(
            "fraction", position - position.floor(), persistent=False
        )
        self.longitudinal_bins = int(longitudinal_bins)
        if self.longitudinal_bins > 1:
            longitudinal_coordinate = (
                -self.support_x[None] * torch.sin(theta)
                + self.support_y[None] * torch.cos(theta)
            ) / (kernel_size / 2)
            longitudinal_position = (
                (longitudinal_coordinate + 1)
                * .5 * (self.longitudinal_bins - 1)
            ).clamp(0, self.longitudinal_bins - 1)
            self.register_buffer(
                "longitudinal_index0",
                longitudinal_position.floor().long(), persistent=False,
            )
            self.register_buffer(
                "longitudinal_index1",
                (longitudinal_position.floor().long() + 1).clamp_max(
                    self.longitudinal_bins - 1
                ),
                persistent=False,
            )
            self.register_buffer(
                "longitudinal_fraction",
                longitudinal_position - longitudinal_position.floor(),
                persistent=False,
            )

    def _dynamic_coordinates(self, angle_offset):
        theta = (
            angle_offset[:, None, None]
            + torch.arange(
                self.directions, device=angle_offset.device,
                dtype=angle_offset.dtype,
            )[None, :, None] * self.direction_step
        )
        across = (
            self.support_x[None, None] * torch.cos(theta)
            + self.support_y[None, None] * torch.sin(theta)
        ) / (self.kernel_size / 2)
        along = (
            -self.support_x[None, None] * torch.sin(theta)
            + self.support_y[None, None] * torch.cos(theta)
        ) / (self.kernel_size / 2)
        return across, along

    @staticmethod
    def _linear_position(coordinate, bins):
        position = ((coordinate + 1) * .5 * (bins - 1)).clamp(0, bins - 1)
        index0 = position.floor().long()
        index1 = (index0 + 1).clamp_max(bins - 1)
        return index0, index1, position - position.floor()

    def _render_with_offset(self, prototype, angle_offset):
        across_coordinate, along_coordinate = self._dynamic_coordinates(angle_offset)
        i0, i1, across = self._linear_position(
            across_coordinate, self.profile_bins
        )
        batch, channels = prototype.shape[:2]
        directions, support = i0.shape[1:]
        if prototype.ndim == 3:
            source = prototype[:, :, None, :].expand(-1, -1, directions, -1)
            gather0 = i0[:, None].expand(-1, channels, -1, -1)
            gather1 = i1[:, None].expand(-1, channels, -1, -1)
            p0 = source.gather(3, gather0)
            p1 = source.gather(3, gather1)
            out = p0 * (1 - across[:, None]) + p1 * across[:, None]
            return out.permute(0, 2, 1, 3).contiguous()

        j0, j1, along = self._linear_position(
            along_coordinate, self.longitudinal_bins
        )
        source = prototype.flatten(2)[:, :, None].expand(
            -1, -1, directions, -1
        )

        def gather(ii, jj):
            index = (ii * self.longitudinal_bins + jj)[:, None].expand(
                -1, channels, -1, -1
            )
            return source.gather(3, index)

        p00, p10 = gather(i0, j0), gather(i1, j0)
        p01, p11 = gather(i0, j1), gather(i1, j1)
        low = p00 * (1 - across[:, None]) + p10 * across[:, None]
        high = p01 * (1 - across[:, None]) + p11 * across[:, None]
        out = low * (1 - along[:, None]) + high * along[:, None]
        return out.permute(0, 2, 1, 3).contiguous()

    def render(self, prototype, angle_offset=None):
        if angle_offset is not None:
            return self._render_with_offset(prototype, angle_offset)
        if prototype.ndim == 4:
            p00 = prototype[..., self.index0, self.longitudinal_index0]
            p10 = prototype[..., self.index1, self.longitudinal_index0]
            p01 = prototype[..., self.index0, self.longitudinal_index1]
            p11 = prototype[..., self.index1, self.longitudinal_index1]
            across = self.fraction[None, None]
            along = self.longitudinal_fraction[None, None]
            low = p00 * (1 - across) + p10 * across
            high = p01 * (1 - across) + p11 * across
            out = low * (1 - along) + high * along
            return out.permute(0, 2, 1, 3).contiguous()
        p0 = prototype[..., self.index0]
        p1 = prototype[..., self.index1]
        out = p0 * (1 - self.fraction[None, None]) + p1 * self.fraction[None, None]
        return out.permute(0, 2, 1, 3).contiguous()
