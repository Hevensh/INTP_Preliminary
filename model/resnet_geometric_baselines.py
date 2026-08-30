from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SharedRotatingConv2d(nn.Module):
    """Rotate one learned Cartesian kernel bank and keep its best response.

    This is a deliberately simple ARF-style comparison layer.  It preserves the
    ordinary ``[B, C, H, W]`` ResNet interface instead of carrying an orientation
    axis through the whole network: the direction axis is removed by max pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        directions: int = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if directions <= 0:
            raise ValueError("directions must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.directions = int(directions)
        self.padding = self.kernel_size // 2

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        angles = torch.arange(self.directions, dtype=torch.float32)
        angles = angles * (math.pi / self.directions)
        matrices = torch.zeros(self.directions, 2, 3, dtype=torch.float32)
        matrices[:, 0, 0] = angles.cos()
        matrices[:, 0, 1] = -angles.sin()
        matrices[:, 1, 0] = angles.sin()
        matrices[:, 1, 1] = angles.cos()
        self.register_buffer("rotation_matrices", matrices, persistent=False)

    def rotated_weights(self) -> torch.Tensor:
        """Return ``[D, Cout, Cin, K, K]`` differentiable rotated kernels."""

        # Every input channel uses the same spatial transform.  Keep Cin as the
        # grid_sample channel axis instead of treating Cout*Cin as its batch;
        # the latter is mathematically redundant and becomes enormous in deep
        # ResNet stages.
        kernels = self.weight.unsqueeze(0).expand(self.directions, -1, -1, -1, -1)
        kernels = kernels.reshape(
            self.directions * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        ).contiguous()
        matrices = self.rotation_matrices.to(dtype=self.weight.dtype)
        matrices = matrices[:, None].expand(-1, self.out_channels, -1, -1)
        matrices = matrices.reshape(-1, 2, 3)
        grid = F.affine_grid(matrices, kernels.shape, align_corners=True)
        rotated = F.grid_sample(
            kernels,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return rotated.reshape(
            self.directions,
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Grouped convolution requires all directional outputs belonging to
        # one input channel to be contiguous: [C, D, 1, K, K].
        weights = self.rotated_weights().permute(1, 0, 2, 3, 4).reshape(
            self.directions * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        bias = self.bias.repeat(self.directions) if self.bias is not None else None
        responses = F.conv2d(
            x,
            weights,
            bias,
            stride=self.stride,
            padding=self.padding,
        )
        responses = responses.reshape(
            x.shape[0],
            self.directions,
            self.out_channels,
            responses.shape[-2],
            responses.shape[-1],
        )
        return responses.amax(dim=1)


class _ResidualComparisonBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        feature_extractor: nn.Module,
        *,
        out_channels: int,
        stride: int,
        downsample: nn.Module | None,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mix = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = int(stride)

        nn.init.kaiming_normal_(self.mix.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.feature_extractor(x)))
        out = self.bn2(self.mix(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class MultiScaleConv2d(nn.Module):
    """Parallel ordinary Cartesian convolutions followed by concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_sizes: Sequence[int],
        stride: int,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        branch_widths = _split_channels(out_channels, len(kernel_sizes))
        self.branches = nn.ModuleList(
            nn.Conv2d(
                in_channels,
                width,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            )
            for width, kernel_size in zip(branch_widths, kernel_sizes, strict=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class MultiScaleRotatingConv2d(nn.Module):
    """Independent scale kernels, each shared across discrete rotations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_sizes: Sequence[int],
        stride: int,
        directions: int,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        branch_widths = _split_channels(out_channels, len(kernel_sizes))
        self.branches = nn.ModuleList(
            SharedRotatingConv2d(
                in_channels,
                width,
                kernel_size=kernel_size,
                stride=stride,
                directions=directions,
                bias=False,
            )
            for width, kernel_size in zip(branch_widths, kernel_sizes, strict=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm used by the ARC routing function."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MixConv4Depthwise(nn.Module):
    """MobileNet-style mixed depthwise convolution with four kernel sizes.

    A pointwise projection first produces the requested ResNet block width.
    The projected channels are then partitioned and processed by independent
    depthwise kernels, matching the central MixConv construction rather than
    using four full-convolution branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        stride: int = 1,
    ) -> None:
        super().__init__()
        if len(kernel_sizes) != 4:
            raise ValueError("MixConv4 requires exactly four kernel sizes")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("MixConv kernel sizes must be positive odd integers")

        self.kernel_sizes = tuple(int(kernel) for kernel in kernel_sizes)
        self.channel_splits = tuple(_split_channels(out_channels, 4))
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            nn.Conv2d(
                width,
                width,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                groups=width,
                bias=False,
            )
            for width, kernel in zip(
                self.channel_splits, self.kernel_sizes, strict=True
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.project(x)
        groups = projected.split(self.channel_splits, dim=1)
        return torch.cat(
            [branch(group) for branch, group in zip(self.branches, groups, strict=True)],
            dim=1,
        )


def _rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    matrices = angles.new_zeros((*angles.shape, 2, 3))
    cosine = angles.cos()
    sine = angles.sin()
    matrices[..., 0, 0] = cosine
    matrices[..., 0, 1] = -sine
    matrices[..., 1, 0] = sine
    matrices[..., 1, 1] = cosine
    return matrices


def _rotate_depthwise_weights(
    weights: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly rotate depthwise banks.

    ``weights`` is ``[N, C, 1, K, K]`` and ``angles`` is ``[..., N]``.
    The returned tensor is ``[..., N, C, 1, K, K]``.
    """

    if weights.ndim != 5 or weights.shape[2] != 1:
        raise ValueError("weights must have shape [N, C, 1, K, K]")
    if angles.shape[-1] != weights.shape[0]:
        raise ValueError("the final angle dimension must equal the kernel count")

    prefix = angles.shape[:-1]
    kernel_count, channels, _, kernel_h, kernel_w = weights.shape
    flat_angles = angles.reshape(-1, kernel_count)
    batch = flat_angles.shape[0]
    expanded = weights.unsqueeze(0).expand(batch, -1, -1, -1, -1, -1)
    expanded = expanded.reshape(batch * kernel_count * channels, 1, kernel_h, kernel_w)
    matrices = _rotation_matrices(flat_angles)
    matrices = matrices[:, :, None].expand(-1, -1, channels, -1, -1)
    matrices = matrices.reshape(batch * kernel_count * channels, 2, 3)
    grid = F.affine_grid(matrices, expanded.shape, align_corners=True)
    rotated = F.grid_sample(
        expanded,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return rotated.reshape(
        *prefix, kernel_count, channels, 1, kernel_h, kernel_w
    )


class FixedRotatedDepthwiseConv2d(nn.Module):
    """Fixed-angle bilinear rotation bank with orientation max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        directions: int = 8,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if directions <= 0:
            raise ValueError("directions must be positive")
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.weight = nn.Parameter(
            torch.empty(1, out_channels, 1, kernel_size, kernel_size)
        )
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.directions = int(directions)
        angles = torch.arange(directions, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / directions)
        self.register_buffer("angles", angles, persistent=False)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def rotated_weights(self) -> torch.Tensor:
        angles = self.angles.to(device=self.weight.device, dtype=self.weight.dtype)
        # Treat every fixed direction as a requested rotation of the same bank.
        return _rotate_depthwise_weights(self.weight, angles.unsqueeze(-1)).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        weights = self.rotated_weights().reshape(
            self.directions * self.out_channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )
        responses = F.conv2d(
            x,
            weights,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=self.out_channels,
        )
        responses = responses.reshape(
            x.shape[0],
            self.out_channels,
            self.directions,
            responses.shape[-2],
            responses.shape[-1],
        )
        return responses.amax(dim=2)


class ARCRoutingFunction(nn.Module):
    """Lightweight routing head following the official ARC implementation."""

    def __init__(
        self,
        channels: int,
        kernel_number: int = 4,
        *,
        dropout: float = 0.2,
        max_angle_degrees: float = 40.0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.norm = LayerNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.alpha_dropout = nn.Dropout(dropout)
        self.angle_dropout = nn.Dropout(dropout)
        self.alpha_head = nn.Linear(channels, kernel_number, bias=True)
        self.angle_head = nn.Linear(channels, kernel_number, bias=False)
        self.max_angle_radians = math.radians(max_angle_degrees)
        for parameter in (
            self.depthwise.weight,
            self.alpha_head.weight,
            self.angle_head.weight,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routed = self.activation(self.norm(self.depthwise(x)))
        routed = self.pool(routed).flatten(1)
        alphas = torch.sigmoid(self.alpha_head(self.alpha_dropout(routed)))
        angles = F.softsign(self.angle_head(self.angle_dropout(routed)))
        return alphas, angles * self.max_angle_radians


class AdaptiveRotatedDepthwiseConv2d(nn.Module):
    """Classification-sized ARC: route, rotate, combine, then convolve once."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        kernel_number: int = 4,
        dropout: float = 0.2,
        max_angle_degrees: float = 40.0,
    ) -> None:
        super().__init__()
        if kernel_size != 3:
            raise ValueError("the literature-aligned ARC baseline uses 3x3 kernels")
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.routing = ARCRoutingFunction(
            out_channels,
            kernel_number,
            dropout=dropout,
            max_angle_degrees=max_angle_degrees,
        )
        self.weight = nn.Parameter(
            torch.empty(kernel_number, out_channels, 1, kernel_size, kernel_size)
        )
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.kernel_number = int(kernel_number)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def adaptive_weights(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alphas, angles = self.routing(x)
        rotated = _rotate_depthwise_weights(self.weight, angles)
        combined = (rotated * alphas[..., None, None, None, None]).sum(dim=1)
        return combined, alphas, angles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        weights, _, _ = self.adaptive_weights(x)
        batch, channels, height, width = x.shape
        grouped_input = x.reshape(1, batch * channels, height, width)
        grouped_weight = weights.reshape(
            batch * channels, 1, self.kernel_size, self.kernel_size
        )
        output = F.conv2d(
            grouped_input,
            grouped_weight,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=batch * channels,
        )
        return output.reshape(batch, channels, output.shape[-2], output.shape[-1])


def _split_channels(channels: int, groups: int) -> list[int]:
    if channels < groups:
        raise ValueError("out_channels must be at least the number of branches")
    quotient, remainder = divmod(channels, groups)
    return [quotient + (index < remainder) for index in range(groups)]


def _replace_resnet18_blocks(
    *,
    num_classes: int,
    mode: str,
    kernel_sizes: tuple[int, ...] = (5, 3),
    directions: int = 4,
) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for index, block in enumerate(stage):
            in_channels = block.conv1.in_channels
            out_channels = block.conv2.out_channels
            if mode == "multiscale":
                extractor: nn.Module = MultiScaleConv2d(
                    in_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    stride=block.stride,
                )
            elif mode == "rotating":
                extractor = SharedRotatingConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=max(kernel_sizes),
                    stride=block.stride,
                    directions=directions,
                    bias=False,
                )
            elif mode == "multiscale_rotating":
                extractor = MultiScaleRotatingConv2d(
                    in_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    stride=block.stride,
                    directions=directions,
                )
            else:
                raise ValueError(f"unknown comparison mode: {mode}")
            stage[index] = _ResidualComparisonBlock(
                extractor,
                out_channels=out_channels,
                stride=block.stride,
                downsample=block.downsample,
            )
    return model


def build_resnet18_multiscale(
    *, num_classes: int, kernel_sizes: tuple[int, ...] = (5, 3)
) -> nn.Module:
    return _replace_resnet18_blocks(
        num_classes=num_classes,
        mode="multiscale",
        kernel_sizes=kernel_sizes,
    )


def build_resnet18_rotconv(
    *, num_classes: int, kernel_size: int = 5, directions: int = 4
) -> nn.Module:
    return _replace_resnet18_blocks(
        num_classes=num_classes,
        mode="rotating",
        kernel_sizes=(kernel_size,),
        directions=directions,
    )


def build_resnet18_multiscale_rotconv(
    *,
    num_classes: int,
    kernel_sizes: tuple[int, ...] = (5, 3),
    directions: int = 4,
) -> nn.Module:
    return _replace_resnet18_blocks(
        num_classes=num_classes,
        mode="multiscale_rotating",
        kernel_sizes=kernel_sizes,
        directions=directions,
    )


def _replace_resnet18_first_spatial_conv(
    *,
    num_classes: int,
    mode: str,
    kernel_sizes: tuple[int, ...] = (3, 5, 7, 9),
    directions: int = 4,
    arc_kernel_number: int = 4,
) -> nn.Module:
    """Keep torchvision's BasicBlock and replace only its first 3x3 conv."""

    model = resnet18(weights=None, num_classes=num_classes)
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in stage:
            in_channels = block.conv1.in_channels
            out_channels = block.conv1.out_channels
            stride = block.conv1.stride[0]
            if mode == "mixconv4":
                replacement: nn.Module = MixConv4Depthwise(
                    in_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    stride=stride,
                )
            elif mode == "fixed_rotinterp8":
                replacement = FixedRotatedDepthwiseConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    directions=directions,
                )
            elif mode == "arc4":
                replacement = AdaptiveRotatedDepthwiseConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    kernel_number=arc_kernel_number,
                )
            else:
                raise ValueError(f"unknown literature comparison mode: {mode}")
            block.conv1 = replacement
    return model


def build_resnet18_mixconv4(
    *,
    num_classes: int,
    kernel_sizes: tuple[int, ...] = (3, 5, 7, 9),
) -> nn.Module:
    return _replace_resnet18_first_spatial_conv(
        num_classes=num_classes,
        mode="mixconv4",
        kernel_sizes=kernel_sizes,
    )


def build_resnet18_fixed_rotinterp8(
    *, num_classes: int, directions: int = 8
) -> nn.Module:
    return _replace_resnet18_first_spatial_conv(
        num_classes=num_classes,
        mode="fixed_rotinterp8",
        directions=directions,
    )


def build_resnet18_arc4bank(
    *, num_classes: int, kernel_number: int = 4
) -> nn.Module:
    return _replace_resnet18_first_spatial_conv(
        num_classes=num_classes,
        mode="arc4",
        arc_kernel_number=kernel_number,
    )
