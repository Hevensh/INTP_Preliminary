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
        weights = self.rotated_weights().reshape(
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
