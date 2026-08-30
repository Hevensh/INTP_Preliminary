from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from layers.cartesian_rotating_harmonic_conv import CartesianRotatingHarmonicConv2d
from layers.cartesian_four_value_paired_mams import (
    CartesianFourValuePairedMAMSConv2d,
    ChannelRMSNorm2d,
    ComplexPointwiseConv2d,
    PairedRMSNorm2d,
)
from layers.cartesian_stage_mams import CartesianStageMAMSRouting2d


class MAMSBasicBlock(nn.Module):
    """Whole-block replacement for a two-3x3 ResNet BasicBlock."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        downsample: nn.Module | None,
        diameters: tuple[int, ...],
        directions: int,
        global_directions: int,
        angular_bins_per_radius: int,
        prototype_chunk_size: int,
        use_null: bool,
        null_initial_score: float,
    ) -> None:
        super().__init__()
        self.mams = CartesianRotatingHarmonicConv2d(
            in_channels,
            out_channels,
            diameters=diameters,
            stride=stride,
            directions=directions,
            global_directions=global_directions,
            angular_bins_per_radius=angular_bins_per_radius,
            prototype_chunk_size=prototype_chunk_size,
            use_null=use_null,
            null_initial_score=null_initial_score,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mix = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

        nn.init.kaiming_normal_(self.mix.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.mams(x)))
        out = self.bn2(self.mix(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def build_resnet18_mams(
    *,
    num_classes: int,
    diameters: tuple[int, ...] = (6, 3),
    directions: int = 4,
    global_directions: int = 8,
    angular_bins_per_radius: int = 4,
    prototype_chunk_size: int = 16,
    use_null: bool = True,
    null_initial_score: float = 0.0,
) -> nn.Module:
    """Replace every two-3x3 ResNet-18 BasicBlock with MAMS then 1x1.

    Stem, stage widths, downsample shortcuts, pooling, and classifier remain
    identical to torchvision ResNet-18.
    """

    model = resnet18(weights=None, num_classes=num_classes)
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for index, block in enumerate(stage):
            in_channels = block.conv1.in_channels
            out_channels = block.conv2.out_channels
            stage[index] = MAMSBasicBlock(
                in_channels,
                out_channels,
                stride=block.stride,
                downsample=block.downsample,
                diameters=diameters,
                directions=directions,
                global_directions=global_directions,
                angular_bins_per_radius=angular_bins_per_radius,
                prototype_chunk_size=prototype_chunk_size,
                use_null=use_null,
                null_initial_score=null_initial_score,
            )
    return model


class FourValuePairedMAMSBasicBlock(nn.Module):
    """A paired MAMS block whose A/B/Vscale values replace the 1x1 mixer.

    The first block lifts ordinary stem channels into cosine/sine pairs.  All
    later blocks preserve that pair structure, including stage shortcuts.
    There is deliberately no post-MAMS ReLU: null-softmax is the block's
    nonlinear routing operation.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        diameters: tuple[int, int],
        directions: int,
        global_directions: int,
        angular_bins_per_radius: int,
        prototype_chunk_size: int,
        paired_input: bool,
        use_residual: bool,
        null_initial_score: float,
    ) -> None:
        super().__init__()
        self.stride = int(stride)
        self.paired_input = bool(paired_input)
        self.use_residual = bool(use_residual)
        self.norm = (
            PairedRMSNorm2d(in_channels)
            if paired_input
            else ChannelRMSNorm2d(in_channels)
        )
        self.mams = CartesianFourValuePairedMAMSConv2d(
            in_channels,
            out_channels,
            diameters=diameters,
            stride=stride,
            directions=directions,
            global_directions=global_directions,
            angular_bins_per_radius=angular_bins_per_radius,
            prototype_chunk_size=prototype_chunk_size,
            paired_input=paired_input,
            null_initial_score=null_initial_score,
        )
        if not use_residual:
            self.shortcut = None
        elif stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ComplexPointwiseConv2d(
                in_channels,
                out_channels,
                stride=stride,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mams(self.norm(x))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


def build_resnet18_mams_fourv_paired(
    *,
    num_classes: int,
    diameters: tuple[int, int] = (6, 3),
    directions: int = 4,
    global_directions: int = 8,
    angular_bins_per_radius: int = 4,
    prototype_chunk_size: int = 256,
    null_initial_score: float = 0.0,
) -> nn.Module:
    """Build ResNet-18 with paired four-value MAMS replacing every block."""

    model = resnet18(weights=None, num_classes=num_classes)
    model.bn1 = ChannelRMSNorm2d(64)
    block_number = 0
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for index, old_block in enumerate(stage):
            in_channels = old_block.conv1.in_channels
            out_channels = old_block.conv2.out_channels
            stage[index] = FourValuePairedMAMSBasicBlock(
                in_channels,
                out_channels,
                stride=old_block.stride,
                diameters=diameters,
                directions=directions,
                global_directions=global_directions,
                angular_bins_per_radius=angular_bins_per_radius,
                prototype_chunk_size=prototype_chunk_size,
                paired_input=block_number > 0,
                use_residual=block_number > 0,
                null_initial_score=null_initial_score,
            )
            block_number += 1
    model.avgpool = nn.Sequential(
        PairedRMSNorm2d(512),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    return model


class StageRoutedMAMS(nn.Module):
    """A standard ResNet stage augmented by one shared large-support router."""

    def __init__(
        self,
        stage: nn.Sequential,
        *,
        diameters: tuple[int, ...],
        directions: int,
        global_directions: int,
        angular_bins_per_radius: int,
        prototype_chunk_size: int,
        null_initial_score: float,
        route_channels: int = 32,
        route_stride: int | None = None,
    ) -> None:
        super().__init__()
        blocks = list(stage.children())
        if not blocks:
            raise ValueError("a routed ResNet stage cannot be empty")
        self.blocks = nn.ModuleList(blocks)
        in_channels = blocks[0].conv1.in_channels
        out_channels = blocks[0].conv2.out_channels
        stride = blocks[0].stride
        if route_stride is None:
            route_stride = stride
        self.router = CartesianStageMAMSRouting2d(
            in_channels,
            route_channels,
            consumers=len(blocks),
            diameters=diameters,
            stride=route_stride,
            directions=directions,
            global_directions=global_directions,
            angular_bins_per_radius=angular_bins_per_radius,
            prototype_chunk_size=prototype_chunk_size,
            null_initial_score=null_initial_score,
        )
        # This is the low-rank value factor attached to each block, not an
        # input-side 1x1 convolution. It is evaluated on the coarse control
        # field before interpolation and is algebraically part of V(r,theta).
        self.value_projection = nn.Parameter(
            torch.empty(len(blocks), out_channels, route_channels)
        )
        nn.init.kaiming_normal_(self.value_projection, nonlinearity="linear")
        self.gate = nn.Parameter(torch.full((len(blocks), out_channels), 0.1))
        self.relu = nn.ReLU(inplace=False)
        # The first 3x3 of every BasicBlock is replaced by the stage route.
        # Removing the module also removes its parameters from the model.
        for block in self.blocks:
            block.conv1 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.router(x)
        out = x
        for index, block in enumerate(self.blocks):
            identity = out
            if block.downsample is not None:
                identity = block.downsample(out)
            correction = F.linear(
                routed[index].permute(0, 2, 3, 1),
                self.value_projection[index].to(routed.dtype),
            ).permute(0, 3, 1, 2)
            if correction.shape[-2:] != out.shape[-2:]:
                correction = F.interpolate(
                    correction,
                    size=identity.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            gate = self.gate[index][None, :, None, None].to(correction.dtype)
            branch = self.relu(block.bn1(identity + correction * gate))
            branch = block.bn2(block.conv2(branch))
            out = self.relu(branch + identity)
        return out


def build_resnet18_stage_mams(
    *,
    num_classes: int,
    stage_diameters: tuple[tuple[int, ...], ...] = (
        (9, 5),
        (9, 5),
        (7, 3),
        (5, 3),
    ),
    route_strides: tuple[int, ...] = (4, 4, 2, 2),
    directions: int = 4,
    global_directions: int = 8,
    angular_bins_per_radius: int = 4,
    prototype_chunk_size: int = 16,
    null_initial_score: float = 0.0,
    route_channels: int = 32,
) -> nn.Module:
    """Add one reusable large-support geometry route to every ResNet stage."""

    if len(stage_diameters) != 4 or len(route_strides) != 4:
        raise ValueError("ResNet-18 requires four stage geometry specifications")
    model = resnet18(weights=None, num_classes=num_classes)
    for index, diameters in enumerate(stage_diameters, start=1):
        stage_name = f"layer{index}"
        setattr(
            model,
            stage_name,
            StageRoutedMAMS(
                getattr(model, stage_name),
                diameters=diameters,
                directions=directions,
                global_directions=global_directions,
                angular_bins_per_radius=angular_bins_per_radius,
                prototype_chunk_size=prototype_chunk_size,
                null_initial_score=null_initial_score,
                route_channels=route_channels,
                route_stride=route_strides[index - 1],
            ),
        )
    return model
