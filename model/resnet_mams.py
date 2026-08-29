from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

from layers.cartesian_rotating_harmonic_conv import CartesianRotatingHarmonicConv2d


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
