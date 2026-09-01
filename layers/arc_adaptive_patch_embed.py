from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ARCPatchRouting(nn.Module):
    """Input-conditioned alpha/angle router following official ARC."""

    def __init__(
        self,
        channels: int,
        kernel_number: int,
        *,
        dropout: float = 0.2,
        max_angle_degrees: float = 40.0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.norm = _LayerNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.alpha_dropout = nn.Dropout(dropout)
        self.angle_dropout = nn.Dropout(dropout)
        self.alpha_head = nn.Linear(channels, kernel_number, bias=True)
        self.angle_head = nn.Linear(channels, kernel_number, bias=False)
        self.max_angle_radians = math.radians(max_angle_degrees)
        nn.init.trunc_normal_(self.depthwise.weight, std=0.02)
        nn.init.trunc_normal_(self.alpha_head.weight, std=0.02)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.trunc_normal_(self.angle_head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routed = self.activation(self.norm(self.depthwise(x)))
        routed = self.pool(routed).flatten(1)
        alpha = torch.sigmoid(self.alpha_head(self.alpha_dropout(routed)))
        angle = F.softsign(self.angle_head(self.angle_dropout(routed)))
        return alpha, angle * self.max_angle_radians


def _rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    matrix = angles.new_zeros((*angles.shape, 2, 3))
    cosine, sine = angles.cos(), angles.sin()
    matrix[..., 0, 0] = cosine
    matrix[..., 0, 1] = -sine
    matrix[..., 1, 0] = sine
    matrix[..., 1, 1] = cosine
    return matrix


class ARCAdaptivePatchEmbed(nn.Module):
    """ARC-style input-adaptive rotated 16x16 DeiT patch projection.

    Each image predicts one mixing coefficient and one bounded angle for every
    canonical kernel bank.  Kernels are bilinearly rotated, combined first, and
    then applied in one grouped convolution per image, matching ARC's central
    combine-and-compute construction.  Batch chunking bounds the temporary
    rotated-kernel tensor on 16 GB T4 GPUs.
    """

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        kernel_number: int = 4,
        max_angle_degrees: float = 40.0,
        routing_dropout: float = 0.2,
        batch_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        if min(in_chans, embed_dim, kernel_number, batch_chunk_size) <= 0:
            raise ValueError("channel counts, kernel_number, and chunk size must be positive")
        self.img_size = (int(img_size), int(img_size))
        self.patch_size = (int(patch_size), int(patch_size))
        side = img_size // patch_size
        self.grid_size = (side, side)
        self.num_patches = side * side
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.kernel_number = int(kernel_number)
        self.batch_chunk_size = int(batch_chunk_size)

        self.routing = ARCPatchRouting(
            in_chans,
            kernel_number,
            dropout=routing_dropout,
            max_angle_degrees=max_angle_degrees,
        )
        self.weight = nn.Parameter(
            torch.empty(
                kernel_number, embed_dim, in_chans, patch_size, patch_size
            )
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Four sigmoid-weighted banks start with aggregate variance close to one
        # standard patch projection when alpha logits are zero (alpha=0.5).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data.div_(math.sqrt(self.kernel_number) * 0.5)
        fan_in = self.in_chans * self.patch_size[0] * self.patch_size[1]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def _adaptive_weights(
        self, alpha: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        batch = alpha.shape[0]
        kernels = self.weight[None].expand(batch, -1, -1, -1, -1, -1)
        kernels = kernels.reshape(
            batch * self.kernel_number * self.embed_dim,
            self.in_chans,
            *self.patch_size,
        )
        matrix = _rotation_matrices(angle)
        matrix = matrix[:, :, None].expand(-1, -1, self.embed_dim, -1, -1)
        matrix = matrix.reshape(-1, 2, 3)
        grid = F.affine_grid(matrix, kernels.shape, align_corners=True)
        rotated = F.grid_sample(
            kernels,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        rotated = rotated.reshape(
            batch,
            self.kernel_number,
            self.embed_dim,
            self.in_chans,
            *self.patch_size,
        )
        return (rotated * alpha[..., None, None, None, None]).sum(dim=1)

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.routing(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_chans:
            raise ValueError(
                f"expected [B,{self.in_chans},H,W], got {tuple(x.shape)}"
            )
        if x.shape[-2:] != self.img_size:
            raise ValueError(
                f"expected image size {self.img_size}, got {tuple(x.shape[-2:])}"
            )
        alpha, angle = self.routing(x)
        outputs: list[torch.Tensor] = []
        for start in range(0, x.shape[0], self.batch_chunk_size):
            stop = min(start + self.batch_chunk_size, x.shape[0])
            chunk = x[start:stop]
            weight = self._adaptive_weights(alpha[start:stop], angle[start:stop])
            chunk_size = stop - start
            grouped_input = chunk.reshape(
                1, chunk_size * self.in_chans, *self.img_size
            )
            grouped_weight = weight.reshape(
                chunk_size * self.embed_dim,
                self.in_chans,
                *self.patch_size,
            )
            output = F.conv2d(
                grouped_input,
                grouped_weight,
                self.bias.repeat(chunk_size),
                stride=self.patch_size,
                groups=chunk_size,
            )
            outputs.append(
                output.reshape(
                    chunk_size, self.embed_dim, *self.grid_size
                )
            )
        x = torch.cat(outputs, dim=0)
        return x.flatten(2).transpose(1, 2)
