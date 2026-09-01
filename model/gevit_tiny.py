"""A DeiT-sized p4 GE-ViT adaptation for the aligned ImageNet-100 study.

The implementation keeps the defining GE-ViT mechanism from Xu et al. (UAI
2023): features live on a joint spatial--orientation domain and local
self-attention uses relative coordinates acted on by the query orientation.
The original repository targets small images with a bespoke staged backbone.
Here a C4 lifting patch projection and twelve constant-width blocks provide a
controlled 224px comparison against DeiT-Tiny; this is therefore an
algorithm-faithful adaptation, not a reproduction of the paper's training
recipe or reported datasets.

Reference (MIT): https://github.com/ZJUCDSYangKaifan/GEVit
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class C4LiftingPatchEmbed(nn.Module):
    """Lift an image to four orientation channels with one shared patch bank."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        orientations: int = 4,
    ) -> None:
        super().__init__()
        if orientations != 4:
            raise ValueError("the aligned GE-ViT adaptation currently uses p4")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.orientations = int(orientations)
        self.grid_size = image_size // patch_size

        self.weight = nn.Parameter(
            torch.empty(embed_dim, in_channels, patch_size, patch_size)
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_channels * self.patch_size**2
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}px input, got {tuple(x.shape[-2:])}"
            )
        rotated = torch.stack(
            [torch.rot90(self.weight, turns, dims=(-2, -1)) for turns in range(4)],
            dim=0,
        )
        output = F.conv2d(
            x,
            rotated.reshape(
                self.orientations * self.embed_dim,
                self.in_channels,
                self.patch_size,
                self.patch_size,
            ),
            self.bias.repeat(self.orientations),
            stride=self.patch_size,
        )
        batch = x.shape[0]
        return output.reshape(
            batch,
            self.orientations,
            self.embed_dim,
            self.grid_size,
            self.grid_size,
        ).permute(0, 2, 1, 3, 4)


class GEViTLocalAttention(nn.Module):
    """Local p4 group attention with orientation-acted relative positions."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 3,
        orientations: int = 4,
        window_size: int = 5,
        grid_size: int = 14,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if orientations != 4:
            raise ValueError("the aligned GE-ViT adaptation currently uses p4")
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.orientations = int(orientations)
        self.window_size = int(window_size)
        self.grid_size = int(grid_size)
        self.points = window_size * window_size

        row_dim = self.head_dim // 3
        col_dim = self.head_dim // 3
        group_dim = self.head_dim - row_dim - col_dim
        self.position_dims = (row_dim, col_dim, group_dim)

        self.qkv = nn.Conv3d(dim, 3 * dim, kernel_size=1, bias=True)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1, bias=True)
        self.row_embedding = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, row_dim)
        )
        self.col_embedding = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, col_dim)
        )
        self.group_embedding = nn.Embedding(orientations, group_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)

        radius = window_size // 2
        coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        offsets = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
        if radius:
            offsets = offsets / float(radius)
        angles = torch.arange(orientations, dtype=torch.float32)
        # Relative coordinates are expressed in the query pose's frame.  The
        # inverse action is what makes a spatial rotation plus an orientation
        # roll leave the positional score unchanged.
        angles = -angles * (2.0 * math.pi / orientations)
        cosine, sine = angles.cos(), angles.sin()
        matrices = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        acted = torch.einsum("gij,pj->gpi", matrices, offsets)
        self.register_buffer("acted_offsets", acted, persistent=False)

        query_orientation = torch.arange(orientations)[:, None]
        key_orientation = torch.arange(orientations)[None, :]
        relative_group = (key_orientation - query_orientation) % orientations
        self.register_buffer("relative_group", relative_group, persistent=False)

        ones = torch.ones(1, 1, grid_size, grid_size)
        valid = F.unfold(
            ones,
            kernel_size=window_size,
            padding=radius,
        ).squeeze(0)
        self.register_buffer(
            "valid_neighbors", valid.transpose(0, 1).bool(), persistent=False
        )

    def _unfold_group(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, depth, groups, height, width = tensor.shape
        tensor = tensor.permute(0, 3, 1, 2, 4, 5).reshape(
            batch * groups, heads * depth, height, width
        )
        tensor = F.unfold(
            tensor,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        )
        return tensor.reshape(
            batch,
            groups,
            heads,
            depth,
            self.points,
            height * width,
        ).permute(0, 2, 1, 3, 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, groups, height, width = x.shape
        if channels != self.dim or groups != self.orientations:
            raise ValueError(
                f"expected [B,{self.dim},{self.orientations},H,W], got {tuple(x.shape)}"
            )
        if height != self.grid_size or width != self.grid_size:
            raise ValueError(
                f"expected {self.grid_size}x{self.grid_size} grid, got {height}x{width}"
            )

        qkv = self.qkv(x).reshape(
            batch,
            3,
            self.num_heads,
            self.head_dim,
            groups,
            height,
            width,
        )
        q = qkv[:, 0].permute(0, 1, 3, 4, 5, 2).reshape(
            batch, self.num_heads, groups, height * width, self.head_dim
        )
        k = self._unfold_group(qkv[:, 1])
        v = self._unfold_group(qkv[:, 2])

        content = torch.einsum("bhgld,bhkdpl->bhglkp", q, k)
        row_dim, col_dim, _ = self.position_dims
        q_row = q[..., :row_dim]
        q_col = q[..., row_dim : row_dim + col_dim]
        q_group = q[..., row_dim + col_dim :]

        row = self.row_embedding(self.acted_offsets[..., 0:1].to(q.dtype))
        col = self.col_embedding(self.acted_offsets[..., 1:2].to(q.dtype))
        group = self.group_embedding(self.relative_group).to(q.dtype)
        spatial_score = torch.einsum("bhgld,gpd->bhglp", q_row, row)
        spatial_score = spatial_score + torch.einsum(
            "bhgld,gpd->bhglp", q_col, col
        )
        group_score = torch.einsum("bhgld,gkd->bhglk", q_group, group)
        score = content + spatial_score.unsqueeze(-2) + group_score.unsqueeze(-1)
        score = score / math.sqrt(float(self.dim))
        valid = self.valid_neighbors[None, None, None, :, None, :]
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        probability = F.softmax(
            score.reshape(
                batch,
                self.num_heads,
                groups,
                height * width,
                groups * self.points,
            ),
            dim=-1,
        ).reshape_as(score)
        probability = self.attention_dropout(probability)
        output = torch.einsum("bhglkp,bhkdpl->bhgld", probability, v)
        output = output.reshape(
            batch, self.num_heads, groups, height, width, self.head_dim
        ).permute(0, 1, 5, 2, 3, 4)
        output = output.reshape(batch, channels, groups, height, width)
        return self.projection_dropout(self.proj(output))


class GEViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        orientations: int,
        window_size: int,
        grid_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim, eps=1e-6)
        self.attention = GEViTLocalAttention(
            dim,
            num_heads=num_heads,
            orientations=orientations,
            window_size=window_size,
            grid_size=grid_size,
            attention_dropout=dropout,
            projection_dropout=dropout,
        )
        self.norm2 = nn.GroupNorm(1, dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GEViTTinyP4(nn.Module):
    """DeiT-Tiny-sized GE-ViT p4 local-attention comparison."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        orientations: int = 4,
        window_size: int = 5,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        grid_size = image_size // patch_size
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.orientations = int(orientations)
        self.patch_embed = C4LiftingPatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            orientations=orientations,
        )
        self.blocks = nn.ModuleList(
            [
                GEViTBlock(
                    embed_dim,
                    num_heads=num_heads,
                    orientations=orientations,
                    window_size=window_size,
                    grid_size=grid_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.GroupNorm(1, embed_dim, eps=1e-6)
        self.head = nn.Conv3d(embed_dim, num_classes, kernel_size=1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.forward_features(x))
        # Match the official GE-ViT readout: sum over space, then select the
        # strongest orientation for each class.
        return logits.sum(dim=(-2, -1)).max(dim=-1).values
