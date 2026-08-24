from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from layers.mini_vit import PatchEmbed, TransformerBlock, init_vit_weights
from layers.square_patch_dense_grid_look import SquarePatchDenseGridLook
from layers.square_patch_low_rank_look import SquarePatchLowRankLook


@dataclass(frozen=True)
class SquareDeiTLoadReport:
    copied_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]


class DeiTTinySquareLook(nn.Module):
    """Ordinary DeiT-Tiny with an optional prototype-conditioned Look Bias.

    ``low_rank`` preserves the historical shared-basis experiment;
    ``dense_grid`` assigns one prototype and one rotating 4x8 Look table to
    every ordinary layer/head.
    """

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        direction_samples: int = 8,
        scales: Sequence[float] = (1.0, math.sqrt(2.0), math.sqrt(3.0), 2.0),
        enable_look: bool = True,
        look_mode: str = "low_rank",
        use_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        if look_mode not in {"low_rank", "dense_grid"}:
            raise ValueError("look_mode must be low_rank or dense_grid")
        self.img_size = 224
        self.embed_dim = 192
        self.num_heads = 3
        self.use_pos_embed = bool(use_pos_embed)
        self.patch_embed = PatchEmbed(224, 16, 3, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, self.embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.embed_dim, self.num_heads, mlp_ratio=4.0, norm_eps=1e-6)
            for _ in range(12)
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Linear(self.embed_dim, num_classes)
        if not enable_look:
            self.look_bank = None
        elif look_mode == "dense_grid":
            # One prototype per ordinary layer/head.  Each owns one dense 4x8
            # canonical Look table which is transformed by the matched pose.
            self.look_bank = SquarePatchDenseGridLook(
                image_size=224,
                patch_size=16,
                num_heads=12 * self.num_heads,
                source_directions=4,
                source_direction_period=8,
                scales=(1.0, 0.5),
            )
        else:
            self.look_bank = SquarePatchLowRankLook(
                image_size=224,
                patch_size=16,
                num_heads=12 * self.num_heads,
                rank=8,
                direction_samples=direction_samples,
                scales=scales,
            )
        self.apply(init_vit_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(image)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        all_bias: torch.Tensor | None = None
        if self.look_bank is not None:
            # One P0 extraction and one shared polar match produce all 12x3
            # ordinary attention-head biases through the rank-8 expansion.
            if isinstance(self.look_bank, SquarePatchDenseGridLook):
                look_is_active = torch.count_nonzero(self.look_bank.look_grid).item() != 0
            else:
                look_is_active = torch.count_nonzero(self.look_bank.head_gain).item() != 0
            if self.training or look_is_active:
                # Polar interpolation/normalization is numerically sensitive
                # in fp16.  It is tiny compared with the ViT backbone, so keep
                # only this branch in fp32 while the ordinary model uses AMP.
                with torch.autocast(device_type=image.device.type, enabled=False):
                    flat_bias, _ = self.look_bank(image.float(), include_cls=True)
                all_bias = flat_bias.reshape(image.shape[0], 12, self.num_heads, 197, 197)

        for index, block in enumerate(self.blocks):
            bias = None if all_bias is None else all_bias[:, index]
            tokens = block(tokens, attn_bias=bias)
        return self.norm(tokens)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image)[:, 0])


@torch.no_grad()
def load_timm_deit_tiny(
    model: DeiTTinySquareLook,
    state_dict: Mapping[str, torch.Tensor],
) -> SquareDeiTLoadReport:
    """Copy every shape-compatible ordinary DeiT parameter into ``model``."""
    target = model.state_dict()
    copied: list[str] = []
    for key, value in state_dict.items():
        if key in target and target[key].shape == value.shape:
            target[key].copy_(value.to(target[key]))
            copied.append(key)
    required = {"cls_token", "pos_embed", "patch_embed.proj.weight", "norm.weight"}
    missing = required.difference(copied)
    if missing:
        raise ValueError(f"source is not a compatible DeiT-Tiny state dict; missing {sorted(missing)}")
    return SquareDeiTLoadReport(
        copied_keys=tuple(copied),
        skipped_keys=tuple(sorted(set(state_dict).difference(copied))),
    )
