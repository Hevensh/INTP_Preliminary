from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from layers.hex_patch_embed import HexPatchEmbed
from layers.hex_linear_patch_embed import HexLinearPatchEmbed
from layers.hex_look_bias import HexLookBias
from layers.mini_vit import TransformerBlock, init_vit_weights
from layers.polar_prototype_look import PolarPrototypeLook
from utils.hex_idx_genr import calc_coo_params, genr_2Dcoo


def estimate_hex_num_patches(
    img_size: Tuple[int, int],
    hex_kernel_size: int,
    hex_stride: int | None = None,
) -> int:
    lattice_stride = hex_kernel_size // 2 if hex_stride is None else hex_stride
    n_x, n_y, stride_x, stride_y = calc_coo_params(img_size, lattice_stride)
    idxs_x, _ = genr_2Dcoo(n_x, n_y, stride_x, stride_y)
    return int(idxs_x.numel())


def build_hex_patch_coordinates(
    img_size: Tuple[int, int],
    hex_kernel_size: int,
    hex_stride: int | None = None,
) -> torch.Tensor:
    """Build normalized xy coordinates in the exact token order used by HexConv2D."""
    lattice_stride = hex_kernel_size // 2 if hex_stride is None else hex_stride
    n_x, n_y, _, _ = calc_coo_params(img_size, lattice_stride)
    coo_x, coo_y = genr_2Dcoo(n_x, n_y, 0.5, 3 ** 0.5 * 0.5, torch.float32)
    return torch.stack((coo_x, coo_y), dim=-1)


class HexViTClassifier(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        hex_kernel_size: int = 16,
        hex_stride: int | None = None,
        patch_embed_mode: Literal["distance", "linear"] = "distance",
        position_mode: Literal[
            "learned",
            "look",
            "learned+look",
            "polar-look",
            "learned+polar-look",
            "none",
        ] = "learned",
        polar_look_strength: float = 1.0,
        polar_look_gate_init: float = 1.0,
        polar_base_radius: float | None = None,
        polar_look_radius: float = 3.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        valid_position_modes = {
            "learned",
            "look",
            "learned+look",
            "polar-look",
            "learned+polar-look",
            "none",
        }
        if position_mode not in valid_position_modes:
            raise ValueError(f"position_mode must be one of {sorted(valid_position_modes)}")
        if polar_look_radius <= 0:
            raise ValueError("polar_look_radius must be positive")
        self.position_mode = position_mode
        self.patch_embed_mode = patch_embed_mode
        self.polar_look_strength = float(polar_look_strength)
        self.polar_base_radius = float(
            (hex_kernel_size - 1) / 2.0 if polar_base_radius is None else polar_base_radius
        )
        if self.polar_base_radius <= 0:
            raise ValueError("polar_base_radius must be positive")
        self.polar_look_radius = float(polar_look_radius)

        if patch_embed_mode == "distance":
            self.patch_embed = HexPatchEmbed(
                in_chans=in_chans,
                embed_dim=embed_dim,
                kernel_size=hex_kernel_size,
                lattice_stride=hex_stride,
            )
        elif patch_embed_mode == "linear":
            if hex_stride is None:
                raise ValueError("hex_stride must be explicit for linear patch embedding")
            self.patch_embed = HexLinearPatchEmbed(
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                kernel_size=hex_kernel_size,
                lattice_stride=hex_stride,
            )
        else:
            raise ValueError("patch_embed_mode must be 'distance' or 'linear'")
        num_patches = estimate_hex_num_patches((img_size, img_size), hex_kernel_size, hex_stride)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if position_mode in {"learned", "learned+look", "learned+polar-look"}:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        else:
            self.register_parameter("pos_embed", None)
        if position_mode in {"look", "learned+look"}:
            coordinates = build_hex_patch_coordinates((img_size, img_size), hex_kernel_size, hex_stride)
            self.look_bias = HexLookBias(coordinates, num_heads=num_heads)
        else:
            self.look_bias = None
        if position_mode in {"polar-look", "learned+polar-look"}:
            self.polar_look_layers = nn.ModuleList(
                PolarPrototypeLook(
                    num_heads=num_heads,
                    in_channels=in_chans,
                )
                for _ in range(depth)
            )
            # One gate per Transformer layer and attention head.  A zero
            # initialization makes a transferred model exactly reproduce its
            # no-look baseline while still allowing the gates to learn first.
            self.polar_look_gate = nn.Parameter(
                torch.full((depth, num_heads), float(polar_look_gate_init))
            )
        else:
            self.polar_look_layers = None
            self.register_parameter("polar_look_gate", None)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(init_vit_weights)

    @property
    def polar_look(self) -> PolarPrototypeLook | None:
        """Compatibility accessor for the first layer's polar-look module."""
        if self.polar_look_layers is None:
            return None
        return self.polar_look_layers[0]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        image = x
        x = self.patch_embed(image)  # (B, N, D)
        bsz, num_patches, _ = x.shape

        cls = self.cls_token.expand(bsz, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)

        if self.pos_embed is not None and x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"pos_embed length mismatch: got tokens={x.shape[1]} but pos_embed={self.pos_embed.shape[1]}; "
                "check img_size/hex_kernel_size."
            )

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        static_attn_bias = self.look_bias(include_cls=True) if self.look_bias is not None else None
        polar_rings = None
        polar_coverage = None
        if self.polar_look_layers is not None and self.polar_look_strength != 0.0:
            # P0 is immutable: extract its multi-scale polar rings once, then
            # let every layer/head match its own independent prototypes.
            with torch.no_grad():
                polar_rings, polar_coverage = self.polar_look_layers[0].ring_sampler(
                    image,
                    self.patch_embed.patch_centers_xy.to(device=image.device),
                    base_radius=self.polar_base_radius,
                    return_coverage=True,
                )

        for layer_index, blk in enumerate(self.blocks):
            attn_bias = static_attn_bias
            if polar_rings is not None and polar_coverage is not None:
                polar_layer = self.polar_look_layers[layer_index]
                patch_coordinates = self.patch_embed.coo_patchs.to(device=image.device)

                def compute_patch_bias(
                    rings: torch.Tensor,
                    coverage: torch.Tensor,
                    *,
                    module: PolarPrototypeLook = polar_layer,
                    coordinates: torch.Tensor = patch_coordinates,
                ) -> torch.Tensor:
                    return module.forward_rings(
                        rings,
                        coverage,
                        coordinates,
                        look_radius=self.polar_look_radius,
                    )[0]

                if self.training and torch.is_grad_enabled():
                    patch_bias = checkpoint(
                        compute_patch_bias,
                        polar_rings,
                        polar_coverage,
                        use_reentrant=False,
                    )
                else:
                    patch_bias = compute_patch_bias(polar_rings, polar_coverage)
                if patch_bias.shape[-2:] != (num_patches, num_patches):
                    raise ValueError(
                        "polar look patch count does not match HexPatchEmbed tokens: "
                        f"bias={tuple(patch_bias.shape[-2:])}, tokens={num_patches}"
                    )
                dynamic_attn_bias = patch_bias.new_zeros(
                    bsz,
                    patch_bias.shape[1],
                    num_patches + 1,
                    num_patches + 1,
                )
                dynamic_attn_bias[:, :, 1:, 1:] = patch_bias
                gate = self.polar_look_gate[layer_index].to(dynamic_attn_bias).view(1, -1, 1, 1)
                attn_bias = dynamic_attn_bias * (self.polar_look_strength * gate)
            x = blk(x, attn_bias=attn_bias)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats)

