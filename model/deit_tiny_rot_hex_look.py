from __future__ import annotations

import torch
import torch.nn as nn

from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed
from layers.mini_vit import TransformerBlock, init_vit_weights
from layers.square_patch_dense_grid_look import SquarePatchDenseGridLook


class DeiTTinyRotHexLook(nn.Module):
    """DeiT-Tiny with the dot-product cos/sin Hex tokenizer and dense Look Bias.

    The 36 Look prototypes correspond exactly to 12 Transformer layers times
    three attention heads. P0 rings are extracted once, then each layer/head
    produces its own directed 8-direction x 4-radius attention bias.
    """

    def __init__(
        self,
        *,
        num_classes: int = 100,
        image_size: int = 224,
        use_pos_embed: bool,
        lattice_stride: int = 18,
        kernel_sizes: tuple[int, ...] = (24, 12),
        bases: int = 96,
        directions: int = 4,
        global_directions: int = 8,
        prototype_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = 192
        self.depth = 12
        self.num_heads = 3
        self.use_pos_embed = bool(use_pos_embed)
        self.patch_embed = HexRotatingHarmonicPatchEmbed(
            img_size=image_size,
            in_chans=3,
            embed_dim=self.embed_dim,
            lattice_stride=lattice_stride,
            kernel_sizes=kernel_sizes,
            bases=bases,
            directions=directions,
            global_directions=global_directions,
            prototype_chunk_size=prototype_chunk_size,
            pose_softmax=False,
            use_null=False,
            match_metric="dot",
        )
        token_count = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, self.embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                self.embed_dim,
                self.num_heads,
                mlp_ratio=4.0,
                norm_eps=1e-6,
            )
            for _ in range(self.depth)
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Linear(self.embed_dim, num_classes)

        # Keep shared-backbone initialization independent of whether PE is
        # active. This makes the two ablation arms differ only by PE usage.
        self.apply(init_vit_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        coo = self.patch_embed.coo_patchs
        patch_coordinates = torch.stack((coo.real, coo.imag), dim=-1)
        self.look_bank = SquarePatchDenseGridLook(
            image_size=image_size,
            patch_size=16,
            in_channels=3,
            num_heads=self.depth * self.num_heads,
            prototype_radial_bins=8,
            prototype_angular_bins=16,
            source_directions=4,
            source_direction_period=8,
            scales=(1.0, 0.5),
            prototype_radius=12.0,
            look_direction_bins=8,
            look_radial_bins=4,
            look_radius=4.0,
            patch_centers_xy=self.patch_embed.patch_centers_xy,
            patch_coordinates_xy=patch_coordinates,
        )

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(image)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # The image-to-ring preprocessing is intentionally immutable. Look
        # prototypes and grids remain trainable, while P0 is shared once.
        with torch.autocast(device_type=image.device.type, enabled=False):
            all_bias, _ = self.look_bank(
                image.float(), include_cls=True, track_input_grad=False
            )
        all_bias = all_bias.reshape(
            image.shape[0], self.depth, self.num_heads,
            tokens.shape[1], tokens.shape[1],
        )
        for layer_index, block in enumerate(self.blocks):
            tokens = block(tokens, attn_bias=all_bias[:, layer_index])
        return self.norm(tokens)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image)[:, 0])
