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
        tokenizer_null_initial_score: float = 0.0,
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
            pose_softmax=True,
            use_null=True,
            null_initial_score=tokenizer_null_initial_score,
            match_metric="dot",
        )
        token_count = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, token_count, self.embed_dim)
            )
        else:
            self.register_parameter("pos_embed", None)
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
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        coo = self.patch_embed.coo_patchs
        patch_coordinates = torch.stack((coo.real, coo.imag), dim=-1)
        self.look_bank = SquarePatchDenseGridLook(
            image_size=image_size,
            patch_size=16,
            in_channels=3,
            num_heads=self.depth * self.num_heads,
            prototype_radial_bins=8,
            # Preserve two stored polar samples per full-period pose.  The
            # default 4/8 half-circle search therefore remains 16 bins, while
            # a 6/12 search uses 24 bins without angular-grid mismatch.
            prototype_angular_bins=2 * global_directions,
            source_directions=directions,
            source_direction_period=global_directions,
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
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # The image-to-ring preprocessing is intentionally immutable. Look
        # prototypes and grids remain trainable, while P0 is shared once.
        with torch.autocast(device_type=image.device.type, enabled=False):
            rings, coverage = self.look_bank.extract_rings(
                image.float(), track_input_grad=False
            )
            pose_weights = self.look_bank.pose_weights(rings, coverage)
            fields = self.look_bank.transformed_look_grids()
        pose_weights = pose_weights.flatten(-2)
        fields = fields.flatten(1, 2)
        for layer_index, block in enumerate(self.blocks):
            start = layer_index * self.num_heads
            stop = start + self.num_heads
            layer_pose = pose_weights[:, :, start:stop]
            layer_fields = fields[start:stop]
            tokens = block(
                tokens,
                structured_look=(layer_pose, layer_fields),
            )
        return self.norm(tokens)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image)[:, 0])
