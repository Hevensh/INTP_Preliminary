from __future__ import annotations

import torch
import torch.nn as nn

from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed
from layers.hex_differentiated_harmonic_patch_embed import (
    HexDifferentiatedHarmonicPatchEmbed,
)
from layers.center_pose_angular_look import CenterPoseAngularLook
from layers.center_pose_grid_look import CenterPoseGridLook
from layers.multiprobe_look import RotatingMultiProbeLook, IndependentMultiProbeLook, aggregate_pose_grids, sample_pose_grids
from layers.mini_vit import TransformerBlock, init_vit_weights
from layers.square_patch_dense_grid_look import SquarePatchDenseGridLook
from layers.two_ring_circular_look import TwoRingCircularLookMatcher


class DeiTTinyRotHexLook(nn.Module):
    """DeiT-Tiny with the dot-product cos/sin Hex tokenizer and dense Look Bias.

    The 36 Look prototypes correspond exactly to 12 Transformer layers times
    three attention heads. P0 rings are extracted once, then each layer/head
    produces its own directed attention bias.  By default, the Look field uses
    the tokenizer's full pose period for its angular resolution and twice the
    tokenizer scale count for its radial resolution.
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
        angular_bins_per_radius: int = 4,
        look_compact_variable_rings: bool = False,
        image_look: bool = True,
        center_pose_look: bool = False,
        center_pose_grid_look: bool = False,
        center_look_layers_per_probe: int = 1,
        image_look_probes: int = 1,
        feature_look_probes: int = 1,
        feature_look_rotating_probes: bool = False,
        feature_ring_look: bool = False,
        feature_ring_start_layer: int = 0,
        feature_ring_group_size: int = 4,
        feature_ring_frequency: bool = False,
        prototype_chunk_size: int = 16,
        tokenizer_null_initial_score: float = 0.0,
        progressive_differentiation: bool = False,
        stripe_longitudinal_bins: int = 3,
        stripe_offset_subdivisions: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = 192
        self.depth = 12
        self.num_heads = 3
        if min(image_look_probes, feature_look_probes) < 1:
            raise ValueError("Look probe counts must be positive")
        multi_feature = feature_look_rotating_probes or feature_look_probes > 1
        if multi_feature and not center_pose_grid_look:
            raise ValueError("multiple/rotating Feature probes require grid Feature Look")
        self.image_look_probes = image_look_probes
        self.grid_first_look = image_look_probes > 1 or multi_feature
        if self.grid_first_look and (global_directions != 12 or len(kernel_sizes) != 2):
            raise ValueError("grid-first experiment currently requires a 4x12 Look grid")
        if self.grid_first_look and feature_ring_look:
            raise ValueError("grid-first experiment does not include legacy ring Look")
        if self.grid_first_look and center_pose_look and not center_pose_grid_look:
            raise ValueError("grid-first experiment requires grid rather than legacy axial Feature Look")
        self.use_pos_embed = bool(use_pos_embed)
        tokenizer_type = (
            HexDifferentiatedHarmonicPatchEmbed
            if progressive_differentiation
            else HexRotatingHarmonicPatchEmbed
        )
        tokenizer_kwargs = dict(
            img_size=image_size,
            in_chans=3,
            embed_dim=self.embed_dim,
            lattice_stride=lattice_stride,
            kernel_sizes=kernel_sizes,
            bases=bases,
            directions=directions,
            global_directions=global_directions,
            angular_bins_per_radius=angular_bins_per_radius,
            prototype_chunk_size=prototype_chunk_size,
            null_initial_score=tokenizer_null_initial_score,
        )
        if progressive_differentiation:
            tokenizer_kwargs.update(
                stripe_longitudinal_bins=stripe_longitudinal_bins,
                stripe_offset_subdivisions=stripe_offset_subdivisions,
            )
        else:
            tokenizer_kwargs.update(
                pose_softmax=True,
                use_null=True,
                match_metric="dot",
            )
        self.patch_embed = tokenizer_type(**tokenizer_kwargs)
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
        if center_pose_look and center_pose_grid_look:
            raise ValueError("simple and grid Center Look are mutually exclusive")
        self.center_pose_grid_look = bool(center_pose_grid_look)
        self.center_pose_look = bool(center_pose_look or center_pose_grid_look)
        if self.center_pose_look and feature_ring_look:
            raise ValueError("center-pose Look and feature-ring Look are exclusive")
        self.image_look = bool(image_look)
        if not self.image_look and not self.center_pose_look:
            raise ValueError("at least one Look branch must be enabled")
        if feature_ring_look and not self.image_look:
            raise ValueError("feature-ring Look requires the image Look branch")
        # Keep the Look lattice structurally coupled to the tokenizer instead
        # of silently retaining an 8 x 4 field when the geometric pose search
        # changes.  A half-6 / full-12 tokenizer therefore uses 12 x 4, while
        # the established half-4 / full-8 tokenizer remains 8 x 4.
        self.look_direction_bins = int(global_directions)
        self.look_radial_bins = 2 * len(kernel_sizes)
        self.look_bank = SquarePatchDenseGridLook(
            image_size=image_size,
            patch_size=16,
            in_channels=3,
            num_heads=self.depth * self.num_heads * image_look_probes,
            prototype_radial_bins=(12 if look_compact_variable_rings else 8),
            # Preserve two stored polar samples per full-period pose.  The
            # default 4/8 half-circle search therefore remains 16 bins, while
            # a 6/12 search uses 24 bins without angular-grid mismatch.
            prototype_angular_bins=2 * global_directions,
            source_directions=directions,
            source_direction_period=global_directions,
            scales=(1.0, 0.5),
            prototype_radius=12.0,
            look_direction_bins=self.look_direction_bins,
            look_radial_bins=self.look_radial_bins,
            look_radius=4.0,
            patch_centers_xy=self.patch_embed.patch_centers_xy,
            patch_coordinates_xy=patch_coordinates,
            compact_angular_bins_per_radius=(
                angular_bins_per_radius if look_compact_variable_rings else None
            ),
            compact_kernel_sizes=(
                kernel_sizes if look_compact_variable_rings else None
            ),
            compact_lattice_stride=(
                lattice_stride if look_compact_variable_rings else None
            ),
        ) if self.image_look else None
        if self.center_pose_look:
            # Patch-to-patch bias in the final block cannot affect that same
            # block's CLS output.  Keep Center Look on the first 11 blocks;
            # the image Look branch, when present, still spans all 12.
            center_builder = (
                (RotatingMultiProbeLook if feature_look_rotating_probes else
                 IndependentMultiProbeLook if feature_look_probes > 1 else CenterPoseGridLook)
                if self.center_pose_grid_look
                else CenterPoseAngularLook
            )
            center_kwargs = {}
            if self.center_pose_grid_look:
                center_kwargs = {
                    "radial_bins": self.look_radial_bins,
                    "direction_bins": self.look_direction_bins,
                    "look_radius": 4.0,
                    "layers_per_probe": center_look_layers_per_probe,
                }
                if multi_feature:
                    center_kwargs["probes"] = feature_look_probes
            self.center_look = center_builder(
                coordinates=patch_coordinates,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                depth=self.depth - 1,
                axes=directions,
                null_initial_score=tokenizer_null_initial_score,
                **center_kwargs,
            )
        else:
            self.center_look = None
        self.feature_ring_look = bool(feature_ring_look)
        if feature_ring_group_size <= 0:
            raise ValueError("feature_ring_group_size must be positive")
        self.feature_ring_group_size = int(feature_ring_group_size)
        self.feature_ring_frequency = bool(feature_ring_frequency)
        if self.feature_ring_look:
            if global_directions != 12:
                raise ValueError(
                    "C6 feature Look requires a full 12-direction Look field"
                )
            self.feature_ring_matcher = TwoRingCircularLookMatcher(
                coordinates=patch_coordinates,
                depth=self.depth,
                num_heads=self.num_heads,
                head_dim=self.embed_dim // self.num_heads,
                start_layer=feature_ring_start_layer,
            )
        else:
            self.feature_ring_matcher = None

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(image)
        if self.center_look is not None:
            # Remove the ordinary output bias before interpreting adjacent
            # dimensions as geometric cosine/sine pairs.
            shared_pose = self.center_look.pose_weights(
                tokens - self.patch_embed.output_bias
            )
        else:
            shared_pose = None
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # The image-to-ring preprocessing is intentionally immutable. Look
        # prototypes and grids remain trainable, while P0 is shared once.
        if self.look_bank is not None:
            with torch.autocast(device_type=image.device.type, enabled=False):
                rings, coverage = self.look_bank.extract_rings(
                    image.float(), track_input_grad=False
                )
                fields = None if self.grid_first_look else self.look_bank.transformed_look_grids()
                if not self.look_bank.compact_variable_rings:
                    # Preserve the established dense-ring baseline exactly.
                    pose_weights = self.look_bank.pose_weights(rings, coverage)
            if self.look_bank.compact_variable_rings:
                # Restore the caller's AMP context for the large K24/K12 GEMMs.
                # pose_weights() promotes only the small routing softmax to fp32.
                pose_weights = self.look_bank.pose_weights(rings, coverage)
            if not self.grid_first_look:
                pose_weights = pose_weights.flatten(-2)
                fields = fields.flatten(1, 2)
        else:
            pose_weights = fields = None
        stage_ring_biases = None
        image_layers = feature_groups = None
        if self.grid_first_look:
            if pose_weights is not None:
                # Unbind once: backward stacks layer gradients once instead of
                # repeatedly zero-filling the entire all-layer response tensor.
                image_layers = pose_weights.reshape(
                    *pose_weights.shape[:2], self.depth, self.num_heads,
                    self.image_look_probes, *pose_weights.shape[-2:]
                ).unbind(2)
            if shared_pose is not None:
                feature_groups = shared_pose.unbind(2)
        for layer_index, block in enumerate(self.blocks):
            layer_pose = layer_fields = None
            if self.grid_first_look:
                dense_bias = None
                grids = None
                if self.look_bank is not None:
                    count = self.num_heads * self.image_look_probes
                    start = layer_index * count
                    p = image_layers[layer_index]
                    grid = self.look_bank.look_grid[start:start+count].reshape(
                        self.num_heads, self.image_look_probes, self.look_radial_bins, self.look_direction_bins)
                    grids = aggregate_pose_grids(p, grid, period=self.look_direction_bins)
                if self.center_look is not None and layer_index < self.center_look.depth:
                    cp = feature_groups[layer_index // self.center_look.layers_per_probe]
                    if isinstance(self.center_look, RotatingMultiProbeLook):
                        center_grid = self.center_look.pose_grids(cp, layer_index)
                        if grids is not None:
                            # Both branches' radius-4 fields share spatial sampling.
                            # Add before interpolation: 2 large samplings, not 2+1.
                            grids = torch.cat((grids[:, :, :, :1] + center_grid,
                                               grids[:, :, :, 1:]), dim=3)
                            cb = None
                        else:
                            cb = sample_pose_grids(center_grid, self.center_look, image=False)
                    else:
                        cb = torch.einsum("bqha,haqk->bhqk", cp, self.center_look.fields(layer_index, dtype=cp.dtype))
                    if cb is not None:
                        dense_bias = cb if dense_bias is None else dense_bias + cb
                if grids is not None:
                    ib = sample_pose_grids(grids, self.look_bank, image=True)
                    dense_bias = ib if dense_bias is None else dense_bias + ib
                if dense_bias is not None:
                    # Keep the existing fused attention and its dense-bias backward.
                    # A zero singleton structured term avoids any M-dependent pose loop.
                    n = tokens.shape[1] - 1
                    zero_pose = tokens.new_zeros(tokens.shape[0], n, self.num_heads, 1)
                    zero_field = tokens.new_zeros(self.num_heads, 1, n, n)
                    tokens = block(tokens, structured_look=(zero_pose, zero_field, dense_bias))
                else:
                    tokens = block(tokens)
                continue
            if self.look_bank is not None:
                start = layer_index * self.num_heads
                stop = start + self.num_heads
                layer_pose = pose_weights[:, :, start:stop]
                layer_fields = fields[start:stop]
            if self.center_look is not None and layer_index < self.center_look.depth:
                center_pose = (
                    self.center_look.pose_for_layer(shared_pose, layer_index)
                    if self.center_pose_grid_look
                    else shared_pose
                )
                center_fields = self.center_look.fields(
                    layer_index, dtype=tokens.dtype
                )
                if layer_pose is None:
                    layer_pose = center_pose
                    layer_fields = center_fields
                else:
                    # Structured attention is linear in the Look terms.  By
                    # concatenating the two pose bases, one Triton call adds
                    # image Look and Center Look exactly, without a dense bias
                    # tensor or a separate branch gate.
                    layer_pose = torch.cat((layer_pose, center_pose), dim=-1)
                    layer_fields = torch.cat((layer_fields, center_fields), dim=1)
            norm1_input = None
            if (
                self.feature_ring_matcher is not None
                and layer_index >= self.feature_ring_matcher.start_layer
            ):
                stage_offset = (
                    layer_index - self.feature_ring_matcher.start_layer
                ) % self.feature_ring_group_size
                if stage_offset == 0:
                    norm1_input = block.norm1(tokens)
                    stop_layer = min(
                        layer_index + self.feature_ring_group_size, self.depth
                    )
                    stage_indices = tuple(range(layer_index, stop_layer))
                    stage_ring_biases = (
                        self.feature_ring_matcher.dense_look_bias_for_layers(
                            norm1_input[:, 1:],
                            layer_indices=stage_indices,
                            frequency_domain=self.feature_ring_frequency,
                        )
                    )
                if stage_ring_biases is None:
                    raise RuntimeError("feature-ring stage cache was not initialized")
                dense_ring_bias = stage_ring_biases[stage_offset]
            else:
                dense_ring_bias = None
            structured_look = None
            if layer_pose is not None:
                structured_look = (layer_pose, layer_fields, dense_ring_bias)
            elif dense_ring_bias is not None:
                raise RuntimeError("dense ring bias requires a structured Look branch")
            tokens = block(tokens, structured_look=structured_look, norm1_input=norm1_input)
        return self.norm(tokens)

    def experiment_diagnostics(self) -> dict[str, object]:
        if self.center_look is None:
            return {}
        name = (
            "center_pose_grid_look"
            if self.center_pose_grid_look
            else "center_pose_angular_look"
        )
        return {name: self.center_look.diagnostics(), "image_look_probes": self.image_look_probes,
                "grid_first_look": self.grid_first_look}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image)[:, 0])
