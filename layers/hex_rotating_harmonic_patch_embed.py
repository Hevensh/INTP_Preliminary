from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.hex_patch_geometry import HexPatchGeometry
from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat
from layers.triton_polar_renderer import triton_polar_render


class HexRotatingHarmonicPatchEmbed(nn.Module):
    """Linear rotating tokenizer: one cosine/sine response pair per prototype."""

    def __init__(
        self,
        *,
        img_size: int,
        in_chans: int,
        embed_dim: int,
        lattice_stride: int = 18,
        kernel_sizes: tuple[int, ...] = (24, 12),
        bases: int = 96,
        directions: int = 4,
        global_directions: int = 8,
        direction_angles_degrees: tuple[float, ...] | None = None,
        radial_bins: int = 12,
        angular_bins_per_radius: int = 4,
        prototype_chunk_size: int = 16,
        prototype_std: float = 0.02,
        pose_softmax: bool = False,
        use_null: bool = False,
        null_initial_score: float = 0.0,
        match_metric: str = "dot",
        raw_direction_output: bool = False,
        harmonic_orders: tuple[int, ...] = (1,),
        output_groups: int = 1,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        self.raw_direction_output = bool(raw_direction_output)
        if not harmonic_orders or any(int(o) != o or o <= 0 for o in harmonic_orders):
            raise ValueError('harmonic_orders must be positive integers')
        if output_groups < 1 or bases % output_groups:
            raise ValueError('bases must divide into output_groups')
        if raw_direction_output and (output_groups != 1 or tuple(harmonic_orders) != (1,)):
            raise ValueError('Raw output cannot group circular moments')
        self.output_groups = output_groups
        self.harmonic_orders = tuple(harmonic_orders)
        if raw_direction_output and (pose_softmax or use_null):
            raise ValueError('Raw direction output does not use pose softmax/null')
        if embed_dim != (bases if raw_direction_output else 2 * len(harmonic_orders) * bases // output_groups):
            raise ValueError('embed_dim must match grouped moment dimensions')
        if not 1 <= directions <= global_directions:
            raise ValueError("directions must be in [1, global_directions]")
        if prototype_chunk_size <= 0 or prototype_std <= 0:
            raise ValueError("prototype_chunk_size and prototype_std must be positive")
        self.embed_dim = int(embed_dim)
        self.bases = int(bases)
        self.directions = int(directions)
        self.scales = len(kernel_sizes)
        self.prototype_chunk_size = int(prototype_chunk_size)
        self.pose_softmax = bool(pose_softmax)
        self.use_null = bool(use_null)
        if match_metric not in {"dot", "relative_l1"}:
            raise ValueError("match_metric must be dot or relative_l1")
        self.match_metric = match_metric
        if self.use_null and not self.pose_softmax:
            raise ValueError("use_null requires pose_softmax")

        self.geometries = nn.ModuleList(
            HexPatchGeometry(img_size, in_chans, int(kernel), lattice_stride)
            for kernel in kernel_sizes
        )
        if len({geometry.num_patches for geometry in self.geometries}) != 1:
            raise ValueError("all scales must produce the same Hex patch centers")
        reference_centers = self.geometries[0].patch_centers_xy
        if any(
            not torch.equal(reference_centers, geometry.patch_centers_xy)
            for geometry in self.geometries[1:]
        ):
            raise ValueError("all scales must share identical Hex patch centers")

        counts = torch.tensor(
            [angular_bins_per_radius * (radius + 1) for radius in range(radial_bins)],
            dtype=torch.long,
        )
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.register_buffer("ring_counts", counts, persistent=False)
        self.register_buffer("ring_offsets", offsets, persistent=False)
        direction_step = 2 * math.pi / global_directions
        theta = torch.arange(directions) * direction_step
        if direction_angles_degrees is not None:
            theta = torch.tensor(direction_angles_degrees, dtype=torch.float32)
            if theta.shape != (directions,) or not torch.isfinite(theta).all():
                raise ValueError("direction_angles_degrees must contain directions finite angles")
            if len(set(float(v) % 360 for v in direction_angles_degrees)) != directions:
                raise ValueError("direction_angles_degrees must be distinct modulo 360")
            theta = torch.deg2rad(theta)
        self.renderers = nn.ModuleList(
            _PolarRenderer(
                geometry,
                radial_bins=radial_bins,
                ring_counts=counts,
                ring_offsets=offsets,
                directions=directions,
                direction_step=direction_step,
                direction_angles=theta,
            )
            for geometry in self.geometries
        )

        self.prototype = nn.Parameter(
            torch.randn(bases, in_chans, int(offsets[-1])) * prototype_std
        )
        if self.use_null:
            self.null_score = nn.Parameter(
                torch.full((bases,), float(null_initial_score))
            )
        self.output_bias = nn.Parameter(torch.zeros(embed_dim))

        reference_cover_mass = self.renderers[0].support_cover.sum()
        for index, renderer in enumerate(self.renderers):
            # Keep the largest kernel's ordinary convolution-like accumulated
            # magnitude.  Compensate smaller kernels up to that same reference
            # mass, instead of shrinking every scale to a unit-sum average.
            raw_cover = renderer.support_cover
            cover = raw_cover * (reference_cover_mass / raw_cover.sum())
            self.register_buffer(f"scale_cover_{index}", cover, persistent=False)

        self.register_buffer(
            "direction_coefficients",
            torch.stack([f(theta * order) for order in harmonic_orders
                         for f in (torch.cos, torch.sin)], dim=-1),
            persistent=False,
        )

    @property
    def num_patches(self) -> int:
        return self.geometries[0].num_patches

    @property
    def patch_centers_xy(self) -> torch.Tensor:
        return self.geometries[0].patch_centers_xy

    @property
    def coo_patchs(self) -> torch.Tensor:
        return self.geometries[0].coo_patchs

    def _chunk_response(
        self,
        patches: list[torch.Tensor],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        prototype = self.prototype[start:stop]
        pose_score = None
        for scale_index, (patch, renderer) in enumerate(zip(patches, self.renderers)):
            cover = getattr(self, f"scale_cover_{scale_index}")
            rendered = triton_polar_render(prototype, renderer)
            if self.match_metric == "dot":
                score = rotating_dot_score(patch, rendered)
            else:
                # Compare each prototype against the zero-prototype baseline:
                #   score = ||x||_1,c - ||x - w||_1,c
                # Multiplying x and w by the non-negative cover makes ordinary
                # L1 distance exactly equal to the required weighted L1. The
                # fused CUDA path streams the reduction without materialising
                # a B*N*P*D*C*M difference tensor.
                patch_flat = patch
                rendered_flat = (
                    rendered * cover[None, None, None]
                ).flatten(2)
                flat_query = patch_flat.reshape(-1, patch_flat.shape[-1])
                flat_prototype = rendered_flat.reshape(
                    -1, rendered_flat.shape[-1]
                )
                if flat_query.is_cuda:
                    from layers.triton_negative_l1 import negative_l1_distance

                    negative_distance = negative_l1_distance(
                        flat_query, flat_prototype
                    )
                else:
                    negative_distance = -torch.cdist(
                        flat_query, flat_prototype, p=1
                    )
                negative_distance = negative_distance.view(
                    patch.shape[0], patch.shape[1], stop - start, self.directions
                )
                zero_distance = patch_flat.abs().sum(-1)
                score = zero_distance[:, :, None, None] + negative_distance
            pose_score = score if pose_score is None else pose_score + score

        if self.pose_softmax:
            if self.use_null:
                null = self.null_score[start:stop][None, None, :, None].expand(
                    pose_score.shape[0], pose_score.shape[1], -1, -1
                )
                pose_score = torch.cat((pose_score, null), dim=-1).softmax(-1)[..., :-1]
            else:
                pose_score = pose_score.softmax(-1)

        if self.raw_direction_output:
            return pose_score.transpose(-1, -2)
        # B,N,P,D times D,2 -> B,N,P,2.  The final dimension is interleaved
        # as prototype0(cos,sin), prototype1(cos,sin), ...
        response = torch.einsum(
            "qnpd,dc->qnpc", pose_score, self.direction_coefficients
        )
        return response.flatten(2, 3)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            patches = [geometry(image) for geometry in self.geometries]
            # Each scale is weighted/flattened once and shared by every
            # prototype chunk. This avoids retaining six duplicate K24/K12
            # patch tensors until backward at the default 96/16 split.
            patches = [
                weighted_patch_flat(
                    patch, getattr(self, f"scale_cover_{scale_index}")
                )
                for scale_index, patch in enumerate(patches)
            ]
            chunks = [
                self._chunk_response(
                    patches,
                    start,
                    min(start + self.prototype_chunk_size, self.bases),
                )
                for start in range(0, self.bases, self.prototype_chunk_size)
            ]
            output = torch.cat(chunks, dim=-1)
            if self.output_groups > 1:
                # Contiguous groups of prototypes. SUM, not mean or a learned map.
                output = output.reshape(*output.shape[:-1], self.output_groups,
                                        self.embed_dim).sum(-2)
            return output + self.output_bias
