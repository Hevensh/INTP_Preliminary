from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.hex_patch_geometry import HexPatchGeometry
from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat


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
        radial_bins: int = 12,
        angular_bins_per_radius: int = 4,
        prototype_chunk_size: int = 16,
        prototype_std: float = 0.02,
        pose_softmax: bool = False,
        use_null: bool = False,
        null_initial_score: float = 0.0,
        match_metric: str = "dot",
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        if embed_dim != 2 * bases:
            raise ValueError("embed_dim must equal 2 * bases for cosine/sine output")
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
        self.renderers = nn.ModuleList(
            _PolarRenderer(
                geometry,
                radial_bins=radial_bins,
                ring_counts=counts,
                ring_offsets=offsets,
                directions=directions,
                direction_step=direction_step,
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

        theta = torch.arange(directions) * direction_step
        self.register_buffer(
            "direction_coefficients",
            torch.stack((theta.cos(), theta.sin()), dim=-1),
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
            rendered = renderer(prototype)
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
            return torch.cat(chunks, dim=-1) + self.output_bias
