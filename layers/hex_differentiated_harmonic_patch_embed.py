from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from layers.hex_patch_geometry import HexPatchGeometry
from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat


FAMILY_NAMES = ("full", "angular", "stripe", "color")
FAMILY_TO_CODE = {name: index for index, name in enumerate(FAMILY_NAMES)}


class _AngularRenderer(nn.Module):
    def __init__(self, geometry: HexPatchGeometry, *, bins: int, directions: int, step: float):
        super().__init__()
        offsets = geometry.patch_offsets_xy
        angle = torch.remainder(torch.atan2(offsets[:, 1], offsets[:, 0]), 2 * math.pi)
        shifted = angle[None] - torch.arange(directions)[:, None] * step
        position = torch.remainder(shifted, 2 * math.pi) / (2 * math.pi) * bins
        raw = position.floor()
        self.register_buffer("index0", raw.long() % bins, persistent=False)
        self.register_buffer("index1", (raw.long() + 1) % bins, persistent=False)
        self.register_buffer("fraction", position - raw, persistent=False)

    def forward(self, prototype: torch.Tensor) -> torch.Tensor:
        low = prototype[..., self.index0]
        high = prototype[..., self.index1]
        rendered = low * (1 - self.fraction[None, None]) + high * self.fraction[None, None]
        return rendered.permute(0, 2, 1, 3).contiguous()


class _StripeRenderer(nn.Module):
    def __init__(
        self,
        geometry: HexPatchGeometry,
        *,
        transverse_bins: int,
        longitudinal_bins: int,
        directions: int,
        step: float,
    ) -> None:
        super().__init__()
        self.transverse_bins = int(transverse_bins)
        self.longitudinal_bins = int(longitudinal_bins)
        self.directions = int(directions)
        self.step = float(step)
        offsets = geometry.patch_offsets_xy
        self.register_buffer("sample_x", offsets[:, 0], persistent=False)
        self.register_buffer("sample_y", offsets[:, 1], persistent=False)
        self.radius = float(geometry.kernel_size) / 2

    @staticmethod
    def _indices(coordinate: torch.Tensor, bins: int) -> tuple[torch.Tensor, ...]:
        position = ((coordinate + 1) * 0.5 * (bins - 1)).clamp(0, bins - 1)
        low = position.floor().long()
        high = (low + 1).clamp_max(bins - 1)
        return low, high, position - position.floor()

    def forward(self, prototype: torch.Tensor, angle_offset: torch.Tensor) -> torch.Tensor:
        if prototype.ndim != 4:
            raise ValueError("stripe prototype must have shape (P,C,T,L)")
        theta = angle_offset[:, None, None] + (
            torch.arange(
                self.directions, device=prototype.device, dtype=prototype.dtype
            )[None, :, None]
            * self.step
        )
        x = self.sample_x.to(prototype.dtype)[None, None]
        y = self.sample_y.to(prototype.dtype)[None, None]
        across = (x * theta.cos() + y * theta.sin()) / self.radius
        along = (-x * theta.sin() + y * theta.cos()) / self.radius
        i0, i1, fi = self._indices(across, self.transverse_bins)
        j0, j1, fj = self._indices(along, self.longitudinal_bins)
        source = prototype.flatten(2)[:, :, None].expand(-1, -1, self.directions, -1)
        channels = prototype.shape[1]

        def gather(ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
            index = (ii * self.longitudinal_bins + jj)[:, None].expand(
                -1, channels, -1, -1
            )
            return source.gather(3, index)

        p00, p10 = gather(i0, j0), gather(i1, j0)
        p01, p11 = gather(i0, j1), gather(i1, j1)
        low = p00 * (1 - fi[:, None]) + p10 * fi[:, None]
        high = p01 * (1 - fi[:, None]) + p11 * fi[:, None]
        rendered = low * (1 - fj[:, None]) + high * fj[:, None]
        return rendered.permute(0, 2, 1, 3).contiguous()


@dataclass(frozen=True)
class PrototypeConversion:
    name: str
    old_parameter: nn.Parameter
    new_parameter: nn.Parameter
    transform: torch.Tensor


class HexDifferentiatedHarmonicPatchEmbed(nn.Module):
    """Hex tokenizer whose Full prototypes progressively specialize.

    Every source prototype permanently owns the same pair of output channels.
    Full, Angular, and Stripe use circular cosine/sine moments.  Color remains
    implemented for old differentiated checkpoints, but new differentiation
    plans only choose between Angular and Stripe.
    """

    def __init__(
        self,
        *,
        img_size: int,
        in_chans: int,
        embed_dim: int,
        lattice_stride: int = 18,
        kernel_sizes: tuple[int, ...] = (24, 12),
        bases: int = 96,
        directions: int = 6,
        global_directions: int = 12,
        radial_bins: int = 12,
        angular_bins_per_radius: int = 3,
        prototype_chunk_size: int = 16,
        prototype_std: float = 0.02,
        null_initial_score: float = 0.0,
        stripe_longitudinal_bins: int = 3,
        stripe_offset_subdivisions: int = 4,
    ) -> None:
        super().__init__()
        if embed_dim != 2 * bases:
            raise ValueError("embed_dim must equal two times bases")
        if len(kernel_sizes) != 2:
            raise ValueError("differentiated Color encoding requires exactly two scales")
        if not 1 <= directions <= global_directions:
            raise ValueError("directions must be in [1, global_directions]")
        if min(prototype_chunk_size, stripe_longitudinal_bins, stripe_offset_subdivisions) <= 0:
            raise ValueError("chunk size and stripe resolutions must be positive")
        self.embed_dim = int(embed_dim)
        self.bases = int(bases)
        self.directions = int(directions)
        self.global_directions = int(global_directions)
        self.scales = len(kernel_sizes)
        self.prototype_chunk_size = int(prototype_chunk_size)
        self.stripe_longitudinal_bins = int(stripe_longitudinal_bins)
        self.stripe_offset_subdivisions = int(stripe_offset_subdivisions)
        self.direction_step = 2 * math.pi / global_directions

        self.geometries = nn.ModuleList(
            HexPatchGeometry(img_size, in_chans, int(kernel), lattice_stride)
            for kernel in kernel_sizes
        )
        if len({geometry.num_patches for geometry in self.geometries}) != 1:
            raise ValueError("all scales must produce identical Hex patch counts")
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
        self.angular_bins = int(counts.max())
        self.stripe_transverse_bins = int(kernel_sizes[0])
        self.full_parameter_width = int(offsets[-1])

        self.renderers = nn.ModuleList(
            _PolarRenderer(
                geometry,
                radial_bins=radial_bins,
                ring_counts=counts,
                ring_offsets=offsets,
                directions=directions,
                direction_step=self.direction_step,
            )
            for geometry in self.geometries
        )
        self.angular_renderers = nn.ModuleList(
            _AngularRenderer(
                geometry,
                bins=self.angular_bins,
                directions=directions,
                step=self.direction_step,
            )
            for geometry in self.geometries
        )
        self.stripe_renderers = nn.ModuleList(
            _StripeRenderer(
                geometry,
                transverse_bins=self.stripe_transverse_bins,
                longitudinal_bins=self.stripe_longitudinal_bins,
                directions=directions,
                step=self.direction_step,
            )
            for geometry in self.geometries
        )

        self.prototype_bank = nn.ParameterList(
            [
                nn.Parameter(torch.randn(in_chans, self.full_parameter_width) * prototype_std)
                for _ in range(bases)
            ]
        )
        self.null_score = nn.Parameter(torch.full((bases,), float(null_initial_score)))
        self.output_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer(
            "family_code", torch.full((bases,), FAMILY_TO_CODE["full"], dtype=torch.long)
        )
        self.register_buffer("stripe_angle_offset", torch.zeros(bases))

        reference_cover_mass = self.renderers[0].support_cover.sum()
        for index, renderer in enumerate(self.renderers):
            raw_cover = renderer.support_cover
            cover = raw_cover * (reference_cover_mass / raw_cover.sum())
            self.register_buffer(f"scale_cover_{index}", cover, persistent=False)

        theta = torch.arange(directions) * self.direction_step
        self.register_buffer(
            "direction_coefficients",
            torch.stack((theta.cos(), theta.sin()), dim=-1),
            persistent=False,
        )
        self._build_projection_operators()

    @property
    def num_patches(self) -> int:
        return self.geometries[0].num_patches

    @property
    def patch_centers_xy(self) -> torch.Tensor:
        return self.geometries[0].patch_centers_xy

    @property
    def coo_patchs(self) -> torch.Tensor:
        return self.geometries[0].coo_patchs

    def family_name(self, base: int) -> str:
        return FAMILY_NAMES[int(self.family_code[base])]

    def family_counts(self) -> dict[str, int]:
        return {
            family: int((self.family_code == code).sum())
            for family, code in FAMILY_TO_CODE.items()
        }

    @staticmethod
    def _renderer_matrix(renderer: nn.Module, shape: tuple[int, ...], offset=None) -> torch.Tensor:
        width = math.prod(shape)
        basis = torch.eye(width).reshape(width, 1, *shape)
        if offset is None:
            rendered = renderer(basis)
        else:
            rendered = renderer(basis, torch.full((width,), float(offset)))
        return rendered[:, 0, 0].T.contiguous()

    def _build_projection_operators(self) -> None:
        """Precompute weighted least-squares Full-to-family maps on K24."""
        with torch.no_grad():
            full_matrix = self._renderer_matrix(
                self.renderers[0], (self.full_parameter_width,)
            )
            angular_matrix = self._renderer_matrix(
                self.angular_renderers[0], (self.angular_bins,)
            )
            color_matrix = torch.ones(full_matrix.shape[0], 1)
            offset_values = torch.arange(self.stripe_offset_subdivisions) * (
                self.direction_step / self.stripe_offset_subdivisions
            )
            stripe_matrices = torch.stack(
                [
                    self._renderer_matrix(
                        self.stripe_renderers[0],
                        (self.stripe_transverse_bins, self.stripe_longitudinal_bins),
                        float(offset),
                    )
                    for offset in offset_values
                ]
            )
            root_cover = self.scale_cover_0.sqrt()[:, None]
            weighted_full = full_matrix * root_cover

            def transform(matrix: torch.Tensor) -> torch.Tensor:
                return torch.linalg.pinv(matrix * root_cover) @ weighted_full

            color_transform = transform(color_matrix)
            angular_transform = transform(angular_matrix)
            stripe_transform = torch.stack([transform(matrix) for matrix in stripe_matrices])

        self.register_buffer("fit_full_matrix", full_matrix, persistent=False)
        self.register_buffer(
            "fit_color_reconstruction", color_matrix @ color_transform, persistent=False
        )
        self.register_buffer(
            "fit_angular_reconstruction", angular_matrix @ angular_transform, persistent=False
        )
        self.register_buffer(
            "fit_stripe_reconstruction",
            torch.einsum("osf,ofm->osm", stripe_matrices, stripe_transform),
            persistent=False,
        )
        self.register_buffer("full_to_color", color_transform, persistent=False)
        self.register_buffer("full_to_angular", angular_transform, persistent=False)
        self.register_buffer("full_to_stripe", stripe_transform, persistent=False)
        self.register_buffer("stripe_offset_candidates", offset_values, persistent=False)

    def _family_ids(self, family: str) -> torch.Tensor:
        return torch.nonzero(
            self.family_code == FAMILY_TO_CODE[family], as_tuple=False
        ).flatten()

    def _stack_family(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.prototype_bank[int(index)] for index in ids.tolist()])

    def _directional_response(
        self,
        patches: list[torch.Tensor],
        ids: torch.Tensor,
        family: str,
    ) -> torch.Tensor:
        outputs = []
        for chunk_ids in ids.split(self.prototype_chunk_size):
            prototype = self._stack_family(chunk_ids)
            score = None
            for scale, patch in enumerate(patches):
                if family == "full":
                    rendered = self.renderers[scale](prototype)
                elif family == "angular":
                    rendered = self.angular_renderers[scale](prototype)
                elif family == "stripe":
                    rendered = self.stripe_renderers[scale](
                        prototype, self.stripe_angle_offset[chunk_ids]
                    )
                else:
                    raise ValueError(f"{family} is not directional")
                scale_score = rotating_dot_score(patch, rendered)
                score = scale_score if score is None else score + scale_score
            null = self.null_score[chunk_ids][None, None, :, None].expand(
                score.shape[0], score.shape[1], -1, -1
            )
            probability = torch.cat((score, null), dim=-1).softmax(-1)[..., :-1]
            if family == "stripe":
                theta = (
                    self.stripe_angle_offset[chunk_ids, None]
                    + torch.arange(
                        self.directions, device=score.device, dtype=score.dtype
                    )[None]
                    * self.direction_step
                )
                coefficients = torch.stack((theta.cos(), theta.sin()), dim=-1)
                response = torch.einsum("bnpd,pdc->bnpc", probability, coefficients)
            else:
                response = torch.einsum(
                    "bnpd,dc->bnpc", probability, self.direction_coefficients
                )
            outputs.append(response)
        return torch.cat(outputs, dim=2)

    def _color_response(self, patches: list[torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
        outputs = []
        for chunk_ids in ids.split(self.prototype_chunk_size):
            prototype = self._stack_family(chunk_ids)
            scores = []
            for patch in patches:
                rendered = prototype[:, None, :, None].expand(
                    -1, 1, -1, patch.shape[-1] // 3
                )
                scores.append(rotating_dot_score(patch, rendered).squeeze(-1))
            score = torch.stack(scores, dim=-1)
            null = self.null_score[chunk_ids][None, None, :, None].expand(
                score.shape[0], score.shape[1], -1, -1
            )
            # K24 -> [1,0], K12 -> [0,1]. Dropping the null probability
            # naturally attenuates both channels when no Color scale matches.
            outputs.append(torch.cat((score, null), dim=-1).softmax(-1)[..., :-1])
        return torch.cat(outputs, dim=2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            patches = [geometry(image.float()) for geometry in self.geometries]
            patches = [
                weighted_patch_flat(patch, getattr(self, f"scale_cover_{scale}"))
                for scale, patch in enumerate(patches)
            ]
            response = image.new_zeros(
                image.shape[0], self.num_patches, self.bases, 2, dtype=torch.float32
            )
            for family in FAMILY_NAMES:
                ids = self._family_ids(family)
                if ids.numel() == 0:
                    continue
                family_response = (
                    self._color_response(patches, ids)
                    if family == "color"
                    else self._directional_response(patches, ids, family)
                )
                response = response.index_copy(2, ids, family_response)
            return response.flatten(2, 3) + self.output_bias

    def plan_differentiation(
        self, *, target_full_count: int, complexity_weight: float
    ) -> dict[str, Any]:
        full_ids = self._family_ids("full")
        current = int(full_ids.numel())
        if not 0 <= target_full_count <= current:
            raise ValueError(f"target Full count must be in [0,{current}]")
        convert_count = current - int(target_full_count)
        if convert_count == 0:
            return {"target_full_count": target_full_count, "assignments": []}
        prototypes = self._stack_family(full_ids)
        target = torch.einsum("pcm,sm->pcs", prototypes, self.fit_full_matrix)
        cover = self.scale_cover_0[None, None]
        denominator = (target.square() * cover).sum((1, 2)).clamp_min(1e-12)

        def relative_error(reconstruction: torch.Tensor) -> torch.Tensor:
            estimate = torch.einsum("pcm,sm->pcs", prototypes, reconstruction)
            return ((target - estimate).square() * cover).sum((1, 2)) / denominator

        angular_error = relative_error(self.fit_angular_reconstruction)
        stripe_estimate = torch.einsum(
            "pcm,osm->pocs", prototypes, self.fit_stripe_reconstruction
        )
        stripe_error = (
            (target[:, None] - stripe_estimate).square() * cover[:, None]
        ).sum((2, 3)) / denominator[:, None]
        best_stripe_error, best_stripe_offset = stripe_error.min(dim=1)
        errors = torch.stack((angular_error, best_stripe_error), dim=1)
        dimensions = torch.tensor(
            [
                self.angular_bins,
                self.stripe_transverse_bins * self.stripe_longitudinal_bins,
            ],
            device=errors.device,
            dtype=errors.dtype,
        )
        costs = errors + float(complexity_weight) * dimensions / self.full_parameter_width
        best_cost, best_family = costs.min(dim=1)
        selected = torch.topk(best_cost, convert_count, largest=False).indices
        target_names = ("angular", "stripe")
        assignments = []
        for local_index in selected.tolist():
            target_index = int(best_family[local_index])
            family = target_names[target_index]
            offset_index = int(best_stripe_offset[local_index]) if family == "stripe" else 0
            assignments.append(
                {
                    "base_id": int(full_ids[local_index]),
                    "family": family,
                    "stripe_offset_index": offset_index,
                    "relative_error": float(errors[local_index, target_index]),
                    "selection_cost": float(best_cost[local_index]),
                }
            )
        return {
            "target_full_count": int(target_full_count),
            "complexity_weight": float(complexity_weight),
            "assignments": assignments,
        }

    def apply_differentiation(
        self, plan: dict[str, Any]
    ) -> tuple[dict[str, Any], list[PrototypeConversion]]:
        conversions: list[PrototypeConversion] = []
        for assignment in plan["assignments"]:
            base = int(assignment["base_id"])
            family = str(assignment["family"])
            if self.family_name(base) != "full":
                raise ValueError(f"base {base} is already {self.family_name(base)}")
            if family == "color":
                transform = self.full_to_color
                shape = (3,)
            elif family == "angular":
                transform = self.full_to_angular
                shape = (3, self.angular_bins)
            elif family == "stripe":
                offset_index = int(assignment["stripe_offset_index"])
                transform = self.full_to_stripe[offset_index]
                shape = (
                    3,
                    self.stripe_transverse_bins,
                    self.stripe_longitudinal_bins,
                )
                self.stripe_angle_offset[base] = self.stripe_offset_candidates[offset_index]
            else:
                raise ValueError(f"unsupported target family {family}")
            old_parameter = self.prototype_bank[base]
            projected = torch.einsum(
                "cm,fm->cf", old_parameter.detach(), transform.to(old_parameter)
            ).reshape(shape)
            new_parameter = nn.Parameter(projected)
            self.prototype_bank[base] = new_parameter
            self.family_code[base] = FAMILY_TO_CODE[family]
            conversions.append(
                PrototypeConversion(
                    name=f"patch_embed.prototype_bank.{base}",
                    old_parameter=old_parameter,
                    new_parameter=new_parameter,
                    transform=transform.detach(),
                )
            )
        audit = dict(plan)
        audit["family_counts"] = self.family_counts()
        audit["effective_geometry_parameters"] = sum(
            parameter.numel() for parameter in self.prototype_bank
        )
        return audit, conversions

    def prepare_for_state_dict(self, state: dict[str, torch.Tensor], prefix: str = "") -> None:
        """Match ParameterList shapes before loading a differentiated checkpoint."""
        for base in range(self.bases):
            key = f"{prefix}prototype_bank.{base}"
            source = state.get(key)
            if source is None:
                continue
            current = self.prototype_bank[base]
            if current.shape != source.shape:
                self.prototype_bank[base] = nn.Parameter(
                    torch.empty(source.shape, device=current.device, dtype=current.dtype)
                )
