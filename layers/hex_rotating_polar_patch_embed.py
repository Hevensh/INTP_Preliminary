from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.hex_patch_geometry import HexPatchGeometry


class _PolarRenderer(nn.Module):
    """Render a compact variable-ring polar prototype on Hex patch samples."""

    def __init__(
        self,
        geometry: HexPatchGeometry,
        *,
        radial_bins: int,
        ring_counts: torch.Tensor,
        ring_offsets: torch.Tensor,
        directions: int,
        direction_step: float,
    ) -> None:
        super().__init__()
        kernel_size = geometry.kernel_size
        offsets = geometry.patch_offsets_xy
        x, y = offsets[:, 0], offsets[:, 1]
        radius = torch.sqrt(x.square() + y.square())
        angle_turn = torch.remainder(torch.atan2(y, x), 2 * math.pi) / (2 * math.pi)
        radial = (radius * radial_bins / (kernel_size / 2) - 0.5).clamp(0, radial_bins - 1)
        r0 = radial.floor().long()
        r1 = (r0 + 1).clamp_max(radial_bins - 1)

        def angular_indices(ring: torch.Tensor):
            count = ring_counts[ring]
            shifts = torch.arange(directions)[:, None] * direction_step / (2 * math.pi)
            position = torch.remainder(angle_turn[None] - shifts, 1.0) * count[None]
            raw = position.floor()
            a0 = torch.remainder(raw.long(), count[None])
            a1 = torch.remainder(a0 + 1, count[None])
            return ring_offsets[ring][None] + a0, ring_offsets[ring][None] + a1, position - raw

        i00, i01, a0_fraction = angular_indices(r0)
        i10, i11, a1_fraction = angular_indices(r1)
        cover = torch.cos(radius * math.pi / kernel_size).clamp_min(0)

        self.register_buffer("index_r0_a0", i00, persistent=False)
        self.register_buffer("index_r0_a1", i01, persistent=False)
        self.register_buffer("index_r1_a0", i10, persistent=False)
        self.register_buffer("index_r1_a1", i11, persistent=False)
        self.register_buffer("angle_fraction_r0", a0_fraction, persistent=False)
        self.register_buffer("angle_fraction_r1", a1_fraction, persistent=False)
        self.register_buffer("radial_fraction", radial - r0, persistent=False)
        self.register_buffer("support_cover", cover, persistent=False)
        n_in = float(geometry.in_chans * cover.sum())
        self.distance_multiplier = 1.0 / (
            (n_in / 6.0) - 0.5 * (n_in * 7.0 / 180.0) ** 0.5
        )

    def forward(self, prototype: torch.Tensor) -> torch.Tensor:
        p00 = prototype[..., self.index_r0_a0]
        p01 = prototype[..., self.index_r0_a1]
        p10 = prototype[..., self.index_r1_a0]
        p11 = prototype[..., self.index_r1_a1]
        a0 = self.angle_fraction_r0[None, None]
        a1 = self.angle_fraction_r1[None, None]
        low = p00 * (1 - a0) + p01 * a0
        high = p10 * (1 - a1) + p11 * a1
        radial = self.radial_fraction[None, None, None]
        return (low * (1 - radial) + high * radial).permute(0, 2, 1, 3).contiguous()


class HexRotatingPolarPatchEmbed(nn.Module):
    """Full-4 polar prototype tokenizer evaluated on fixed Hex patch centers."""

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
        use_null: bool = True,
        null_initial_score: float = -1.0,
        score_normalization: str = "none",
        routing_score_mode: str = "same",
        response_gate: str = "exp2",
        score_clamp: float = 4.0,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        if not 1 <= directions <= global_directions:
            raise ValueError("directions must be in [1, global_directions]")
        if prototype_chunk_size <= 0:
            raise ValueError("prototype_chunk_size must be positive")
        self.embed_dim = int(embed_dim)
        self.bases = int(bases)
        self.directions = int(directions)
        self.scales = len(kernel_sizes)
        self.prototype_chunk_size = int(prototype_chunk_size)
        self.use_null = bool(use_null)
        if score_normalization not in {"none", "patch_global"}:
            raise ValueError("score_normalization must be none or patch_global")
        if response_gate not in {"exp2", "exp"}:
            raise ValueError("response_gate must be exp2 or exp")
        if routing_score_mode not in {"same", "centered_raw"}:
            raise ValueError("routing_score_mode must be same or centered_raw")
        if score_clamp <= 0:
            raise ValueError("score_clamp must be positive")
        self.score_normalization = score_normalization
        self.routing_score_mode = routing_score_mode
        self.response_gate = response_gate
        self.score_clamp = float(score_clamp)

        self.geometries = nn.ModuleList(
            HexPatchGeometry(img_size, in_chans, int(kernel), lattice_stride)
            for kernel in kernel_sizes
        )
        patch_counts = {geometry.num_patches for geometry in self.geometries}
        if len(patch_counts) != 1:
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

        self.prototype = nn.Parameter(torch.randn(bases, in_chans, int(offsets[-1])) * 0.02)
        self.log2_distance_scale = nn.Parameter(torch.zeros(bases, self.scales))
        # Four high-dimensional values per base: one cosine/sine pair and one
        # independent scale value for each of the two default scales.
        self.direction_pair = nn.Parameter(torch.empty(bases, 2, embed_dim))
        self.scale_value = nn.Parameter(torch.empty(bases, self.scales, embed_dim))
        self.null_score = nn.Parameter(torch.full((bases,), float(null_initial_score)))
        self.output_bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.direction_pair, std=0.02)
        nn.init.trunc_normal_(self.scale_value, std=0.02)

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

    def _chunk_scores(
        self,
        patches: list[torch.Tensor],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        prototype = self.prototype[start:stop]
        scale_scores = []
        for scale_index, (patch, renderer) in enumerate(zip(patches, self.renderers)):
            rendered = renderer(prototype)
            cover = renderer.support_cover
            weighted_patch = patch * cover[None, None, None]
            cross = torch.einsum("qncm,pdcm->qnpd", weighted_patch, rendered)
            patch_energy = (patch.square() * cover[None, None, None]).sum((2, 3))
            prototype_energy = (rendered.square() * cover[None, None, None, :]).sum((2, 3))
            distance = (
                patch_energy[:, :, None, None]
                + prototype_energy[None, None]
                - 2 * cross
            ).clamp_min(0)
            score = -distance * renderer.distance_multiplier
            score = score * torch.exp2(
                self.log2_distance_scale[start:stop, scale_index]
            )[None, None, :, None]
            scale_scores.append(score)
        return torch.stack(scale_scores, dim=3)  # B, N, P, S, D

    def _chunk_output(
        self,
        routing_scores: torch.Tensor,
        gate_scores: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        flat_scores = routing_scores.flatten(3, 4)
        if self.use_null:
            null = self.null_score[start:stop][None, None, :, None].expand(
                flat_scores.shape[0], flat_scores.shape[1], -1, -1
            )
            weights = torch.cat((flat_scores, null), dim=-1).softmax(-1)[..., :-1]
        else:
            weights = flat_scores.softmax(-1)
        weights = weights.view_as(routing_scores)
        gate = (
            torch.exp(gate_scores)
            if self.response_gate == "exp"
            else torch.exp2(gate_scores)
        )
        weights = weights * gate

        # Algebraically factor pose values instead of materializing P x S x D x C:
        # V[p,s,d] = A[p] cos(theta[d]) + B[p] sin(theta[d]) + Vscale[p,s].
        cosine_mass = torch.einsum(
            "qnpsd,d->qnp", weights, self.direction_coefficients[:, 0]
        )
        sine_mass = torch.einsum(
            "qnpsd,d->qnp", weights, self.direction_coefficients[:, 1]
        )
        scale_mass = weights.sum(-1)
        pair = self.direction_pair[start:stop]
        output = torch.einsum("qnp,pc->qnc", cosine_mass, pair[:, 0])
        output = output + torch.einsum("qnp,pc->qnc", sine_mass, pair[:, 1])
        output = output + torch.einsum(
            "qnps,psc->qnc", scale_mass, self.scale_value[start:stop]
        )
        return output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Distance calibration is deliberately float32; the transformer still
        # runs under the caller's AMP autocast context.
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            patches = [geometry(image) for geometry in self.geometries]
            score_chunks = [
                self._chunk_scores(
                    patches, start, min(start + self.prototype_chunk_size, self.bases)
                )
                for start in range(0, self.bases, self.prototype_chunk_size)
            ]
            raw_scores = torch.cat(score_chunks, dim=2)
            gate_scores = raw_scores
            if self.score_normalization == "patch_global":
                variance, mean = torch.var_mean(
                    raw_scores, dim=(2, 3, 4), unbiased=False, keepdim=True
                )
                gate_scores = (raw_scores - mean) * torch.rsqrt(variance + 1e-6)
                gate_scores = gate_scores.clamp(-self.score_clamp, self.score_clamp)
            routing_scores = gate_scores
            if self.routing_score_mode == "centered_raw":
                # Subtracting one common value from a base's real poses leaves
                # their softmax ratios unchanged while putting null in a stable
                # relative coordinate system.  Do not divide by std here:
                # that would silently change routing temperature.
                routing_scores = raw_scores - raw_scores.mean(
                    dim=(3, 4), keepdim=True
                )
            output = None
            for start in range(0, self.bases, self.prototype_chunk_size):
                chunk = self._chunk_output(
                    routing_scores[:, :, start : start + self.prototype_chunk_size],
                    gate_scores[:, :, start : start + self.prototype_chunk_size],
                    start,
                    min(start + self.prototype_chunk_size, self.bases),
                )
                output = chunk if output is None else output + chunk
            return output + self.output_bias
