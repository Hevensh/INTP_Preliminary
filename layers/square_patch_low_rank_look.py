from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .polar_prototype_look import PolarPrototypeLook


def build_square_patch_centers(
    image_size: int = 224,
    patch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pixel centers and unit-grid coordinates in ViT token order."""
    if min(image_size, patch_size) <= 0 or image_size % patch_size:
        raise ValueError("image_size must be divisible by a positive patch_size")
    side = image_size // patch_size
    grid_y, grid_x = torch.meshgrid(
        torch.arange(side, dtype=torch.float32),
        torch.arange(side, dtype=torch.float32),
        indexing="ij",
    )
    grid_xy = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    pixel_xy = (grid_xy + 0.5) * float(patch_size) - 0.5
    return pixel_xy, grid_xy


class SquarePatchLowRankLook(nn.Module):
    """Low-rank polar look basis for an ordinary square ViT patch grid.

    A single learned polar matcher produces ``S*T`` signed pose responses per
    query patch.  The paired pose fields are normalized over the actually
    visible target patches, so large scales and image edges do not gain or lose
    amplitude merely because their support contains a different point count.
    The ordinary ViT attention heads then mix this shared pose bank with
    independent L2-normalized coefficients.
    """

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_heads: int = 3,
        rank: int | None = None,
        radial_bins: int = 8,
        angular_bins: int = 16,
        direction_samples: int = 8,
        scales: Sequence[float] = (
            1.0,
            2.0**0.5,
            3.0**0.5,
            2.0,
        ),
        look_radius: float = 3.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if rank is not None and rank <= 0:
            raise ValueError("rank must be positive when provided")
        if look_radius <= 0:
            raise ValueError("look_radius must be positive")
        if angular_bins % direction_samples:
            raise ValueError("angular_bins must be divisible by direction_samples")

        pixel_xy, grid_xy = build_square_patch_centers(image_size, patch_size)
        self.register_buffer("patch_centers_xy", pixel_xy, persistent=True)
        self.register_buffer("patch_coordinates_xy", grid_xy, persistent=True)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_heads = int(num_heads)
        self.rank = int(rank) if rank is not None else None
        self.look_radius = float(look_radius)
        self.eps = float(eps)

        # With rank=R these are R genuinely independent learned polar look
        # prototypes, each evaluated at all S*T poses.  This is not a linear
        # compression of one prototype's pose responses.
        self.pose = PolarPrototypeLook(
            num_heads=self.rank or 1,
            in_channels=in_channels,
            radial_bins=radial_bins,
            angular_bins=angular_bins,
            rotation_samples=direction_samples,
            scales=scales,
        )
        self.num_poses = self.pose.num_scales * self.pose.num_rotations
        if self.rank is None:
            self.head_mix = nn.Parameter(torch.empty(self.num_heads, self.num_poses))
            nn.init.normal_(self.head_mix, std=1.0 / math.sqrt(self.num_poses))
        else:
            self.head_mix = nn.Parameter(torch.empty(self.num_heads, self.rank))
            nn.init.normal_(self.head_mix, std=1.0 / math.sqrt(self.rank))
        # Zero preserves an existing attention model at insertion time.
        self.head_gain = nn.Parameter(torch.zeros(self.num_heads))

    @property
    def num_patches(self) -> int:
        return int(self.patch_coordinates_xy.shape[0])

    @property
    def normalized_head_mix(self) -> torch.Tensor:
        return F.normalize(self.head_mix, dim=-1, eps=self.eps)

    def pose_masks(self) -> torch.Tensor:
        """Return normalized fields as ``(P,N,N)`` or ``(R,P,N,N)``.

        Normalization is performed independently for every pose and query.
        The self/center target is always invisible.
        """
        coordinates = self.patch_coordinates_xy.to(
            device=self.pose.look_prototype_logits.device,
            dtype=self.pose.look_prototype_logits.dtype,
        )
        relative_xy = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
        field = torch.sigmoid(self.pose.look_prototype_logits)
        sampled = self.pose._sample_square_polar_field(
            field,
            relative_xy,
            base_radius=self.look_radius,
            rho_min=0.0,
            center_mode="zero",
        ).squeeze(3)  # (R,S,T,N,N), with R=1 in the unfactored case
        if self.rank is None:
            sampled = sampled.squeeze(0).flatten(0, 1)
        else:
            sampled = sampled.flatten(1, 2)
        diagonal = torch.eye(
            self.num_patches,
            dtype=torch.bool,
            device=sampled.device,
        )
        sampled = sampled.masked_fill(diagonal, 0.0)
        mass = sampled.sum(dim=-1, keepdim=True)
        return torch.where(mass > self.eps, sampled / mass.clamp_min(self.eps), sampled)

    def match_image(
        self,
        image: torch.Tensor,
        *,
        track_input_grad: bool = False,
    ) -> torch.Tensor:
        """Return signed pose responses shaped ``(B,N,P)``."""
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected image spatial size {(self.image_size, self.image_size)}, "
                f"got {tuple(image.shape[-2:])}"
            )
        rings, coverage = self.extract_rings(image, track_input_grad=track_input_grad)
        return self.match_rings(rings, coverage)

    def extract_rings(
        self,
        image: torch.Tensor,
        *,
        track_input_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the reusable unencoded square-grid P0 polar cache."""
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected image spatial size {(self.image_size, self.image_size)}, "
                f"got {tuple(image.shape[-2:])}"
            )
        context = torch.enable_grad if track_input_grad else torch.no_grad
        with context():
            rings, coverage = self.pose.ring_sampler(
                image,
                self.patch_centers_xy,
                base_radius=(self.patch_size - 1) / 2.0,
                return_coverage=True,
            )
        return rings, coverage

    def match_rings(
        self,
        rings: torch.Tensor,
        coverage: torch.Tensor,
    ) -> torch.Tensor:
        """Match this layer's prototype against a shared P0 cache."""
        response = self.pose.match_rings(rings, coverage)
        if self.rank is None:
            return response.squeeze(2).flatten(-2)
        return response.flatten(-2)  # (B,N,R,P)

    def pose_bias(self, pose_response: torch.Tensor) -> torch.Tensor:
        """Apply each response to its normalized field, returning ``(B,P,N,N)``."""
        masks = self.pose_masks().to(pose_response)
        if self.rank is None:
            if pose_response.ndim != 3 or pose_response.shape[1:] != (
                self.num_patches,
                self.num_poses,
            ):
                raise ValueError(
                    f"pose_response must have shape (B,{self.num_patches},{self.num_poses})"
                )
            return torch.einsum("bip,pij->bpij", pose_response, masks)
        if pose_response.ndim != 4 or pose_response.shape[1:] != (
            self.num_patches,
            self.rank,
            self.num_poses,
        ):
            raise ValueError(
                "pose_response must have shape "
                f"(B,{self.num_patches},{self.rank},{self.num_poses})"
            )
        # Each independent prototype's 32 posed response/field pairs become
        # one spatial look basis.  Only after this aggregation do we mix R->H.
        return torch.einsum("birp,rpij->brij", pose_response, masks)

    def mix_head_bias(
        self,
        pose_bias: torch.Tensor,
        *,
        include_cls: bool = True,
    ) -> torch.Tensor:
        """Mix the pose bank into ordinary ViT head biases ``(B,H,N,N)``."""
        basis_count = self.num_poses if self.rank is None else self.rank
        if pose_bias.ndim != 4 or pose_bias.shape[1:] != (
            basis_count,
            self.num_patches,
            self.num_patches,
        ):
            raise ValueError(
                "pose_bias must have shape "
                f"(B,{basis_count},{self.num_patches},{self.num_patches})"
            )
        mixed = torch.einsum("hp,bpij->bhij", self.normalized_head_mix, pose_bias)
        mixed = mixed * self.head_gain.to(mixed).view(1, -1, 1, 1)
        if not include_cls:
            return mixed
        result = mixed.new_zeros(
            mixed.shape[0],
            mixed.shape[1],
            self.num_patches + 1,
            self.num_patches + 1,
        )
        result[:, :, 1:, 1:] = mixed
        return result

    def forward(
        self,
        image: torch.Tensor,
        *,
        include_cls: bool = True,
        track_input_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response = self.match_image(image, track_input_grad=track_input_grad)
        basis = self.pose_bias(response)
        return self.mix_head_bias(basis, include_cls=include_cls), response

    def forward_rings(
        self,
        rings: torch.Tensor,
        coverage: torch.Tensor,
        *,
        include_cls: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response = self.match_rings(rings, coverage)
        basis = self.pose_bias(response)
        return self.mix_head_bias(basis, include_cls=include_cls), response
