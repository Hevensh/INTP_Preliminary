"""Shared GEMM path for rotating rendered dot-product prototypes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_patch_flat(
    patch: torch.Tensor, cover: torch.Tensor
) -> torch.Tensor:
    """Apply one spatial cover and flatten [B,N,C,M] to contiguous [B,N,F]."""
    return (patch * cover[None, None, None]).flatten(2).contiguous()


def rotating_dot_score(
    patch_flat: torch.Tensor, rendered: torch.Tensor
) -> torch.Tensor:
    """Return [B,N,P,D] dot scores through the optimized linear/GEMM path."""
    if patch_flat.ndim != 3 or rendered.ndim != 4:
        raise ValueError("expected patch [B,N,F] and rendered [P,D,C,M]")
    bases, directions = rendered.shape[:2]
    rendered_flat = rendered.flatten(2).reshape(
        bases * directions, patch_flat.shape[-1]
    )
    return F.linear(patch_flat, rendered_flat).view(
        patch_flat.shape[0], patch_flat.shape[1], bases, directions
    )
