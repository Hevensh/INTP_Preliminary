from __future__ import annotations

import math

import torch


def complex_to_xy(coordinates: torch.Tensor) -> torch.Tensor:
    """Convert complex hex-center coordinates to ``(..., 2)`` xy values."""
    if not torch.is_complex(coordinates):
        raise TypeError("coordinates must be a complex tensor")
    return torch.stack((coordinates.real, coordinates.imag), dim=-1)


def xy_to_axial(coordinates: torch.Tensor) -> torch.Tensor:
    """Map the triangular-lattice xy convention used by ``genr_2Dcoo`` to axial integers."""
    if coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (..., 2)")
    x, y = coordinates.unbind(dim=-1)
    axial_r = torch.round(2.0 * y / math.sqrt(3.0))
    axial_q = torch.round(x - 0.5 * axial_r)
    return torch.stack((axial_q, axial_r), dim=-1).to(torch.long)


def hex_relative_bins(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return directed graph-ring and six-sector indices for every ``query -> key`` pair.

    Args:
        coordinates: Hex-center xy coordinates shaped ``(N, 2)``.

    Returns:
        ``rings`` and ``directions``, both shaped ``(N, N)``. Direction zero
        points along positive x and increases counter-clockwise. Self pairs use
        ring zero and direction zero.
    """
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    axial = xy_to_axial(coordinates)
    delta = axial.unsqueeze(0) - axial.unsqueeze(1)  # query i -> key j
    dq, dr = delta.unbind(dim=-1)
    rings = (dq.abs() + dr.abs() + (dq + dr).abs()) // 2

    xy_delta = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
    angles = torch.atan2(xy_delta[..., 1], xy_delta[..., 0])
    directions = torch.floor((angles + math.pi / 6.0) / (math.pi / 3.0)).to(torch.long) % 6
    directions = torch.where(rings == 0, torch.zeros_like(directions), directions)
    return rings, directions

