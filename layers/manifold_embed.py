"""Manifold-embedding wrapper modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .HexConv import HexConv2D
from .SVLinear import SVLinear
from .Sim2cooManifoldLinear import Sim2cooManifoldLayer
from .activation_counter import PerLabelActivationCounter


class HexConvManifoldEmbed(nn.Module):
    """
    Component: HexConv2D -> Sim2cooManifoldLayer

    Input:  image tensor (B, C_in, H, W)
    Output: manifold embedding (B, N, D)
    """

    def __init__(
        self,
        *,
        in_chans: int = 3,
        num_sv: int = 128,
        manifold_dim: int = 2,
        hex_kernel_size: int = 16,
        num_class: int | None = None,
        record_active: bool = True,
        record_grad: bool = False,
        record_prefer: bool = False,
    ) -> None:
        super().__init__()
        num_sv_i = int(num_sv)
        self.hexconv = HexConv2D(in_channels=in_chans, out_channels=num_sv_i, kernel_size=hex_kernel_size)
        self.num_class: int | None = None
        self.hexconv_counter: PerLabelActivationCounter | None = None
        self.set_counter(num_class, record_active=record_active, record_grad=record_grad, record_prefer=record_prefer)
        self.manifold = Sim2cooManifoldLayer(in_channels=num_sv_i, embed_dim=manifold_dim)

        self.in_chans = int(in_chans)
        self.num_sv = int(num_sv_i)
        self.manifold_dim = int(manifold_dim)
        self.hex_kernel_size = int(hex_kernel_size)

    @property
    def num_patches(self) -> int | None:
        return getattr(self.hexconv, "num_patchs", None)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None or self.hexconv_counter is None:
            sim = self.hexconv(x)
        else:
            sim = self.hexconv_counter(x, labels)
        return self.manifold(sim)  # (B, N, manifold_dim)

    def set_counter(
        self,
        num_class: int | None = None,
        *,
        record_active: bool = True,
        record_grad: bool = False,
        record_prefer: bool = False,
    ) -> None:
        if num_class is None:
            self.num_class = None
            self.hexconv_counter = None
            return

        num_class_i = int(num_class)
        if num_class_i <= 0:
            raise ValueError("num_class must be a positive integer or None")

        self.num_class = num_class_i
        self.hexconv_counter = PerLabelActivationCounter(
            self.hexconv,
            num_class=self.num_class,
            record_active=bool(record_active),
            record_grad=bool(record_grad),
            record_prefer=bool(record_prefer),
        )

    def extra_repr(self) -> str:
        return (
            f"in_chans={self.in_chans}, num_sv={self.num_sv}, "
            f"manifold_dim={self.manifold_dim}, hex_kernel_size={self.hex_kernel_size}, num_class={self.num_class}"
        )


class SVLinearManifoldEmbed(nn.Module):
    """
    Component: SVLinear -> Sim2cooManifoldLayer

    Input:  token tensor (..., in_features)
    Output: manifold embedding (..., manifold_dim)
    """

    def __init__(
        self,
        *,
        in_features: int,
        num_sv: int = 32,
        manifold_dim: int = 64,
        scale_init: float = 1.0,
        num_class: int | None = None,
        record_active: bool = True,
        record_grad: bool = False,
        record_prefer: bool = False,
    ) -> None:
        super().__init__()
        num_sv_i = int(num_sv)
        self.sv = SVLinear(in_features=in_features, out_features=num_sv_i, scale_init=scale_init)
        self.num_class: int | None = None
        self.sv_counter: PerLabelActivationCounter | None = None
        self.set_counter(num_class, record_active=record_active, record_grad=record_grad, record_prefer=record_prefer)
        self.manifold = Sim2cooManifoldLayer(in_channels=num_sv_i, embed_dim=manifold_dim)

        self.in_features = int(in_features)
        self.num_sv = int(num_sv_i)
        self.manifold_dim = int(manifold_dim)
        self.scale_init = float(scale_init)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None or self.sv_counter is None:
            sim = self.sv(x)
        else:
            sim = self.sv_counter(x, labels)
        return self.manifold(sim)

    def set_counter(
        self,
        num_class: int | None = None,
        *,
        record_active: bool = True,
        record_grad: bool = False,
        record_prefer: bool = False,
    ) -> None:
        if num_class is None:
            self.num_class = None
            self.sv_counter = None
            return

        num_class_i = int(num_class)
        if num_class_i <= 0:
            raise ValueError("num_class must be a positive integer or None")

        self.num_class = num_class_i
        self.sv_counter = PerLabelActivationCounter(
            self.sv,
            num_class=self.num_class,
            record_active=bool(record_active),
            record_grad=bool(record_grad),
            record_prefer=bool(record_prefer),
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, num_sv={self.num_sv}, manifold_dim={self.manifold_dim}, "
            f"scale_init={self.scale_init:g}, num_class={self.num_class}"
        )
