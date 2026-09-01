from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMixtureRingConv2d(nn.Module):
    """Compact GMR convolution following the official GMR-Conv parameterization.

    A spatial kernel is represented by ``num_rings`` concentric indicator bands.
    The bands are smoothed by a trainable Gaussian mixture, convolved depthwise
    over each input channel, and finally mixed with a learned 1x1 projection.
    Moving ``stride`` to the ring convolution is algebraically equivalent to the
    official depthwise-like forward, while avoiding dense intermediate maps.

    This is a small self-contained adaptation of the MIT-licensed reference at
    https://github.com/XYPB/GMR-Conv.  Only the 2-D, non-grouped path needed by
    the ImageNet-100 patch-embedding comparison is included here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        gaussian_sigma_scale: float = 2.355,
        train_gaussian_sigma: bool = True,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, kernel_size, stride) <= 0:
            raise ValueError("channel counts, kernel_size, and stride must be positive")
        if padding < 0:
            raise ValueError("padding must be non-negative")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.num_rings = self.kernel_size // 2 + 1

        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.num_rings)
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None

        ring_masks = self._make_ring_masks(self.kernel_size, self.num_rings)
        self.register_buffer("ring_masks", ring_masks, persistent=True)
        self.register_buffer(
            "ring_centers", torch.arange(self.num_rings, dtype=torch.float32),
            persistent=False,
        )
        band_width = self.kernel_size / (2.0 * (self.num_rings - 1))
        sigma = torch.full(
            (self.num_rings,), band_width / float(gaussian_sigma_scale)
        )
        if train_gaussian_sigma:
            self.log_sigmas = nn.Parameter(sigma.log())
        else:
            self.register_buffer("log_sigmas", sigma.log(), persistent=True)
        self.reset_parameters()

    @staticmethod
    def _make_ring_masks(kernel_size: int, num_rings: int) -> torch.Tensor:
        # Match the reference implementation: the discrete center is H//2,
        # including for even kernels, and the circular support ends at H//2+0.5.
        coordinate = torch.arange(kernel_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        center = float(kernel_size // 2)
        distance = ((yy - center).square() + (xx - center).square()).sqrt()
        levels = torch.linspace(0.0, center + 0.5, num_rings + 1)
        masks = [
            ((distance >= levels[index]) & (distance < levels[index + 1])).float()
            for index in range(num_rings)
        ]
        return torch.stack(masks, dim=0)

    def reset_parameters(self) -> None:
        # The rendered kernel has the same fan-in as a dense KxK convolution.
        bound = 1.0 / math.sqrt(self.in_channels * self.kernel_size**2)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def ring_filters(self) -> torch.Tensor:
        """Return the differentiable Gaussian-smoothed ring bank [R,1,K,K]."""

        sigma = self.log_sigmas.exp().clamp(1e-2, 2.0 * self.num_rings)
        locations = torch.arange(
            self.num_rings,
            device=sigma.device,
            dtype=sigma.dtype,
        )
        delta = locations[None, :] - self.ring_centers.to(sigma)[:, None]
        probability = torch.exp(-0.5 * (delta / sigma[:, None]).square())
        probability = probability / (sigma[:, None] * math.sqrt(2.0 * math.pi))
        masks = self.ring_masks.to(device=sigma.device, dtype=sigma.dtype)
        return (probability @ masks.flatten(1)).reshape(
            self.num_rings, 1, self.kernel_size, self.kernel_size
        )

    def rendered_weight(self) -> torch.Tensor:
        """Materialize the equivalent dense [Cout,Cin,K,K] kernel for QA."""

        filters = self.ring_filters().squeeze(1)
        return torch.einsum("oir,rhw->oihw", self.weight, filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        batch, channels, height, width = x.shape
        ring_response = F.conv2d(
            x.reshape(batch * channels, 1, height, width),
            self.ring_filters().to(device=x.device, dtype=x.dtype),
            stride=self.stride,
            padding=self.padding,
        )
        out_h, out_w = ring_response.shape[-2:]
        ring_response = ring_response.reshape(
            batch, channels * self.num_rings, out_h, out_w
        )
        pointwise = self.weight.reshape(
            self.out_channels, self.in_channels * self.num_rings, 1, 1
        )
        return F.conv2d(
            ring_response,
            pointwise.to(dtype=x.dtype),
            self.bias.to(dtype=x.dtype) if self.bias is not None else None,
        )


class EquiVitGMRPatchEmbed(nn.Module):
    """Equi-ViT-style two-stage GMR patch embedding for 224px DeiT.

    The Equi-ViT paper specifies sequential 6x6 and 11x11 GMR kernels but does
    not publish its ViT glue code.  The strides 6 and 2 reproduce the required
    14x14 token grid: 224 -> 37 -> 14.  The intermediate width of 24 mirrors
    the parameter count implied by the paper's matched two-layer Conv-ViT stem.
    """

    def __init__(
        self,
        *,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: int = 192,
        hidden_channels: int = 24,
    ) -> None:
        super().__init__()
        if min(img_size, in_chans, embed_dim, hidden_channels) <= 0:
            raise ValueError("all dimensions must be positive")
        self.img_size = (int(img_size), int(img_size))
        self.patch_size = (16, 16)
        self.proj1 = GaussianMixtureRingConv2d(
            in_chans, hidden_channels, kernel_size=6, stride=6
        )
        self.proj2 = GaussianMixtureRingConv2d(
            hidden_channels, embed_dim, kernel_size=11, stride=2
        )
        first = (img_size - 6) // 6 + 1
        final = (first - 11) // 2 + 1
        self.grid_size = (final, final)
        self.num_patches = final * final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.img_size:
            raise ValueError(
                f"expected image size {self.img_size}, got {tuple(x.shape[-2:])}"
            )
        x = self.proj2(self.proj1(x))
        return x.flatten(2).transpose(1, 2)
