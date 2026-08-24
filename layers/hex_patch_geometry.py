from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hex_idx_genr import calc_coo_params, genr_2Dcoo, genr_2Didx


class HexPatchGeometry(nn.Module):
    """Fixed circular patch sampling on a staggered hexagonal lattice."""

    def __init__(
        self,
        img_size: int | tuple[int, int],
        in_chans: int,
        kernel_size: int,
        lattice_stride: int,
    ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if min(*img_size, in_chans, kernel_size, lattice_stride) <= 0:
            raise ValueError("image size, channels, kernel size, and lattice stride must be positive")

        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.in_chans = int(in_chans)
        self.kernel_size = int(kernel_size)
        self.lattice_stride = int(lattice_stride)

        idx_x, idx_y, _ = genr_2Didx(self.kernel_size)
        self.register_buffer("idx_x", idx_x.to(torch.long), persistent=False)
        self.register_buffer("idx_y", idx_y.to(torch.long), persistent=False)

        n_x, n_y, stride_x, stride_y = calc_coo_params(self.img_size, self.lattice_stride)
        starts_x, starts_y = genr_2Dcoo(n_x, n_y, stride_x, stride_y)

        pad_x = ((n_x - 1) * stride_x + self.kernel_size - self.img_size[0] + 1) // 2
        pad_y = ((n_y - 1) * stride_y + self.kernel_size - self.img_size[1] + 1) // 2
        if pad_x >= self.img_size[0] or pad_y >= self.img_size[1]:
            raise ValueError("reflection padding must be smaller than the corresponding image dimension")
        self.pad_x = int(pad_x)
        self.pad_y = int(pad_y)

        self.register_buffer(
            "sample_x",
            starts_x[:, None] + self.idx_x[None, :],
            persistent=False,
        )
        self.register_buffer(
            "sample_y",
            starts_y[:, None] + self.idx_y[None, :],
            persistent=False,
        )

        center = (self.kernel_size - 1) / 2.0
        centers_xy = torch.stack(
            (
                starts_y.to(torch.float32) - self.pad_y + center,
                starts_x.to(torch.float32) - self.pad_x + center,
            ),
            dim=-1,
        )
        self.register_buffer("patch_centers_xy", centers_xy, persistent=False)

        coo_x, coo_y = genr_2Dcoo(n_x, n_y, 0.5, 3**0.5 * 0.5, torch.float32)
        self.register_buffer("coo_patchs", torch.complex(coo_x, coo_y), persistent=False)

    @property
    def num_patches(self) -> int:
        return int(self.patch_centers_xy.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.idx_x.numel())

    @property
    def patch_offsets_xy(self) -> torch.Tensor:
        center = (self.kernel_size - 1) / 2.0
        return torch.stack((self.idx_y - center, self.idx_x - center), dim=-1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape (B, C, H, W)")
        if tuple(image.shape[1:]) != (self.in_chans, *self.img_size):
            raise ValueError(
                f"expected image tail {(self.in_chans, *self.img_size)}, got {tuple(image.shape[1:])}"
            )
        padded = F.pad(image, (self.pad_y, self.pad_y, self.pad_x, self.pad_x), mode="reflect")
        return padded[..., self.sample_x, self.sample_y].transpose(1, 2)
