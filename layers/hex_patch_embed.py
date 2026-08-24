import torch
import torch.nn as nn

from layers.HexConv import HexConv2D


class HexPatchEmbed(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 128,
        kernel_size: int = 16,
        token_mode: str = "neg",
        lattice_stride: int | None = None,
    ):
        super().__init__()
        self.hexconv = HexConv2D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            lattice_stride=lattice_stride,
        )
        if token_mode not in {"neg", "rbf", "raw"}:
            raise ValueError("token_mode must be one of: 'neg', 'rbf', 'raw'")
        self.token_mode = token_mode

    @property
    def num_patches(self) -> int | None:
        return getattr(self.hexconv, "num_patchs", None)

    @property
    def patch_centers_xy(self) -> torch.Tensor:
        return self.hexconv.patch_centers_xy

    @property
    def coo_patchs(self) -> torch.Tensor:
        return self.hexconv.coo_patchs

    @property
    def patch_offsets_xy(self) -> torch.Tensor:
        return self.hexconv.patch_offsets_xy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.hexconv(x)  # (B, N, D) in the current simplified HexConv2D
        if self.token_mode == "raw":
            return dist
        if self.token_mode == "rbf":
            return torch.exp(-dist)
        return -dist
