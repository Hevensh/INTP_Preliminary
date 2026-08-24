from typing import List, Optional

import torch
from torch import Tensor, einsum, empty, full, no_grad, ones, pi, stack, zeros, zeros_like
from torch.nn import Buffer, Module, Parameter, ReflectionPad2d, init

from utils.hex_idx_genr import genr_2Didx, calc_coo_params, genr_2Dcoo


class HexConv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        lattice_stride: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.lattice_stride = kernel_size // 2 if lattice_stride is None else int(lattice_stride)
        if self.lattice_stride <= 0:
            raise ValueError("lattice_stride must be positive")

        idx_x, idx_y, self.dist_in_patch = genr_2Didx(kernel_size)
        self.idx_x = Parameter(idx_x, requires_grad=False)
        self.idx_y = Parameter(idx_y, requires_grad=False)
        self.weight_in_patch = Parameter(
            (self.dist_in_patch * (pi / kernel_size)).cos(),
            requires_grad=False)

        N_in = in_channels * self.weight_in_patch.sum()

        self.W = Parameter(empty(out_channels, in_channels, self.weight_in_patch.shape[0]))
        self.Wr = Parameter(empty(out_channels))
        self.reset_filters_()
        
        self.multi = 1.0 / ((N_in / 6) - 0.5 * (N_in * 7 /180) ** (0.5)) # uniform
        # self.multi = 1.0 / ((2 * N_in) - 0.5 * (8 * N_in) ** (0.5)) # normal

        self.img_sz: List[int] = [0,0]
        self.idxs_x: Tensor
        self.idxs_y: Tensor
        self.pad_layer: ReflectionPad2d
        self.coo_patchs: Tensor
        self.patch_centers_xy: Tensor
        self.num_patchs: int

        self.dWr_increase = Buffer(full((), 0.1))
        class HexConv2dFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, img:Tensor, W:Tensor, Wr:Tensor):
                token_BNCL = self.process_img(img)

                dist2_list: list[Tensor] = []
                for Wi in W.unbind():
                    dist2_ = ((token_BNCL - Wi).pow(2) * self.weight_in_patch)
                    dist2_list.append(dist2_.sum(dim=(-1, -2)))
                dist2 = stack(dist2_list, -1)

                similarity = (-dist2 * (Wr.exp2() * self.multi)).exp2()

                ctx.save_for_backward(img, dist2, similarity)
                return similarity

            @staticmethod
            def backward(ctx, dLdY: Tensor):
                img, dist2, similarity = ctx.saved_tensors
                token_BNCL = self.process_img(img)

                dLdW = []
                dLdWr = []
                for Wi, d2i, Yi, Gi in zip(self.W, dist2.unbind(-1), similarity.unbind(-1), dLdY.unbind(-1)):
                    prefer_i = Yi * Gi  # (B, N)

                    direction_ = (token_BNCL - Wi)  # (B, N, C, L)
                    coef_ = (prefer_i + 0.1) / (d2i + 0.1)  # (B, N)

                    dLdW.append(einsum("bn,bncl->cl", coef_, direction_))
                    dLdWr.append(-(prefer_i * d2i).mean())


                dLdW = stack(dLdW)
                dLdWr = stack(dLdWr)

                return None, dLdW, dLdWr


        self.calc_func = HexConv2dFunc

    @no_grad()
    def reset_filters_(self, idx: Optional[Tensor] = None) -> None:
        """
        Reset (re-initialize) output filters in-place.

        idx:
          - None: reset all filters
          - LongTensor indices shaped (k,) or scalar
          - BoolTensor mask shaped (out_channels,)
        Resets:
          - self.W[idx] ~ Uniform(0, 1)
          - self.Wr[idx] = 0
        """
        if idx is None:
            init.uniform_(self.W, 0.0, 1.0)
            init.zeros_(self.Wr)
            return

        if not torch.is_tensor(idx):
            raise TypeError("idx must be a torch.Tensor")

        if idx.dtype == torch.bool:
            if idx.ndim != 1 or idx.numel() != self.out_channels:
                raise ValueError(f"bool mask idx must have shape ({self.out_channels},)")
            idx_ = idx.nonzero(as_tuple=False).flatten()
        else:
            idx_ = idx.to(dtype=torch.long).flatten()

        if idx_.numel() == 0:
            return

        idx_ = idx_.to(device=self.W.device)
        if int(idx_.min().item()) < 0 or int(idx_.max().item()) >= self.out_channels:
            raise ValueError(f"idx out of range [0, {self.out_channels - 1}]")

        init.uniform_(self.W[idx_], 0.0, 1.0)
        init.zeros_(self.Wr[idx_])

    def process_img(self, img: Tensor):
        img_pad = self.pad_layer(img)
        token_BCNL = img_pad[..., self.idxs_x, self.idxs_y]
        token_BNCL = token_BCNL.transpose(1, 2)
        return token_BNCL

    @property
    def patch_offsets_xy(self) -> Tensor:
        """Sampling offsets inside each raw patch, shaped ``(L, 2)``."""
        center = (self.kernel_size - 1) / 2.0
        return stack((self.idx_x - center, self.idx_y - center), dim=-1)

    def extract_patches(self, img: Tensor) -> tuple[Tensor, Tensor]:
        """Return unencoded circular patches and their hex-center coordinates.

        The returned patches have shape ``(B, N, C, L)`` and can be retained as
        the immutable ``P0`` memory for later layers. Coordinates are complex
        xy values shaped ``(N,)`` in exactly the same token order.
        """
        if img.ndim != 4:
            raise ValueError("img must have shape (B, C, H, W)")
        _, channels, *img_sz = img.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Input channel {channels} does not match with the expected input channel {self.in_channels}"
            )
        if self.img_sz != img_sz:
            self.img_sz = img_sz
            self.genr_idx_patchs(img_sz)
        return self.process_img(img), self.coo_patchs

    def forward(self, img:Tensor):
        self.extract_patches(img)

        return self.calc_func.apply(img, self.W, self.Wr)

    def genr_idx_patchs(self, img_sz):
        N_x, N_y, stride_x, stride_y = calc_coo_params(img_sz, self.lattice_stride)

        idxs_x, idxs_y = genr_2Dcoo(N_x, N_y, stride_x, stride_y)

        self.idxs_x = Buffer(
            idxs_x.unsqueeze(-1).to(self.idx_x.device) + self.idx_x, 
            persistent=False)
        self.idxs_y = Buffer(
            idxs_y.unsqueeze(-1).to(self.idx_x.device) + self.idx_y, 
            persistent=False)

        pad_x = ((N_x - 1) * stride_x + self.kernel_size - img_sz[0] + 1) // 2
        pad_y = ((N_y - 1) * stride_y + self.kernel_size - img_sz[1] + 1) // 2
        self.pad_layer = ReflectionPad2d((pad_y, pad_y, pad_x, pad_x))

        # Original-image pixel coordinates, ordered as (x=column, y=row), for
        # differentiable samplers such as PolarRingSampler.
        centers_xy = stack((idxs_y - pad_y + (self.kernel_size - 1) / 2,
                            idxs_x - pad_x + (self.kernel_size - 1) / 2), dim=-1)
        self.patch_centers_xy = Buffer(centers_xy.to(torch.float), persistent=False)


        coo_x, coo_y = genr_2Dcoo(N_x, N_y, 0.5, 3 ** (0.5) * 0.5, torch.float)
        coordinates = zeros(coo_x.shape, dtype=torch.complex64)
        coordinates.real = coo_x
        coordinates.imag = coo_y
        self.coo_patchs = Buffer(coordinates, persistent=False)

        self.num_patchs = self.coo_patchs.shape[0]

        
