import torch
from math import ceil
from typing import Tuple

def genr_2Didx(kernel_size: int):
    coordinates = torch.zeros(kernel_size, kernel_size, dtype=torch.complex64)
    idx = torch.arange(kernel_size).tile(kernel_size,1)
    dist1D = idx.float() - (kernel_size - 1) / 2

    coordinates.real = dist1D
    coordinates.imag = dist1D.T
    dist2D = coordinates.abs()
    in_ring = dist2D < kernel_size/2
    # return in_ring
    idx_x = idx[in_ring]
    idx_y = idx.T[in_ring]
    idx_dist = dist2D[in_ring]

    return idx_x, idx_y, idx_dist

def calc_coo_params(img_size: Tuple[int, int], stride: int):
    sz_x, sz_y = img_size
    stride_x: int = stride // 2
    stride_y: int = round(stride * (3 ** (1/2) / 2))

    N_x: int = ceil(sz_x / stride_x) + 1
    N_y: int = ceil(sz_y / stride_y) + 1

    return N_x, N_y, stride_x, stride_y

def genr_2Dcoo(N_x:int, N_y:int, stride_x:int, stride_y:int, dtype:torch.dtype=torch.int64):
    idxs_x_2row = torch.arange(N_x, dtype=dtype) * stride_x
    idxs_y_2row = torch.zeros(N_x, dtype=dtype)
    idxs_y_2row[1::2] += stride_y

    idxs_x_2row = idxs_x_2row.tile(N_y // 2, 1)
    idxs_y_2row = idxs_y_2row.tile(N_y // 2, 1)
    idxs_y_2row += torch.arange(N_y // 2, dtype=dtype).unsqueeze(-1) * (2 * stride_y) 

    idxs_x_2row = idxs_x_2row.view(-1)
    idxs_y_2row = idxs_y_2row.view(-1)

    if N_y % 2:
        idxs_x_1row = idxs_x_2row[:N_x:2]
        idxs_y_1row = torch.zeros_like(idxs_x_1row, dtype=dtype) + (N_y - 1) * stride_y

        idxs_x = torch.concat((idxs_x_2row, idxs_x_1row))
        idxs_y = torch.concat((idxs_y_2row, idxs_y_1row))

    else:
        idxs_x = idxs_x_2row
        idxs_y = idxs_y_2row

    return idxs_x, idxs_y

