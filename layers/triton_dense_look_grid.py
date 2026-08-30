"""Fused bilinear sampling from per-query 4x12 Look grids."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "intpcore_triton_cache"),
)
import triton
import triton.language as tl


def reference_dense_look_grid_sample(
    grid: torch.Tensor,
    radial0: torch.Tensor,
    radial1: torch.Tensor,
    angular0: torch.Tensor,
    angular1: torch.Tensor,
    radial_fraction: torch.Tensor,
    angular_fraction: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Reference sampler: grid [B,Q,H,4,12] -> bias [B,H,Q,K]."""

    flat = grid.permute(0, 2, 1, 3, 4).flatten(-2)

    def gather(radial: torch.Tensor, angular: torch.Tensor) -> torch.Tensor:
        index = (radial * 12 + angular)[None, None]
        return torch.gather(
            flat,
            dim=-1,
            index=index.expand(flat.shape[0], flat.shape[1], -1, -1),
        )

    rw = radial_fraction[None, None]
    aw = angular_fraction[None, None]
    sampled = (
        gather(radial0, angular0) * (1.0 - rw) * (1.0 - aw)
        + gather(radial0, angular1) * (1.0 - rw) * aw
        + gather(radial1, angular0) * rw * (1.0 - aw)
        + gather(radial1, angular1) * rw * aw
    )
    return sampled * valid[None, None].to(sampled.dtype)


@triton.jit
def _dense_look_grid_forward(
    grid_ptr,
    r0_ptr, r1_ptr, a0_ptr, a1_ptr, rw_ptr, aw_ptr, valid_ptr,
    output_ptr,
    stride_gb, stride_gq, stride_gh, stride_gr, stride_ga,
    stride_sq, stride_sk,
    stride_ob, stride_oh, stride_oq, stride_ok,
    heads: tl.constexpr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    key_block = tl.program_id(1)
    batch = row // (heads * queries)
    head_query = row % (heads * queries)
    head = head_query // queries
    query = head_query % queries
    key = key_block * block_k + tl.arange(0, block_k)
    key_mask = key < keys
    sample_offset = query * stride_sq + key * stride_sk
    sample_valid = tl.load(valid_ptr + sample_offset, mask=key_mask, other=0)
    mask = key_mask & sample_valid

    r0 = tl.load(r0_ptr + sample_offset, mask=key_mask, other=0)
    r1 = tl.load(r1_ptr + sample_offset, mask=key_mask, other=0)
    a0 = tl.load(a0_ptr + sample_offset, mask=key_mask, other=0)
    a1 = tl.load(a1_ptr + sample_offset, mask=key_mask, other=0)
    rw = tl.load(rw_ptr + sample_offset, mask=key_mask, other=0.0)
    aw = tl.load(aw_ptr + sample_offset, mask=key_mask, other=0.0)
    grid_base = (
        batch * stride_gb + query * stride_gq + head * stride_gh
    )

    g00 = tl.load(
        grid_ptr + grid_base + r0 * stride_gr + a0 * stride_ga,
        mask=mask, other=0.0,
    ).to(tl.float32)
    g01 = tl.load(
        grid_ptr + grid_base + r0 * stride_gr + a1 * stride_ga,
        mask=mask, other=0.0,
    ).to(tl.float32)
    g10 = tl.load(
        grid_ptr + grid_base + r1 * stride_gr + a0 * stride_ga,
        mask=mask, other=0.0,
    ).to(tl.float32)
    g11 = tl.load(
        grid_ptr + grid_base + r1 * stride_gr + a1 * stride_ga,
        mask=mask, other=0.0,
    ).to(tl.float32)
    value = g00 * (1.0 - rw) * (1.0 - aw)
    value += g01 * (1.0 - rw) * aw
    value += g10 * rw * (1.0 - aw)
    value += g11 * rw * aw
    output_offset = (
        batch * stride_ob + head * stride_oh
        + query * stride_oq + key * stride_ok
    )
    tl.store(output_ptr + output_offset, value, mask=key_mask)


class _DenseLookGridSample(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grid,
        radial0,
        radial1,
        angular0,
        angular1,
        radial_fraction,
        angular_fraction,
        valid,
    ):
        batch, queries, heads, radial_bins, angular_bins = grid.shape
        if (radial_bins, angular_bins) != (4, 12):
            raise ValueError("dense Look grid must have shape [B,Q,H,4,12]")
        if radial0.shape != (queries, queries):
            raise ValueError("Look sampling buffers must have shape [Q,Q]")
        output = torch.empty(
            batch, heads, queries, queries,
            device=grid.device,
            dtype=grid.dtype,
        )
        block_k = 64
        _dense_look_grid_forward[
            (batch * heads * queries, triton.cdiv(queries, block_k))
        ](
            grid,
            radial0, radial1, angular0, angular1,
            radial_fraction, angular_fraction, valid,
            output,
            *grid.stride(), *radial0.stride(), *output.stride(),
            heads=heads, queries=queries, keys=queries, block_k=block_k,
            num_warps=4,
        )
        ctx.grid_shape = grid.shape
        ctx.save_for_backward(
            radial0, radial1, angular0, angular1,
            radial_fraction, angular_fraction, valid,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        radial0, radial1, angular0, angular1, rw, aw, valid = ctx.saved_tensors
        batch, queries, heads, _, _ = ctx.grid_shape
        grad_flat = grad_output.new_zeros(batch, queries, heads, 48)
        source = grad_output.permute(0, 2, 1, 3)
        valid_weight = valid.to(source.dtype)
        corners = (
            (radial0, angular0, (1.0 - rw) * (1.0 - aw)),
            (radial0, angular1, (1.0 - rw) * aw),
            (radial1, angular0, rw * (1.0 - aw)),
            (radial1, angular1, rw * aw),
        )
        for radial, angular, weight in corners:
            index = (radial * 12 + angular)[None, :, None].expand(
                batch, -1, heads, -1
            )
            grad_flat.scatter_add_(
                -1,
                index,
                source * (
                    weight.to(source.dtype) * valid_weight
                )[None, :, None],
            )
        grad_grid = grad_flat.reshape(batch, queries, heads, 4, 12)
        return grad_grid, None, None, None, None, None, None, None


def dense_look_grid_sample(
    grid: torch.Tensor,
    radial0: torch.Tensor,
    radial1: torch.Tensor,
    angular0: torch.Tensor,
    angular1: torch.Tensor,
    radial_fraction: torch.Tensor,
    angular_fraction: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if not grid.is_cuda:
        return reference_dense_look_grid_sample(
            grid, radial0, radial1, angular0, angular1,
            radial_fraction, angular_fraction, valid,
        )
    return _DenseLookGridSample.apply(
        grid, radial0, radial1, angular0, angular1,
        radial_fraction, angular_fraction, valid,
    )
