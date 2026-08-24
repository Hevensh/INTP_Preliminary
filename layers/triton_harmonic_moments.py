"""Fused pose-to-harmonic moment reduction for CUDA tensors."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

# Triton's Windows default lives under the user profile, which may be
# read-only in managed runs. Keep generated binaries in the system temp area.
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "intpcore_triton_cache"),
)
import triton
import triton.language as tl


@triton.jit
def _moment_forward(
    weight_ptr, coefficient_ptr, output_ptr,
    bases: tl.constexpr, poses: tl.constexpr, spatial: tl.constexpr,
    harmonics: tl.constexpr, rows: tl.constexpr,
    block_poses: tl.constexpr, block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    row_mask = row < rows
    spatial_index = row % spatial
    base_query = row // spatial
    pose_offsets = tl.arange(0, block_poses)
    pose_mask = pose_offsets[None, :] < poses
    weight = tl.load(
        weight_ptr
        + base_query[:, None] * poses * spatial
        + pose_offsets[None, :] * spatial
        + spatial_index[:, None],
        mask=row_mask[:, None] & pose_mask, other=0.0,
    ).to(tl.float32)
    for harmonic in tl.static_range(0, harmonics):
        coefficient = tl.load(
            coefficient_ptr + pose_offsets * harmonics + harmonic,
            mask=pose_offsets < poses, other=0.0,
        ).to(tl.float32)
        tl.store(
            output_ptr + row * harmonics + harmonic,
            tl.sum(weight * coefficient[None, :], axis=1),
            mask=row_mask,
        )


@triton.jit
def _moment_backward_weight(
    grad_output_ptr, coefficient_ptr, grad_weight_ptr,
    poses: tl.constexpr, spatial: tl.constexpr,
    harmonics: tl.constexpr, rows: tl.constexpr,
    block_poses: tl.constexpr, block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    row_mask = row < rows
    spatial_index = row % spatial
    base_query = row // spatial
    pose_offsets = tl.arange(0, block_poses)
    pose_mask = pose_offsets[None, :] < poses
    accumulator = tl.zeros((block_rows, block_poses), tl.float32)
    for harmonic in tl.static_range(0, harmonics):
        grad_output = tl.load(
            grad_output_ptr + row * harmonics + harmonic,
            mask=row_mask, other=0.0,
        ).to(tl.float32)
        coefficient = tl.load(
            coefficient_ptr + pose_offsets * harmonics + harmonic,
            mask=pose_offsets < poses, other=0.0,
        ).to(tl.float32)
        accumulator += grad_output[:, None] * coefficient[None, :]
    tl.store(
        grad_weight_ptr
        + base_query[:, None] * poses * spatial
        + pose_offsets[None, :] * spatial
        + spatial_index[:, None],
        accumulator, mask=row_mask[:, None] & pose_mask,
    )


class _HarmonicMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, coefficients):
        if weight.ndim != 5 or coefficients.ndim != 2:
            raise ValueError("expected weight [Q,B,P,H,W] and coefficients [P,K]")
        queries, bases, poses, height, width = weight.shape
        if coefficients.shape[0] != poses:
            raise ValueError("pose and coefficient counts differ")
        harmonics = coefficients.shape[1]
        output = torch.empty(
            queries, bases, height, width, harmonics,
            device=weight.device, dtype=weight.dtype,
        )
        spatial = height * width
        rows = queries * bases * spatial
        block_poses = triton.next_power_of_2(poses)
        block_rows = 16
        _moment_forward[(triton.cdiv(rows, block_rows),)](
            weight, coefficients, output,
            bases=bases, poses=poses, spatial=spatial,
            harmonics=harmonics, rows=rows,
            block_poses=block_poses, block_rows=block_rows,
        )
        ctx.save_for_backward(coefficients)
        ctx.weight_shape = weight.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (coefficients,) = ctx.saved_tensors
        queries, bases, poses, height, width = ctx.weight_shape
        spatial = height * width
        rows = queries * bases * spatial
        grad_weight = torch.empty(
            ctx.weight_shape, device=grad_output.device,
            dtype=grad_output.dtype,
        )
        block_rows = 16
        _moment_backward_weight[(triton.cdiv(rows, block_rows),)](
            grad_output.contiguous(), coefficients, grad_weight,
            poses=poses, spatial=spatial,
            harmonics=coefficients.shape[1],
            rows=rows,
            block_poses=triton.next_power_of_2(poses),
            block_rows=block_rows,
        )
        return grad_weight, None


def triton_harmonic_moments(weight, coefficients):
    """Return [Q,B,H,W,K], falling back to PyTorch away from CUDA."""
    if not weight.is_cuda:
        return torch.einsum("qbphw,pk->qbhwk", weight, coefficients)
    return _HarmonicMoments.apply(weight, coefficients.contiguous())
