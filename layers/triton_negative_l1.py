"""Memory-bounded negative L1 distance with a fused CUDA backward."""

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


@triton.jit
def _negative_l1_forward(
    x_ptr,
    w_ptr,
    output_ptr,
    query_size,
    pose_size: tl.constexpr,
    feature_size: tl.constexpr,
    block_pose: tl.constexpr,
    block_feature: tl.constexpr,
):
    query = tl.program_id(0)
    poses = tl.program_id(1) * block_pose + tl.arange(0, block_pose)
    pose_mask = poses < pose_size
    accumulator = tl.zeros((block_pose,), tl.float32)
    for feature_start in tl.static_range(0, feature_size, block_feature):
        features = feature_start + tl.arange(0, block_feature)
        feature_mask = features < feature_size
        x = tl.load(
            x_ptr + query * feature_size + features,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + poses[:, None] * feature_size + features[None, :],
            mask=pose_mask[:, None] & feature_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator -= tl.sum(tl.abs(x[None, :] - w), axis=1)
    tl.store(
        output_ptr + query * pose_size + poses,
        accumulator,
        mask=pose_mask,
    )


@triton.jit
def _negative_l1_backward_w(
    x_ptr,
    w_ptr,
    grad_output_ptr,
    grad_w_ptr,
    query_size,
    pose_size: tl.constexpr,
    feature_size: tl.constexpr,
    block_query: tl.constexpr,
    block_feature: tl.constexpr,
):
    pose = tl.program_id(0)
    features = tl.program_id(1) * block_feature + tl.arange(0, block_feature)
    feature_mask = features < feature_size
    w = tl.load(
        w_ptr + pose * feature_size + features,
        mask=feature_mask,
        other=0.0,
    ).to(tl.float32)
    accumulator = tl.zeros((block_feature,), tl.float32)
    for query_start in tl.range(0, query_size, block_query):
        queries = query_start + tl.arange(0, block_query)
        query_mask = queries < query_size
        x = tl.load(
            x_ptr + queries[:, None] * feature_size + features[None, :],
            mask=query_mask[:, None] & feature_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr + queries * pose_size + pose,
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        delta = x - w[None, :]
        derivative = tl.where(
            delta > 0.0,
            1.0,
            tl.where(delta < 0.0, -1.0, 0.0),
        )
        accumulator += tl.sum(
            grad_output[:, None] * derivative,
            axis=0,
        )
    tl.store(
        grad_w_ptr + pose * feature_size + features,
        accumulator,
        mask=feature_mask,
    )


@triton.jit
def _negative_l1_backward_x(
    x_ptr,
    w_ptr,
    grad_output_ptr,
    grad_x_ptr,
    query_size,
    pose_size: tl.constexpr,
    feature_size: tl.constexpr,
    block_pose: tl.constexpr,
    block_feature: tl.constexpr,
):
    query = tl.program_id(0)
    features = tl.program_id(1) * block_feature + tl.arange(0, block_feature)
    feature_mask = features < feature_size
    x = tl.load(
        x_ptr + query * feature_size + features,
        mask=feature_mask,
        other=0.0,
    ).to(tl.float32)
    accumulator = tl.zeros((block_feature,), tl.float32)
    for pose_start in tl.static_range(0, pose_size, block_pose):
        poses = pose_start + tl.arange(0, block_pose)
        pose_mask = poses < pose_size
        w = tl.load(
            w_ptr + poses[:, None] * feature_size + features[None, :],
            mask=pose_mask[:, None] & feature_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr + query * pose_size + poses,
            mask=pose_mask,
            other=0.0,
        ).to(tl.float32)
        delta = x[None, :] - w
        derivative = tl.where(
            delta > 0.0,
            -1.0,
            tl.where(delta < 0.0, 1.0, 0.0),
        )
        accumulator += tl.sum(
            grad_output[:, None] * derivative,
            axis=0,
        )
    tl.store(
        grad_x_ptr + query * feature_size + features,
        accumulator,
        mask=feature_mask,
    )


class _TritonNegativeL1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or prototype.ndim != 2:
            raise ValueError("expected x [Q,F] and prototype [H,F]")
        if x.shape[1] != prototype.shape[1]:
            raise ValueError("x and prototype feature dimensions differ")
        if not all(
            tensor.is_cuda and tensor.dtype == torch.float32 and tensor.is_contiguous()
            for tensor in (x, prototype)
        ):
            raise ValueError("Triton negative L1 expects contiguous CUDA float32")
        query_size, feature_size = x.shape
        pose_size = prototype.shape[0]
        output = torch.empty(
            (query_size, pose_size), device=x.device, dtype=torch.float32
        )
        block_pose = 8
        _negative_l1_forward[(query_size, triton.cdiv(pose_size, block_pose))](
            x,
            prototype,
            output,
            query_size,
            pose_size=pose_size,
            feature_size=feature_size,
            block_pose=block_pose,
            block_feature=128,
            num_warps=8,
        )
        ctx.save_for_backward(x, prototype)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, prototype = ctx.saved_tensors
        grad_output = grad_output.contiguous().float()
        query_size, feature_size = x.shape
        pose_size = prototype.shape[0]
        grad_x = grad_prototype = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.empty_like(x)
            block_feature = 64
            _negative_l1_backward_x[
                (query_size, triton.cdiv(feature_size, block_feature))
            ](
                x,
                prototype,
                grad_output,
                grad_x,
                query_size,
                pose_size=pose_size,
                feature_size=feature_size,
                block_pose=16,
                block_feature=block_feature,
                num_warps=8,
            )
        if ctx.needs_input_grad[1]:
            grad_prototype = torch.empty_like(prototype)
            block_feature = 32
            _negative_l1_backward_w[
                (pose_size, triton.cdiv(feature_size, block_feature))
            ](
                x,
                prototype,
                grad_output,
                grad_prototype,
                query_size,
                pose_size=pose_size,
                feature_size=feature_size,
                block_query=32,
                block_feature=block_feature,
                num_warps=8,
            )
        return grad_x, grad_prototype


def negative_l1_distance(
    x: torch.Tensor, prototype: torch.Tensor
) -> torch.Tensor:
    """Return ``-||x-w||_1`` as [Q,H] without a broadcast difference tensor."""
    if not x.is_cuda:
        return -torch.cdist(x, prototype, p=1)
    return _TritonNegativeL1.apply(
        x.contiguous().float(), prototype.contiguous().float()
    )
