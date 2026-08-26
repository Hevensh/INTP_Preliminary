"""Flash-style attention with an implicit pose-conditioned Look bias.

The forward kernel never materialises ``B x H x N x N`` attention logits or
Look bias.  The current backward deliberately recomputes the small reference
formula one layer at a time; this keeps activation memory bounded while giving
us an exact gradient oracle before replacing the backward with Triton.
"""

from __future__ import annotations

import math
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


def reference_structured_look_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pose: torch.Tensor,
    fields: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Explicit correctness oracle.

    ``q/k/v`` are ``(B,H,N+1,D)``, pose is ``(B,N,H,P)``, and fields are
    ``(H,P,N,N)``.  CLS has no Look bias.
    """
    patch_count = pose.shape[1]
    if q.shape[-2] != patch_count + 1:
        raise ValueError("q sequence must contain one CLS token plus pose patches")
    look = torch.einsum("bqhp,hpqk->bhqk", pose, fields)
    scores = q.float() @ k.float().transpose(-2, -1)
    scores = scores * float(scale)
    scores = scores.clone()
    scores[:, :, 1:, 1:] += look.float()
    probability = scores.softmax(dim=-1)
    return (probability @ v.float()).to(q.dtype)


@triton.jit
def _structured_look_forward(
    q_ptr, k_ptr, v_ptr, pose_ptr, field_ptr, output_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_pb, stride_pq, stride_ph, stride_pp,
    stride_fh, stride_fp, stride_fq, stride_fk,
    stride_ob, stride_oh, stride_on, stride_od,
    heads: tl.constexpr,
    sequence: tl.constexpr,
    patch_count: tl.constexpr,
    head_dim: tl.constexpr,
    poses: tl.constexpr,
    scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    batch_head = tl.program_id(0)
    block_q = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    query = block_q * block_m + tl.arange(0, block_m)
    feature = tl.arange(0, block_d)
    query_mask = query < sequence
    feature_mask = feature < head_dim

    q = tl.load(
        q_ptr + batch * stride_qb + head * stride_qh
        + query[:, None] * stride_qn + feature[None, :] * stride_qd,
        mask=query_mask[:, None] & feature_mask[None, :],
        other=0.0,
    )
    row_max = tl.full((block_m,), -float("inf"), tl.float32)
    row_sum = tl.zeros((block_m,), tl.float32)
    accumulator = tl.zeros((block_m, block_d), tl.float32)

    for key_start in tl.static_range(0, sequence, block_n):
        key = key_start + tl.arange(0, block_n)
        key_mask = key < sequence
        k = tl.load(
            k_ptr + batch * stride_kb + head * stride_kh
            + key[:, None] * stride_kn + feature[None, :] * stride_kd,
            mask=key_mask[:, None] & feature_mask[None, :],
            other=0.0,
        )
        score = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        patch_query = query - 1
        patch_key = key - 1
        look_valid = (
            query_mask[:, None] & key_mask[None, :]
            & (query[:, None] > 0) & (key[None, :] > 0)
        )
        for pose_index in tl.static_range(0, poses):
            probability = tl.load(
                pose_ptr + batch * stride_pb + patch_query * stride_pq
                + head * stride_ph + pose_index * stride_pp,
                mask=query_mask & (query > 0) & (patch_query < patch_count),
                other=0.0,
            ).to(tl.float32)
            field = tl.load(
                field_ptr + head * stride_fh + pose_index * stride_fp
                + patch_query[:, None] * stride_fq
                + patch_key[None, :] * stride_fk,
                mask=look_valid,
                other=0.0,
            ).to(tl.float32)
            score += probability[:, None] * field
        score = tl.where(query_mask[:, None] & key_mask[None, :], score, -float("inf"))

        new_max = tl.maximum(row_max, tl.max(score, axis=1))
        old_scale = tl.exp(row_max - new_max)
        probability = tl.exp(score - new_max[:, None])
        new_sum = row_sum * old_scale + tl.sum(probability, axis=1)
        value = tl.load(
            v_ptr + batch * stride_vb + head * stride_vh
            + key[:, None] * stride_vn + feature[None, :] * stride_vd,
            mask=key_mask[:, None] & feature_mask[None, :],
            other=0.0,
        )
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probability.to(value.dtype), value, input_precision="ieee"
        )
        row_max = new_max
        row_sum = new_sum

    accumulator /= row_sum[:, None]
    tl.store(
        output_ptr + batch * stride_ob + head * stride_oh
        + query[:, None] * stride_on + feature[None, :] * stride_od,
        accumulator,
        mask=query_mask[:, None] & feature_mask[None, :],
    )


class _StructuredLookAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pose, fields, scale):
        if not all(tensor.is_cuda for tensor in (q, k, v, pose, fields)):
            raise ValueError("Triton structured Look attention requires CUDA tensors")
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
            raise ValueError("q, k, and v must share shape (B,H,N,D)")
        if pose.ndim != 4 or fields.ndim != 4:
            raise ValueError("pose and fields must have shapes (B,N,H,P) and (H,P,N,N)")
        batch, heads, sequence, head_dim = q.shape
        patch_count, poses = pose.shape[1], pose.shape[-1]
        if pose.shape != (batch, patch_count, heads, poses):
            raise ValueError("pose shape is inconsistent with q")
        if fields.shape != (heads, poses, patch_count, patch_count):
            raise ValueError("fields shape is inconsistent with pose")
        if sequence != patch_count + 1:
            raise ValueError("sequence must be patch_count + one CLS token")
        block_d = triton.next_power_of_2(head_dim)
        if block_d > 128:
            raise ValueError("head_dim above 128 is not supported")
        output = torch.empty_like(q)
        block_m, block_n = 16, 32
        _structured_look_forward[(batch * heads, triton.cdiv(sequence, block_m))](
            q, k, v, pose, fields, output,
            *q.stride(), *k.stride(), *v.stride(), *pose.stride(), *fields.stride(),
            *output.stride(),
            heads=heads, sequence=sequence, patch_count=patch_count,
            head_dim=head_dim, poses=poses, scale=float(scale),
            block_m=block_m, block_n=block_n, block_d=block_d,
            num_warps=4,
        )
        ctx.scale = float(scale)
        ctx.save_for_backward(q, k, v, pose, fields)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        needs = ctx.needs_input_grad[:5]
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_(need) for tensor, need in zip(saved, needs)]
            output = reference_structured_look_attention(
                *inputs, scale=ctx.scale
            )
            requested = [tensor for tensor, need in zip(inputs, needs) if need]
            gradients = torch.autograd.grad(
                output, requested, grad_output, allow_unused=False
            )
        result = []
        iterator = iter(gradients)
        for need in needs:
            result.append(next(iterator) if need else None)
        return (*result, None)


def structured_look_attention(q, k, v, pose, fields, *, scale):
    """Memory-bounded attention; CPU uses the explicit reference oracle."""
    if not q.is_cuda:
        return reference_structured_look_attention(
            q, k, v, pose, fields, scale=scale
        )
    return _StructuredLookAttention.apply(q, k, v, pose, fields, float(scale))
