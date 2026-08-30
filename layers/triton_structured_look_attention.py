"""Flash-style attention with an implicit pose-conditioned Look bias.

The forward kernel never materialises ``B x H x N x N`` attention logits or
Look bias.  Backward recomputes probabilities and score gradients in a Triton
kernel, then uses GEMMs for Q/K/V and compact reductions for pose/grid grads.
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
    dense_bias: torch.Tensor | None = None,
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
    if dense_bias is not None:
        expected = (q.shape[0], q.shape[1], patch_count, patch_count)
        if dense_bias.shape != expected:
            raise ValueError(f"dense_bias must have shape {expected}")
        scores[:, :, 1:, 1:] += dense_bias.float()
    probability = scores.softmax(dim=-1)
    return (probability @ v.float()).to(q.dtype)


@triton.jit
def _structured_look_forward(
    q_ptr, k_ptr, v_ptr, pose_ptr, field_ptr, dense_ptr, output_ptr, lse_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_pb, stride_pq, stride_ph, stride_pp,
    stride_fh, stride_fp, stride_fq, stride_fk,
    stride_db, stride_dh, stride_dq, stride_dk,
    stride_ob, stride_oh, stride_on, stride_od,
    heads: tl.constexpr,
    sequence: tl.constexpr,
    patch_count: tl.constexpr,
    head_dim: tl.constexpr,
    poses: tl.constexpr,
    has_dense: tl.constexpr,
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
        if has_dense:
            dense = tl.load(
                dense_ptr + batch * stride_db + head * stride_dh
                + patch_query[:, None] * stride_dq
                + patch_key[None, :] * stride_dk,
                mask=look_valid,
                other=0.0,
            ).to(tl.float32)
            score += dense
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
    tl.store(
        lse_ptr + batch_head * sequence + query,
        row_max + tl.log(row_sum),
        mask=query_mask,
    )


@triton.jit
def _structured_look_probability_score_grad(
    q_ptr, k_ptr, v_ptr, pose_ptr, field_ptr, dense_ptr,
    grad_output_ptr, output_ptr, lse_ptr,
    probability_ptr, score_grad_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_pb, stride_pq, stride_ph, stride_pp,
    stride_fh, stride_fp, stride_fq, stride_fk,
    stride_db, stride_dh, stride_dq, stride_dk,
    stride_gob, stride_goh, stride_gon, stride_god,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_prob_b, stride_prob_h, stride_prob_q, stride_prob_k,
    heads: tl.constexpr,
    sequence: tl.constexpr,
    patch_count: tl.constexpr,
    head_dim: tl.constexpr,
    poses: tl.constexpr,
    has_dense: tl.constexpr,
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
    lse = tl.load(
        lse_ptr + batch_head * sequence + query,
        mask=query_mask,
        other=0.0,
    )
    grad_output = tl.load(
        grad_output_ptr + batch * stride_gob + head * stride_goh
        + query[:, None] * stride_gon + feature[None, :] * stride_god,
        mask=query_mask[:, None] & feature_mask[None, :],
        other=0.0,
    )
    output = tl.load(
        output_ptr + batch * stride_ob + head * stride_oh
        + query[:, None] * stride_on + feature[None, :] * stride_od,
        mask=query_mask[:, None] & feature_mask[None, :],
        other=0.0,
    )
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    patch_query = query - 1

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
        patch_key = key - 1
        look_valid = (
            query_mask[:, None] & key_mask[None, :]
            & (query[:, None] > 0) & (key[None, :] > 0)
        )
        for pose_index in tl.static_range(0, poses):
            pose_probability = tl.load(
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
            score += pose_probability[:, None] * field
        if has_dense:
            dense = tl.load(
                dense_ptr + batch * stride_db + head * stride_dh
                + patch_query[:, None] * stride_dq
                + patch_key[None, :] * stride_dk,
                mask=look_valid,
                other=0.0,
            ).to(tl.float32)
            score += dense

        probability = tl.exp(score - lse[:, None])
        probability = tl.where(
            query_mask[:, None] & key_mask[None, :], probability, 0.0
        )
        value = tl.load(
            v_ptr + batch * stride_vb + head * stride_vh
            + key[:, None] * stride_vn + feature[None, :] * stride_vd,
            mask=key_mask[:, None] & feature_mask[None, :],
            other=0.0,
        )
        grad_probability = tl.dot(
            grad_output, tl.trans(value), input_precision="ieee"
        )
        score_grad = probability * (grad_probability - delta[:, None])
        matrix_offset = (
            batch * stride_prob_b + head * stride_prob_h
            + query[:, None] * stride_prob_q + key[None, :] * stride_prob_k
        )
        matrix_mask = query_mask[:, None] & key_mask[None, :]
        tl.store(probability_ptr + matrix_offset, probability, mask=matrix_mask)
        tl.store(score_grad_ptr + matrix_offset, score_grad, mask=matrix_mask)


class _StructuredLookAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pose, fields, dense_bias, scale):
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
        has_dense = dense_bias.numel() > 0
        if has_dense and dense_bias.shape != (
            batch, heads, patch_count, patch_count
        ):
            raise ValueError("dense_bias shape is inconsistent with q")
        if sequence != patch_count + 1:
            raise ValueError("sequence must be patch_count + one CLS token")
        block_d = triton.next_power_of_2(head_dim)
        if block_d > 128:
            raise ValueError("head_dim above 128 is not supported")
        output = torch.empty_like(q)
        lse = torch.empty(
            (batch, heads, sequence), device=q.device, dtype=torch.float32
        )
        block_m, block_n = 16, 32
        _structured_look_forward[(batch * heads, triton.cdiv(sequence, block_m))](
            q, k, v, pose, fields, dense_bias, output, lse,
            *q.stride(), *k.stride(), *v.stride(), *pose.stride(), *fields.stride(),
            *(dense_bias.stride() if has_dense else (0, 0, 0, 0)),
            *output.stride(),
            heads=heads, sequence=sequence, patch_count=patch_count,
            head_dim=head_dim, poses=poses, has_dense=has_dense, scale=float(scale),
            block_m=block_m, block_n=block_n, block_d=block_d,
            num_warps=4,
        )
        ctx.scale = float(scale)
        ctx.has_dense = has_dense
        ctx.save_for_backward(q, k, v, pose, fields, dense_bias, output, lse)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, pose, fields, dense_bias, output, lse = ctx.saved_tensors
        needs = ctx.needs_input_grad[:6]
        batch, heads, sequence, head_dim = q.shape
        patch_count, poses = pose.shape[1], pose.shape[-1]
        block_m, block_n = 16, 32
        block_d = triton.next_power_of_2(head_dim)

        # Storing these two transient matrices makes the expensive score/Look
        # recomputation a single fused pass.  They live only during backward.
        probability = torch.empty(
            (batch, heads, sequence, sequence), device=q.device, dtype=q.dtype
        )
        score_grad = torch.empty_like(probability)
        grid = (
            batch * heads,
            triton.cdiv(sequence, block_m),
        )
        _structured_look_probability_score_grad[grid](
            q, k, v, pose, fields, dense_bias, grad_output, output, lse,
            probability, score_grad,
            *q.stride(), *k.stride(), *v.stride(), *pose.stride(), *fields.stride(),
            *(dense_bias.stride() if ctx.has_dense else (0, 0, 0, 0)),
            *grad_output.stride(), *output.stride(), *probability.stride(),
            heads=heads, sequence=sequence, patch_count=patch_count,
            head_dim=head_dim, poses=poses, has_dense=ctx.has_dense, scale=ctx.scale,
            block_m=block_m, block_n=block_n, block_d=block_d,
            num_warps=4,
        )

        grad_q = grad_k = grad_v = grad_pose = grad_fields = grad_dense = None
        if needs[0]:
            grad_q = (score_grad @ k) * ctx.scale
        if needs[1]:
            grad_k = (score_grad.transpose(-2, -1) @ q) * ctx.scale
        if needs[2]:
            grad_v = probability.transpose(-2, -1) @ grad_output

        patch_score_grad = score_grad[:, :, 1:, 1:].float()
        if needs[3]:
            grad_pose = torch.einsum(
                "bhqk,hpqk->bqhp", patch_score_grad, fields.float()
            ).to(pose.dtype)
        if needs[4]:
            grad_fields = torch.einsum(
                "bhqk,bqhp->hpqk", patch_score_grad, pose.float()
            ).to(fields.dtype)
        if needs[5] and ctx.has_dense:
            grad_dense = patch_score_grad.to(dense_bias.dtype)
        return grad_q, grad_k, grad_v, grad_pose, grad_fields, grad_dense, None


def structured_look_attention(
    q, k, v, pose, fields, *, scale, dense_bias=None
):
    """Memory-bounded attention; CPU uses the explicit reference oracle."""
    if not q.is_cuda:
        return reference_structured_look_attention(
            q, k, v, pose, fields, scale=scale, dense_bias=dense_bias
        )
    if dense_bias is None:
        dense_bias = q.new_empty(0)
    return _StructuredLookAttention.apply(
        q, k, v, pose, fields, dense_bias, float(scale)
    )
