from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _distance_forward(
    x_ptr, w_ptr, cover_ptr, log_scale_ptr, y_ptr, dist_ptr,
    m_size: tl.constexpr, h_size: tl.constexpr, d_size: tl.constexpr,
    multi: tl.constexpr, metric: tl.constexpr, block_d: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    offsets = tl.arange(0, block_d)
    mask = offsets < d_size
    x = tl.load(x_ptr + m * d_size + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + h * d_size + offsets, mask=mask, other=0.0).to(tl.float32)
    cover = tl.load(cover_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    delta = x - w
    if metric == 1:
        element = tl.abs(delta)
    else:
        element = delta * delta
    distance = tl.sum(element * cover, axis=0)
    scale = tl.exp2(tl.load(log_scale_ptr + h).to(tl.float32)) * multi
    similarity = tl.exp2(-distance * scale)
    index = m * h_size + h
    tl.store(y_ptr + index, similarity)
    tl.store(dist_ptr + index, distance)


@triton.jit
def _distance_backward_x(
    x_ptr, w_ptr, cover_ptr, log_scale_ptr, y_ptr, grad_y_ptr, grad_x_ptr,
    m_size: tl.constexpr, h_size: tl.constexpr, d_size: tl.constexpr,
    multi: tl.constexpr, metric: tl.constexpr,
    block_d: tl.constexpr, block_h: tl.constexpr,
):
    m = tl.program_id(0)
    d_offsets = tl.program_id(1) * block_d + tl.arange(0, block_d)
    d_mask = d_offsets < d_size
    x = tl.load(x_ptr + m * d_size + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    cover = tl.load(cover_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    accumulator = tl.zeros((block_d,), tl.float32)
    log2 = 0.6931471805599453
    for h_start in range(0, h_size, block_h):
        h_offsets = h_start + tl.arange(0, block_h)
        h_mask = h_offsets < h_size
        w = tl.load(
            w_ptr + h_offsets[:, None] * d_size + d_offsets[None, :],
            mask=h_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        index = m * h_size + h_offsets
        y = tl.load(y_ptr + index, mask=h_mask, other=0.0).to(tl.float32)
        grad_y = tl.load(grad_y_ptr + index, mask=h_mask, other=0.0).to(tl.float32)
        scale = tl.exp2(tl.load(log_scale_ptr + h_offsets, mask=h_mask, other=0.0)) * multi
        grad_dist = -grad_y * y * log2 * scale
        delta = x[None, :] - w
        if metric == 1:
            derivative = tl.where(delta > 0, 1.0, tl.where(delta < 0, -1.0, 0.0))
        else:
            derivative = 2.0 * delta
        accumulator += tl.sum(grad_dist[:, None] * derivative * cover[None, :], axis=0)
    tl.store(grad_x_ptr + m * d_size + d_offsets, accumulator, mask=d_mask)


@triton.jit
def _distance_backward_w(
    x_ptr, w_ptr, cover_ptr, log_scale_ptr, y_ptr, grad_y_ptr, grad_w_ptr,
    m_size: tl.constexpr, h_size: tl.constexpr, d_size: tl.constexpr,
    multi: tl.constexpr, metric: tl.constexpr,
    block_d: tl.constexpr, block_m: tl.constexpr,
):
    h = tl.program_id(0)
    d_offsets = tl.program_id(1) * block_d + tl.arange(0, block_d)
    d_mask = d_offsets < d_size
    w = tl.load(w_ptr + h * d_size + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    cover = tl.load(cover_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    accumulator = tl.zeros((block_d,), tl.float32)
    log2 = 0.6931471805599453
    for m_start in range(0, m_size, block_m):
        m_offsets = m_start + tl.arange(0, block_m)
        m_mask = m_offsets < m_size
        x = tl.load(
            x_ptr + m_offsets[:, None] * d_size + d_offsets[None, :],
            mask=m_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        index = m_offsets * h_size + h
        y = tl.load(y_ptr + index, mask=m_mask, other=0.0).to(tl.float32)
        grad_y = tl.load(grad_y_ptr + index, mask=m_mask, other=0.0).to(tl.float32)
        scale = tl.exp2(tl.load(log_scale_ptr + h).to(tl.float32)) * multi
        grad_dist = -grad_y * y * log2 * scale
        delta = x - w[None, :]
        if metric == 1:
            derivative = tl.where(delta > 0, -1.0, tl.where(delta < 0, 1.0, 0.0))
        else:
            derivative = -2.0 * delta
        accumulator += tl.sum(grad_dist[:, None] * derivative * cover[None, :], axis=0)
    tl.store(grad_w_ptr + h * d_size + d_offsets, accumulator, mask=d_mask)


@triton.jit
def _distance_backward_scale(
    y_ptr, dist_ptr, grad_y_ptr, log_scale_ptr, grad_scale_ptr,
    m_size: tl.constexpr, h_size: tl.constexpr, multi: tl.constexpr,
    block_m: tl.constexpr,
):
    h = tl.program_id(0)
    accumulator = tl.zeros((block_m,), tl.float32)
    log2 = 0.6931471805599453
    scale = tl.exp2(tl.load(log_scale_ptr + h).to(tl.float32)) * multi
    for m_start in range(0, m_size, block_m):
        m_offsets = m_start + tl.arange(0, block_m)
        mask = m_offsets < m_size
        index = m_offsets * h_size + h
        y = tl.load(y_ptr + index, mask=mask, other=0.0).to(tl.float32)
        distance = tl.load(dist_ptr + index, mask=mask, other=0.0).to(tl.float32)
        grad_y = tl.load(grad_y_ptr + index, mask=mask, other=0.0).to(tl.float32)
        accumulator += -grad_y * y * distance * scale * log2 * log2
    tl.store(grad_scale_ptr + h, tl.sum(accumulator, axis=0))


class _TritonDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, prototype, log2_scale, cover, multi: float, metric_code: int):
        if not all(t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() for t in (x, prototype, log2_scale, cover)):
            raise ValueError("Triton distance expects contiguous CUDA float32 tensors")
        m_size, d_size = x.shape
        h_size = prototype.shape[0]
        block_d = triton.next_power_of_2(d_size)
        y = torch.empty((m_size, h_size), device=x.device, dtype=torch.float32)
        distance = torch.empty_like(y)
        _distance_forward[(m_size, h_size)](
            x, prototype, cover, log2_scale, y, distance,
            m_size=m_size, h_size=h_size, d_size=d_size,
            multi=multi, metric=metric_code, block_d=block_d,
            num_warps=8,
        )
        ctx.save_for_backward(x, prototype, log2_scale, cover, y, distance)
        ctx.multi = multi
        ctx.metric_code = metric_code
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, prototype, log2_scale, cover, y, distance = ctx.saved_tensors
        grad_y = grad_y.contiguous().float()
        m_size, d_size = x.shape
        h_size = prototype.shape[0]
        grad_x = torch.empty_like(x)
        grad_w = torch.empty_like(prototype)
        grad_scale = torch.empty_like(log2_scale)
        block_d = 32
        _distance_backward_x[(m_size, triton.cdiv(d_size, block_d))](
            x, prototype, cover, log2_scale, y, grad_y, grad_x,
            m_size=m_size, h_size=h_size, d_size=d_size,
            multi=ctx.multi, metric=ctx.metric_code,
            block_d=block_d, block_h=16, num_warps=8,
        )
        _distance_backward_w[(h_size, triton.cdiv(d_size, block_d))](
            x, prototype, cover, log2_scale, y, grad_y, grad_w,
            m_size=m_size, h_size=h_size, d_size=d_size,
            multi=ctx.multi, metric=ctx.metric_code,
            block_d=block_d, block_m=16, num_warps=8,
        )
        _distance_backward_scale[(h_size,)](
            y, distance, grad_y, log2_scale, grad_scale,
            m_size=m_size, h_size=h_size, multi=ctx.multi,
            block_m=256, num_warps=8,
        )
        return grad_x, grad_w, grad_scale, None, None, None


def triton_distance_similarity(
    x: torch.Tensor,
    prototype: torch.Tensor,
    log2_scale: torch.Tensor,
    cover: torch.Tensor,
    multi: float,
    metric: str = "l2",
) -> torch.Tensor:
    """Fused indexed L1/L2 distance similarity for flattened patches."""
    if metric not in {"l1", "l2"}:
        raise ValueError("metric must be l1 or l2")
    return _TritonDistanceFunction.apply(
        x.contiguous().float(), prototype.contiguous().float(),
        log2_scale.contiguous().float(), cover.contiguous().float(),
        float(multi), 1 if metric == "l1" else 2,
    )

