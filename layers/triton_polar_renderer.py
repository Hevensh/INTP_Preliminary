"""Fused polar interpolation with a scatter-add backward.

The PyTorch reference renderer uses four advanced-index operations. Their
backward launches repeated ``index_put`` kernels, which dominates paired MAMS
training. This implementation performs the same four-point interpolation in
one forward and one backward kernel.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "intpcore_triton_cache"),
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installations
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _polar_forward(
        prototype, i00, i01, i10, i11, a0, a1, radial, output,
        elements: tl.constexpr, channels: tl.constexpr,
        directions: tl.constexpr, samples: tl.constexpr,
        stored: tl.constexpr, block: tl.constexpr,
    ):
        offset = tl.program_id(0) * block + tl.arange(0, block)
        mask = offset < elements
        sample = offset % samples
        rest = offset // samples
        channel = rest % channels
        rest = rest // channels
        direction = rest % directions
        base = rest // directions
        lookup = direction * samples + sample
        index00 = tl.load(i00 + lookup, mask=mask, other=0).to(tl.int32)
        index01 = tl.load(i01 + lookup, mask=mask, other=0).to(tl.int32)
        index10 = tl.load(i10 + lookup, mask=mask, other=0).to(tl.int32)
        index11 = tl.load(i11 + lookup, mask=mask, other=0).to(tl.int32)
        angle0 = tl.load(a0 + lookup, mask=mask, other=0.0).to(tl.float32)
        angle1 = tl.load(a1 + lookup, mask=mask, other=0.0).to(tl.float32)
        radius = tl.load(radial + sample, mask=mask, other=0.0).to(tl.float32)
        origin = (base * channels + channel) * stored
        p00 = tl.load(prototype + origin + index00, mask=mask, other=0.0).to(tl.float32)
        p01 = tl.load(prototype + origin + index01, mask=mask, other=0.0).to(tl.float32)
        p10 = tl.load(prototype + origin + index10, mask=mask, other=0.0).to(tl.float32)
        p11 = tl.load(prototype + origin + index11, mask=mask, other=0.0).to(tl.float32)
        low = p00 * (1.0 - angle0) + p01 * angle0
        high = p10 * (1.0 - angle1) + p11 * angle1
        tl.store(output + offset, low * (1.0 - radius) + high * radius, mask=mask)


    @triton.jit
    def _polar_backward_gather(
        grad_output, reverse_lookup, reverse_weight, grad_prototype,
        elements: tl.constexpr, channels: tl.constexpr,
        directions: tl.constexpr, samples: tl.constexpr,
        stored: tl.constexpr, contributions: tl.constexpr,
        block_contributions: tl.constexpr, block: tl.constexpr,
    ):
        offset = tl.program_id(0) * block + tl.arange(0, block)
        mask = offset < elements
        compact = offset % stored
        rest = offset // stored
        channel = rest % channels
        base = rest // channels
        contribution = tl.arange(0, block_contributions)
        contribution_mask = contribution < contributions
        reverse_offset = compact[:, None] * contributions + contribution[None]
        lookup = tl.load(
            reverse_lookup + reverse_offset,
            mask=mask[:, None] & contribution_mask[None], other=-1,
        ).to(tl.int32)
        weight = tl.load(
            reverse_weight + reverse_offset,
            mask=mask[:, None] & contribution_mask[None], other=0.0,
        ).to(tl.float32)
        direction = lookup // samples
        sample = lookup % samples
        grad_offset = (
            ((base[:, None] * directions + direction) * channels
             + channel[:, None]) * samples + sample
        )
        valid = mask[:, None] & contribution_mask[None] & (lookup >= 0)
        grad = tl.load(grad_output + grad_offset, mask=valid, other=0.0).to(tl.float32)
        tl.store(
            grad_prototype + offset,
            tl.sum(grad * weight, axis=1),
            mask=mask,
        )


class _TritonPolarRender(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, prototype, i00, i01, i10, i11, a0, a1, radial,
        reverse_lookup, reverse_weight,
    ):
        bases, channels, stored = prototype.shape
        directions, samples = i00.shape
        output = torch.empty(
            bases, directions, channels, samples,
            device=prototype.device,
            dtype=prototype.dtype,
        )
        elements = output.numel()
        block = 256
        _polar_forward[(triton.cdiv(elements, block),)](
            prototype, i00, i01, i10, i11, a0, a1, radial, output,
            elements=elements, channels=channels, directions=directions,
            samples=samples, stored=stored, block=block,
        )
        ctx.save_for_backward(reverse_lookup, reverse_weight)
        ctx.prototype_shape = prototype.shape
        ctx.output_shape = output.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        reverse_lookup, reverse_weight = ctx.saved_tensors
        bases, channels, stored = ctx.prototype_shape
        _, directions, _, samples = ctx.output_shape
        grad_prototype = torch.zeros(
            ctx.prototype_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        elements = grad_prototype.numel()
        block = 256
        contributions = reverse_lookup.shape[1]
        _polar_backward_gather[(triton.cdiv(elements, block),)](
            grad_output.contiguous(), reverse_lookup, reverse_weight,
            grad_prototype, elements=elements, channels=channels,
            directions=directions, samples=samples, stored=stored,
            contributions=contributions,
            block_contributions=triton.next_power_of_2(contributions), block=block,
        )
        return (
            grad_prototype, None, None, None, None, None, None, None,
            None, None,
        )


def triton_polar_render(prototype: torch.Tensor, renderer) -> torch.Tensor:
    """Render with Triton on CUDA and the exact PyTorch reference on CPU."""

    if triton is None or not prototype.is_cuda:
        return renderer(prototype)
    return _TritonPolarRender.apply(
        prototype,
        renderer.index_r0_a0,
        renderer.index_r0_a1,
        renderer.index_r1_a0,
        renderer.index_r1_a1,
        renderer.angle_fraction_r0,
        renderer.angle_fraction_r1,
        renderer.radial_fraction,
        renderer.reverse_lookup,
        renderer.reverse_weight,
    )
