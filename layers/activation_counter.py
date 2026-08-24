from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import Parameter


class PerLabelActivationCounter(nn.Module):
    """
    Wrapper that records activation amount per label (sum of similarity).

    Expects:
    - wrapped module returns a Tensor shaped (B, N, out_channels)
    - `labels` is a LongTensor shaped (B,) with values in [0, num_class-1]
    Counters are updated only during training by default.
    """

    def __init__(
        self,
        module: nn.Module,
        num_class: int,
        record_active: bool = True,
        record_grad: bool = False,
        record_prefer: bool = False,
    ):
        super().__init__()
        if num_class <= 0:
            raise ValueError("num_class must be a positive integer")

        self.module = module
        self.num_class = int(num_class)
        self.record_active = bool(record_active)
        self.record_grad = bool(record_grad)
        self.record_prefer = bool(record_prefer)

        out_channels = getattr(module, "out_channels", None)
        if out_channels is None:
            raise ValueError("wrapped module must have attribute `out_channels`")

        self.num_active_ = Parameter(
            torch.zeros(self.num_class, int(out_channels), dtype=torch.float32),
            requires_grad=False,
        )
        self.num_grad_ = Parameter(
            torch.zeros(self.num_class, int(out_channels), dtype=torch.float32),
            requires_grad=False,
        )
        self.num_prefer_ = Parameter(
            torch.zeros(self.num_class, int(out_channels), dtype=torch.float32),
            requires_grad=False,
        )
        self.total_possible_active_ = Parameter(torch.zeros(self.num_class, dtype=torch.long), requires_grad=False)
        self.total_possible_grad_ = Parameter(torch.zeros(self.num_class, dtype=torch.long), requires_grad=False)
        self.total_possible_prefer_ = Parameter(torch.zeros(self.num_class, dtype=torch.long), requires_grad=False)

        class _PerLabelCounterFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, sim: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(sim, lbl)
                ctx.num_token = self._num_token(sim)
                if self.record_active:
                    with torch.no_grad():
                        self._accum_(
                            lbl,
                            self._sum_per_sample(sim),
                            total_possible=self.total_possible_active_,
                            counter=self.num_active_,
                            num_token=ctx.num_token,
                        )
                return sim

            @staticmethod
            def backward(ctx, dLdY: torch.Tensor):
                sim, lbl = ctx.saved_tensors
                num_token = int(getattr(ctx, "num_token", 1))
                with torch.no_grad():
                    if self.record_grad:
                        self._accum_(
                            lbl,
                            self._sum_per_sample(dLdY.abs()),
                            total_possible=self.total_possible_grad_,
                            counter=self.num_grad_,
                            num_token=num_token,
                        )
                    if self.record_prefer:
                        self._accum_(
                            lbl,
                            self._sum_per_sample(sim.mul(dLdY)),
                            total_possible=self.total_possible_prefer_,
                            counter=self.num_prefer_,
                            num_token=num_token,
                        )
                return dLdY, None

        self._counter_func = _PerLabelCounterFunc

    def reset_num_active_(self) -> None:
        self.num_active_.zero_()
        self.total_possible_active_.zero_()

    def reset_num_grad_(self) -> None:
        self.num_grad_.zero_()
        self.total_possible_grad_.zero_()

    def reset_num_prefer_(self) -> None:
        self.num_prefer_.zero_()
        self.total_possible_prefer_.zero_()

    def reset_grad_active_(self) -> None:
        self.reset_num_grad_()

    @staticmethod
    def _sum_per_sample(x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.ndim - 1))
        return x.sum(dim=dims) if dims else x

    @staticmethod
    def _num_token(x: torch.Tensor) -> int:
        # x is interpreted as (B, ..., C); token count per sample excludes batch and the last dim C.
        if x.shape[0] == 0:
            return 0
        return int(x.numel() // (int(x.shape[0]) * int(x.shape[-1])))

    def _accum_(
        self,
        labels: torch.Tensor,
        num_per_sample: torch.Tensor,
        total_possible: Parameter,
        counter: Parameter,
        num_token: int,
    ) -> None:
        counter.index_add_(0, labels, num_per_sample.to(counter.dtype))
        counts = torch.bincount(labels, minlength=int(total_possible.numel())).to(device=total_possible.device)
        total_possible.add_(counts.to(dtype=total_possible.dtype) * int(num_token))

    def forward(self, img: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        similarity = self.module(img)
        if labels is None:
            return similarity
        if not (self.record_active or self.record_grad or self.record_prefer):
            return similarity

        return self._counter_func.apply(similarity, labels)
