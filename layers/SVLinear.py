from __future__ import annotations

import torch
from torch import Tensor, einsum, empty, full, stack, zeros_like
from torch.autograd import Function
from torch.nn import Buffer, Module, Parameter, init


class SVLinear(Module):
    """
    Support-vector linear layer (distance-based similarity).

    Similarity matches the current HexConv2D implementation style:
        y_j = exp2(-(||x - W_j||^2) * (exp2(Wr_j) * multi))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.W = Parameter(empty(out_features, in_features))
        self.Wr = Parameter(empty(out_features))
        self.reset_filters_()

        # Keep the same normalization style as HexConv.
        n_in = float(in_features)
        # self.multi = 1.0 / ((N_in / 6) - 0.5 * (N_in * 7 /180) ** (0.5)) # uniform
        self.multi = 1.0 / ((2 * n_in) - 0.5 * (8 * n_in) ** (0.5)) # normal

        if scale_init != 1.0:
            self.Wr.data.mul_(float(scale_init))

        class SVLinearFunc(Function):
            @staticmethod
            def forward(ctx, x: Tensor, W: Tensor, Wr: Tensor) -> Tensor:
                if x.size(-1) != self.in_features:
                    raise ValueError(
                        f"Input feature {x.size(-1)} does not match expected in_features {self.in_features}"
                    )

                dist2_list: list[Tensor] = []
                for Wi in W.unbind():
                    dist2_ = (x - Wi).pow(2)
                    dist2_list.append(dist2_.sum(dim=-1))
                dist2 = stack(dist2_list, -1)

                similarity = (-dist2 * (Wr.exp2() * self.multi)).exp2()

                ctx.save_for_backward(x, dist2, similarity)
                return similarity

            @staticmethod
            def backward(ctx, dLdY: Tensor):
                x, dist2, similarity = ctx.saved_tensors

                dLdX = zeros_like(x)
                dLdW = []
                dLdWr = []
                for Wi, d2i, Yi, Gi in zip(self.W, dist2.unbind(-1), similarity.unbind(-1), dLdY.unbind(-1)):
                    prefer_i = Yi * Gi

                    direction_ = x - Wi  # (..., C)
                    d2i = direction_.pow(2).sum(dim=-1)  # (...)
                    coef_ = prefer_i / (d2i + 0.1)  # (...)

                    dLdX -= coef_.unsqueeze(-1) * direction_
                    dLdW.append(einsum("...,...c->c", coef_, direction_))
                    dLdWr.append(-(prefer_i * d2i).mean())

                dLdW = stack(dLdW)
                dLdWr = stack(dLdWr)

                return dLdX, dLdW, dLdWr

        self.calc_func = SVLinearFunc

    @torch.no_grad()
    def reset_filters_(self, idx: Tensor | None = None) -> None:
        """
        Reset (re-initialize) output filters in-place.

        idx:
          - None: reset all filters
          - LongTensor indices shaped (k,) or scalar
          - BoolTensor mask shaped (out_features,)
        Resets:
          - self.W[idx] ~ Normal(0, 1)
          - self.Wr[idx] = 0
        """
        if idx is None:
            init.normal_(self.W, mean=0.0, std=1.0)
            init.zeros_(self.Wr)
            return

        if not torch.is_tensor(idx):
            raise TypeError("idx must be a torch.Tensor")

        if idx.dtype == torch.bool:
            if idx.ndim != 1 or idx.numel() != self.out_features:
                raise ValueError(f"bool mask idx must have shape ({self.out_features},)")
            idx_ = idx.nonzero(as_tuple=False).flatten()
        else:
            idx_ = idx.to(dtype=torch.long).flatten()

        if idx_.numel() == 0:
            return

        idx_ = idx_.to(device=self.W.device)
        if int(idx_.min().item()) < 0 or int(idx_.max().item()) >= self.out_features:
            raise ValueError(f"idx out of range [0, {self.out_features - 1}]")

        init.normal_(self.W[idx_], mean=0.0, std=1.0)
        init.zeros_(self.Wr[idx_])

    def forward(self, x: Tensor) -> Tensor:
        return self.calc_func.apply(x, self.W, self.Wr)

    @property
    def out_channels(self) -> int:
        # For compatibility with PerLabelActivationCounter (which expects `out_channels`).
        return self.out_features

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, multi={self.multi:.6g}"
        )
