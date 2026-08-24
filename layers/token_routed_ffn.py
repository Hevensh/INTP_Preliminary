from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TokenRouteIndices:
    """Disjoint parameter ownership for shared, CLS-only, and patch-only neurons."""

    shared: tuple[int, ...]
    cls_only: tuple[int, ...]
    patch_only: tuple[int, ...]

    @property
    def hidden_features(self) -> int:
        return len(self.shared) + len(self.cls_only) + len(self.patch_only)

    @property
    def cls_features(self) -> int:
        return len(self.shared) + len(self.cls_only)

    @property
    def patch_features(self) -> int:
        return len(self.shared) + len(self.patch_only)

    def validate(self, hidden_features: int | None = None) -> None:
        groups = self.shared, self.cls_only, self.patch_only
        flat = tuple(index for group in groups for index in group)
        expected = self.hidden_features if hidden_features is None else hidden_features
        if len(flat) != expected:
            raise ValueError(f"routes contain {len(flat)} indices, expected {expected}")
        if len(set(flat)) != len(flat):
            raise ValueError("route index groups must be disjoint")
        if sorted(flat) != list(range(expected)):
            raise ValueError(f"route indices must cover [0, {expected - 1}]")

    def route_codes(self) -> torch.Tensor:
        """Return 0=shared, 1=CLS-only, and 2=patch-only codes in source order."""
        self.validate()
        codes = torch.empty(self.hidden_features, dtype=torch.uint8)
        codes[list(self.shared)] = 0
        codes[list(self.cls_only)] = 1
        codes[list(self.patch_only)] = 2
        return codes


def select_token_route_indices(
    cls_energy: torch.Tensor,
    patch_energy: torch.Tensor,
    *,
    shared_features: int,
    cls_only_features: int,
) -> TokenRouteIndices:
    """Choose per-layer routes from non-negative contribution energies.

    Shared neurons are selected by the contribution retained for the weaker token
    type, ``min(E_cls, E_patch)``. Among the remaining neurons, the requested
    number of CLS-only slots goes to the largest ``E_cls - E_patch`` values and
    the rest become patch-only.
    """
    cls = torch.as_tensor(cls_energy, dtype=torch.float64).flatten().cpu()
    patch = torch.as_tensor(patch_energy, dtype=torch.float64).flatten().cpu()
    if cls.shape != patch.shape or cls.ndim != 1:
        raise ValueError("CLS and patch energies must be matching one-dimensional tensors")
    if not torch.isfinite(cls).all() or not torch.isfinite(patch).all():
        raise ValueError("contribution energies must be finite")
    if (cls < 0).any() or (patch < 0).any():
        raise ValueError("contribution energies must be non-negative")

    hidden = int(cls.numel())
    if shared_features < 0 or cls_only_features < 0:
        raise ValueError("route sizes must be non-negative")
    if shared_features + cls_only_features > hidden:
        raise ValueError("shared and CLS-only sizes exceed the hidden width")

    shared_order = torch.argsort(
        torch.minimum(cls, patch), descending=True, stable=True
    )
    shared = shared_order[:shared_features]
    remaining_mask = torch.ones(hidden, dtype=torch.bool)
    remaining_mask[shared] = False
    remaining = torch.arange(hidden)[remaining_mask]
    cls_order = torch.argsort(
        cls[remaining] - patch[remaining], descending=True, stable=True
    )
    cls_only = remaining[cls_order[:cls_only_features]]
    patch_only = remaining[cls_order[cls_only_features:]]

    routes = TokenRouteIndices(
        shared=tuple(sorted(shared.tolist())),
        cls_only=tuple(sorted(cls_only.tolist())),
        patch_only=tuple(sorted(patch_only.tolist())),
    )
    routes.validate(hidden)
    return routes


def select_token_routes_by_retained_energy(
    cls_energy: torch.Tensor,
    patch_energy: torch.Tensor,
    *,
    min_cls_retained: float,
    min_patch_retained: float,
) -> TokenRouteIndices:
    """Choose variable route widths while bounding discarded contribution energy.

    Starting from an all-shared FFN, neurons with the smallest patch energy move
    to the CLS-only route until the patch loss budget is exhausted. From the
    remainder, neurons with the smallest CLS energy move to the patch-only route.
    This directly minimizes routed token work because skipping a neuron for all
    patch tokens is the dominant saving in ViT.
    """
    if not 0.0 <= min_cls_retained <= 1.0 or not 0.0 <= min_patch_retained <= 1.0:
        raise ValueError("retained-energy targets must be in [0, 1]")
    cls = torch.as_tensor(cls_energy, dtype=torch.float64).flatten().cpu()
    patch = torch.as_tensor(patch_energy, dtype=torch.float64).flatten().cpu()
    if cls.shape != patch.shape or cls.ndim != 1:
        raise ValueError("CLS and patch energies must be matching one-dimensional tensors")
    if not torch.isfinite(cls).all() or not torch.isfinite(patch).all():
        raise ValueError("contribution energies must be finite")
    if (cls < 0).any() or (patch < 0).any():
        raise ValueError("contribution energies must be non-negative")

    hidden = int(cls.numel())

    def within_budget(values: torch.Tensor, budget: float) -> torch.Tensor:
        order = torch.argsort(values, descending=False, stable=True)
        cumulative = torch.cumsum(values[order], dim=0)
        count = int((cumulative <= budget + 1e-15).sum())
        return order[:count]

    patch_loss_budget = float(patch.sum()) * (1.0 - min_patch_retained)
    cls_only = within_budget(patch, patch_loss_budget)
    remaining_mask = torch.ones(hidden, dtype=torch.bool)
    remaining_mask[cls_only] = False
    remaining = torch.arange(hidden)[remaining_mask]
    cls_loss_budget = float(cls.sum()) * (1.0 - min_cls_retained)
    patch_choice = within_budget(cls[remaining], cls_loss_budget)
    patch_only = remaining[patch_choice]
    shared_mask = remaining_mask.clone()
    shared_mask[patch_only] = False
    shared = torch.arange(hidden)[shared_mask]

    routes = TokenRouteIndices(
        shared=tuple(sorted(shared.tolist())),
        cls_only=tuple(sorted(cls_only.tolist())),
        patch_only=tuple(sorted(patch_only.tolist())),
    )
    routes.validate(hidden)
    return routes


def select_progressive_route_update(
    cls_energy: torch.Tensor,
    patch_energy: torch.Tensor,
    cls_positive_distinction: torch.Tensor,
    patch_positive_distinction: torch.Tensor,
    route_code: torch.Tensor,
    *,
    split_fraction: float,
    protect_top_fraction: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a fraction of shared neurons using energy and distinction ranks.

    Low common utility makes a shared neuron eligible for specialization. A
    neuron in the protected top positive-distinction fraction receives maximal
    utility for that token type, preventing the update from excluding it there.
    The neuron remains on whichever token type has larger protected utility.
    """
    if not 0.0 < split_fraction <= 1.0:
        raise ValueError("split_fraction must be in (0, 1]")
    if not 0.0 <= protect_top_fraction < 1.0:
        raise ValueError("protect_top_fraction must be in [0, 1)")
    tensors = [
        torch.as_tensor(value, dtype=torch.float64).flatten().cpu()
        for value in (
            cls_energy,
            patch_energy,
            cls_positive_distinction,
            patch_positive_distinction,
        )
    ]
    codes = torch.as_tensor(route_code, dtype=torch.uint8).flatten().cpu()
    hidden = int(codes.numel())
    if any(value.numel() != hidden for value in tensors):
        raise ValueError("all progressive statistics must match route_code")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("progressive route statistics must be finite")
    if any((value < 0).any() for value in tensors):
        raise ValueError("progressive route statistics must be non-negative")
    if not torch.isin(codes, torch.tensor([0, 1, 2], dtype=torch.uint8)).all():
        raise ValueError("route_code must use 0=shared, 1=CLS-only, 2=patch-only")
    shared = torch.nonzero(codes == 0, as_tuple=False).flatten()
    if shared.numel() == 0:
        return (), ()

    def percentile_rank(values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values, stable=True)
        ranks = torch.empty_like(values)
        if values.numel() == 1:
            ranks.zero_()
        else:
            ranks[order] = torch.linspace(
                0.0, 1.0, values.numel(), dtype=values.dtype
            )
        return ranks

    cls_energy_rank = percentile_rank(tensors[0][shared])
    patch_energy_rank = percentile_rank(tensors[1][shared])
    cls_distinction_rank = percentile_rank(tensors[2][shared])
    patch_distinction_rank = percentile_rank(tensors[3][shared])
    protection_start = 1.0 - protect_top_fraction
    cls_utility = cls_energy_rank.clone()
    patch_utility = patch_energy_rank.clone()
    if protect_top_fraction > 0.0:
        cls_utility[cls_distinction_rank >= protection_start] = 1.0
        patch_utility[patch_distinction_rank >= protection_start] = 1.0
    common_utility = torch.minimum(cls_utility, patch_utility)
    split_count = min(
        int(shared.numel()), max(1, int(round(float(shared.numel()) * split_fraction)))
    )
    selected_local = torch.argsort(common_utility, stable=True)[:split_count]
    selected = shared[selected_local]
    keep_cls = cls_utility[selected_local] >= patch_utility[selected_local]
    cls_only = selected[keep_cls]
    patch_only = selected[~keep_cls]
    return tuple(sorted(cls_only.tolist())), tuple(sorted(patch_only.tolist()))


class TokenRoutedMlp(nn.Module):
    """ViT MLP with shared and token-type-specific hidden neuron subsets.

    The union of the three branches is exactly the original hidden width. CLS
    tokens use ``shared + cls_only`` and spatial tokens use
    ``shared + patch_only``. The second projection bias stays shared, keeping the
    trainable parameter count identical to the source MLP.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        routes: TokenRouteIndices,
        first_bias: bool,
        second_bias: bool,
        act: nn.Module,
        drop1: nn.Module,
        drop2: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        routes.validate()
        self.routes = routes
        self.in_features = in_features
        self.out_features = out_features
        self.cls_only_features = len(routes.cls_only)
        self.shared_features = len(routes.shared)
        self.cls_end = self.cls_only_features + self.shared_features
        self.patch_start = self.cls_only_features
        hidden_features = routes.hidden_features
        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
            bias=first_bias,
            device=device,
            dtype=dtype,
        )
        self.fc2 = nn.Linear(
            hidden_features,
            out_features,
            bias=second_bias,
            device=device,
            dtype=dtype,
        )
        self.act = act
        self.drop1 = drop1
        self.drop2 = drop2

    @classmethod
    def from_mlp(cls, mlp: nn.Module, routes: TokenRouteIndices) -> "TokenRoutedMlp":
        if not isinstance(getattr(mlp, "fc1", None), nn.Linear) or not isinstance(
            getattr(mlp, "fc2", None), nn.Linear
        ):
            raise TypeError("source MLP must expose linear fc1 and fc2 modules")
        if not isinstance(getattr(mlp, "norm", nn.Identity()), nn.Identity):
            raise TypeError("hidden normalization must be Identity for structured routing")
        hidden = int(mlp.fc1.out_features)
        if mlp.fc2.in_features != hidden:
            raise ValueError("source MLP projections disagree on hidden width")
        routes.validate(hidden)
        result = cls(
            in_features=int(mlp.fc1.in_features),
            out_features=int(mlp.fc2.out_features),
            routes=routes,
            first_bias=mlp.fc1.bias is not None,
            second_bias=mlp.fc2.bias is not None,
            act=copy.deepcopy(mlp.act),
            drop1=copy.deepcopy(mlp.drop1),
            drop2=copy.deepcopy(mlp.drop2),
            device=mlp.fc1.weight.device,
            dtype=mlp.fc1.weight.dtype,
        )
        with torch.no_grad():
            # [CLS-only | shared | patch-only] makes both token routes contiguous:
            # CLS uses [: cls_end], patches use [patch_start :].
            source_order = torch.tensor(
                routes.cls_only + routes.shared + routes.patch_only,
                device=mlp.fc1.weight.device,
            )
            result.fc1.weight.copy_(mlp.fc1.weight.index_select(0, source_order))
            if result.fc1.bias is not None:
                result.fc1.bias.copy_(mlp.fc1.bias.index_select(0, source_order))
            result.fc2.weight.copy_(mlp.fc2.weight.index_select(1, source_order))
            if result.fc2.bias is not None:
                result.fc2.bias.copy_(mlp.fc2.bias)
        result.train(mlp.training)
        return result

    def _route(
        self, x: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        first_bias = self.fc1.bias[start:end] if self.fc1.bias is not None else None
        hidden = torch.nn.functional.linear(x, self.fc1.weight[start:end], first_bias)
        hidden = self.drop1(self.act(hidden))
        return torch.nn.functional.linear(
            hidden,
            self.fc2.weight[:, start:end],
            self.fc2.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected [..., tokens, {self.in_features}], got {tuple(x.shape)}"
            )
        if x.shape[-2] < 1:
            raise ValueError("token dimension must contain a CLS token")

        cls_output = self._route(x[..., :1, :], 0, self.cls_end)
        if x.shape[-2] == 1:
            output = cls_output
        else:
            patch_output = self._route(
                x[..., 1:, :], self.patch_start, self.routes.hidden_features
            )
            output = torch.cat((cls_output, patch_output), dim=-2)
        return self.drop2(output)

    def theoretical_compute_ratio(self, patch_tokens: int) -> float:
        if patch_tokens < 0:
            raise ValueError("patch token count must be non-negative")
        full = (patch_tokens + 1) * self.routes.hidden_features
        routed = self.routes.cls_features + patch_tokens * self.routes.patch_features
        return routed / full


class ProgressiveTokenRoutedMlp(nn.Module):
    """Training-time routed MLP with stable neuron and optimizer-state identity.

    Unlike :class:`TokenRoutedMlp`, this module never repacks parameters while
    routes evolve. It masks post-activation hidden values in source-neuron order,
    allowing route codes to change without replacing Parameters or invalidating
    Adam moments. The final routes can be repacked after training for inference.
    """

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        if not isinstance(getattr(mlp, "fc1", None), nn.Linear) or not isinstance(
            getattr(mlp, "fc2", None), nn.Linear
        ):
            raise TypeError("source MLP must expose linear fc1 and fc2 modules")
        if not isinstance(getattr(mlp, "norm", nn.Identity()), nn.Identity):
            raise TypeError("hidden normalization must be Identity for structured routing")
        if mlp.fc1.out_features != mlp.fc2.in_features:
            raise ValueError("source MLP projections disagree on hidden width")
        self.fc1 = copy.deepcopy(mlp.fc1)
        self.fc2 = copy.deepcopy(mlp.fc2)
        self.act = copy.deepcopy(mlp.act)
        self.drop1 = copy.deepcopy(mlp.drop1)
        self.drop2 = copy.deepcopy(mlp.drop2)
        hidden = int(self.fc1.out_features)
        self.register_buffer("route_code", torch.zeros(hidden, dtype=torch.uint8))
        self.register_buffer("transition_mask", torch.zeros(hidden, dtype=torch.bool))
        self.register_buffer("transition_keep", torch.tensor(0.0, dtype=torch.float32))
        self._stat_labels: torch.Tensor | None = None
        self._stat_num_classes = 0
        self._stat_cls_squared: torch.Tensor | None = None
        self._stat_patch_squared: torch.Tensor | None = None
        self._stat_cls_count: torch.Tensor | None = None
        self._stat_patch_count: torch.Tensor | None = None
        self._stat_cls_positive: torch.Tensor | None = None
        self._stat_patch_positive: torch.Tensor | None = None
        self._stat_cls_class_count: torch.Tensor | None = None
        self._stat_patch_class_count: torch.Tensor | None = None
        self.train(mlp.training)

    @property
    def hidden_features(self) -> int:
        return int(self.route_code.numel())

    def route_indices(self) -> TokenRouteIndices:
        codes = self.route_code.detach().cpu()
        route = TokenRouteIndices(
            shared=tuple(torch.nonzero(codes == 0, as_tuple=False).flatten().tolist()),
            cls_only=tuple(torch.nonzero(codes == 1, as_tuple=False).flatten().tolist()),
            patch_only=tuple(torch.nonzero(codes == 2, as_tuple=False).flatten().tolist()),
        )
        route.validate(self.hidden_features)
        return route

    @torch.no_grad()
    def update_routes(
        self,
        *,
        cls_only: torch.Tensor | tuple[int, ...] | list[int] = (),
        patch_only: torch.Tensor | tuple[int, ...] | list[int] = (),
    ) -> None:
        cls_indices = torch.as_tensor(cls_only, dtype=torch.long, device=self.route_code.device)
        patch_indices = torch.as_tensor(
            patch_only, dtype=torch.long, device=self.route_code.device
        )
        if cls_indices.ndim != 1 or patch_indices.ndim != 1:
            raise ValueError("route update indices must be one-dimensional")
        combined = torch.cat((cls_indices, patch_indices))
        if combined.numel() == 0:
            return
        if (combined < 0).any() or (combined >= self.hidden_features).any():
            raise IndexError("route update index is outside the hidden width")
        if torch.unique(combined).numel() != combined.numel():
            raise ValueError("route update indices must be disjoint")
        if (self.route_code[combined] != 0).any():
            raise ValueError("only currently shared neurons can become token-specific")
        self.route_code[cls_indices] = 1
        self.route_code[patch_indices] = 2
        self.transition_mask[combined] = True
        self.transition_keep.fill_(1.0)

    @torch.no_grad()
    def set_transition_progress(self, progress: float) -> None:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("transition progress must be in [0, 1]")
        self.transition_keep.fill_(1.0 - progress)

    @torch.no_grad()
    def finish_transition(self) -> None:
        self.transition_mask.zero_()
        self.transition_keep.zero_()

    @torch.no_grad()
    def begin_statistics(self, num_classes: int) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        device = self.fc1.weight.device
        hidden = self.hidden_features
        self._stat_num_classes = int(num_classes)
        self._stat_cls_squared = torch.zeros(hidden, device=device, dtype=torch.float32)
        self._stat_patch_squared = torch.zeros(hidden, device=device, dtype=torch.float32)
        self._stat_cls_count = torch.zeros((), device=device, dtype=torch.float32)
        self._stat_patch_count = torch.zeros((), device=device, dtype=torch.float32)
        self._stat_cls_positive = torch.zeros(
            num_classes, hidden, device=device, dtype=torch.float32
        )
        self._stat_patch_positive = torch.zeros(
            num_classes, hidden, device=device, dtype=torch.float32
        )
        self._stat_cls_class_count = torch.zeros(
            num_classes, device=device, dtype=torch.float32
        )
        self._stat_patch_class_count = torch.zeros(
            num_classes, device=device, dtype=torch.float32
        )

    def set_stat_labels(self, labels: torch.Tensor | None) -> None:
        self._stat_labels = labels

    @torch.no_grad()
    def _record_statistics(
        self, pre_activation: torch.Tensor, post_activation: torch.Tensor
    ) -> None:
        if self._stat_cls_squared is None or self._stat_labels is None:
            return
        if pre_activation.ndim != 3:
            raise ValueError("online routed statistics require [batch, tokens, hidden]")
        labels = self._stat_labels
        if labels.ndim != 1 or labels.shape[0] != pre_activation.shape[0]:
            raise ValueError("stat labels must match the activation batch")
        if (labels < 0).any() or (labels >= self._stat_num_classes).any():
            raise ValueError("stat labels are outside the configured class range")
        post = post_activation.detach().float()
        pre = pre_activation.detach()
        patch_tokens = int(pre.shape[1] - 1)
        self._stat_cls_squared.add_(post[:, 0].square().sum(dim=0))
        self._stat_cls_count.add_(float(pre.shape[0]))
        if patch_tokens:
            self._stat_patch_squared.add_(post[:, 1:].square().sum(dim=(0, 1)))
            self._stat_patch_count.add_(float(pre.shape[0] * patch_tokens))
        one_hot = torch.nn.functional.one_hot(
            labels, num_classes=self._stat_num_classes
        ).to(torch.float32)
        cls_positive = (pre[:, 0] > 0).to(torch.float32)
        self._stat_cls_positive.add_(one_hot.transpose(0, 1) @ cls_positive)
        self._stat_cls_class_count.add_(one_hot.sum(dim=0))
        if patch_tokens:
            patch_positive = (pre[:, 1:] > 0).to(torch.float32).sum(dim=1)
            self._stat_patch_positive.add_(one_hot.transpose(0, 1) @ patch_positive)
            self._stat_patch_class_count.add_(one_hot.sum(dim=0) * patch_tokens)

    @torch.no_grad()
    def statistics(self) -> dict[str, torch.Tensor]:
        required = (
            self._stat_cls_squared,
            self._stat_patch_squared,
            self._stat_cls_count,
            self._stat_patch_count,
            self._stat_cls_positive,
            self._stat_patch_positive,
            getattr(self, "_stat_cls_class_count", None),
            getattr(self, "_stat_patch_class_count", None),
        )
        if any(value is None for value in required):
            raise RuntimeError("begin_statistics must be called before statistics")
        output_norm_squared = self.fc2.weight.detach().float().square().sum(dim=0)
        cls_energy = (
            self._stat_cls_squared / self._stat_cls_count.clamp_min(1.0)
        ) * output_norm_squared
        patch_energy = (
            self._stat_patch_squared / self._stat_patch_count.clamp_min(1.0)
        ) * output_norm_squared
        cls_rate = self._stat_cls_positive / self._stat_cls_class_count.clamp_min(1.0)[
            :, None
        ]
        patch_rate = self._stat_patch_positive / self._stat_patch_class_count.clamp_min(
            1.0
        )[:, None]
        return {
            "cls_energy": cls_energy.detach().cpu(),
            "patch_energy": patch_energy.detach().cpu(),
            "cls_positive_distinction": cls_rate.std(dim=0, unbiased=False).detach().cpu(),
            "patch_positive_distinction": patch_rate.std(
                dim=0, unbiased=False
            ).detach().cpu(),
            "cls_positive_rate": cls_rate.detach().cpu(),
            "patch_positive_rate": patch_rate.detach().cpu(),
        }

    def _token_gate(self, excluded_code: int) -> torch.Tensor:
        excluded = self.route_code == excluded_code
        gate = torch.ones(
            self.hidden_features, device=self.route_code.device, dtype=self.fc1.weight.dtype
        )
        gate[excluded] = 0.0
        transitioning = excluded & self.transition_mask
        gate[transitioning] = self.transition_keep.to(gate.dtype)
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3 or x.shape[-2] < 1:
            raise ValueError("expected an input with a non-empty token dimension")
        pre_activation = self.fc1(x)
        post_activation = self.act(pre_activation)
        self._record_statistics(pre_activation, post_activation)
        hidden = self.drop1(post_activation)
        cls_hidden = hidden[..., :1, :] * self._token_gate(2)
        if x.shape[-2] == 1:
            hidden = cls_hidden
        else:
            patch_hidden = hidden[..., 1:, :] * self._token_gate(1)
            hidden = torch.cat((cls_hidden, patch_hidden), dim=-2)
        return self.drop2(self.fc2(hidden))
