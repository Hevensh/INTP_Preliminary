from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import torch

from layers.hex_differentiated_harmonic_patch_embed import PrototypeConversion


def _clone_state_value(value: Any, device: torch.device) -> Any:
    return value.detach().clone().to(device) if torch.is_tensor(value) else deepcopy(value)


def rebuild_adamw_and_scheduler(
    *,
    model: torch.nn.Module,
    old_optimizer: torch.optim.AdamW,
    old_scheduler: torch.optim.lr_scheduler.LRScheduler,
    conversions: list[PrototypeConversion],
    learning_rate: float,
    weight_decay: float,
    scheduler_factory: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
) -> tuple[
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LRScheduler,
    dict[str, Any],
]:
    """Rebind AdamW after prototype shapes change and transport its moments."""
    conversion_by_parameter = {
        id(conversion.new_parameter): conversion for conversion in conversions
    }
    old_groups = [dict(group) for group in old_optimizer.param_groups]
    new_optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    installed = transformed = missing = 0
    installed_numel = total_numel = 0

    for _name, parameter in model.named_parameters():
        total_numel += parameter.numel()
        conversion = conversion_by_parameter.get(id(parameter))
        source_parameter = (
            conversion.old_parameter if conversion is not None else parameter
        )
        source_state = old_optimizer.state.get(source_parameter)
        if source_state is None:
            missing += 1
            continue
        if conversion is None:
            tensors = [
                value
                for key, value in source_state.items()
                if key != "step" and torch.is_tensor(value) and value.ndim > 0
            ]
            if any(value.shape != parameter.shape for value in tensors):
                missing += 1
                continue
            new_optimizer.state[parameter] = {
                key: _clone_state_value(value, parameter.device)
                for key, value in source_state.items()
            }
            installed += 1
            installed_numel += parameter.numel()
            continue

        transform = conversion.transform.to(device=parameter.device)
        projected_state: dict[str, Any] = {}
        for key, value in source_state.items():
            if key == "step" or not torch.is_tensor(value) or value.ndim == 0:
                projected_state[key] = _clone_state_value(value, parameter.device)
                continue
            source = value.to(parameter.device)
            projection = transform.square() if key in {"exp_avg_sq", "max_exp_avg_sq"} else transform
            projected = torch.einsum("cm,fm->cf", source, projection).reshape(parameter.shape)
            if key in {"exp_avg_sq", "max_exp_avg_sq"}:
                projected = projected.clamp_min_(0)
            projected_state[key] = projected
        new_optimizer.state[parameter] = projected_state
        transformed += 1
        installed_numel += parameter.numel()

    new_scheduler = scheduler_factory(new_optimizer)
    new_scheduler.load_state_dict(old_scheduler.state_dict())
    for index, group in enumerate(new_optimizer.param_groups):
        source = old_groups[min(index, len(old_groups) - 1)]
        group["lr"] = source["lr"]
        if "initial_lr" in source:
            group["initial_lr"] = source["initial_lr"]
    audit = {
        "installed_parameter_states": installed,
        "transformed_parameter_states": transformed,
        "parameters_without_state": missing,
        "state_numel_coverage": installed_numel / max(total_numel, 1),
    }
    return new_optimizer, new_scheduler, audit
