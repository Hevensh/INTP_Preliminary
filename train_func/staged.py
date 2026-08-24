from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _forward_classifier(model: nn.Module, images: torch.Tensor, targets: torch.Tensor, *, use_labels: bool) -> torch.Tensor:
    if not use_labels:
        return model(images)
    try:
        return model(images, labels=targets)
    except TypeError:
        return model(images)

def _iter_loader(loader: DataLoader, *, desc: str, progress: bool):
    if not progress or tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_labels: bool = False,
    progress: bool = True,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, targets in _iter_loader(loader, desc="train", progress=progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = _forward_classifier(model, images, targets, use_labels=use_labels)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_labels: bool = False,
    progress: bool = True,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, targets in _iter_loader(loader, desc="eval", progress=progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = _forward_classifier(model, images, targets, use_labels=use_labels)
        loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def fit(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    use_labels: bool = False,
    print_prefix: str = "",
    progress: bool = True,
) -> Tuple[float, List[Tuple[float, float, float, float]]]:
    """
    Minimal training loop with small stdout footprint.

    Returns:
      best_test_acc, history[(train_loss, train_acc, test_loss, test_acc), ...]
    """
    history: List[Tuple[float, float, float, float]] = []
    best_test_acc = 0.0

    for epoch in range(1, int(epochs) + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_labels=use_labels, progress=progress
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, use_labels=use_labels, progress=progress
        )
        history.append((train_loss, train_acc, test_loss, test_acc))
        best_test_acc = max(best_test_acc, test_acc)

        prefix = f"{print_prefix} " if print_prefix else ""
        print(
            f"{prefix}Epoch {epoch:02d}/{epochs} | "
            f"train acc {train_acc:.4f} loss {train_loss:.4f} | "
            f"test acc {test_acc:.4f} loss {test_loss:.4f}"
        )

    return best_test_acc, history
