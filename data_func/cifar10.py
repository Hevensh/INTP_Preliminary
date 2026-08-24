from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cifar10_transform_div255() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float32) / 255.0),
        ]
    )


def cifar10_train_transform_div255(
    *,
    translate_px: int = 4,
    rotate_deg: float = 30.0,
    brightness: float = 0.2,
) -> transforms.Compose:
    translate_frac = translate_px / 32.0
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=rotate_deg,
                translate=(translate_frac, translate_frac),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=brightness),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float32) / 255.0),
        ]
    )


def cifar10_imagenet224_transform(*, train: bool) -> transforms.Compose:
    """Resize CIFAR-10 for an ImageNet-pretrained 224px ViT."""
    operations: list[object] = []
    if train:
        operations.extend(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
            ]
        )
    operations.extend(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def build_cifar10_loaders(
    root: str,
    batch_size: int,
    num_workers: int = 0,
    device: torch.device | None = None,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_transform = cifar10_train_transform_div255()
    test_transform = cifar10_transform_div255()

    train_set = datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)

    pin_memory = bool(device is not None and getattr(device, "type", None) == "cuda")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_set.classes
