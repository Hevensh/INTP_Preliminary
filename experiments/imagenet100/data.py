from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_NAMES = {"train", "training"}
VAL_NAMES = {"val", "valid", "validation"}


@dataclass(frozen=True)
class ImageFolderSplits:
    train: tuple[Path, ...]
    val: Path
    classes: tuple[str, ...]


class MultiRootImageFolder(Dataset):
    """Combine sharded ImageFolder roots under one global class mapping."""

    def __init__(
        self,
        roots: tuple[Path, ...],
        *,
        classes: tuple[str, ...],
        transform: transforms.Compose,
    ) -> None:
        self.roots = roots
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.transform = transform
        self.loader = ImageFolder(roots[0]).loader
        samples: list[tuple[str, int]] = []
        for root in roots:
            local = ImageFolder(root)
            for path, local_index in local.samples:
                class_name = local.classes[local_index]
                samples.append((path, self.class_to_idx[class_name]))
        self.samples = samples
        self.targets = [target for _, target in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        return self.transform(image), target


def _directories_to_depth(root: Path, max_depth: int = 5) -> list[Path]:
    result = [root]
    frontier = [(root, 0)]
    while frontier:
        parent, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        lowered = parent.name.lower()
        if lowered in TRAIN_NAMES | VAL_NAMES or lowered.startswith(("train.", "val.")):
            # Split roots can contain hundreds of thousands of images. Once a
            # split root is found, class discovery handles its immediate
            # children; recursively enumerating image files is unnecessary.
            continue
        try:
            children = [path for path in parent.iterdir() if path.is_dir()]
        except OSError:
            continue
        result.extend(children)
        frontier.extend((child, depth + 1) for child in children)
    return result


def _class_names(split: Path) -> tuple[str, ...]:
    names: list[str] = []
    for candidate in split.iterdir():
        if not candidate.is_dir():
            continue
        try:
            has_image = any(
                path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                for path in candidate.iterdir()
            )
        except OSError:
            has_image = False
        if has_image:
            names.append(candidate.name)
    return tuple(sorted(names))


def discover_imagefolder_splits(
    data_root: str | Path,
    *,
    expected_classes: int = 100,
) -> ImageFolderSplits:
    """Find matching ImageFolder train/validation trees below a Kaggle input.

    Kaggle dataset revisions often add an extra top-level directory. This keeps
    the notebook command stable while still rejecting the 64 px TinyImageNet
    component when the requested dataset is ImageNet-100.
    """

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")

    directories = _directories_to_depth(root)
    train_dirs = [
        path
        for path in directories
        if path.name.lower() in TRAIN_NAMES or path.name.lower().startswith("train.")
    ]
    val_dirs = [
        path
        for path in directories
        if path.name.lower() in VAL_NAMES or path.name.lower().startswith("val.")
    ]
    candidates: list[tuple[int, ImageFolderSplits]] = []
    inspected: list[str] = []
    train_groups: dict[Path, list[Path]] = {}
    for train in train_dirs:
        train_groups.setdefault(train.parent, []).append(train)
    for train_parent, trains in train_groups.items():
        trains = sorted(trains)
        class_union = tuple(sorted({name for train in trains for name in _class_names(train)}))
        if class_union:
            inspected.append(
                f"{', '.join(str(path) for path in trains)} ({len(class_union)} unique classes)"
            )
        for val in val_dirs:
            # Prefer sibling splits, but permit one wrapper directory mismatch.
            if train_parent != val.parent and train_parent.parent != val.parent.parent:
                continue
            val_classes = _class_names(val)
            if class_union != val_classes:
                continue
            split = ImageFolderSplits(train=tuple(trains), val=val, classes=class_union)
            class_score = 10_000 if len(class_union) == expected_classes else 0
            sibling_score = 100 if train_parent == val.parent else 0
            candidates.append((class_score + sibling_score + len(class_union), split))

    if not candidates:
        details = "\n  ".join(inspected[:20]) or "no ImageFolder-style train directory found"
        raise FileNotFoundError(
            f"Could not find matching train/val ImageFolder splits below {root}.\n"
            f"Inspected:\n  {details}\n"
            "Expected train/<class>/*.jpg and val/<class>/*.jpg."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]
    if len(selected.classes) != expected_classes:
        raise ValueError(
            f"best dataset candidate has {len(selected.classes)} classes, expected "
            f"{expected_classes}: {selected.train}"
        )
    return selected


def imagenet100_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, value="random"),
        ]
    )
    val = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, val


def _subset(dataset: Dataset, count: int | None) -> Dataset:
    if count is None or count >= len(dataset):
        return dataset
    if count <= 0:
        raise ValueError("sample limit must be positive")
    return Subset(dataset, range(count))


def build_imagefolder_loaders(
    splits: ImageFolderSplits,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_samples: int | None = None,
    val_samples: int | None = None,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> tuple[DataLoader, DataLoader, int, int]:
    train_transform, val_transform = imagenet100_transforms(image_size)
    train_dataset = MultiRootImageFolder(
        splits.train,
        classes=splits.classes,
        transform=train_transform,
    )
    val_dataset = ImageFolder(splits.val, transform=val_transform)
    if tuple(train_dataset.classes) != splits.classes or tuple(val_dataset.classes) != splits.classes:
        raise RuntimeError("class ordering changed while constructing ImageFolder datasets")
    full_train_size, full_val_size = len(train_dataset), len(val_dataset)
    train_dataset = _subset(train_dataset, train_samples)
    val_dataset = _subset(val_dataset, val_samples)
    if distributed_world_size <= 0:
        raise ValueError("distributed_world_size must be positive")
    if not 0 <= distributed_rank < distributed_world_size:
        raise ValueError("distributed_rank must be in [0, distributed_world_size)")
    train_sampler = val_sampler = None
    if distributed_world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=False,
            drop_last=False,
        )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            drop_last=True,
            **common,
        ),
        DataLoader(
            val_dataset,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        full_train_size,
        full_val_size,
    )
