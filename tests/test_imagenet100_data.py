from pathlib import Path

from PIL import Image

from experiments.imagenet100.data import (
    build_imagefolder_loaders,
    discover_imagefolder_splits,
    index_imagefolder_samples,
    load_or_index_imagefolder_samples,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (128, 64, 32)).save(path)


def test_discovers_wrapped_imagefolder_splits(tmp_path: Path) -> None:
    for split in ("train", "val"):
        for class_name in ("n0001", "n0002"):
            _write_image(tmp_path / "wrapper" / "imagenet100" / split / class_name / "sample.jpg")
    result = discover_imagefolder_splits(tmp_path, expected_classes=2)
    assert len(result.train) == 1
    assert result.train[0].name == "train"
    assert result.val.name == "val"
    assert result.classes == ("n0001", "n0002")


def test_rejects_wrong_class_count(tmp_path: Path) -> None:
    for split in ("train", "val"):
        _write_image(tmp_path / split / "only_class" / "sample.jpg")
    try:
        discover_imagefolder_splits(tmp_path, expected_classes=2)
    except ValueError as error:
        assert "expected 2" in str(error)
    else:
        raise AssertionError("wrong class count should be rejected")


def test_discovers_four_training_shards_with_global_classes(tmp_path: Path) -> None:
    classes = tuple(f"n{index:04d}" for index in range(8))
    for shard in range(4):
        for class_name in classes[shard * 2 : shard * 2 + 2]:
            _write_image(tmp_path / f"train.X{shard + 1}" / class_name / "sample.jpg")
    for class_name in classes:
        _write_image(tmp_path / "val.X" / class_name / "sample.jpg")
    result = discover_imagefolder_splits(tmp_path, expected_classes=8)
    assert tuple(path.name for path in result.train) == (
        "train.X1",
        "train.X2",
        "train.X3",
        "train.X4",
    )
    assert result.classes == classes


def test_distributed_loaders_partition_training_indices(tmp_path: Path) -> None:
    for split in ("train", "val"):
        for class_name in ("n0001", "n0002"):
            for index in range(4):
                _write_image(tmp_path / split / class_name / f"{index}.jpg")
    splits = discover_imagefolder_splits(tmp_path, expected_classes=2)
    train_index, val_index = index_imagefolder_samples(splits)
    loaders = []
    for rank in range(2):
        train, val, _, _ = build_imagefolder_loaders(
            splits,
            image_size=8,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            distributed_rank=rank,
            distributed_world_size=2,
            train_index=train_index,
            val_index=val_index,
        )
        loaders.append((train, val))

    rank0 = set(iter(loaders[0][0].sampler))
    rank1 = set(iter(loaders[1][0].sampler))
    assert rank0.isdisjoint(rank1)
    assert rank0 | rank1 == set(range(8))
    assert len(loaders[0][1].sampler) == len(loaders[1][1].sampler) == 4


def test_image_index_cache_is_reused(tmp_path: Path, monkeypatch) -> None:
    for split in ("train", "val"):
        for class_name in ("n0001", "n0002"):
            _write_image(tmp_path / split / class_name / "sample.jpg")
    splits = discover_imagefolder_splits(tmp_path, expected_classes=2)
    cache_dir = tmp_path / "cache"

    expected_train, expected_val, hit, cache_path = load_or_index_imagefolder_samples(
        splits, cache_dir=cache_dir
    )
    assert not hit
    assert cache_path.is_file()

    def fail_if_reindexed(*args, **kwargs):
        raise AssertionError("cache hit should not walk the image folders again")

    monkeypatch.setattr("experiments.imagenet100.data._index_roots", fail_if_reindexed)
    train, val, hit, reused_path = load_or_index_imagefolder_samples(
        splits, cache_dir=cache_dir
    )
    assert hit
    assert reused_path == cache_path
    assert train == expected_train
    assert val == expected_val


def test_image_index_cache_recovers_from_invalid_json(tmp_path: Path) -> None:
    for split in ("train", "val"):
        for class_name in ("n0001", "n0002"):
            _write_image(tmp_path / split / class_name / "sample.jpg")
    splits = discover_imagefolder_splits(tmp_path, expected_classes=2)
    cache_dir = tmp_path / "cache"
    _, _, _, cache_path = load_or_index_imagefolder_samples(
        splits, cache_dir=cache_dir
    )
    cache_path.write_text("not-json", encoding="utf-8")

    train, val, hit, _ = load_or_index_imagefolder_samples(
        splits, cache_dir=cache_dir
    )
    assert not hit
    assert len(train) == len(val) == 2
