from pathlib import Path

from PIL import Image

from experiments.imagenet100.data import discover_imagefolder_splits


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
