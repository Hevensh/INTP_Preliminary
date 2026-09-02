import torch

from experiments.imagenet100.eval_rotation_consistency import (
    _aggregate_rows,
    _angle_metrics,
    _balanced_subset_indices,
    _parse_angles,
)


def test_parse_dense_angle_range() -> None:
    assert _parse_angles("0:45:15") == [0.0, 15.0, 30.0, 45.0]


def test_identical_probabilities_have_perfect_consistency() -> None:
    probabilities = torch.tensor(
        [
            [0.60, 0.10, 0.10, 0.10, 0.05, 0.05],
            [0.10, 0.60, 0.10, 0.10, 0.05, 0.05],
        ]
    )
    targets = torch.tensor([0, 1])
    metrics = _angle_metrics(probabilities, targets, probabilities)

    assert metrics["top1"] == 100.0
    assert metrics["agreement"] == 100.0
    assert abs(metrics["js_divergence"]) < 1e-7

    summary = _aggregate_rows(
        [{"angle_degrees": 90.0, **metrics}],
        base_top1=100.0,
    )
    assert summary["mean_top1"] == 100.0
    assert summary["mean_top1_drop"] == 0.0


def test_sample_limit_is_balanced_across_classes() -> None:
    targets = [0] * 5 + [1] * 5 + [2] * 5
    selected = _balanced_subset_indices(targets, 6)
    assert [targets[index] for index in selected] == [0, 0, 1, 1, 2, 2]
