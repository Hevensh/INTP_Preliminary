import torch

from layers.mixed_geometry_distance_projection import (
    MixedGeometryDistanceProjection,
)


def small_model(color_kernel_sizes):
    return MixedGeometryDistanceProjection(
        angular_bases=1,
        radial_bases=1,
        color_bases=2,
        stripe_bases=1,
        full_bases=1,
        kernel_sizes=(24, 12),
        color_kernel_sizes=color_kernel_sizes,
    )


def test_two_scale_color_scores_and_values_align():
    model = small_model((24, 12))
    image = torch.randn(2, 3, 224, 224)

    scores = model.family_scores(image)

    assert scores["color"].shape == (2, 2, 2, 14, 14)
    assert model.color_log2_scale.shape == (2, 2)
    assert model.color_value.shape == (2, 2, 192)
    assert model(image).shape == (2, 192, 14, 14)
    assert model.direction_pose_inventory()["color"] == 4


def test_single_scale_color_retains_legacy_parameter_shapes():
    model = small_model((24,))

    assert model.color_log2_scale.shape == (2,)
    assert model.color_value.shape == (2, 1, 192)
    assert model.direction_pose_inventory()["color"] == 2


def test_half_circle_stripe_keeps_original_angular_spacing():
    model = MixedGeometryDistanceProjection(
        angular_bases=1,
        radial_bases=1,
        color_bases=1,
        stripe_bases=2,
        full_bases=1,
        directions=8,
        stripe_directions=4,
        kernel_sizes=(24, 12),
    )

    scores = model.family_scores(torch.randn(1, 3, 224, 224))

    assert scores["stripe"].shape == (1, 2, 2, 4, 14, 14)
    assert model.stripe_value.shape == (2, 2, 4, 192)
    assert model.direction_pose_inventory()["stripe"] == 16


def test_two_fixed_directions_remain_soft_selectable():
    model = MixedGeometryDistanceProjection(
        angular_bases=1,
        radial_bases=1,
        color_bases=1,
        stripe_bases=1,
        full_bases=1,
        directions=8,
        stripe_directions=4,
        kernel_sizes=(24, 12),
    )
    model.set_pose_degradation(direction_fixed={"stripe": {0: [1, 3]}})

    output = model(torch.randn(1, 3, 224, 224))

    assert output.shape == (1, 192, 14, 14)
    assert model.stripe_direction_fixed[0] == -1
    assert model.stripe_direction_subset[0].tolist() == [False, True, False, True]
    assert model.direction_pose_inventory()["stripe"] == 4


def test_all_directional_families_can_use_half_circle():
    model = MixedGeometryDistanceProjection(
        angular_bases=1,
        radial_bases=1,
        color_bases=1,
        stripe_bases=1,
        full_bases=1,
        directions=8,
        angular_directions=4,
        stripe_directions=4,
        full_directions=4,
        kernel_sizes=(24, 12),
    )

    scores = model.family_scores(torch.randn(1, 3, 224, 224))

    assert scores["angular"].shape[2] == 4
    assert scores["stripe"].shape[3] == 4
    assert scores["full"].shape[3] == 4
    assert model.direction_pose_inventory()["total"] == 23


def test_stripe_can_keep_two_directions_independently_per_scale():
    model = MixedGeometryDistanceProjection(
        angular_bases=1,
        radial_bases=1,
        color_bases=1,
        stripe_bases=1,
        full_bases=1,
        directions=8,
        stripe_directions=4,
        kernel_sizes=(24, 12),
    )
    model.set_pose_degradation(
        scale_direction_fixed={
            "stripe": {0: {0: [0, 2], 1: [1, 3]}}
        }
    )

    output = model(torch.randn(1, 3, 224, 224))

    assert output.shape == (1, 192, 14, 14)
    assert model.stripe_scale_direction_subset[0, 0].tolist() == [True, False, True, False]
    assert model.stripe_scale_direction_subset[0, 1].tolist() == [False, True, False, True]
    assert model.direction_pose_inventory()["stripe"] == 4


def test_angular_and_full_can_keep_two_directions_per_scale():
    model = MixedGeometryDistanceProjection(
        angular_bases=1,
        radial_bases=1,
        color_bases=1,
        stripe_bases=1,
        full_bases=1,
        directions=8,
        angular_directions=4,
        angular_kernel_sizes=(24, 12),
        stripe_directions=4,
        full_directions=4,
        kernel_sizes=(24, 12),
    )
    model.set_pose_degradation(scale_direction_fixed={
        "angular": {0: {0: [0, 2], 1: [1, 3]}},
        "full": {0: {0: [0, 1], 1: [2, 3]}},
    })

    scores = model.family_scores(torch.randn(1, 3, 224, 224))
    output = model(torch.randn(1, 3, 224, 224))

    assert scores["angular"].shape == (1, 1, 2, 4, 14, 14)
    assert model.angular_value.shape == (1, 2, 4, 192)
    assert output.shape == (1, 192, 14, 14)
    assert model.direction_pose_inventory()["angular"] == 4
    assert model.direction_pose_inventory()["full"] == 4
