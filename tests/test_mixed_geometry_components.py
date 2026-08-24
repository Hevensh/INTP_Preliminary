import torch

from layers.mixed_geometry_distance_projection import (
    MixedGeometryDistanceProjection,
)


def build_projection(*, learnable_cover=False):
    return MixedGeometryDistanceProjection(
        out_channels=8,
        angular_bases=2,
        radial_bases=1,
        color_bases=1,
        stripe_bases=2,
        full_bases=1,
        directions=4,
        angular_directions=2,
        stripe_directions=2,
        full_directions=2,
        kernel_sizes=(8, 4),
        color_kernel_sizes=(8, 4),
        angular_bins=8,
        radial_bins=4,
        stripe_bins=6,
        ring_counts=(4, 8, 12, 16),
        stride=4,
        learnable_cover=learnable_cover,
    )


def test_component_split_preserves_historical_cover_checkpoint_keys():
    model = build_projection(learnable_cover=True)
    keys = model.state_dict().keys()
    assert "cover_mask_logits.angular" in keys
    assert "cover_mask_logits.full" in keys
    assert not any(key.startswith("cover_mask_logits.logits.") for key in keys)
    assert model.cover_mask_logits["angular"].shape == (2, 3, 8)
    assert model.cover_mask_logits["radial"].shape == (1, 3, 4)
    assert model.cover_mask_logits["color"].shape == (1, 3)
    assert model.cover_mask_logits["stripe"].shape == (2, 3, 6)
    assert model.cover_mask_logits["full"].shape == (1, 3, 40)


def test_mask_logits_follow_the_exact_prototype_interpolation_path():
    model = build_projection(learnable_cover=True)
    geometry = model.angular_geometry
    with torch.no_grad():
        model.cover_mask_logits["angular"][0, 0].copy_(
            torch.tensor([-4.0, 4.0, -3.0, 3.0, -2.0, 2.0, -1.0, 1.0])
        )
    rendered = model.cover_mask_logits.render(
        "angular", geometry, geometry.render, normalize=False
    )
    expected = geometry.render(
        model.cover_mask_logits["angular"]
    ).sigmoid()
    assert torch.allclose(rendered, expected)


def test_rendered_mask_keeps_cosine_cover_mass_at_every_pose():
    model = build_projection(learnable_cover=True)
    geometry = model.stripe_geometries[1]
    rendered = model._cover_weight("stripe", geometry, geometry.render)
    target = (3 * geometry.support_cover.sum()).expand_as(
        rendered.sum((-2, -1))
    )
    assert torch.allclose(
        rendered.sum((-2, -1)), target, atol=1e-5, rtol=1e-5
    )


def test_prototype_states_report_common_starting_inventory():
    model = build_projection()
    states = model.prototype_states()
    assert len(states) == sum(model.family_counts.values())
    assert sum(state.active_pose_count for state in states) == (
        model.direction_pose_inventory()["total"]
    )
    angular = [state for state in states if state.family == "angular"]
    assert all(state.direction_mode == "free" for state in angular)


def test_per_scale_sparse_stripe_is_visible_to_state_differentiator():
    model = build_projection()
    model.set_pose_degradation(
        scale_direction_fixed={"stripe": {0: {0: [0]}}},
    )
    state = next(
        state for state in model.prototype_states()
        if state.family == "stripe" and state.base == 0
    )
    assert state.direction_mode == "per_scale_sparse"
    assert state.active_pose_count == int(
        model.stripe_scale_direction_subset[0].sum()
    )


def test_learnable_cover_still_runs_after_component_split():
    model = build_projection(learnable_cover=True)
    image = torch.randn(1, 3, 16, 16)
    output = model(image)
    assert output.shape == (1, 8, 4, 4)
    output.square().mean().backward()
    assert model.cover_mask_logits["angular"].grad is not None
