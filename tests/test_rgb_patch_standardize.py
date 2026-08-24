import torch

from layers.mixed_geometry.rgb_patch_standardize import RGBPatchStandardizer
from layers.mixed_geometry_distance_projection import MixedGeometryDistanceProjection


def test_shared_rgb_mean_std_standardization_preserves_weighted_moments():
    module = RGBPatchStandardizer(stride=2, variance_epsilon=1e-4)
    samples = torch.tensor(
        [[[[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]]]]
    )
    weight = torch.tensor([1.0, 2.0])
    normalized, state = module.decompose(samples, weight)
    weight_sum = 3.0 * weight.sum()
    assert torch.allclose(
        (normalized * weight).sum(dim=(-2, -1)),
        torch.zeros_like(state.mean),
        atol=2e-6,
    )
    expected_variance = (
        (samples - state.mean[..., None, None]).square() * weight
    ).sum(dim=(-2, -1)) / weight_sum
    assert torch.allclose(
        state.sigma_safe.square(), expected_variance + 1e-4
    )


def stats_model(out_channels=12):
    return MixedGeometryDistanceProjection(
        out_channels=out_channels,
        angular_bases=0,
        radial_bases=0,
        color_bases=0,
        stripe_bases=0,
        full_bases=2,
        directions=4,
        full_directions=4,
        kernel_sizes=(24, 12),
        radial_bins=3,
        ring_counts=(4, 8, 12),
        value_mode="shared_scale_affine_harmonic_stats",
        rgb_patch_standardize=True,
        use_triton_harmonic=False,
    )


def test_six_value_vectors_start_pairwise_orthogonal_per_base():
    model = stats_model()
    value = model.full_value.detach()
    gram = value @ value.transpose(1, 2)
    off_diagonal = gram - torch.diag_embed(gram.diagonal(dim1=1, dim2=2))
    assert value.shape == (2, 6, 12)
    assert off_diagonal.abs().max() < 1e-6


def test_mean_and_std_values_are_independent_additive_moments():
    model = stats_model(out_channels=7)
    weight = torch.rand(1, 2, 8, 3, 3)
    model._input_stats = torch.rand(1, 3, 3, 2)
    actual = model._weighted_value_sum(
        weight, model.full_value, "full", (2, 4)
    )
    coefficients = model._harmonic_pose_coefficients(
        "full", (2, 4), device=weight.device, dtype=weight.dtype
    )
    compact = torch.einsum("qbphw,pk->qbkhw", weight, coefficients)
    route_mass = weight.sum(2, keepdim=True)
    stats = model._input_stats.permute(0, 3, 1, 2)[:, None]
    compact = torch.cat((compact, route_mass * stats), dim=2)
    expected = torch.einsum("qbkhw,bkc->qhwc", compact, model.full_value)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_stats_mode_requires_rgb_standardization():
    try:
        MixedGeometryDistanceProjection(
            value_mode="shared_scale_affine_harmonic_stats",
            rgb_patch_standardize=False,
        )
    except ValueError as error:
        assert "requires rgb_patch_standardize" in str(error)
    else:
        raise AssertionError("missing incompatible-mode validation")
