import torch

from layers.mixed_geometry.geometries import DiskGeometry
from layers.mixed_geometry.rgb_patch_detrend import RGBPatchDetrender
from layers.mixed_geometry_distance_projection import (
    MixedGeometryDistanceProjection,
)


def test_cosine_weighted_moments_remove_shared_dc_and_rgb_xy_trends():
    geometry = DiskGeometry(kernel_size=8, stride=8)
    detrender = RGBPatchDetrender(out_channels=5, stride=8)
    radius = max(
        float(geometry.support_x.abs().max()),
        float(geometry.support_y.abs().max()),
    )
    x = geometry.support_x / radius
    y = geometry.support_y / radius
    shared_mean = torch.tensor([0.4, 0.6])
    channel_offset = torch.tensor([-0.12, 0.02, 0.10])
    gradient_x = torch.tensor(
        [[0.10, -0.04, 0.02], [-0.03, 0.08, 0.01]]
    )
    gradient_y = torch.tensor(
        [[-0.02, 0.06, 0.04], [0.07, -0.01, -0.05]]
    )
    samples = (
        shared_mean[:, None, None]
        + channel_offset[None, :, None]
        + gradient_x[:, :, None] * x
        + gradient_y[:, :, None] * y
    )

    normalized, state = detrender.decompose(
        samples,
        geometry.support_x,
        geometry.support_y,
        geometry.support_cover,
    )

    torch.testing.assert_close(state.mean, shared_mean, atol=1e-6, rtol=0)
    torch.testing.assert_close(state.gradient_x, gradient_x, atol=1e-6, rtol=0)
    torch.testing.assert_close(state.gradient_y, gradient_y, atol=1e-6, rtol=0)
    expected_sigma = torch.sqrt(channel_offset.square().mean() + 1e-4)
    torch.testing.assert_close(
        state.sigma_safe,
        torch.full_like(state.sigma_safe, expected_sigma),
        atol=1e-6,
        rtol=0,
    )

    # Per-channel DC color offsets remain; only the shared RGB mean and the
    # six xy trend moments are removed.
    expected = channel_offset[None, :, None] / expected_sigma
    torch.testing.assert_close(
        normalized,
        expected.expand_as(normalized),
        atol=2e-5,
        rtol=0,
    )


def test_detrender_adds_exactly_six_value_vectors_and_no_mean_parameters():
    detrender = RGBPatchDetrender(out_channels=7, stride=4)
    parameters = dict(detrender.named_parameters())
    assert set(parameters) == {"trend_value"}
    assert parameters["trend_value"].shape == (3, 2, 7)
    assert parameters["trend_value"].numel() == 6 * 7


def test_mixed_projection_detrend_forward_and_gradients():
    model = MixedGeometryDistanceProjection(
        out_channels=8,
        angular_bases=1,
        radial_bases=1,
        color_bases=1,
        stripe_bases=1,
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
        rgb_patch_detrend=True,
        detrend_variance_epsilon=1e-4,
    )
    unit_image = torch.rand(2, 3, 16, 16)
    mean = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
    image = (unit_image - mean) / std
    output = model(image)
    assert output.shape == (2, 8, 4, 4)
    output.square().mean().backward()
    assert model.rgb_detrender.trend_value.grad is not None
    assert model.full_prototype.grad is not None
