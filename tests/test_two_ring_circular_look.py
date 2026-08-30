import math

import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.square_patch_dense_grid_look import SquarePatchDenseGridLook
from layers.two_ring_circular_look import TwoRingCircularLookMatcher


def _build(variant: str, **kwargs):
    return build_imagenet100_model(
        variant=variant,
        model_name="deit_tiny_patch16_224",
        pretrained=False,
        num_classes=100,
        image_size=224,
        rot_kernel_sizes=(24, 12),
        rot_bases=96,
        rot_directions=6,
        rot_global_directions=12,
        rot_angular_bins_per_radius=3,
        look_compact_variable_rings=True,
        rot_prototype_chunk_size=16,
        rot_null_initial_score=0.0,
        **kwargs,
    )


def _coordinates():
    base = _build("rot_hex_harmonic_pe_look")
    return torch.stack(
        (base.patch_embed.coo_patchs.real, base.patch_embed.coo_patchs.imag),
        dim=-1,
    )


def test_feature_ring_stores_only_one_c6_probe_and_one_full_look_field():
    coordinates = _coordinates()
    matcher = TwoRingCircularLookMatcher(
        coordinates=coordinates,
        depth=2,
        num_heads=3,
        head_dim=64,
        start_layer=1,
    )
    assert matcher.radius.shape == (1, 3, 6, 32)
    assert matcher.phase.shape == (1, 3, 6, 32)
    assert matcher.radius[0, 0].numel() + matcher.phase[0, 0].numel() == 6 * 64
    assert matcher.relative.shape == (6, 6)
    assert matcher.look_grid.shape == (1, 3, 4, 12)
    assert matcher.look_radial0.shape == (
        coordinates.shape[0], coordinates.shape[0]
    )
    assert (matcher.valid.sum(dim=1) == 6).any()
    assert not hasattr(matcher, "outer_neighbors")
    assert not hasattr(matcher, "outer_radius")


def test_c6_probe_outputs_centered_coefficients_and_gradients():
    coordinates = _coordinates()
    matcher = TwoRingCircularLookMatcher(
        coordinates=coordinates,
        depth=1,
        num_heads=3,
        head_dim=64,
        start_layer=0,
    )
    with torch.no_grad():
        matcher.gate.fill_(0.1)
    features = torch.randn(2, coordinates.shape[0], 192, requires_grad=True)
    scores = matcher(features, layer_index=0)
    dense_bias = matcher.dense_look_bias(scores, layer_index=0)
    combined_bias = matcher.dense_look_bias_from_features(features, layer_index=0)
    assert scores.shape == (2, coordinates.shape[0], 3, 6)
    assert dense_bias.shape == (
        2, 3, coordinates.shape[0], coordinates.shape[0]
    )
    torch.testing.assert_close(combined_bias, dense_bias)
    torch.testing.assert_close(scores.mean(dim=-1), torch.zeros_like(scores[..., 0]))
    (scores.square().mean() + combined_bias.square().mean()).backward()
    for gradient in (
        features.grad,
        matcher.radius.grad,
        matcher.phase.grad,
        matcher.look_grid.grad,
        matcher.gate.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_four_layer_batch_matches_independent_direct_fields():
    coordinates = _coordinates()
    matcher = TwoRingCircularLookMatcher(
        coordinates=coordinates,
        depth=4,
        num_heads=3,
        head_dim=64,
        start_layer=0,
    )
    with torch.no_grad():
        matcher.gate.normal_(mean=0.0, std=0.1)
    features = torch.randn(2, coordinates.shape[0], 192, requires_grad=True)
    layer_indices = (0, 1, 2, 3)
    expected = torch.stack(
        tuple(
            matcher.dense_look_bias_from_features(features, layer_index=index)
            for index in layer_indices
        )
    )
    actual = matcher.dense_look_bias_for_layers(
        features, layer_indices=layer_indices
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.square().mean().backward()
    for gradient in (
        features.grad,
        matcher.radius.grad,
        matcher.phase.grad,
        matcher.look_grid.grad,
        matcher.gate.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_frequency_batch_matches_four_layer_direct_fields():
    coordinates = _coordinates()
    matcher = TwoRingCircularLookMatcher(
        coordinates=coordinates,
        depth=4,
        num_heads=3,
        head_dim=64,
        start_layer=0,
    )
    with torch.no_grad():
        matcher.gate.normal_(mean=0.0, std=0.1)
    features = torch.randn(1, coordinates.shape[0], 192)
    layer_indices = (0, 1, 2, 3)
    direct = matcher.dense_look_bias_for_layers(
        features, layer_indices=layer_indices, frequency_domain=False
    )
    frequency = matcher.dense_look_bias_for_layers(
        features, layer_indices=layer_indices, frequency_domain=True
    )
    torch.testing.assert_close(frequency, direct, rtol=2e-4, atol=2e-4)


def test_spatial_and_paired_weight_rotation_are_synchronized():
    coordinates = _coordinates()
    matcher = TwoRingCircularLookMatcher(
        coordinates=coordinates,
        depth=1,
        num_heads=3,
        head_dim=64,
        start_layer=0,
    )
    rendered = matcher._render_polar_weight(
        matcher.radius[0],
        matcher.phase[0],
        matcher.relative,
        matcher.candidate_angle,
    )
    step = 2.0 * torch.pi / 6.0
    spatially_rotated = torch.roll(rendered[:, 0], shifts=1, dims=1)
    paired = spatially_rotated.reshape(3, 6, 32, 2)
    cosine, sine = math.cos(step), math.sin(step)
    expected = torch.stack(
        (
            paired[..., 0] * cosine - paired[..., 1] * sine,
            paired[..., 0] * sine + paired[..., 1] * cosine,
        ),
        dim=-1,
    ).flatten(-2)
    torch.testing.assert_close(rendered[:, 1], expected)


def test_c6_pose_steers_the_original_full_four_ring_look_transform():
    coordinates = _coordinates()
    matcher = TwoRingCircularLookMatcher(
        coordinates=coordinates,
        depth=1,
        num_heads=3,
        head_dim=64,
        start_layer=0,
    )
    reference = SquarePatchDenseGridLook(
        image_size=224,
        patch_size=16,
        num_heads=3,
        prototype_angular_bins=24,
        source_directions=6,
        source_direction_period=6,
        scales=(1.0,),
        prototype_radius=12.0,
        look_direction_bins=12,
        look_radial_bins=4,
        look_radius=4.0,
        patch_centers_xy=coordinates,
        patch_coordinates_xy=coordinates,
    )
    canonical = torch.randn(3, 4, 12)
    with torch.no_grad():
        matcher.look_grid[0].copy_(canonical)
        reference.look_grid.copy_(canonical)
    fields = reference.transformed_look_grids()
    scores = torch.zeros(1, coordinates.shape[0], 3, 6)
    scores[..., 2] = 1.0
    actual = matcher.dense_look_bias(scores, layer_index=0)
    expected = fields[:, 0, 2].unsqueeze(0)
    torch.testing.assert_close(actual, expected)
    assert matcher.look_valid.sum() == reference.look_valid[0, 0].sum()


def test_zero_gated_ring_variant_matches_existing_pe_look_model():
    torch.manual_seed(7)
    baseline = _build("rot_hex_harmonic_pe_look").eval()
    torch.manual_seed(7)
    refined = _build("rot_hex_harmonic_pe_look_ring").eval()
    assert refined.feature_ring_matcher.start_layer == 0
    assert refined.feature_ring_matcher.active_depth == refined.depth
    assert refined.feature_ring_matcher.look_grid.shape == (
        refined.depth, refined.num_heads, 4, 12
    )
    missing, unexpected = refined.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert all("feature_ring_matcher" in key for key in missing)
    assert torch.count_nonzero(refined.feature_ring_matcher.gate) == 0

    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        expected = baseline(image)
        actual = refined(image)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
