import math

import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.square_patch_dense_grid_look import SquarePatchDenseGridLook
from layers.two_ring_circular_look import TwoRingCircularLookMatcher


def _build(variant: str):
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
    )


def test_two_ring_matcher_stores_only_c6_and_c12_banks():
    base = _build("rot_hex_harmonic_pe_look")
    coordinates = torch.stack(
        (base.patch_embed.coo_patchs.real, base.patch_embed.coo_patchs.imag), dim=-1
    )
    matcher = TwoRingCircularLookMatcher(
        coordinates=coordinates,
        depth=2,
        num_heads=3,
        head_dim=64,
        start_layer=1,
    )
    assert matcher.inner_radius.shape == (1, 3, 6, 32)
    assert matcher.inner_phase.shape == (1, 3, 6, 32)
    assert matcher.outer_radius.shape == (1, 3, 12, 32)
    assert matcher.outer_phase.shape == (1, 3, 12, 32)
    assert (
        matcher.inner_radius[0, 0].numel()
        + matcher.inner_phase[0, 0].numel()
    ) == 6 * 64
    assert (
        matcher.outer_radius[0, 0].numel()
        + matcher.outer_phase[0, 0].numel()
    ) == 12 * 64
    assert matcher.inner_relative.shape == (6, 6)
    assert matcher.outer_relative.shape == (12, 12)
    assert matcher.look_grid.shape == (1, 3, 4, 12)
    assert matcher.look_radial0.shape == (
        2, coordinates.shape[0], coordinates.shape[0]
    )
    # The two graph rings are two transformed supports of one canonical map:
    # C12 uses the larger support and C6 the contracted support.
    assert matcher.look_valid[0].sum() > matcher.look_valid[1].sum()
    assert not hasattr(matcher, "inner_look_grid")
    assert not hasattr(matcher, "outer_look_grid")
    assert (matcher.inner_valid.sum(dim=1) == 6).any()
    assert (matcher.outer_valid.sum(dim=1) == 12).any()


def test_two_ring_matcher_outputs_centered_coefficients_and_gradients():
    base = _build("rot_hex_harmonic_pe_look")
    coordinates = torch.stack(
        (base.patch_embed.coo_patchs.real, base.patch_embed.coo_patchs.imag), dim=-1
    )
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
    inner, outer = matcher(features, layer_index=0)
    assert inner.shape == (2, coordinates.shape[0], 3, 6)
    assert outer.shape == (2, coordinates.shape[0], 3, 12)
    dense_bias = matcher.dense_look_bias(inner, outer, layer_index=0)
    assert dense_bias.shape == (
        2, 3, coordinates.shape[0], coordinates.shape[0]
    )
    torch.testing.assert_close(inner.mean(dim=-1), torch.zeros_like(inner[..., 0]))
    torch.testing.assert_close(outer.mean(dim=-1), torch.zeros_like(outer[..., 0]))
    loss = inner.square().mean() + outer.square().mean()
    loss = loss + dense_bias.square().mean()
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(matcher.inner_radius.grad).all()
    assert torch.isfinite(matcher.inner_phase.grad).all()
    assert torch.isfinite(matcher.outer_radius.grad).all()
    assert torch.isfinite(matcher.outer_phase.grad).all()
    assert torch.isfinite(matcher.look_grid.grad).all()


def test_spatial_and_paired_weight_rotation_are_synchronized():
    base = _build("rot_hex_harmonic_pe_look")
    coordinates = torch.stack(
        (base.patch_embed.coo_patchs.real, base.patch_embed.coo_patchs.imag), dim=-1
    )
    matcher = TwoRingCircularLookMatcher(
        coordinates=coordinates,
        depth=1,
        num_heads=3,
        head_dim=64,
        start_layer=0,
    )
    rendered = matcher._render_polar_weight(
        matcher.outer_radius[0],
        matcher.outer_phase[0],
        matcher.outer_relative,
        matcher.outer_candidate_angle,
    )
    step = 2.0 * torch.pi / 12.0
    spatially_rotated = torch.roll(rendered[:, 0], shifts=1, dims=1)
    paired = spatially_rotated.reshape(3, 12, 32, 2)
    cosine, sine = math.cos(step), math.sin(step)
    expected = torch.stack(
        (
            paired[..., 0] * cosine - paired[..., 1] * sine,
            paired[..., 0] * sine + paired[..., 1] * cosine,
        ),
        dim=-1,
    ).flatten(-2)
    torch.testing.assert_close(rendered[:, 1], expected)


def test_ring_scales_match_original_shared_look_grid_transforms():
    base = _build("rot_hex_harmonic_pe_look")
    coordinates = torch.stack(
        (base.patch_embed.coo_patchs.real, base.patch_embed.coo_patchs.imag), dim=-1
    )
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
        source_directions=12,
        source_direction_period=12,
        scales=(1.0, 0.5),
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
    patches = coordinates.shape[0]

    # C12 direction 3 is the large-scale transform at 90 degrees.
    inner = torch.zeros(1, patches, 3, 6)
    outer = torch.zeros(1, patches, 3, 12)
    outer[..., 3] = 1.0
    actual_large = matcher.dense_look_bias(inner, outer, layer_index=0)
    expected_large = fields[:, 0, 3].unsqueeze(0)
    torch.testing.assert_close(actual_large, expected_large)

    # C6 direction 2 is 120 degrees, i.e. full-period direction index 4,
    # while using the contracted 0.5-scale support of the same table.
    inner.zero_()
    outer.zero_()
    inner[..., 2] = 1.0
    actual_small = matcher.dense_look_bias(inner, outer, layer_index=0)
    expected_small = fields[:, 1, 4].unsqueeze(0)
    torch.testing.assert_close(actual_small, expected_small)


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
