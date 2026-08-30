import math

import torch

from experiments.imagenet100.models import build_imagenet100_model
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
    assert matcher.inner_to_pose.shape == (1, 3, 12)
    assert matcher.outer_to_pose.shape == (1, 3, 12)
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
    pose = matcher.project_to_pose(inner, outer, layer_index=0)
    assert pose.shape == (2, coordinates.shape[0], 3, 2, 6)
    torch.testing.assert_close(inner.mean(dim=-1), torch.zeros_like(inner[..., 0]))
    torch.testing.assert_close(outer.mean(dim=-1), torch.zeros_like(outer[..., 0]))
    pose.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(matcher.inner_radius.grad).all()
    assert torch.isfinite(matcher.inner_phase.grad).all()
    assert torch.isfinite(matcher.outer_radius.grad).all()
    assert torch.isfinite(matcher.outer_phase.grad).all()
    assert torch.isfinite(matcher.inner_to_pose.grad).all()
    assert torch.isfinite(matcher.outer_to_pose.grad).all()


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


def test_zero_gated_ring_variant_matches_existing_pe_look_model():
    torch.manual_seed(7)
    baseline = _build("rot_hex_harmonic_pe_look").eval()
    torch.manual_seed(7)
    refined = _build("rot_hex_harmonic_pe_look_ring").eval()
    missing, unexpected = refined.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert all("feature_ring_matcher" in key for key in missing)
    assert torch.count_nonzero(refined.feature_ring_matcher.gate) == 0

    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        expected = baseline(image)
        actual = refined(image)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
