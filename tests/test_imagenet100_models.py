import math

import pytest
import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.hex_linear_patch_embed import HexLinearPatchEmbed
from layers.hex_rotating_polar_patch_embed import HexRotatingPolarPatchEmbed
from layers.hex_rotating_dot_patch_embed import HexRotatingDotPatchEmbed
from layers.hex_rotating_grouped_dot_patch_embed import HexRotatingGroupedDotPatchEmbed
from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed
from layers.gmr_patch_embed import EquiVitGMRPatchEmbed, GaussianMixtureRingConv2d
from layers.arc_adaptive_patch_embed import ARCAdaptivePatchEmbed
from model.gevit_tiny import C4LiftingPatchEmbed, GEViTTinyP4


def _build(variant: str, **kwargs):
    return build_imagenet100_model(
        variant=variant,
        model_name="deit_tiny_patch16_224",
        pretrained=False,
        num_classes=100,
        image_size=224,
        hex_kernel_size=21,
        hex_stride=18,
        **kwargs,
    )


def test_hex_comparison_only_replaces_patch_embedding_and_position_length():
    baseline = _build("deit_tiny")
    hex_model = _build("hex_patch")

    assert isinstance(hex_model.patch_embed, HexLinearPatchEmbed)
    assert baseline.patch_embed.num_patches == 196
    assert hex_model.patch_embed.num_patches == 195
    assert baseline.pos_embed.shape == (1, 197, 192)
    assert hex_model.pos_embed.shape == (1, 196, 192)
    assert len(baseline.blocks) == len(hex_model.blocks) == 12
    assert baseline.blocks[0].attn.num_heads == hex_model.blocks[0].attn.num_heads == 3
    assert baseline.head.out_features == hex_model.head.out_features == 100


def test_hex_comparison_forward_shape():
    model = _build("hex_patch").eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 100)
    assert torch.isfinite(output).all()


def test_gmr_efficient_forward_matches_rendered_dense_kernel():
    layer = GaussianMixtureRingConv2d(3, 5, kernel_size=6, stride=2).eval()
    image = torch.randn(2, 3, 20, 20)
    actual = layer(image)
    expected = torch.nn.functional.conv2d(
        image,
        layer.rendered_weight(),
        layer.bias,
        stride=2,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_equi_gmr_model_uses_paper_sized_stem_and_standard_token_grid():
    model = _build("equi_gmr_pe")
    assert isinstance(model.patch_embed, EquiVitGMRPatchEmbed)
    assert model.patch_embed.proj1.kernel_size == 6
    assert model.patch_embed.proj1.stride == 6
    assert model.patch_embed.proj2.kernel_size == 11
    assert model.patch_embed.proj2.stride == 2
    assert model.patch_embed.num_patches == 196
    assert model.pos_embed.shape == (1, 197, 192)


def test_arc_patch_embed_routes_rotates_and_backpropagates():
    embed = ARCAdaptivePatchEmbed(
        img_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=12,
        kernel_number=4,
        batch_chunk_size=1,
    )
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = embed(image)
    alpha, angle = embed.route(image.detach())
    assert output.shape == (2, 4, 12)
    assert alpha.shape == angle.shape == (2, 4)
    assert torch.all((alpha > 0) & (alpha < 1))
    assert angle.abs().max() <= math.radians(40.0)
    output.square().mean().backward()
    assert torch.isfinite(embed.weight.grad).all()
    assert torch.isfinite(embed.routing.angle_head.weight.grad).all()


def test_arc_model_keeps_standard_token_and_position_shapes():
    model = _build("arc_adaptive_pe", arc_batch_chunk_size=2)
    assert isinstance(model.patch_embed, ARCAdaptivePatchEmbed)
    assert model.patch_embed.kernel_number == 4
    assert model.patch_embed.num_patches == 196
    assert model.pos_embed.shape == (1, 197, 192)


def test_gevit_builder_uses_p4_group_field_without_absolute_position_embedding():
    model = _build("gevit_p4_local")
    assert isinstance(model, GEViTTinyP4)
    assert isinstance(model.patch_embed, C4LiftingPatchEmbed)
    assert model.orientations == 4
    assert len(model.blocks) == 12
    assert model.blocks[0].attention.window_size == 5
    assert not hasattr(model, "pos_embed")
    assert 5_400_000 < sum(parameter.numel() for parameter in model.parameters()) < 5_700_000


def test_gevit_small_forward_backward_and_c4_equivariance():
    torch.manual_seed(0)
    model = GEViTTinyP4(
        image_size=32,
        patch_size=8,
        num_classes=10,
        embed_dim=24,
        depth=1,
        num_heads=3,
        window_size=3,
    )
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    lifted = model.patch_embed(image)
    rotated_lifted = model.patch_embed(torch.rot90(image, 1, dims=(-2, -1)))
    expected_lifted = torch.roll(
        torch.rot90(lifted, 1, dims=(-2, -1)),
        shifts=1,
        dims=2,
    )
    torch.testing.assert_close(rotated_lifted, expected_lifted, atol=2e-5, rtol=2e-5)

    output = model(image)
    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(image.grad).all()


def test_gevit_local_attention_preserves_c4_action():
    torch.manual_seed(0)
    model = GEViTTinyP4(
        image_size=32,
        patch_size=8,
        num_classes=10,
        embed_dim=24,
        depth=1,
        num_heads=3,
        window_size=3,
    ).eval()
    field = torch.randn(1, 24, 4, 4, 4)
    actual = model.blocks[0].attention(
        torch.roll(torch.rot90(field, 1, dims=(-2, -1)), shifts=1, dims=2)
    )
    expected = torch.roll(
        torch.rot90(model.blocks[0].attention(field), 1, dims=(-2, -1)),
        shifts=1,
        dims=2,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("variant", ["equi_gmr_pe", "arc_adaptive_pe"])
def test_new_rotation_comparison_model_forward_shape(variant: str):
    model = _build(variant, arc_batch_chunk_size=1).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 100)
    assert torch.isfinite(output).all()


def test_hex_pretrained_mode_is_rejected_for_from_scratch_comparison():
    with pytest.raises(ValueError, match="from-scratch"):
        build_imagenet100_model(
            variant="hex_patch",
            model_name="deit_tiny_patch16_224",
            pretrained=True,
            num_classes=100,
            image_size=224,
        )


def test_rotating_hex_pe_uses_same_token_count_and_learned_position_shape():
    model = _build("rot_hex_pe")
    assert isinstance(model.patch_embed, HexRotatingPolarPatchEmbed)
    assert model.patch_embed.num_patches == 195
    assert model.pos_embed.shape == (1, 196, 192)
    assert model.patch_embed.bases == 96
    assert model.patch_embed.scales == 2
    assert model.patch_embed.directions == 4


def test_rotating_hex_patch_embed_forward_and_backward():
    embed = HexRotatingPolarPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=12,
        lattice_stride=8,
        kernel_sizes=(12, 6),
        bases=4,
        directions=4,
        global_directions=8,
        radial_bins=4,
        prototype_chunk_size=2,
    )
    output = embed(torch.randn(2, 3, 32, 32, requires_grad=True))
    assert output.shape == (2, embed.num_patches, 12)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert embed.prototype.grad is not None
    assert torch.isfinite(embed.prototype.grad).all()


def test_rotating_hex_group_exp_matches_explicit_pose_aggregation():
    embed = HexRotatingPolarPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=12,
        lattice_stride=8,
        kernel_sizes=(12, 6),
        bases=4,
        directions=4,
        global_directions=8,
        radial_bins=4,
        prototype_chunk_size=4,
        null_initial_score=-1,
        score_normalization="patch_global",
        response_gate="exp",
        response_gate_location="group",
    )
    scores = torch.randn(2, 3, 4, 2, 4)
    actual = embed._chunk_output(scores, 0, 4)

    null = embed.null_score[None, None, :, None].expand(2, 3, -1, -1)
    probabilities = torch.cat((scores.flatten(3, 4), null), -1).softmax(-1)[..., :-1]
    probabilities = probabilities.view_as(scores)
    group_amplitude = torch.exp((probabilities * scores).sum((3, 4)))
    direction_value = torch.einsum(
        "dk,pkc->pdc", embed.direction_coefficients, embed.direction_pair
    )
    pose_value = direction_value[:, None] + embed.scale_value[:, :, None]
    expected = torch.einsum("qnpsd,psdc->qnpc", probabilities, pose_value)
    expected = (expected * group_amplitude[..., None]).sum(2)
    torch.testing.assert_close(actual, expected)


def test_simple_rotating_dot_embed_forward_backward():
    embed = HexRotatingDotPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=12,
        lattice_stride=8,
        kernel_sizes=(12, 6),
        bases=4,
        directions=4,
        global_directions=8,
        radial_bins=4,
        prototype_chunk_size=2,
    )
    output = embed(torch.randn(2, 3, 32, 32, requires_grad=True))
    assert output.shape == (2, embed.num_patches, 12)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert embed.prototype.grad is not None
    assert torch.isfinite(embed.prototype.grad).all()


def test_simple_rotating_dot_model_keeps_hex_token_and_pe_shapes():
    model = _build("rot_hex_dot_simple_pe")
    assert isinstance(model.patch_embed, HexRotatingDotPatchEmbed)
    assert model.patch_embed.num_patches == 195
    assert model.pos_embed.shape == (1, 196, 192)


def test_grouped_rotating_dot_embed_forward_backward_and_parameter_shapes():
    embed = HexRotatingGroupedDotPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=12,
        lattice_stride=8,
        kernel_sizes=(12, 6),
        bases=6,
        directions=4,
        global_directions=8,
        groups=3,
        radial_bins=4,
        prototype_chunk_size=2,
    )
    output = embed(torch.randn(2, 3, 32, 32, requires_grad=True))
    assert output.shape == (2, embed.num_patches, 12)
    assert embed.direction_pair.shape == (6, 2, 4)
    assert embed.scale_value.shape == (6, 2, 4)
    assert torch.all(embed.null_score == 0)
    output.square().mean().backward()
    assert torch.isfinite(embed.prototype.grad).all()
    assert torch.isfinite(embed.direction_pair.grad).all()


def test_grouped_rotating_dot_model_keeps_hex_token_and_pe_shapes():
    model = _build("rot_hex_dot_grouped_pe")
    assert isinstance(model.patch_embed, HexRotatingGroupedDotPatchEmbed)
    assert model.patch_embed.num_patches == 195
    assert model.patch_embed.group_dim == 64
    assert model.pos_embed.shape == (1, 196, 192)


def test_rotating_harmonic_embed_is_only_prototype_and_cosine_sine_output():
    embed = HexRotatingHarmonicPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=12,
        lattice_stride=8,
        kernel_sizes=(12, 6),
        bases=6,
        directions=4,
        global_directions=8,
        radial_bins=4,
        prototype_chunk_size=2,
    )
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = embed(image)
    assert output.shape == (2, embed.num_patches, 12)
    assert not hasattr(embed, "null_score")
    assert not hasattr(embed, "direction_pair")
    assert not hasattr(embed, "scale_value")
    assert torch.allclose(embed.scale_cover_0.sum(), embed.scale_cover_1.sum())
    assert torch.allclose(embed(2 * image), 2 * output, atol=1e-6, rtol=1e-5)
    output.square().mean().backward()
    assert torch.isfinite(embed.prototype.grad).all()


def test_rotating_harmonic_model_keeps_hex_token_and_pe_shapes():
    model = _build("rot_hex_harmonic_pe")
    assert isinstance(model.patch_embed, HexRotatingHarmonicPatchEmbed)
    assert model.patch_embed.num_patches == 195
    assert model.patch_embed.bases * 2 == 192
    assert model.pos_embed.shape == (1, 196, 192)


def test_rotating_harmonic_softmax_adds_only_per_prototype_null_score():
    embed = HexRotatingHarmonicPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=12,
        lattice_stride=8,
        kernel_sizes=(12, 6),
        bases=6,
        directions=4,
        global_directions=8,
        radial_bins=4,
        prototype_chunk_size=2,
        pose_softmax=True,
        use_null=True,
        null_initial_score=0.0,
    )
    output = embed(torch.randn(2, 3, 32, 32, requires_grad=True))
    assert output.shape == (2, embed.num_patches, 12)
    assert embed.null_score.shape == (6,)
    assert torch.all(embed.null_score == 0)
    assert not hasattr(embed, "direction_pair")
    assert not hasattr(embed, "scale_value")
    output.square().mean().backward()
    assert torch.isfinite(embed.prototype.grad).all()
    assert torch.isfinite(embed.null_score.grad).all()


def test_rotating_harmonic_softmax_model_keeps_hex_token_and_pe_shapes():
    model = _build("rot_hex_harmonic_softmax_pe")
    assert isinstance(model.patch_embed, HexRotatingHarmonicPatchEmbed)
    assert model.patch_embed.pose_softmax
    assert model.patch_embed.use_null
    assert model.patch_embed.num_patches == 195
    assert model.pos_embed.shape == (1, 196, 192)


@pytest.mark.parametrize(
    ("variant", "use_pos_embed"),
    [
        ("rot_hex_harmonic_look", False),
        ("rot_hex_harmonic_pe_look", True),
    ],
)
def test_rotating_harmonic_look_variants_use_null_softmax_in_both_routes(
    variant: str,
    use_pos_embed: bool,
):
    model = _build(variant, rot_null_initial_score=0.0)
    assert model.use_pos_embed is use_pos_embed
    assert (model.pos_embed is not None) is use_pos_embed
    if model.pos_embed is not None:
        assert model.pos_embed.requires_grad
    assert model.patch_embed.pose_softmax
    assert model.patch_embed.use_null
    assert torch.all(model.patch_embed.null_score == 0)
    assert model.look_bank.null_score.shape == (36,)
    assert torch.all(model.look_bank.null_score == 0)


def test_half_six_look_model_uses_matching_twelve_pose_period():
    model = _build(
        "rot_hex_harmonic_look",
        rot_directions=6,
        rot_global_directions=12,
        rot_null_initial_score=0.0,
    )
    assert model.pos_embed is None
    assert model.patch_embed.directions == 6
    assert model.patch_embed.direction_coefficients.shape == (6, 2)
    assert model.look_bank.source_directions == 6
    assert model.look_bank.source_direction_period == 12
    assert model.look_bank.ring_sampler.angular_bins == 24
    assert model.look_bank.ring_sampler.rotation_samples == 12
    assert model.look_direction_bins == 12
    assert model.look_radial_bins == 4
    assert model.look_bank.look_grid.shape == (36, 4, 12)


def test_half_four_look_model_keeps_eight_by_four_look_field():
    model = _build(
        "rot_hex_harmonic_pe_look",
        rot_directions=4,
        rot_global_directions=8,
        rot_null_initial_score=0.0,
    )
    assert model.look_direction_bins == 8
    assert model.look_radial_bins == 4
    assert model.look_bank.look_grid.shape == (36, 4, 8)


def test_half_six_compact_r3_look_shares_tokenizer_variable_ring_storage():
    model = _build(
        "rot_hex_harmonic_pe_look",
        rot_directions=6,
        rot_global_directions=12,
        rot_angular_bins_per_radius=3,
        look_compact_variable_rings=True,
        rot_null_initial_score=0.0,
    )
    expected_counts = torch.arange(3, 37, 3)

    assert model.patch_embed.directions == 6
    assert model.look_bank.source_directions == 6
    assert model.look_bank.source_direction_period == 12
    assert model.look_bank.compact_variable_rings
    torch.testing.assert_close(model.patch_embed.ring_counts, expected_counts)
    torch.testing.assert_close(model.look_bank.ring_counts, expected_counts)
    assert model.patch_embed.prototype.shape == (96, 3, 234)
    assert model.look_bank.match_prototype.shape == (36, 3, 234)
    for geometry in model.look_bank.compact_geometries:
        torch.testing.assert_close(
            geometry.patch_centers_xy,
            model.patch_embed.patch_centers_xy,
        )


def test_relative_l1_harmonic_uses_zero_baseline_and_negative_null():
    embed = HexRotatingHarmonicPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=12,
        lattice_stride=8,
        kernel_sizes=(12, 6),
        bases=6,
        directions=4,
        global_directions=8,
        radial_bins=4,
        prototype_chunk_size=2,
        pose_softmax=True,
        use_null=True,
        null_initial_score=-0.1,
        match_metric="relative_l1",
    )
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = embed(image)
    assert output.shape == (2, embed.num_patches, 12)
    assert embed.match_metric == "relative_l1"
    assert torch.allclose(embed.null_score, torch.full((6,), -0.1))
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(embed.prototype.grad).all()
    assert torch.isfinite(embed.null_score.grad).all()


def test_relative_l1_harmonic_score_is_improvement_over_zero_prototype():
    embed = HexRotatingHarmonicPatchEmbed(
        img_size=24,
        in_chans=3,
        embed_dim=2,
        lattice_stride=8,
        kernel_sizes=(12,),
        bases=1,
        directions=1,
        global_directions=8,
        radial_bins=4,
        prototype_chunk_size=1,
        match_metric="relative_l1",
    )
    image = torch.randn(1, 3, 24, 24)
    patch = embed.geometries[0](image.float())
    rendered = embed.renderers[0](embed.prototype)
    cover = embed.scale_cover_0
    expected = (
        patch.abs() * cover[None, None, None]
    ).sum((2, 3)) - (
        (patch[:, :, None, None] - rendered[None, None]).abs()
        * cover[None, None, None, None, None]
    ).sum((4, 5)).squeeze((2, 3))
    actual = embed(image)[..., 0] - embed.output_bias[0]
    # cdist(p=1) uses a different parallel reduction order than the explicit
    # reference sum, so allow the expected float32 accumulation difference.
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_relative_l1_harmonic_model_keeps_hex_token_and_pe_shapes():
    model = _build("rot_hex_harmonic_l1_softmax_pe")
    assert isinstance(model.patch_embed, HexRotatingHarmonicPatchEmbed)
    assert model.patch_embed.match_metric == "relative_l1"
    assert model.patch_embed.pose_softmax
    assert model.patch_embed.use_null
    assert model.patch_embed.num_patches == 195
    assert model.pos_embed.shape == (1, 196, 192)


def test_grouped_compensated_model_keeps_large_cover_mass_for_both_scales():
    model = _build("rot_hex_dot_grouped_compensated_pe")
    embed = model.patch_embed
    assert isinstance(embed, HexRotatingGroupedDotPatchEmbed)
    assert embed.compensate_small_scales
    assert torch.allclose(embed.scale_cover_0.sum(), embed.scale_cover_1.sum())
    assert embed.scale_cover_0.sum() > 100
    assert embed.group_dim == 64
    assert embed.bases_per_group == 32
    assert embed.null_score.shape == (96,)
    assert model.pos_embed.shape == (1, 196, 192)
