import pytest
import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.hex_linear_patch_embed import HexLinearPatchEmbed
from layers.hex_rotating_polar_patch_embed import HexRotatingPolarPatchEmbed
from layers.hex_rotating_dot_patch_embed import HexRotatingDotPatchEmbed
from layers.hex_rotating_grouped_dot_patch_embed import HexRotatingGroupedDotPatchEmbed
from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed


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
