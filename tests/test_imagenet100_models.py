import pytest
import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.hex_linear_patch_embed import HexLinearPatchEmbed
from layers.hex_rotating_polar_patch_embed import HexRotatingPolarPatchEmbed
from layers.hex_rotating_dot_patch_embed import HexRotatingDotPatchEmbed


def _build(variant: str):
    return build_imagenet100_model(
        variant=variant,
        model_name="deit_tiny_patch16_224",
        pretrained=False,
        num_classes=100,
        image_size=224,
        hex_kernel_size=21,
        hex_stride=18,
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
