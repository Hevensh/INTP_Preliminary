import pytest
import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.hex_linear_patch_embed import HexLinearPatchEmbed
from layers.hex_rotating_polar_patch_embed import HexRotatingPolarPatchEmbed


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
