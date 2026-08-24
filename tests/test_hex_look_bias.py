import math

import torch

from layers.hex_look_bias import HexLookBias
from layers.mini_vit import MultiHeadSelfAttention
from model.hex_vit_classifier import HexViTClassifier, build_hex_patch_coordinates
from utils.hex_graph import hex_relative_bins


def test_hex_relative_bins_are_directed() -> None:
    coordinates = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
        ]
    )
    rings, directions = hex_relative_bins(coordinates)
    assert rings[0, 1].item() == 1
    assert rings[1, 0].item() == 1
    assert directions[0, 1].item() == 0
    assert directions[1, 0].item() == 3
    assert directions[0, 2].item() == 1


def test_look_bias_can_weight_reverse_directions_differently() -> None:
    coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    look = HexLookBias(coordinates, num_heads=1)
    with torch.no_grad():
        look.weight[0, 1, 0] = 2.0
        look.weight[0, 1, 3] = -1.0
    bias = look(include_cls=False)
    assert bias[0, 0, 1].item() == 2.0
    assert bias[0, 1, 0].item() == -1.0


def test_coordinates_match_expected_patch_count() -> None:
    coordinates = build_hex_patch_coordinates((32, 32), 16)
    assert coordinates.shape == (27, 2)


def test_coordinates_match_hexconv_token_order() -> None:
    model = HexViTClassifier(
        img_size=32,
        embed_dim=24,
        depth=1,
        num_heads=4,
        hex_kernel_size=16,
        position_mode="look",
    )
    model.patch_embed(torch.rand(1, 3, 32, 32))
    stored = model.patch_embed.hexconv.coo_patchs
    expected = build_hex_patch_coordinates((32, 32), 16)
    assert torch.allclose(torch.view_as_real(stored), expected)


def test_hex_vit_position_modes_forward() -> None:
    images = torch.rand(2, 3, 32, 32)
    for mode in ("learned", "look", "learned+look", "none"):
        model = HexViTClassifier(
            img_size=32,
            embed_dim=24,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            hex_kernel_size=16,
            position_mode=mode,
        )
        logits = model(images)
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()


def test_look_bias_receives_gradient() -> None:
    model = HexViTClassifier(
        img_size=32,
        embed_dim=24,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        hex_kernel_size=16,
        position_mode="look",
    )
    model(torch.rand(2, 3, 32, 32)).sum().backward()
    assert model.look_bias is not None
    assert model.look_bias.weight.grad is not None
    assert torch.isfinite(model.look_bias.weight.grad).all()


def test_attention_accepts_equivalent_static_and_batch_bias() -> None:
    torch.manual_seed(7)
    attention = MultiHeadSelfAttention(dim=12, num_heads=3).eval()
    tokens = torch.randn(2, 5, 12)
    static_bias = torch.randn(3, 5, 5)
    batch_bias = static_bias.unsqueeze(0).expand(2, -1, -1, -1)

    static_output = attention(tokens, attn_bias=static_bias)
    batch_output = attention(tokens, attn_bias=batch_bias)
    assert torch.allclose(static_output, batch_output, atol=1e-6, rtol=1e-5)


def test_hex_vit_dynamic_polar_modes_forward() -> None:
    images = torch.rand(1, 3, 32, 32)
    for mode in ("polar-look", "learned+polar-look"):
        model = HexViTClassifier(
            img_size=32,
            embed_dim=12,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            hex_kernel_size=16,
            position_mode=mode,
        )
        logits = model(images)
        assert logits.shape == (1, 10)
        assert torch.isfinite(logits).all()


def test_zero_polar_strength_matches_model_without_position_bias() -> None:
    torch.manual_seed(11)
    baseline = HexViTClassifier(
        img_size=32,
        embed_dim=12,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        hex_kernel_size=16,
        position_mode="none",
    ).eval()
    polar = HexViTClassifier(
        img_size=32,
        embed_dim=12,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        hex_kernel_size=16,
        position_mode="polar-look",
        polar_look_strength=0.0,
    ).eval()
    polar.load_state_dict(baseline.state_dict(), strict=False)

    images = torch.rand(1, 3, 32, 32)
    assert torch.allclose(baseline(images), polar(images), atol=1e-7, rtol=1e-6)


def test_zero_polar_gate_preserves_learned_position_baseline() -> None:
    torch.manual_seed(13)
    baseline = HexViTClassifier(
        img_size=32,
        embed_dim=12,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        hex_kernel_size=16,
        position_mode="learned",
    ).eval()
    polar = HexViTClassifier(
        img_size=32,
        embed_dim=12,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        hex_kernel_size=16,
        position_mode="learned+polar-look",
        polar_look_gate_init=0.0,
    ).eval()
    polar.load_state_dict(baseline.state_dict(), strict=False)

    images = torch.rand(1, 3, 32, 32)
    assert polar.polar_look_gate.shape == (2, 2)
    assert torch.count_nonzero(polar.polar_look_gate) == 0
    assert torch.allclose(baseline(images), polar(images), atol=1e-7, rtol=1e-6)


def test_dynamic_polar_bias_receives_gradient() -> None:
    model = HexViTClassifier(
        img_size=32,
        embed_dim=12,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        hex_kernel_size=16,
        position_mode="polar-look",
    )
    model(torch.rand(1, 3, 32, 32)).square().mean().backward()
    assert model.polar_look is not None
    assert model.polar_look.match_prototype.grad is not None
    assert model.polar_look.look_prototype_logits.grad is not None
    assert torch.isfinite(model.polar_look.match_prototype.grad).all()
    assert torch.isfinite(model.polar_look.look_prototype_logits.grad).all()
