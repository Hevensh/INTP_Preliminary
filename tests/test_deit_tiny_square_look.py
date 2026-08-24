import torch
import timm

from model.deit_tiny_square_look import DeiTTinySquareLook, load_timm_deit_tiny


def test_zero_gain_look_is_exactly_no_look() -> None:
    torch.manual_seed(3)
    plain = DeiTTinySquareLook(num_classes=10, enable_look=False).eval()
    look = DeiTTinySquareLook(num_classes=10, enable_look=True).eval()
    report = load_timm_deit_tiny(look, plain.state_dict())
    assert "patch_embed.proj.weight" in report.copied_keys
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = plain(image)
        actual = look(image)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_transferred_model_matches_timm_deit() -> None:
    torch.manual_seed(4)
    source = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=10).eval()
    target = DeiTTinySquareLook(num_classes=10, enable_look=True).eval()
    load_timm_deit_tiny(target, source.state_dict())
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = source(image)
        actual = target(image)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_absolute_position_embedding_can_be_removed() -> None:
    model = DeiTTinySquareLook(num_classes=10, enable_look=False, use_pos_embed=False).eval()
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(image)
    assert output.shape == (1, 10)


def test_each_layer_and_head_has_an_independent_gain() -> None:
    model = DeiTTinySquareLook(num_classes=10)
    assert model.look_bank is not None
    assert model.look_bank.rank == 8
    assert model.look_bank.pose.match_prototype.shape[:2] == (8, 3)
    assert model.look_bank.pose.look_prototype_logits.shape[:2] == (8, 1)
    assert model.look_bank.head_mix.shape == (36, 8)
    assert model.look_bank.head_gain.shape == (36,)


def test_nonzero_bias_has_cls_border_zero() -> None:
    model = DeiTTinySquareLook(num_classes=10)
    assert model.look_bank is not None
    layer = model.look_bank
    layer.head_gain.data.fill_(0.1)
    image = torch.randn(1, 3, 224, 224)
    rings, coverage = layer.extract_rings(image)
    bias, response = layer.forward_rings(rings, coverage)
    assert bias.shape == (1, 36, 197, 197)
    assert response.shape == (1, 196, 8, 32)
    assert torch.count_nonzero(bias[:, :, 0]).item() == 0
    assert torch.count_nonzero(bias[:, :, :, 0]).item() == 0
