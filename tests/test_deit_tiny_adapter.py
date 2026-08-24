import torch

from layers.hex_linear_patch_embed import HexLinearPatchEmbed
from model.deit_tiny_adapter import interpolate_deit_pos_embed, load_deit_tiny_state_dict
from model.hex_vit_classifier import HexViTClassifier, estimate_hex_num_patches
from train_func.deit_tiny_hex224 import (
    build_hex224_model,
    configure_train_stage,
    load_hex_stage_checkpoint,
)


def test_hex224_geometry_stays_close_to_deit_patch_count() -> None:
    assert estimate_hex_num_patches((224, 224), 21, 18) == 195
    embed = HexLinearPatchEmbed(
        img_size=224,
        in_chans=3,
        embed_dim=8,
        kernel_size=21,
        lattice_stride=18,
    )
    assert embed.num_patches == 195
    assert embed.patch_centers_xy.shape == (195, 2)


def test_vit_patch_projection_resampling_preserves_output_filter_norm() -> None:
    torch.manual_seed(0)
    embed = HexLinearPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=8,
        kernel_size=4,
        lattice_stride=3,
    )
    source_weight = torch.randn(8, 3, 16, 16)
    source_bias = torch.randn(8)
    embed.load_vit_patch_projection(source_weight, source_bias)

    source_norm = source_weight.flatten(1).norm(dim=1)
    target_norm = embed.weight.flatten(1).norm(dim=1)
    torch.testing.assert_close(target_norm, source_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(embed.bias, source_bias)


def test_position_interpolation_keeps_cls_and_targets_hex_count() -> None:
    pos_embed = torch.randn(1, 197, 192)
    embed = HexLinearPatchEmbed(
        img_size=224,
        in_chans=3,
        embed_dim=192,
        kernel_size=21,
        lattice_stride=18,
    )
    target = interpolate_deit_pos_embed(pos_embed, embed.patch_centers_xy, (224, 224))
    assert target.shape == (1, 196, 192)
    torch.testing.assert_close(target[:, :1], pos_embed[:, :1])
    assert torch.isfinite(target).all()


def test_plain_timm_deit_tiny_state_dict_maps_to_hex_model() -> None:
    timm = __import__("timm")
    source = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    target = HexViTClassifier(
        img_size=224,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
        hex_kernel_size=21,
        hex_stride=18,
        patch_embed_mode="linear",
        position_mode="learned",
        polar_look_strength=0.0,
    )

    report = load_deit_tiny_state_dict(target, source.state_dict())
    assert len(report.copied_keys) == 150
    assert report.skipped_keys == ("head.bias", "head.weight")
    assert target.pos_embed.shape == (1, 196, 192)
    torch.testing.assert_close(target.blocks[0].attn.qkv.weight, source.blocks[0].attn.qkv.weight)
    torch.testing.assert_close(target.blocks[-1].mlp.fc2.bias, source.blocks[-1].mlp.fc2.bias)
    torch.testing.assert_close(target.norm.weight, source.norm.weight)


def test_tokenizer_stage_never_unfreezes_transformer_blocks() -> None:
    model = HexViTClassifier(
        img_size=224,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
        hex_kernel_size=21,
        hex_stride=18,
        patch_embed_mode="linear",
        position_mode="learned",
        polar_look_strength=0.0,
    )
    trainable = configure_train_stage(model, "tokenizer")
    assert set(trainable) == {
        "pos_embed",
        "patch_embed.weight",
        "patch_embed.bias",
        "head.weight",
        "head.bias",
    }
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 240_778
    assert not any(parameter.requires_grad for block in model.blocks for parameter in block.parameters())
    assert not any(parameter.requires_grad for parameter in model.norm.parameters())

    tokenizer_only = configure_train_stage(model, "tokenizer", train_head=False)
    assert set(tokenizer_only) == {
        "pos_embed",
        "patch_embed.weight",
        "patch_embed.bias",
    }
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 238_848


def test_look_stage_only_trains_polar_fields_and_zero_initialized_gates() -> None:
    model = HexViTClassifier(
        img_size=32,
        num_classes=10,
        embed_dim=24,
        depth=2,
        num_heads=3,
        hex_kernel_size=16,
        position_mode="learned+polar-look",
        polar_look_gate_init=0.0,
    )
    trainable = configure_train_stage(model, "look", train_head=False)
    assert "polar_look_gate" in trainable
    assert len(trainable) == 5
    assert all(
        name == "polar_look_gate" or name.startswith("polar_look_layers.")
        for name in trainable
    )
    assert torch.count_nonzero(model.polar_look_gate) == 0
    assert len(model.polar_look_layers) == 2
    assert (
        model.polar_look_layers[0].match_prototype.data_ptr()
        != model.polar_look_layers[1].match_prototype.data_ptr()
    )
    assert not model.pos_embed.requires_grad
    assert not any(parameter.requires_grad for block in model.blocks for parameter in block.parameters())


def test_full_hex_stage_checkpoint_restores_position_and_every_look_layer(tmp_path) -> None:
    source = HexViTClassifier(
        img_size=32,
        embed_dim=24,
        depth=2,
        num_heads=3,
        hex_kernel_size=16,
        position_mode="learned+polar-look",
        polar_look_gate_init=0.25,
    )
    checkpoint = tmp_path / "hex_stage.pt"
    torch.save(
        {
            "patch_embed": source.patch_embed.state_dict(),
            "pos_embed": source.pos_embed.detach().clone(),
            "polar_look_layers": source.polar_look_layers.state_dict(),
            "polar_look_gate": source.polar_look_gate.detach().clone(),
        },
        checkpoint,
    )
    target = HexViTClassifier(
        img_size=32,
        embed_dim=24,
        depth=2,
        num_heads=3,
        hex_kernel_size=16,
        position_mode="learned+polar-look",
        polar_look_gate_init=0.0,
    )
    assert load_hex_stage_checkpoint(target, checkpoint) == "full-stage"
    torch.testing.assert_close(target.pos_embed, source.pos_embed)
    torch.testing.assert_close(target.polar_look_gate, source.polar_look_gate)
    torch.testing.assert_close(
        target.polar_look_layers[1].match_prototype,
        source.polar_look_layers[1].match_prototype,
    )


def test_random_tokenizer_ablation_keeps_backbone_and_position_transfer_fixed() -> None:
    torch.manual_seed(7)
    transferred, _ = build_hex224_model(pretrained=False, tokenizer_init="transferred")
    torch.manual_seed(7)
    random_tokenizer, _ = build_hex224_model(pretrained=False, tokenizer_init="random")

    torch.testing.assert_close(
        transferred.blocks[0].attn.qkv.weight,
        random_tokenizer.blocks[0].attn.qkv.weight,
    )
    torch.testing.assert_close(transferred.pos_embed, random_tokenizer.pos_embed)
    assert not torch.equal(transferred.patch_embed.weight, random_tokenizer.patch_embed.weight)

    torch.manual_seed(7)
    norm_matched, _ = build_hex224_model(pretrained=False, tokenizer_init="random_norm_matched")
    torch.testing.assert_close(
        transferred.patch_embed.weight.flatten(1).norm(dim=1),
        norm_matched.patch_embed.weight.flatten(1).norm(dim=1),
    )
    torch.testing.assert_close(transferred.patch_embed.bias, norm_matched.patch_embed.bias)
    assert not torch.equal(transferred.patch_embed.weight, norm_matched.patch_embed.weight)
