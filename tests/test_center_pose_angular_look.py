import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.center_pose_angular_look import CenterPoseAngularLook


def test_center_pose_look_groups_complete_pairs_and_keeps_null_mass():
    coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [-0.5, 0.866]]
    )
    module = CenterPoseAngularLook(
        coordinates=coordinates,
        embed_dim=12,
        num_heads=3,
        depth=2,
        axes=2,
    )
    tokens = torch.randn(2, 4, 12, requires_grad=True)
    probability = module.pose_weights(tokens)
    assert probability.shape == (2, 4, 3, 2)
    assert torch.all(probability.sum(dim=-1) < 1.0)
    assert module.fields(0, dtype=tokens.dtype).shape == (3, 2, 4, 4)
    (probability.square().mean() + module.fields(0, dtype=tokens.dtype).square().mean()).backward()
    assert torch.isfinite(tokens.grad).all()
    assert torch.isfinite(module.axis_weight.grad).all()
    assert torch.isfinite(module.layer_axis_gain.grad).all()


def test_center_pose_model_uses_six_axes_without_image_look_bank():
    model = build_imagenet100_model(
        variant="rot_hex_harmonic_pe_center_look",
        model_name="deit_tiny_patch16_224",
        pretrained=False,
        num_classes=100,
        image_size=224,
        hex_stride=18,
        rot_kernel_sizes=(24, 12),
        rot_bases=96,
        rot_directions=6,
        rot_global_directions=12,
        rot_angular_bins_per_radius=3,
        look_compact_variable_rings=True,
        rot_prototype_chunk_size=16,
        rot_null_initial_score=0.0,
    )
    assert model.look_bank is None
    assert model.center_look.axes == 6
    assert model.center_look.pairs_per_head == 32
    assert model.center_look.angular_fields.shape == (6, 195, 195)
    assert model.center_look.layer_axis_gain.shape == (12, 3, 6)
    assert model.look_radial_bins == 4  # tokenizer metadata remains unchanged
    diagnostics = model.experiment_diagnostics()["center_pose_angular_look"]
    assert diagnostics["directed_angles"] == 12
    assert len(diagnostics["layer_mean_abs_gain"]) == 12
