import pytest
import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.center_pose_grid_look import CenterPoseGridLook


@pytest.mark.parametrize(
    ("layers_per_probe", "probe_groups"),
    [(1, 11), (2, 6), (3, 4), (4, 3), (6, 2)],
)
def test_center_grid_look_shares_probe_but_keeps_one_grid_per_layer(
    layers_per_probe: int,
    probe_groups: int,
):
    coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [-0.5, 0.866]]
    )
    module = CenterPoseGridLook(
        coordinates=coordinates,
        embed_dim=12,
        num_heads=3,
        depth=11,
        axes=6,
        layers_per_probe=layers_per_probe,
    )
    assert module.look_grid.shape == (11, 3, 4, 12)
    pose = module.pose_weights(torch.randn(2, 4, 12))
    assert module.axis_weight.shape == (probe_groups, 3, 6, 2, 2)
    assert pose.shape == (2, 4, probe_groups, 3, 6)
    assert module.pose_for_layer(pose, 0).shape == (2, 4, 3, 6)
    assert module.pose_for_layer(pose, 10).shape == (2, 4, 3, 6)
    for layer in range(11):
        expected_group = layer // layers_per_probe
        assert torch.equal(module.pose_for_layer(pose, layer), pose[:, :, expected_group])
    assert module.fields(0, dtype=torch.float32).shape == (3, 6, 4, 4)
    assert module.fields(10, dtype=torch.float32).shape == (3, 6, 4, 4)


def test_center_grid_model_replaces_image_look_with_four_by_twelve_templates():
    model = build_imagenet100_model(
        variant="rot_hex_harmonic_pe_center_grid_look",
        model_name="deit_tiny_patch16_224",
        pretrained=False,
        num_classes=100,
        image_size=224,
        rot_directions=6,
        rot_global_directions=12,
        rot_angular_bins_per_radius=3,
        look_compact_variable_rings=True,
        center_look_layers_per_probe=3,
        rot_null_initial_score=0.0,
    )
    assert model.look_bank is None
    assert model.center_pose_grid_look
    assert model.center_look.look_grid.shape == (11, 3, 4, 12)
    assert model.center_look.axis_weight.shape == (4, 3, 6, 32, 2)
    diagnostics = model.experiment_diagnostics()["center_pose_grid_look"]
    assert diagnostics["layers_per_probe"] == 3
    assert diagnostics["probe_groups"] == 4
    assert diagnostics["look_grid_shape"] == [11, 3, 4, 12]
