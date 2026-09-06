import json
from pathlib import Path

from experiments.imagenet100.train_vit import TrainConfig
from model.deit_tiny_rot_hex_look import DeiTTinyRotHexLook


def test_half7_dual_look_field_override():
    path = Path(__file__).resolve().parents[1] / 'configs/imagenet100/half7d5r_pe_dual_look_g3_field4x12_ddp_e20.json'
    c = TrainConfig(**json.loads(path.read_text()))
    m = DeiTTinyRotHexLook(
        image_size=48, use_pos_embed=True, directions=c.rot_directions,
        global_directions=c.rot_global_directions,
        angular_bins_per_radius=c.rot_angular_bins_per_radius,
        look_compact_variable_rings=c.look_compact_variable_rings,
        image_look_field_direction_bins=c.image_look_field_direction_bins,
        feature_look_field_direction_bins=c.feature_look_field_direction_bins,
        center_pose_grid_look=True,
        center_look_layers_per_probe=c.center_look_layers_per_probe,
    )
    assert m.look_bank.look_grid.shape == (36, 4, 12)
    assert m.look_bank.match_prototype.shape == (36, 3, 390)
    assert m.look_bank.source_direction_period == 14
    assert m.center_look.look_grid.shape == (11, 3, 4, 12)
    assert m.center_look.axis_weight.shape == (4, 3, 7, 32, 2)
    assert m.center_look.layers_per_probe == 3
    assert not c.rot_progressive_differentiation
    assert not c.sparse_hex_look
