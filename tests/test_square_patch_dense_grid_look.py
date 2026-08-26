import torch

from layers.square_patch_dense_grid_look import SquarePatchDenseGridLook
from model.deit_tiny_square_look import DeiTTinySquareLook


def small_look(num_heads: int = 2) -> SquarePatchDenseGridLook:
    return SquarePatchDenseGridLook(
        image_size=48,
        patch_size=16,
        num_heads=num_heads,
        prototype_radial_bins=4,
        prototype_angular_bins=8,
        source_directions=4,
        source_direction_period=8,
        scales=(1.0, 0.5),
        prototype_radius=8.0,
        look_direction_bins=8,
        look_radial_bins=4,
        look_radius=2.0,
    )


def test_deit_dense_grid_has_one_prototype_and_grid_per_layer_head() -> None:
    model = DeiTTinySquareLook(num_classes=10, look_mode="dense_grid")
    look = model.look_bank
    assert isinstance(look, SquarePatchDenseGridLook)
    assert look.match_prototype.shape == (36, 3, 8, 16)
    assert look.look_grid.shape == (36, 4, 8)
    assert look.null_score.shape == (36,)


def test_null_route_is_dropped_without_real_pose_renormalization() -> None:
    look = small_look(1)
    image = torch.randn(1, 3, 48, 48)
    rings, coverage = look.extract_rings(image)
    with torch.no_grad():
        look.null_score.fill_(100.0)
        weights = look.pose_weights(rings, coverage)
    assert weights.shape == (1, 9, 1, 2, 4)
    assert weights.sum(dim=(-2, -1)).max().item() < 1e-10


def test_same_dense_grid_rotates_and_contracts_with_pose() -> None:
    look = small_look(1)
    with torch.no_grad():
        look.look_grid.zero_()
        look.look_grid[0, :, 0] = 1.0
    fields = look.transformed_look_grids()
    center, right, down_right = 4, 5, 8
    # Direction 0 points right; direction 1 is the same table rotated 45°.
    assert fields[0, 0, 0, center, right].item() > 0.9
    assert fields[0, 0, 1, center, down_right].item() > 0.9
    # A two-cell displacement is visible at scale 1 and removed when the same
    # Look table is contracted to scale 0.5.
    left, far_right = 3, 5
    assert fields[0, 0, 0, left, far_right].item() > 0.0
    assert fields[0, 1, 0, left, far_right].item() == 0.0


def test_dense_grid_and_match_prototype_receive_gradients() -> None:
    torch.manual_seed(5)
    look = small_look(2)
    with torch.no_grad():
        look.look_grid.normal_(std=0.01)
    bias, weights = look(torch.randn(1, 3, 48, 48), include_cls=True)
    assert bias.shape == (1, 2, 10, 10)
    assert weights.shape == (1, 9, 2, 2, 4)
    bias.square().mean().backward()
    for parameter in (look.match_prototype, look.look_grid, look.null_score):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_pose_softmax_promotes_half_precision_match_scores_to_float32(
    monkeypatch,
) -> None:
    look = small_look(2)
    response = torch.randn(
        1,
        look.num_patches,
        look.num_heads,
        look.num_scales,
        look.source_directions,
        dtype=torch.float16,
    )
    monkeypatch.setattr(
        look,
        "raw_pose_response",
        lambda rings, coverage: response,
    )
    weights = look.pose_weights(torch.empty(0), torch.empty(0))
    assert weights.dtype == torch.float32
    assert weights.shape == response.shape
    assert torch.isfinite(weights).all()
