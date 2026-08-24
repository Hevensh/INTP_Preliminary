import torch

from layers.square_patch_low_rank_look import (
    SquarePatchLowRankLook,
    build_square_patch_centers,
)


def test_square_patch_centers_follow_vit_row_major_order() -> None:
    pixel, grid = build_square_patch_centers(32, 16)
    torch.testing.assert_close(
        grid,
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    torch.testing.assert_close(pixel[0], torch.tensor([7.5, 7.5]))
    torch.testing.assert_close(pixel[-1], torch.tensor([23.5, 23.5]))


def test_pose_masks_hide_center_and_normalize_every_edge_query() -> None:
    module = SquarePatchLowRankLook(
        image_size=80,
        patch_size=16,
        in_channels=1,
        num_heads=3,
        angular_bins=8,
        direction_samples=8,
    )
    masks = module.pose_masks()
    assert masks.shape == (32, 25, 25)
    diagonal = masks.diagonal(dim1=-2, dim2=-1)
    assert torch.count_nonzero(diagonal) == 0
    torch.testing.assert_close(
        masks.sum(dim=-1),
        torch.ones(32, 25),
        atol=1e-6,
        rtol=1e-6,
    )


def test_eight_square_directions_rotate_the_peak_by_quarter_turn() -> None:
    module = SquarePatchLowRankLook(
        image_size=80,
        patch_size=16,
        in_channels=1,
        num_heads=1,
        radial_bins=5,
        angular_bins=8,
        direction_samples=8,
        scales=(1.0,),
        look_radius=2.0,
    )
    with torch.no_grad():
        module.pose.look_prototype_logits.fill_(-12.0)
        module.pose.look_prototype_logits[0, 0, -1, 0] = 12.0

    masks = module.pose_masks()
    center = 12
    assert masks[0, center].argmax().item() == 14  # positive x
    assert masks[2, center].argmax().item() == 22  # positive y


def test_signed_response_and_l2_mix_preserve_head_amplitude() -> None:
    module = SquarePatchLowRankLook(
        image_size=32,
        patch_size=16,
        in_channels=1,
        num_heads=2,
        angular_bins=8,
        direction_samples=4,
        scales=(1.0,),
        look_radius=1.5,
    )
    with torch.no_grad():
        module.head_mix.zero_()
        module.head_mix[0, 0] = 1.0
        module.head_mix[1].fill_(1.0)
        module.head_gain.fill_(1.0)
    response = torch.zeros(1, 4, 4)
    response[:, :, 0] = 0.5
    positive = module.mix_head_bias(module.pose_bias(response), include_cls=False)
    negative = module.mix_head_bias(module.pose_bias(-response), include_cls=False)
    torch.testing.assert_close(negative, -positive)
    assert torch.isfinite(positive).all()
    torch.testing.assert_close(module.normalized_head_mix.norm(dim=-1), torch.ones(2))


def test_image_forward_shapes_and_gradients() -> None:
    module = SquarePatchLowRankLook(
        image_size=32,
        patch_size=16,
        in_channels=3,
        num_heads=3,
        angular_bins=8,
        direction_samples=4,
        scales=(1.0, 2.0**0.5),
        look_radius=1.5,
    )
    with torch.no_grad():
        module.head_gain.fill_(1.0)
    head_bias, response = module(torch.randn(2, 3, 32, 32), include_cls=True)
    assert response.shape == (2, 4, 8)
    assert head_bias.shape == (2, 3, 5, 5)
    head_bias.square().mean().backward()
    assert module.pose.match_prototype.grad is not None
    assert module.pose.look_prototype_logits.grad is not None
    assert module.head_mix.grad is not None
    assert module.head_gain.grad is not None
    assert torch.isfinite(head_bias).all()
