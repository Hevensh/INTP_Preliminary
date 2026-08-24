import math

import torch

from layers.HexConv import HexConv2D
from layers.polar_prototype_look import PolarPrototypeLook


def _ring_coordinates() -> torch.Tensor:
    angles = torch.arange(6, dtype=torch.float32) * (math.pi / 3.0)
    ring = torch.stack((angles.cos(), angles.sin()), dim=-1)
    return torch.cat((torch.zeros(1, 2), ring), dim=0)


def test_hexconv_exposes_unencoded_p0_and_coordinates() -> None:
    layer = HexConv2D(in_channels=3, out_channels=4, kernel_size=16)
    patches, coordinates = layer.extract_patches(torch.rand(2, 3, 32, 32))
    assert patches.shape == (2, 27, 3, layer.idx_x.numel())
    assert coordinates.shape == (27,)
    assert torch.is_complex(coordinates)
    assert layer.patch_offsets_xy.shape == (layer.idx_x.numel(), 2)


def test_square_polar_match_sampling_excludes_center() -> None:
    module = PolarPrototypeLook(
        num_heads=1,
        in_channels=1,
        radial_bins=4,
        angular_bins=6,
        rotation_samples=6,
        scales=(1.0,),
        rho_min=0.2,
    )
    field = torch.ones(1, 1, 4, 6)
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    sampled = module._sample_square_polar_field(field, points, base_radius=1.0)
    assert sampled.shape == (1, 1, 6, 1, 2)
    assert torch.count_nonzero(sampled[..., 0]) == 0
    assert torch.all(sampled[..., 1] > 0)


def test_square_polar_sampling_keeps_inner_ring_empty_at_every_scale() -> None:
    module = PolarPrototypeLook(
        num_heads=1,
        in_channels=1,
        radial_bins=3,
        angular_bins=4,
        rotation_samples=1,
        scales=(0.5, 1.0),
        rho_min=0.125,
    )
    field = torch.ones(1, 1, 3, 4)
    sampled = module._sample_square_polar_field(
        field,
        torch.tensor([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [0.125, 0.0]]),
        base_radius=1.0,
    )
    # The physical blind radius is scale * rho_min: 0.0625 and 0.125 here.
    assert torch.count_nonzero(sampled[..., 0]) == 0
    assert sampled[0, 0, 0, 0, 1].item() == 0.0
    assert sampled[0, 1, 0, 0, 2].item() == 0.0
    assert sampled[0, 0, 0, 0, 2].item() > 0.0
    assert sampled[0, 1, 0, 0, 3].item() > 0.0


def test_rotated_match_votes_for_rotated_look_region() -> None:
    module = PolarPrototypeLook(
        num_heads=1,
        in_channels=1,
        radial_bins=5,
        angular_bins=6,
        rotation_samples=6,
        scales=(1.0,),
        rho_min=0.1,
        match_temperature=0.05,
    )
    with torch.no_grad():
        module.look_prototype_logits.fill_(-12.0)
        # Canonical look region: outer radius, direction zero (positive x).
        module.look_prototype_logits[0, 0, -1, 0] = 12.0

    coordinates = _ring_coordinates()
    response = torch.zeros((1, 7, 1, 1, 6))
    response[0, 0, 0, 0, 0] = 1.0
    look_zero = module.project_look(response, coordinates, look_radius=1.0)
    response[0, 0, 0, 0, 0] = 0.0
    response[0, 0, 0, 0, 1] = 1.0
    look_sixty = module.project_look(response, coordinates, look_radius=1.0)
    assert look_zero[0, 0, 0, 1:].argmax().item() == 0
    assert look_sixty[0, 0, 0, 1:].argmax().item() == 1
    assert torch.isfinite(look_zero[0, 0, 0, 0])


def test_default_look_region_is_directional_and_preserves_response_sign() -> None:
    module = PolarPrototypeLook(
        num_heads=1,
        in_channels=1,
        radial_bins=5,
        angular_bins=6,
        rotation_samples=6,
        scales=(1.0,),
    )
    field = torch.sigmoid(module.look_prototype_logits[0, 0])
    assert field[:, 0].max() > field[:, 3].max()

    coordinates = _ring_coordinates()
    positive = torch.zeros((1, 7, 1, 1, 6))
    positive[0, 0, 0, 0, 0] = 0.25
    positive_look = module.project_look(positive, coordinates, look_radius=1.0)
    negative_look = module.project_look(-positive, coordinates, look_radius=1.0)

    assert positive_look[0, 0, 0, 1:].argmax().item() == 0
    assert torch.allclose(negative_look, -positive_look, atol=1e-7, rtol=1e-6)


def test_match_and_project_shapes_and_gradients() -> None:
    module = PolarPrototypeLook(
        num_heads=2,
        in_channels=3,
        radial_bins=4,
        angular_bins=6,
        rotation_samples=3,
        scales=(0.8, 1.0),
    )
    patches = torch.rand(2, 5, 3, 8)
    angles = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8.0)
    offsets = torch.stack((angles.cos(), angles.sin()), dim=-1)
    coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254], [-0.5, 0.8660254], [-1.0, 0.0]]
    )
    look, response = module(
        patches,
        offsets,
        coordinates,
        patch_radius=1.0,
        look_radius=2.0,
    )
    assert response.shape == (2, 5, 2, 2, 3)
    assert look.shape == (2, 2, 5, 5)
    (look.sum() + response.sum()).backward()
    assert module.match_prototype.grad is not None
    assert module.look_prototype_logits.grad is not None
    assert torch.isfinite(module.match_prototype.grad).all()
    assert torch.isfinite(module.look_prototype_logits.grad).all()


def _polar_sample_offsets(radial_count: int = 6, angular_count: int = 48) -> torch.Tensor:
    radii = torch.linspace(0.15, 1.0, radial_count)
    angles = torch.arange(angular_count, dtype=torch.float32) * (2.0 * math.pi / angular_count)
    radius_grid, angle_grid = torch.meshgrid(radii, angles, indexing="ij")
    return torch.stack(
        (radius_grid * angle_grid.cos(), radius_grid * angle_grid.sin()), dim=-1
    ).reshape(-1, 2)


def test_default_probe_grid_is_hex_aligned() -> None:
    module = PolarPrototypeLook(num_heads=1, in_channels=3)
    assert module.num_rotations == 12
    assert module.num_scales == 4
    assert math.isclose(
        float(module.rotations[2]), math.pi / 3.0, rel_tol=0.0, abs_tol=1e-6
    )
    assert torch.allclose(
        module.scales,
        torch.tensor([1.0, math.sqrt(2.0), math.sqrt(3.0), 2.0]),
        atol=1e-6,
    )


def test_radial_match_weight_hides_center_and_decays_outward() -> None:
    module = PolarPrototypeLook(num_heads=1, in_channels=3, rho_min=0.1)
    offsets = torch.tensor([[0.0, 0.0], [0.25, 0.0], [0.75, 0.0], [1.0, 0.0]])
    weight = module.radial_match_weight(offsets, patch_radius=1.0)
    assert weight[0].item() == 0.0
    assert weight[1] > weight[2] > weight[3]
    assert weight[3].item() < 1e-6


def test_real_match_recovers_known_rotation_and_scale_probe() -> None:
    torch.manual_seed(7)
    module = PolarPrototypeLook(
        num_heads=1,
        in_channels=3,
        radial_bins=8,
        angular_bins=12,
        rotation_samples=12,
    )
    offsets = _polar_sample_offsets()
    templates = module._sample_square_polar_field(
        module.match_prototype, offsets, base_radius=1.0
    )
    target_scale = 2
    target_rotation = 5
    patch = templates[0, target_scale, target_rotation].unsqueeze(0).unsqueeze(0)
    response = module.match(patch, offsets, patch_radius=1.0)
    best = response[0, 0, 0].reshape(-1).argmax().item()
    assert best == target_scale * module.num_rotations + target_rotation
    assert response[0, 0, 0, target_scale, target_rotation].item() > 0.999


def test_sixty_degree_rotation_moves_peak_by_two_slots() -> None:
    torch.manual_seed(11)
    module = PolarPrototypeLook(num_heads=1, in_channels=3)
    offsets = _polar_sample_offsets()
    templates = module._sample_square_polar_field(
        module.match_prototype, offsets, base_radius=1.0
    )
    target_scale = 1
    patch_zero = templates[0, target_scale, 0].unsqueeze(0).unsqueeze(0)
    patch_sixty = templates[0, target_scale, 2].unsqueeze(0).unsqueeze(0)
    response_zero = module.match(patch_zero, offsets, patch_radius=1.0)[0, 0, 0, target_scale]
    response_sixty = module.match(patch_sixty, offsets, patch_radius=1.0)[0, 0, 0, target_scale]
    assert response_zero.argmax().item() == 0
    assert response_sixty.argmax().item() == 2


def test_image_soft_splat_path_recovers_its_scale_and_rotation() -> None:
    torch.manual_seed(19)
    module = PolarPrototypeLook(num_heads=1, in_channels=1)
    image = torch.randn(1, 1, 41, 41)
    centers = torch.tensor([[20.0, 20.0]])
    _, rings, coverage = module.match_image(
        image,
        centers,
        base_radius=8.0,
        return_details=True,
    )
    target_scale = 2
    with torch.no_grad():
        module.match_prototype.copy_(rings[0, 0, :, target_scale].unsqueeze(0))
    response = module.match_image(image, centers, base_radius=8.0)
    best = response[0, 0, 0].reshape(-1).argmax().item()
    assert best == target_scale * module.num_rotations
    assert torch.isfinite(response).all()
    assert coverage[-1].sum() > coverage[0].sum()


def test_image_path_detaches_patch_preprocessing_by_default() -> None:
    module = PolarPrototypeLook(num_heads=1, in_channels=1)
    centers = torch.tensor([[10.0, 10.0]])
    image = torch.randn(1, 1, 21, 21, requires_grad=True)
    module.match_image(image, centers, base_radius=4.0).sum().backward()
    assert image.grad is None
    assert module.match_prototype.grad is not None

    module.zero_grad(set_to_none=True)
    tracked_image = image.detach().clone().requires_grad_(True)
    module.match_image(
        tracked_image,
        centers,
        base_radius=4.0,
        track_input_grad=True,
    ).sum().backward()
    assert tracked_image.grad is not None
