import torch

from layers import PolarRingSampler


def test_polar_ring_sampler_shapes_and_constant_image() -> None:
    sampler = PolarRingSampler(
        radial_bins=5,
        angular_bins=24,
        rotation_samples=12,
        scales=(0.8, 1.0, 1.25),
    )
    image = torch.full((2, 3, 25, 25), 0.37)
    centers = torch.tensor([[12.0, 12.0], [8.0, 15.0]])
    rings, coverage = sampler(image, centers, base_radius=5.0, return_coverage=True)
    assert rings.shape == (2, 2, 3, 3, 5, 24)
    assert coverage.shape == (3, 5, 24)
    covered = coverage > 1e-6
    assert torch.allclose(
        rings.permute(0, 1, 2, 4, 5, 3)[..., covered.permute(1, 2, 0)],
        torch.full_like(rings.permute(0, 1, 2, 4, 5, 3)[..., covered.permute(1, 2, 0)], 0.37),
        atol=1e-5,
    )
    # Outer polar cells aggregate a larger Cartesian footprint than inner cells.
    assert coverage[-1, -1].mean() > coverage[-1, 0].mean()


def test_circular_match_recovers_known_angular_shift() -> None:
    torch.manual_seed(3)
    sampler = PolarRingSampler(
        radial_bins=4,
        angular_bins=24,
        rotation_samples=12,
        scales=(1.0,),
    )
    prototype = torch.randn(1, 2, 4, 24)
    target_rotation = 7
    angular_shift = target_rotation * 2
    shifted = torch.roll(prototype[0], shifts=angular_shift, dims=-1)
    rings = shifted[None, None, :, None]
    response = sampler.circular_match(rings, prototype)
    assert response.shape == (1, 1, 1, 1, 12)
    assert response[0, 0, 0, 0].argmax().item() == target_rotation
    assert response[0, 0, 0, 0, target_rotation].item() > 0.999


def test_partial_rotation_match_is_exact_prefix_of_full_period() -> None:
    torch.manual_seed(4)
    sampler = PolarRingSampler(
        radial_bins=4,
        angular_bins=24,
        rotation_samples=12,
        scales=(1.0, 0.5),
    )
    rings = torch.randn(2, 3, 2, 2, 4, 24)
    prototype = torch.randn(5, 2, 4, 24)
    coverage = torch.ones(2, 4, 24)
    full = sampler.circular_match(rings, prototype, coverage)
    half = sampler.circular_match(
        rings, prototype, coverage, rotation_count=6
    )
    assert half.shape == (2, 3, 5, 2, 6)
    torch.testing.assert_close(half, full[..., :6])


def test_sampling_and_circular_match_are_differentiable() -> None:
    torch.manual_seed(5)
    sampler = PolarRingSampler(
        radial_bins=4,
        angular_bins=12,
        rotation_samples=12,
        scales=(0.8, 1.3),
    )
    image = torch.randn(1, 3, 21, 21, requires_grad=True)
    centers = torch.tensor([[10.0, 10.0], [7.5, 12.25]], requires_grad=True)
    prototype = torch.randn(2, 3, 4, 12, requires_grad=True)
    rings, coverage = sampler(image, centers, base_radius=5.0, return_coverage=True)
    response = sampler.circular_match(rings, prototype, coverage)
    response.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert centers.grad is not None and torch.isfinite(centers.grad).all()
    assert prototype.grad is not None and torch.isfinite(prototype.grad).all()


def test_large_scale_point_count_does_not_increase_match_amplitude() -> None:
    sampler = PolarRingSampler(
        radial_bins=8,
        angular_bins=24,
        rotation_samples=12,
    )
    image = torch.full((1, 1, 35, 35), 0.6)
    centers = torch.tensor([[17.0, 17.0]])
    rings, coverage = sampler(image, centers, base_radius=7.0, return_coverage=True)
    prototype = torch.ones(1, 1, 8, 24)
    response = sampler.circular_match(rings, prototype, coverage)
    assert torch.allclose(response, torch.full_like(response, 0.6), atol=1e-5)
    assert coverage[-1].sum() > coverage[0].sum() * 3.0


def test_match_preserves_input_ring_magnitude() -> None:
    sampler = PolarRingSampler(
        radial_bins=4,
        angular_bins=12,
        rotation_samples=12,
        scales=(1.0,),
    )
    prototype = torch.ones(1, 1, 4, 12)
    coverage = torch.ones(1, 4, 12)
    weak = torch.full((1, 1, 1, 1, 4, 12), 0.25)
    strong = weak * 3.0
    weak_response = sampler.circular_match(weak, prototype, coverage)
    strong_response = sampler.circular_match(strong, prototype, coverage)
    assert torch.allclose(strong_response, weak_response * 3.0, atol=1e-6)
