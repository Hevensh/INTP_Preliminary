import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.cartesian_rotating_harmonic_conv import (
    CartesianCircularPatchGeometry,
    CartesianRotatingHarmonicConv2d,
)
from model.resnet_mams import MAMSBasicBlock
from model.resnet_mams import FourValuePairedMAMSBasicBlock
from layers.cartesian_four_value_paired_mams import (
    CartesianFourValuePairedMAMSConv2d,
    ComplexPointwiseConv2d,
    PairedRMSNorm2d,
    _SharedWindowPatchExtractor,
)
from layers.hex_rotating_polar_patch_embed import _PolarRenderer
from layers.triton_polar_renderer import triton_polar_render


def test_cartesian_diameter_scales_share_pixel_centers():
    large = CartesianCircularPatchGeometry(8, diameter=6, stride=2)
    small = CartesianCircularPatchGeometry(8, diameter=3, stride=2)

    assert large.window_size == 7
    assert small.window_size == 3
    assert large.num_samples == 25
    assert small.num_samples == 9
    assert large.output_size(16, 16) == small.output_size(16, 16) == (8, 8)


def test_cartesian_mams_conv_forward_backward_and_storage():
    layer = CartesianRotatingHarmonicConv2d(
        8,
        12,
        diameters=(6, 3),
        stride=2,
        directions=4,
        global_directions=8,
        angular_bins_per_radius=4,
        prototype_chunk_size=3,
        use_null=True,
        null_initial_score=0.0,
        bias=False,
    )
    image = torch.randn(2, 8, 16, 16, requires_grad=True)
    output = layer(image)

    assert output.shape == (2, 12, 8, 8)
    assert layer.bases == 6
    assert layer.ring_counts.tolist() == [4, 8, 12]
    assert layer.prototype.shape == (6, 8, 24)
    assert layer.null_score.shape == (6,)
    assert layer.output_bias is None
    torch.testing.assert_close(
        layer.scale_cover_0.sum(), layer.scale_cover_1.sum()
    )
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert torch.isfinite(layer.prototype.grad).all()
    assert torch.isfinite(layer.null_score.grad).all()
    assert torch.isfinite(image.grad).all()


def test_resnet18_mams_replaces_whole_basic_blocks():
    baseline = build_imagenet100_model(
        variant="resnet18",
        model_name="resnet18",
        pretrained=False,
        num_classes=100,
        image_size=224,
    )
    model = build_imagenet100_model(
        variant="resnet18_mams",
        model_name="resnet18_mams_4d4r_d6d3",
        pretrained=False,
        num_classes=100,
        image_size=224,
        rot_kernel_sizes=(6, 3),
        rot_directions=4,
        rot_global_directions=8,
        rot_angular_bins_per_radius=4,
        rot_use_null=True,
        rot_null_initial_score=0.0,
    )

    blocks = [module for module in model.modules() if isinstance(module, MAMSBasicBlock)]
    assert len(blocks) == 8
    assert blocks[0].mams.in_channels == blocks[0].mams.out_channels == 64
    assert blocks[2].mams.in_channels == 64
    assert blocks[2].mams.out_channels == 128
    assert blocks[2].stride == 2
    assert sum(parameter.numel() for parameter in model.parameters()) < sum(
        parameter.numel() for parameter in baseline.parameters()
    )

    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 100)
    assert torch.isfinite(output).all()


def _rotate_pairs(x: torch.Tensor, angle: float) -> torch.Tensor:
    pair = x.reshape(x.shape[0], x.shape[1] // 2, 2, *x.shape[-2:])
    cosine = torch.cos(torch.tensor(angle, dtype=x.dtype))
    sine = torch.sin(torch.tensor(angle, dtype=x.dtype))
    real = pair[:, :, 0] * cosine - pair[:, :, 1] * sine
    imag = pair[:, :, 0] * sine + pair[:, :, 1] * cosine
    return torch.stack((real, imag), dim=2).reshape_as(x)


def test_shared_window_extractor_is_identical_to_old_geometry():
    geometries = torch.nn.ModuleList([
        CartesianCircularPatchGeometry(8, diameter=6, stride=2),
        CartesianCircularPatchGeometry(8, diameter=3, stride=2),
    ])
    image = torch.randn(2, 8, 16, 16)
    shared = _SharedWindowPatchExtractor(geometries)(image)
    for actual, geometry in zip(shared, geometries, strict=True):
        torch.testing.assert_close(actual, geometry(image))


def test_paired_norm_and_complex_shortcut_commute_with_rotation():
    image = torch.randn(2, 8, 9, 9)
    angle = 0.73
    norm = PairedRMSNorm2d(8)
    shortcut = ComplexPointwiseConv2d(8, 12, stride=2)
    torch.testing.assert_close(
        norm(_rotate_pairs(image, angle)),
        _rotate_pairs(norm(image), angle),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        shortcut(_rotate_pairs(image, angle)),
        _rotate_pairs(shortcut(image), angle),
        atol=1e-5,
        rtol=1e-5,
    )


def test_four_value_paired_mams_forward_backward():
    layer = CartesianFourValuePairedMAMSConv2d(
        8,
        12,
        diameters=(6, 3),
        stride=2,
        directions=4,
        global_directions=8,
        prototype_chunk_size=64,
        paired_input=True,
    )
    image = torch.randn(2, 8, 16, 16, requires_grad=True)
    output = layer(image)
    assert output.shape == (2, 12, 8, 8)
    assert layer.prototype_radius_raw.shape == (6, 4, 24)
    assert layer.prototype_theta_offset.shape == (6, 4, 24)
    assert layer.direction_value.shape == (6, 2, 2)
    assert layer.scale_value.shape == (6, 2, 2)
    output.square().mean().backward()
    assert torch.isfinite(image.grad).all()
    assert torch.isfinite(layer.prototype_radius_raw.grad).all()


def test_resnet18_four_value_paired_mams_structure():
    model = build_imagenet100_model(
        variant="resnet18_mams_fourv_paired",
        model_name="resnet18_mams_fourv_paired_4d4r_d6d3",
        pretrained=False,
        num_classes=100,
        image_size=224,
        rot_kernel_sizes=(6, 3),
        rot_directions=4,
        rot_global_directions=8,
        rot_prototype_chunk_size=64,
        rot_null_initial_score=0.0,
    )
    blocks = [
        module for module in model.modules()
        if isinstance(module, FourValuePairedMAMSBasicBlock)
    ]
    assert len(blocks) == 8
    assert blocks[0].paired_input is False
    assert blocks[0].shortcut is None
    assert all(block.paired_input for block in blocks[1:])
    assert isinstance(blocks[2].shortcut, ComplexPointwiseConv2d)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())

    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 100)
    assert torch.isfinite(output).all()


def test_triton_polar_renderer_matches_reference_forward_backward():
    if not torch.cuda.is_available():
        return
    counts = torch.tensor([4, 8, 12])
    offsets = torch.tensor([0, 4, 12, 24])
    geometry = CartesianCircularPatchGeometry(8, diameter=6, stride=1)
    renderer = _PolarRenderer(
        geometry,
        radial_bins=3,
        ring_counts=counts,
        ring_offsets=offsets,
        directions=4,
        direction_step=torch.pi / 4,
    ).cuda()
    reference_input = torch.randn(5, 8, 24, device="cuda", requires_grad=True)
    triton_input = reference_input.detach().clone().requires_grad_()
    reference = renderer(reference_input)
    actual = triton_polar_render(triton_input, renderer)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)

    gradient = torch.randn_like(reference)
    (reference * gradient).sum().backward()
    (actual * gradient).sum().backward()
    torch.testing.assert_close(
        triton_input.grad,
        reference_input.grad,
        atol=2e-6,
        rtol=2e-6,
    )
