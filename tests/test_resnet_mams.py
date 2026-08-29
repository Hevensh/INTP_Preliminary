import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.cartesian_rotating_harmonic_conv import (
    CartesianCircularPatchGeometry,
    CartesianRotatingHarmonicConv2d,
)
from model.resnet_mams import MAMSBasicBlock


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
