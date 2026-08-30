import torch

from experiments.imagenet100.models import build_imagenet100_model
from model.resnet_geometric_baselines import (
    AdaptiveRotatedDepthwiseConv2d,
    FixedRotatedDepthwiseConv2d,
    MixConv4Depthwise,
    MultiScaleConv2d,
    MultiScaleRotatingConv2d,
    SharedRotatingConv2d,
    _ResidualComparisonBlock,
)


def test_shared_rotating_conv_has_differentiable_shared_bank():
    layer = SharedRotatingConv2d(
        3,
        6,
        kernel_size=5,
        stride=2,
        directions=4,
        bias=False,
    )
    image = torch.randn(2, 3, 17, 17, requires_grad=True)
    rotated = layer.rotated_weights()
    output = layer(image)

    assert rotated.shape == (4, 6, 3, 5, 5)
    assert output.shape == (2, 6, 9, 9)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(image.grad).all()


def test_multiscale_extractors_preserve_requested_width():
    image = torch.randn(2, 8, 16, 16)
    multiscale = MultiScaleConv2d(
        8, 12, kernel_sizes=(5, 3), stride=2
    )
    rotating = MultiScaleRotatingConv2d(
        8, 12, kernel_sizes=(5, 3), stride=2, directions=4
    )

    assert multiscale(image).shape == (2, 12, 8, 8)
    assert rotating(image).shape == (2, 12, 8, 8)


def test_all_geometric_resnet_comparisons_replace_eight_blocks():
    variants = (
        "resnet18_multiscale",
        "resnet18_rotconv4",
        "resnet18_multiscale_rotconv4",
    )
    for variant in variants:
        model = build_imagenet100_model(
            variant=variant,
            model_name=variant,
            pretrained=False,
            num_classes=100,
            image_size=224,
        )
        blocks = [
            module
            for module in model.modules()
            if isinstance(module, _ResidualComparisonBlock)
        ]
        assert len(blocks) == 8
        assert blocks[2].stride == 2
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 32, 32))
        assert output.shape == (1, 100)
        assert torch.isfinite(output).all()


def test_literature_aligned_lightweight_operators_are_differentiable():
    image = torch.randn(2, 8, 17, 17, requires_grad=True)
    operators = (
        MixConv4Depthwise(8, 12, kernel_sizes=(3, 5, 7, 9), stride=2),
        FixedRotatedDepthwiseConv2d(8, 12, stride=2, directions=4),
        AdaptiveRotatedDepthwiseConv2d(8, 12, stride=2, kernel_number=4),
    )
    for operator in operators:
        output = operator(image)
        assert output.shape == (2, 12, 9, 9)
        assert torch.isfinite(output).all()
        output.square().mean().backward(retain_graph=True)


def test_fixed_rotation_depthwise_bank_does_not_mix_projected_channels():
    layer = FixedRotatedDepthwiseConv2d(2, 2, stride=1, directions=8)
    with torch.no_grad():
        layer.project.weight.zero_()
        layer.project.weight[0, 0, 0, 0] = 1.0
        layer.project.weight[1, 1, 0, 0] = 1.0
        layer.weight.zero_()
        layer.weight[0, 0, 0, 1, 1] = 1.0
        layer.weight[0, 1, 0, 1, 1] = 2.0
    image = torch.zeros(1, 2, 5, 5)
    image[:, 0] = 1.0
    output = layer(image)
    assert torch.all(output[:, 0] > 0)
    assert torch.count_nonzero(output[:, 1]) == 0


def test_new_resnet_baselines_keep_basicblock_structure():
    variants = (
        "resnet18_mixconv4",
        "resnet18_fixed_rotinterp8",
        "resnet18_arc4bank",
    )
    expected_types = (
        MixConv4Depthwise,
        FixedRotatedDepthwiseConv2d,
        AdaptiveRotatedDepthwiseConv2d,
    )
    for variant, expected_type in zip(variants, expected_types, strict=True):
        model = build_imagenet100_model(
            variant=variant,
            model_name=variant,
            pretrained=False,
            num_classes=100,
            image_size=224,
        )
        assert isinstance(model.layer1[0].conv1, expected_type)
        assert isinstance(model.layer1[0].conv2, torch.nn.Conv2d)
        assert sum(isinstance(module, expected_type) for module in model.modules()) == 8
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 32, 32))
        assert output.shape == (1, 100)
        assert torch.isfinite(output).all()
