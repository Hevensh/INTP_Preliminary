import torch

from experiments.imagenet100.models import build_imagenet100_model
from model.resnet_geometric_baselines import (
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
