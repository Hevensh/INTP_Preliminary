import pytest
import torch

from experiments.imagenet100.models import build_imagenet100_model
from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed


@pytest.mark.parametrize('global_directions,angles', [(13, None), (14, (0,-5,5,-15,15,-30,30))])
def test_custom_angles_render_and_project_consistently(global_directions, angles):
    args = dict(img_size=32, in_chans=3, embed_dim=4, bases=2,
                directions=7, global_directions=global_directions,
                angular_bins_per_radius=5, pose_softmax=True, use_null=True)
    layer = HexRotatingHarmonicPatchEmbed(**args, direction_angles_degrees=angles)
    theta = (torch.arange(7)*2*torch.pi/global_directions if angles is None
             else torch.deg2rad(torch.tensor(angles, dtype=torch.float32)))
    torch.testing.assert_close(layer.direction_coefficients, torch.stack((theta.cos(),theta.sin()),-1))
    # Each rendered pose must equal rendering that exact angle alone.
    for d in range(7):
        single = HexRotatingHarmonicPatchEmbed(**{**args, 'directions':1},
            direction_angles_degrees=(float(torch.rad2deg(theta[d])),))
        for renderer, reference in zip(layer.renderers, single.renderers):
            torch.testing.assert_close(renderer(layer.prototype)[:,d:d+1],
                                       reference(layer.prototype), atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke')
@pytest.mark.parametrize('grid,angles', [(13,None),(14,(0,-5,5,-15,15,-30,30))])
def test_custom_full_model_backward(grid, angles):
    model = build_imagenet100_model(variant='rot_hex_harmonic_softmax_pe',
        model_name='deit_tiny_patch16_224', pretrained=False, num_classes=100,
        image_size=224, rot_directions=7, rot_global_directions=grid,
        rot_direction_angles_degrees=angles, rot_angular_bins_per_radius=5).cuda()
    with torch.autocast('cuda', dtype=torch.float16):
        result = model(torch.randn(2,3,224,224,device='cuda'))
    assert result.shape == (2,100) and result.isfinite().all()
    result.float().square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
