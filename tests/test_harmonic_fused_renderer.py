import pytest
import torch

from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed
from layers.triton_polar_renderer import triton_polar_render


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('directions,grid,resolution', [(11,23,7),(7,17,5),(6,12,3)])
def test_fused_render_values_and_gradients(directions,grid,resolution):
    layer = HexRotatingHarmonicPatchEmbed(img_size=32,in_chans=3,embed_dim=8,
        bases=4,directions=directions,global_directions=grid,
        angular_bins_per_radius=resolution).cuda()
    for renderer in layer.renderers:
        p=layer.prototype.detach().clone().requires_grad_()
        ref=renderer(p)
        actual=triton_polar_render(p,renderer)
        weight=torch.randn_like(ref)
        g_ref=torch.autograd.grad(ref,p,weight)[0]
        g_actual=torch.autograd.grad(actual,p,weight)[0]
        torch.testing.assert_close(actual,ref,atol=1e-7,rtol=2e-5)
        torch.testing.assert_close(g_actual,g_ref,atol=3e-5,rtol=2e-4)
