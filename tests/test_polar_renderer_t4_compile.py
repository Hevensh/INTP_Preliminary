"""Compile the actual 11/23-r7 backward tiles for T4, even on newer GPUs."""
import pytest
import torch


def test_t4_backward_shared_memory_budget():
    triton = pytest.importorskip('triton')
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from layers.triton_polar_renderer import _polar_backward_gather
    from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed
    model = HexRotatingHarmonicPatchEmbed(img_size=224,in_chans=3,embed_dim=192,
        directions=11,global_directions=23,angular_bins_per_radius=7)
    for renderer in model.renderers:
        contributions=renderer.reverse_lookup.shape[1]
        width=triton.next_power_of_2(contributions)
        constants=dict(elements=16*3*546,channels=3,directions=11,
            samples=renderer.index_r0_a0.shape[1],stored=546,
            contributions=contributions,block_contributions=width,
            block=min(32,max(1,2048//width)))
        signature=dict(grad_output='*fp32',reverse_lookup='*i64',
            reverse_weight='*fp32',grad_prototype='*fp32')
        kernel=triton.compile(ASTSource(_polar_backward_gather,signature,
            constexprs=constants),target=GPUTarget('cuda',75,32),
            options=dict(num_warps=4,num_stages=1))
        print('T4 backward:', 'width',width,'block',constants['block'],
              'shared bytes',kernel.metadata.shared)
        assert kernel.metadata.shared <= 65536
