import torch
from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed


def test_group_sum_matches_ungrouped_values_and_gradients():
    kwargs=dict(img_size=48,in_chans=3,bases=8,directions=14,
                global_directions=14,angular_bins_per_radius=5,
                harmonic_orders=(1,2),pose_softmax=True,use_null=True,
                prototype_chunk_size=3)
    a=HexRotatingHarmonicPatchEmbed(**kwargs,embed_dim=8,output_groups=4)
    b=HexRotatingHarmonicPatchEmbed(**kwargs,embed_dim=32)
    with torch.no_grad():
        b.prototype.copy_(a.prototype)
        b.null_score.copy_(a.null_score)
    x=torch.randn(1,3,48,48)
    y=a(x)
    z=b(x).reshape(1,b.num_patches,4,8).sum(-2)
    torch.testing.assert_close(y,z)
    y.square().sum().backward();z.square().sum().backward()
    torch.testing.assert_close(a.prototype.grad,b.prototype.grad)
    torch.testing.assert_close(a.null_score.grad,b.null_score.grad)


def test_opposite_peaks_second_order_survives():
    a=HexRotatingHarmonicPatchEmbed(img_size=48,in_chans=3,embed_dim=4,
        bases=1,directions=14,global_directions=14,harmonic_orders=(1,2))
    p=torch.zeros(14);p[0]=p[7]=.5
    moments=p@a.direction_coefficients
    torch.testing.assert_close(moments,torch.tensor([0.,0.,1.,0.]),atol=1e-6,rtol=0)
