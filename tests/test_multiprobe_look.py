import pytest
import torch
from layers.multiprobe_look import RotatingMultiProbeLook, aggregate_pose_grids, sample_pose_grids
from layers.center_pose_grid_look import CenterPoseGridLook


def test_unbind_layer_and_repeated_group_gradients_match_slices():
    torch.manual_seed(6)
    image = torch.randn(2,5,12*3*4,2,6,requires_grad=True)
    feature = torch.randn(2,5,4,3,4,6,requires_grad=True)
    layers = image.reshape(2,5,12,3,4,2,6).unbind(2)
    groups = feature.unbind(2)
    old = new = 0
    for i in range(12):
        old = old + (image[:,:,i*12:(i+1)*12].sin()*(i+1)).sum()
        new = new + (layers[i].sin()*(i+1)).sum()
        if i < 11:
            old = old + (feature[:,:,i//3].cos()*(i+1)).sum()
            new = new + (groups[i//3].cos()*(i+1)).sum()
    torch.testing.assert_close(old,new)
    for tensor in (image,feature):
        torch.testing.assert_close(torch.autograd.grad(old,tensor,retain_graph=True)[0],
                                   torch.autograd.grad(new,tensor,retain_graph=True)[0])


def make(probes=4, device="cpu"):
    torch.manual_seed(12)
    coords = torch.randn(9, 2)
    return RotatingMultiProbeLook(coordinates=coords, embed_dim=12, num_heads=3,
        depth=2, axes=6, probes=probes).to(device)


def test_rotated_weights_equal_two_projections():
    m = make().double()
    x = torch.randn(2, 9, 12, dtype=torch.double, requires_grad=True)
    w = m.axis_weight
    # Match buffer precision used by implementation, independently materialize Wtheta.
    c, s = m.direction_cos, m.direction_sin
    jw = torch.stack((-w[..., 1], w[..., 0]), -1)
    rotated = w[..., None, :, :]*c[:, None, None] + jw[..., None, :, :]*s[:, None, None]
    score = torch.einsum("bqhpc,ghmapc->bqghma", x.reshape(2,9,3,2,2), rotated) + m.axis_bias
    null = m.null_score[None,None,...,None].expand(*score.shape[:-1],1)
    expected = torch.cat((score,null),-1).float().softmax(-1)[...,:-1].double()
    actual = m.pose_weights(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), x)[0])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU"))])
def test_grid_first_equals_interpolate_every_probe(device):
    m = make(device=device)
    m.look_grid.data.normal_()
    pose = torch.randn(2,9,3,4,6,device=device).softmax(-1).requires_grad_()
    actual = m.dense_bias(pose, 0)
    fields = []
    for probe in range(4):
        # Reuse the old interpolation implementation with one template slice.
        proxy = make(1,device)
        proxy.look_grid = torch.nn.Parameter(m.look_grid.detach()[:,:,probe])
        fields.append(CenterPoseGridLook._transformed_fields(proxy,0))
    expected = torch.einsum("bqhma,hmaqk->bhqk", pose, torch.stack(fields,1))/4
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(),pose,retain_graph=True)[0],
                               torch.autograd.grad(expected.square().sum(),pose)[0],atol=2e-5,rtol=5e-5)
    actual.sum().backward()
    assert torch.isfinite(m.look_grid.grad).all()


def test_null_is_not_renormalized_away():
    m = make()
    m.null_score.data.fill_(30)
    assert m.pose_weights(torch.zeros(1,9,12)).sum(-1).max() < 1e-10


def test_image_two_scales_grid_first_values_and_gradients():
    from layers.square_patch_dense_grid_look import SquarePatchDenseGridLook
    bank = SquarePatchDenseGridLook(image_size=48, patch_size=16, num_heads=6, prototype_angular_bins=24,
        source_directions=6, source_direction_period=12, look_direction_bins=12,
        look_radial_bins=4, look_radius=3.)
    bank.look_grid.data.normal_()
    p = torch.randn(2,9,2,3,2,6).softmax(-1).requires_grad_()
    grids = aggregate_pose_grids(p, bank.look_grid.reshape(2,3,4,12),period=12)
    actual = sample_pose_grids(grids,bank,image=True)
    fields = bank.transformed_look_grids().reshape(2,3,2,6,9,9)
    expected = torch.einsum("bqhmsa,hmsaqk->bhqk",p,fields)/3
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    for parameter in (p,bank.look_grid):
        a = torch.autograd.grad(actual.square().sum(),parameter,retain_graph=True)[0]
        b = torch.autograd.grad(expected.square().sum(),parameter,retain_graph=True)[0]
        torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)


def test_dual_grid_merge_uses_identical_coordinates():
    from model.deit_tiny_rot_hex_look import DeiTTinyRotHexLook
    model = DeiTTinyRotHexLook(use_pos_embed=True, directions=6,global_directions=12,
        angular_bins_per_radius=3,look_compact_variable_rings=True,
        center_pose_grid_look=True,feature_look_rotating_probes=True,
        image_look_probes=4,feature_look_probes=4)
    bank,center=model.look_bank,model.center_look
    for name in ("look_radial0","look_radial1","look_angular0","look_angular1",
                 "look_radial_fraction","look_angular_fraction","look_valid"):
        image=getattr(bank,name)[0,0]
        feature=getattr(center,name)
        if "angular" in name: feature=feature[0]
        torch.testing.assert_close(image,feature)
    n=model.patch_embed.num_patches
    ig=torch.randn(1,n,3,2,4,12,requires_grad=True)
    fg=torch.randn(1,n,3,1,4,12,requires_grad=True)
    separate=sample_pose_grids(ig,bank,image=True)+sample_pose_grids(fg,center,image=False)
    merged=sample_pose_grids(torch.cat((ig[:,:,:,:1]+fg,ig[:,:,:,1:]),3),bank,image=True)
    torch.testing.assert_close(merged,separate,atol=2e-6,rtol=1e-5)
    for parameter in (ig,fg):
        a=torch.autograd.grad(merged.square().sum(),parameter,retain_graph=True)[0]
        b=torch.autograd.grad(separate.square().sum(),parameter,retain_graph=True)[0]
        torch.testing.assert_close(a,b,atol=3e-5,rtol=2e-5)
