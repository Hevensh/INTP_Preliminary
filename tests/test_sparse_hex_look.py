import torch
import pytest
from layers.sparse_hex_look import SparseHexLookLayout,sparse_query_attention


def test_centers_share_only_three_outer_ring_points():
    qr=torch.tensor([(q,r) for q in range(-10,11) for r in range(-10,11)])
    xy=torch.stack((qr[:,0]+qr[:,1]/2,qr[:,1]*3**.5/2),-1)
    layout=SparseHexLookLayout(xy)
    rings=[{tuple((layout.axial[c]+d).tolist()) for d in layout.offsets} for c in layout.centers]
    for i in range(len(rings)):
        for j in range(i):
            common=rings[i]&rings[j]
            assert len(common) in (0,3)
            for v in common:
                for c in (layout.centers[i],layout.centers[j]):
                    q,r=(torch.tensor(v)-layout.axial[c]).tolist()
                    assert max(abs(q),abs(r),abs(q+r))==2
    for perm in layout.permutations:
        assert sorted(perm.tolist())==list(range(18))


def test_sparse_attention_matches_dense_values_and_all_gradients():
    torch.manual_seed(3)
    q,k,v=[torch.randn(2,3,13,8,requires_grad=True) for _ in range(3)]
    centers=torch.tensor([2,7]);keys=torch.tensor([[1,3,0],[4,8,9]])
    valid=torch.tensor([[True,True,False],[True,True,True]])
    values=torch.randn(2,3,2,3,requires_grad=True)
    actual=sparse_query_attention(q,k,v,centers,keys,valid,values,scale=8**-.5)
    rows=torch.zeros(2,3,2,13).scatter_add(-1,keys[None,None].expand(2,3,-1,-1),values*valid[None,None])
    bias=torch.zeros(2,3,13,13).index_copy(2,centers,rows)
    expected=((q@k.transpose(-2,-1))*8**-.5+bias).softmax(-1)@v
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=1e-5)
    coeff=torch.randn_like(actual)
    for x in (q,k,v,values):
        a=torch.autograd.grad((actual*coeff).sum(),x,retain_graph=True)[0]
        b=torch.autograd.grad((expected*coeff).sum(),x,retain_graph=True)[0]
        torch.testing.assert_close(a,b,atol=4e-6,rtol=3e-5)


@pytest.mark.parametrize('G',[1,3,12])
def test_sparse_model_forward_backward_and_roundtrip(G):
    from model.deit_tiny_rot_hex_look import DeiTTinyRotHexLook
    kw=dict(use_pos_embed=True,image_size=48,directions=6,global_directions=12,
        angular_bins_per_radius=3,look_compact_variable_rings=True,center_pose_grid_look=True,
        center_look_layers_per_probe=G,image_look_probes=4,feature_look_probes=4,
        sparse_hex_look=True,progressive_differentiation=True)
    m=DeiTTinyRotHexLook(**kw)
    n=len(m.sparse_layout.centers)
    assert all(g.sample_x.shape[0]==n for g in m.look_bank.compact_geometries)
    assert m.patch_embed.num_patches>n
    with torch.no_grad():
        m.look_bank.look_grid.normal_(std=.02);m.center_look.look_grid.normal_(std=.02)
    x=torch.randn(2,3,48,48)
    y=m(x);y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.center_look.axis_weight.grad.abs().sum()>0
    assert m.experiment_diagnostics()['scale_mapping']=={'K12':'inner6','K24':'outer12'}
    restored=DeiTTinyRotHexLook(**kw);restored.load_state_dict(m.state_dict())
    torch.testing.assert_close(restored(x),y)


def test_scale_ring_aggregation_matches_explicit_sum():
    coords=torch.tensor([[0.,0.],[1.,0.]])
    m=SparseHexLookLayout(coords)
    p=torch.rand(2,1,3,4,2,6)
    t=torch.randn(3,4,18,requires_grad=True)
    small=m.aggregate(p[...,1,:],t);large=m.aggregate(p[...,0,:],t)
    actual=torch.cat((small[...,:6],large[...,6:]),-1)
    expected=torch.zeros_like(actual)
    for h in range(3):
        for probe in range(4):
            for a in range(6):
                for e in range(18):
                    scale=1 if e<6 else 0
                    expected[:,h,:,e]+=p[:,:,h,probe,scale,a]*t[h,probe,m.permutations[a,e]]/4
    torch.testing.assert_close(actual,expected)
