import math
import torch
from model.hex_direction_vit import TokenGroupNorm, HexDirectionAttention, HexDirectionPyramid


def test_group_norm_layout_and_gradients():
    a = TokenGroupNorm(12)
    b = torch.nn.GroupNorm(1, 12, eps=1e-6)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 7, 6, 12, requires_grad=True)
    y = a(x)
    z = b(x.permute(0,3,2,1).unsqueeze(-1)).squeeze(-1).permute(0,3,2,1)
    torch.testing.assert_close(y,z)
    torch.testing.assert_close(torch.autograd.grad(y.square().sum(),x)[0],
                               torch.autograd.grad(z.square().sum(),x)[0], atol=1e-5, rtol=1e-3)


def test_aligned_attention_explicit_reference():
    xy = torch.tensor([(0.,0.), (1.,0.), (.5,math.sqrt(3)/2)])
    a = HexDirectionAttention(12,3,xy,torch.arange(6)*math.pi/3,ge_aligned=True)
    x = torch.randn(2,3,6,12,requires_grad=True)
    q,k,v = a.qkv(x).reshape(2,3,6,3,3,4).unbind(3)
    rows=a.row_embedding(a.local_coordinates[...,:1])
    cols=a.col_embedding(a.local_coordinates[...,1:])
    groups=a.group_embedding(a.relative_group)
    outputs=[]
    for n in range(3):
        scores=torch.einsum('bghd,bpkhd->bhgpk',q[:,n],k[:,a.indices[n]])
        scores += torch.einsum('bghd,gpd->bhgp',q[:,n,...,:1],rows)[...,None]
        scores += torch.einsum('bghd,gpd->bhgp',q[:,n,...,1:2],cols)[...,None]
        scores += torch.einsum('bghd,gkd->bhgk',q[:,n,...,2:],groups)[:,:,:,None,:]
        scores=(scores/math.sqrt(12)).masked_fill(~a.valid[n][None,None,None,:,None],float('-inf'))
        p=scores.flatten(-2).softmax(-1).reshape_as(scores)
        outputs.append(torch.einsum('bhgpk,bpkhd->bghd',p,v[:,a.indices[n]]).reshape(2,6,12))
    expected=a.proj(torch.stack(outputs,1))
    actual=a(x)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(),x)[0],
                               torch.autograd.grad(expected.square().sum(),x)[0],atol=1e-6,rtol=1e-4)


def test_aligned_pyramid_readout():
    m=HexDirectionPyramid(image_size=48, full_circle=True, ge_aligned=True)
    image=torch.randn(1,3,48,48)
    x=m.patch_embed(image)
    for i,stage in enumerate(m.stages):
        x=stage(x)
        if i<2:
            x=m.transitions[i](x)
    expected=m.head(m.norm(x)).sum(1).max(1).values
    actual=m(image)
    torch.testing.assert_close(actual,expected)
    actual.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
    assert isinstance(m.norm,TokenGroupNorm)
    assert all(torch.count_nonzero(p)==0 for name,p in m.named_parameters()
               if name.endswith('bias') and 'patch_embed' not in name)
