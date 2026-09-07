import math
import torch
from model.hex_direction_vit import HexSubsample, HexDirectionPyramid


def test_pool7_reference_and_gradients():
    qr=[(q,r) for q in range(-2,3) for r in range(-2,3)]
    xy=torch.tensor([(q+r/2,math.sqrt(3)*r/2) for q,r in qr])
    m=HexSubsample(xy,4,7,pool_neighbors=True)
    x=(-torch.rand(2,len(qr),6,4)-1).requires_grad_()
    reference=[]
    for center in m.keep.tolist():
        q,r=qr[center]
        ids=[i for i,(a,b) in enumerate(qr)
             if max(abs(a-q),abs(b-r),abs(a-q+b-r))<=1]
        reference.append(x[:,ids].max(1).values)
    expected=m.proj(torch.stack(reference,1))
    actual=m(x)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(),x)[0],
                               torch.autograd.grad(expected.square().sum(),x)[0])
    assert m.pool_indices.shape==(len(m.keep),7)
    assert m.pool_valid.sum(1).min()<7
    # The old selection variant and the pooled variant retain identical centers.
    old=HexSubsample(xy,4,7)
    torch.testing.assert_close(m.keep,old.keep)
    torch.testing.assert_close(m.coordinates,old.coordinates)


def test_pool7_pyramid():
    m=HexDirectionPyramid(full_circle=True,ge_aligned=True,pool_neighbors=True)
    assert m.token_counts==[195,52,14]
    assert sum(p.numel() for p in m.parameters())==5499456
    assert all(t.pool_neighbors for t in m.transitions)
