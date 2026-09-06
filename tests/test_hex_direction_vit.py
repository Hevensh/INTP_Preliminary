import math
import torch
from model.hex_direction_vit import HexDirectionViT, HexDirectionAttention, hex_neighbors


def test_hex_neighbors():
    xy = torch.tensor([(q+r/2, math.sqrt(3)*r/2) for q in range(-3,4) for r in range(-3,4)])
    ix, valid, _ = hex_neighbors(xy)
    center = xy.square().sum(-1).argmin()
    assert valid[center].sum() == 19
    assert len(ix[center].unique()) == 19


def test_chunk_values_and_gradients():
    xy = torch.tensor([(0.,0.), (1.,0.), (.5,math.sqrt(3)/2)])
    a = HexDirectionAttention(12, 3, xy, torch.arange(6)*math.pi/6, query_chunk=1)
    b = HexDirectionAttention(12, 3, xy, torch.arange(6)*math.pi/6, query_chunk=20)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2,3,6,12,requires_grad=True)
    y, z = a(x), b(x)
    torch.testing.assert_close(y,z)
    ga = torch.autograd.grad(y.square().sum(), x)[0]
    gb = torch.autograd.grad(z.square().sum(), x)[0]
    torch.testing.assert_close(ga,gb)


def test_raw_features_and_model():
    m = HexDirectionViT(image_size=48,depth=1)
    x = torch.randn(1,3,48,48)
    f = m.patch_embed(x)
    assert f.shape == (1,m.patch_embed.num_patches,6,96)
    y = m(x)
    assert y.shape == (1,100)
    y.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
