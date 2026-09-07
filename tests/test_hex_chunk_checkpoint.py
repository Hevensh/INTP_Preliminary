import math
import torch
from model.hex_direction_vit import HexDirectionAttention


def test_checkpoint_chunks_match_full_attention_gradients():
    xy=torch.tensor([(q+r/2,math.sqrt(3)*r/2) for q in range(3) for r in range(3)])
    a=HexDirectionAttention(12,3,xy,torch.arange(6)*math.pi/3,query_chunk=9,ge_aligned=True)
    b=HexDirectionAttention(12,3,xy,torch.arange(6)*math.pi/3,query_chunk=2,ge_aligned=True)
    b.load_state_dict(a.state_dict()); b.checkpoint_chunks=True
    x=torch.randn(2,9,6,12,requires_grad=True)
    z=x.detach().clone().requires_grad_()
    y,w=a(x),b(z)
    torch.testing.assert_close(y,w)
    y.square().sum().backward();w.square().sum().backward()
    torch.testing.assert_close(x.grad,z.grad,atol=1e-6,rtol=1e-4)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-4)
