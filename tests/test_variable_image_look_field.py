import pytest
import torch
from layers.square_patch_dense_grid_look import SquarePatchDenseGridLook


def reference(look):
    xy=look.patch_coordinates_xy
    delta=xy.unsqueeze(0)-xy.unsqueeze(1)
    distance=delta.norm(dim=-1)
    angle=torch.atan2(delta[...,1],delta[...,0])
    outputs=[]
    for scale in (1.,.5):
        poses=[]
        position=distance/(look.look_radius*scale)*4-1
        r0=position.floor().clamp(0,3).long()
        r1=(r0+1).clamp_max(3)
        rw=position-position.floor()
        for d in range(7):
            turn=((angle-d*2*torch.pi/14)/(2*torch.pi))%1
            result=0
            for ring,w in ((r0,1-rw),(r1,rw)):
                count=5*(ring+1)
                offset=5*ring*(ring+1)//2
                a=turn*count
                left=a.floor().long()
                fraction=a-a.floor()
                result=result+w*((1-fraction)*look.look_grid[:,offset+left%count]
                                 +fraction*look.look_grid[:,offset+(left+1)%count])
            poses.append(result*((distance>0)&(distance<=look.look_radius*scale)))
        outputs.append(torch.stack(poses,1))
    return torch.stack(outputs,1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_variable_field_values_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA')
    look=SquarePatchDenseGridLook(image_size=48,num_heads=2,source_directions=7,
        source_direction_period=14,prototype_angular_bins=28,look_radial_bins=4,
        look_angular_bins_per_radius=5).to(device)
    assert look.look_grid.shape==(2,50)
    assert look.field_ring_counts.tolist()==[5,10,15,20]
    torch.nn.init.normal_(look.look_grid)
    a=look.transformed_look_grids()
    b=reference(look)
    torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
    weight=torch.randn_like(a)
    ga=torch.autograd.grad(a,look.look_grid,weight)[0]
    gb=torch.autograd.grad(b,look.look_grid,weight)[0]
    torch.testing.assert_close(ga,gb,atol=2e-5,rtol=2e-4)
