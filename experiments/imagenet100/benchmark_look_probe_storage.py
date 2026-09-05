"""Check polar-W equivalence and benchmark probe-only storage overhead."""
import json
import math
import statistics
import torch
from layers.multiprobe_look import rotating_probe_scores


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    # Small float64 check: direct polar-angle formula vs Cartesian two-dot form.
    x=torch.randn(2,5,3,32,2,dtype=torch.double,requires_grad=True)
    r=torch.rand(4,3,4,32,dtype=torch.double,requires_grad=True)
    phi=torch.randn_like(r,requires_grad=True)
    angle=torch.arange(6,dtype=torch.double)*math.pi/6
    w=torch.stack((r*phi.cos(),r*phi.sin()),-1)
    factored=rotating_probe_scores(x,w,angle.cos(),angle.sin())
    ra=phi[...,None,:]+angle[:,None]
    rotated=torch.stack((r[...,None,:]*ra.cos(),r[...,None,:]*ra.sin()),-1)
    direct=torch.einsum("bqhpc,ghmapc->bqghma",x,rotated)
    torch.testing.assert_close(factored,direct)
    for value in (x,r,phi):
        torch.testing.assert_close(torch.autograd.grad(factored.square().sum(),value,retain_graph=True)[0],
                                   torch.autograd.grad(direct.square().sum(),value,retain_graph=True)[0])
    print("polar/Cartesian outputs and gradients: equivalent (float64)",flush=True)
    # Realistic B128, 195 tokens, G3 => 4 detector groups, M4, three heads.
    x=torch.randn(128,195,3,32,2,device="cuda",requires_grad=True)
    initial=torch.randn(4,3,4,32,2,device="cuda")/8
    angle=(torch.arange(6,device="cuda")*math.pi/6)
    c,s=angle.cos(),angle.sin()
    results=[]
    for mode in ("cartesian","polar_reconstruct","cartesian","polar_reconstruct"):
        w=initial.detach().clone().requires_grad_()
        r=initial.norm(dim=-1).detach().requires_grad_()
        phi=initial[...,1].atan2(initial[...,0]).detach().requires_grad_()
        def step():
            for value in (x,w,r,phi):value.grad=None
            weight=w if mode=="cartesian" else torch.stack((r*phi.cos(),r*phi.sin()),-1)
            with torch.autocast("cuda",dtype=torch.float16):
                score=rotating_probe_scores(x,weight,c,s)
                p=torch.cat((score,torch.zeros_like(score[...,:1])),-1).float().softmax(-1)
                loss=p.square().mean()
            loss.backward()
        for _ in range(5):step()
        times=[]
        for _ in range(40):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();step();end.record();end.synchronize()
            times.append(start.elapsed_time(end))
        result={"mode":mode,"probe_forward_backward_ms":statistics.median(times)}
        results.append(result)
        print(json.dumps(result),flush=True)
    print(json.dumps({"gpu":torch.cuda.get_device_name(),"results":results}),flush=True)


if __name__=="__main__":main()
