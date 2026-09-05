"""Two-ring Hex Look centers, sharing precisely a three-token outer edge.

Coordinates use x=q+r/2, y=sqrt(3)*r/2. Center lattice basis (2,2),
(-2,4) has index 12. Boundaries are clipped, never wrapped or renormalized.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class SparseHexLookLayout(nn.Module):
    def __init__(self, coordinates, axes=6):
        super().__init__()
        xy=coordinates.detach().cpu().double()
        r=xy[:,1]*2/math.sqrt(3)
        qr=torch.stack((xy[:,0]-r/2,r),-1)
        if not torch.allclose(qr,qr.round(),atol=1e-4,rtol=0):
            raise ValueError('expected unit Hex lattice coordinates')
        qr=qr.round().long()
        origin=qr[(xy-xy.mean(0)).square().sum(-1).argmin()]
        delta=qr-origin
        chosen=((2*delta[:,0]+delta[:,1])%6==0)&((delta[:,1]-delta[:,0])%6==0)
        centers=chosen.nonzero().flatten()
        lookup={tuple(v):i for i,v in enumerate(qr.tolist())}
        offsets=[]
        for radius in (1,2):
            ring=[(q,r) for q in range(-radius,radius+1) for r in range(-radius,radius+1)
                  if max(abs(q),abs(r),abs(q+r))==radius]
            ring.sort(key=lambda t: math.atan2(math.sqrt(3)*t[1]/2,t[0]+t[1]/2)%(2*math.pi))
            offsets.extend(ring)
        keys=[]
        for c in qr[centers].tolist():
            keys.append([lookup.get((c[0]+dq,c[1]+dr),-1) for dq,dr in offsets])
        keys=torch.tensor(keys,dtype=torch.long)
        # Ring-local one-to-one permutations. Half-step ties round forward;
        # this is an explicitly approximate 30-degree mapping on the inner ring.
        permutations=[]
        for a in range(axes):
            permutations.append(torch.cat([
                (torch.arange(6)-int(math.floor(a*6/(2*axes)+.5)))%6,
                6+(torch.arange(12)-int(math.floor(a*12/(2*axes)+.5)))%12]))
        for name,value in dict(centers=centers,keys=keys.clamp_min(0),valid=keys>=0,
                axial=qr,offsets=torch.tensor(offsets),permutations=torch.stack(permutations)).items():
            self.register_buffer(name,value)
        keep=torch.ones(len(qr)+1,dtype=torch.bool)
        keep[centers+1]=False
        self.register_buffer('ordinary_queries',keep.nonzero().flatten())

    def aggregate(self, pose, template):
        """B,C,H,M,A and H,M,18 -> B,H,C,18; null already removed."""
        rotated=template[...,self.permutations].to(pose.dtype)
        return torch.einsum('bchma,hmae->bhce',pose,rotated)/pose.shape[3]


def sparse_query_attention(q,k,v,centers,keys,valid,values,ordinary=None,*,scale,dropout_p=0.):
    """Global attention remains global. Only selected query rows carry bias.

    Ordinary rows use SDPA; selected rows use a small explicit FP32 softmax.
    No dense N*N Look tensor, no spatial interpolation, no masked-out keys.
    Indices include CLS. Invalid boundary neighbors contribute zero.
    """
    n=q.shape[2]
    selected=q.index_select(2,centers)
    mask=values.new_zeros(*values.shape[:-1],n)
    indices=keys[None,None].expand(values.shape[0],values.shape[1],-1,-1)
    mask=mask.scatter_add(-1,indices,values*valid[None,None])
    with torch.autocast(device_type=q.device.type,enabled=False):
        logits=(selected.float()@k.float().transpose(-2,-1))*scale+mask.float()
        prob=F.dropout(logits.softmax(-1),p=dropout_p,training=dropout_p>0)
        special=(prob@v.float()).to(q.dtype)
    if ordinary is None:
        keep=torch.ones(n,dtype=torch.bool,device=q.device)
        keep[centers]=False
        ordinary=keep.nonzero().flatten()
    normal=F.scaled_dot_product_attention(q.index_select(2,ordinary),k,v,dropout_p=dropout_p,scale=scale)
    out=torch.zeros_like(q).index_copy(2,ordinary,normal)
    return out.index_copy(2,centers,special)
