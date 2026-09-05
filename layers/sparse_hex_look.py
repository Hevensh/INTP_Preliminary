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


class _SparseBias(torch.autograd.Function):
    """Expand sparse entries once; gather their gradients without dense copies."""

    @staticmethod
    def forward(ctx, values, centers, keys, valid, sequence):
        indices = (centers[:, None] * sequence + keys).flatten()
        ctx.save_for_backward(indices, valid)
        ctx.value_shape = values.shape
        bias = values.new_zeros(*values.shape[:2], sequence * sequence)
        bias.scatter_add_(2, indices[None, None].expand(*values.shape[:2], -1),
                          (values * valid).flatten(2))
        return bias.view(*values.shape[:2], sequence, sequence)

    @staticmethod
    def backward(ctx, grad):
        indices, valid = ctx.saved_tensors
        # Index the two spatial axes directly: SDPA may return a padded stride.
        n = grad.shape[-1]
        selected = grad[:, :, indices // n, indices % n]
        return selected.reshape(ctx.value_shape) * valid, None, None, None, None


def sparse_query_attention(q,k,v,centers,keys,valid,values,ordinary=None,*,scale,dropout_p=0.):
    """Global attention remains global. Only selected query rows carry bias.

    A single SDPA call keeps all queries in the same fused attention path.
    The temporary dense bias trades memory for avoiding a separate FP32
    attention branch and duplicated K/V gradients. No spatial interpolation
    or masked-out keys. ``ordinary`` remains accepted for checkpoint callers.
    Indices include CLS. Invalid boundary neighbors contribute zero.
    """
    bias = _SparseBias.apply(values.to(q.dtype), centers, keys, valid, q.shape[2])
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                         dropout_p=dropout_p, scale=scale)
