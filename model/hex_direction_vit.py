"""Experimental half-circle direction features; NOT a strict C6-equivariant model.

Raw two-scale responses are summed as in the harmonic tokenizer, but all six
direction slots survive. Local attention uses exact Hex two-ring neighbors
and query-frame continuous coordinates, not rounded 30-degree neighbor rolls.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from layers.hex_rotating_harmonic_patch_embed import HexRotatingHarmonicPatchEmbed


def hex_neighbors(coordinates):
    xy = coordinates.detach().cpu().double()
    r = xy[:, 1] * 2 / math.sqrt(3)
    qr = torch.stack((xy[:, 0] - r / 2, r), -1)
    # A common fractional origin is harmless; only lattice differences matter.
    qr = qr - qr[0]
    if not torch.allclose(qr, qr.round(), atol=1e-4, rtol=0):
        raise ValueError('Expected a regular Hex lattice')
    qr = qr.round().long()
    offsets = [(q, r) for q in range(-2, 3) for r in range(-2, 3)
               if max(abs(q), abs(r), abs(q+r)) <= 2]
    lookup = {tuple(v): i for i, v in enumerate(qr.tolist())}
    indices = torch.tensor([[lookup.get((q+dq, r+dr), -1) for dq, dr in offsets]
                            for q, r in qr.tolist()])
    delta = torch.tensor([(q+r/2, math.sqrt(3)*r/2) for q, r in offsets])
    return indices.clamp_min(0), indices >= 0, delta


class HexDirectionAttention(nn.Module):
    def __init__(self, dim, heads, coordinates, angles, query_chunk=16):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must divide into heads')
        self.heads, self.dh = heads, dim // heads
        self.groups, self.query_chunk = len(angles), query_chunk
        indices, valid, delta = hex_neighbors(coordinates)
        self.register_buffer('indices', indices, persistent=False)
        self.register_buffer('valid', valid, persistent=False)
        a = torch.as_tensor(angles)
        c, s = a.cos(), a.sin()
        rotation = torch.stack((torch.stack((c, s), -1), torch.stack((-s, c), -1)), -2)
        local = torch.einsum('gij,kj->gki', rotation, delta / 2)
        self.register_buffer('local_coordinates', local, persistent=False)
        # Actual signed angular differences: do not wrap six half-circle slots
        # as though they were a full circle. Encode differences continuously.
        difference = a[None, :] - a[:, None]
        relative = torch.stack((difference.cos(), difference.sin()), -1)
        self.register_buffer('relative_direction', relative, persistent=False)
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        self.position = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, self.dh))
        self.direction = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, self.dh))

    def forward(self, x):
        b, n, g, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, g, 3, self.heads, self.dh).unbind(3)
        # [query direction, neighbor position, key direction, head dim]
        e = self.position(self.local_coordinates)[:, :, None, :] + self.direction(self.relative_direction)[:, None, :, :]
        e = e.reshape(g, 19*g, self.dh).to(q.dtype)
        outputs = []
        for start in range(0, n, self.query_chunk):
            stop = min(n, start+self.query_chunk)
            ix = self.indices[start:stop]
            qc = q[:, start:stop].permute(0, 1, 3, 2, 4)
            kc = k[:, ix].permute(0, 1, 4, 2, 3, 5).flatten(3, 4)
            vc = v[:, ix].permute(0, 1, 4, 2, 3, 5).flatten(3, 4)
            bias = torch.einsum('bnhgd,gkd->bnhgk', qc, e) / math.sqrt(self.dh)
            valid = self.valid[start:stop, :, None].expand(-1, -1, g).flatten(1)
            bias = bias.masked_fill(~valid[None, :, None, None, :], float('-inf'))
            y = F.scaled_dot_product_attention(
                qc.flatten(0, 1), kc.flatten(0, 1), vc.flatten(0, 1),
                attn_mask=bias.flatten(0, 1)).reshape(b, stop-start, self.heads, g, self.dh)
            outputs.append(y.permute(0, 1, 3, 2, 4).reshape(b, stop-start, g, c))
        return self.proj(torch.cat(outputs, 1))


class DirectionBlock(nn.Module):
    def __init__(self, coordinates, angles, dim=96, heads=3, query_chunk=16,
                 checkpoint_attention=False, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = HexDirectionAttention(dim, heads, coordinates, angles, query_chunk)
        self.checkpoint_attention = checkpoint_attention
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        z = self.norm1(x)
        z = (checkpoint(self.attn, z, use_reentrant=False)
             if self.training and self.checkpoint_attention else self.attn(z))
        x = x + z
        return x + self.mlp(self.norm2(x))


class HexDirectionViT(nn.Module):
    def __init__(self, image_size=224, num_classes=100, depth=12, checkpoint_blocks=True):
        super().__init__()
        self.embed_dim = 96
        self.checkpoint_blocks = checkpoint_blocks
        self.patch_embed = HexRotatingHarmonicPatchEmbed(
            img_size=image_size, in_chans=3, embed_dim=96, bases=96,
            directions=6, global_directions=12, angular_bins_per_radius=3,
            raw_direction_output=True, kernel_sizes=(24, 12))
        coo = self.patch_embed.coo_patchs
        coordinates = torch.stack((coo.real, coo.imag), -1)
        angles = torch.arange(6)*math.pi/6
        self.blocks = nn.ModuleList([DirectionBlock(coordinates, angles) for _ in range(depth)])
        self.norm = nn.LayerNorm(96, eps=1e-6)
        self.head = nn.Linear(96, num_classes)

    def forward_features(self, image):
        x = self.patch_embed(image)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.checkpoint_blocks else block(x)
        return self.norm(x)

    def forward(self, image):
        return self.head(self.forward_features(image).mean(dim=(1, 2)))

    def experiment_diagnostics(self):
        return dict(direction_width=96, directions_degrees=[0,30,60,90,120,150],
                    depth=len(self.blocks), heads=3, neighborhood=19,
                    candidates_per_query=114, strict_equivariance=False,
                    tokenizer='raw dot, two scales summed, no null/softmax/circular projection',
                    readout='space and direction mean', checkpoint_blocks=self.checkpoint_blocks)


class HexSubsample(nn.Module):
    """Keep the even/even axial sublattice after two neighborhood blocks.

No additional spatial pooling: preceding attention already aggregates neighbors.
New coordinates are divided by two so a two-ring window grows in image units.
"""
    def __init__(self, coordinates, in_dim, out_dim):
        super().__init__()
        xy = coordinates.detach().cpu().double()
        r = xy[:, 1]*2/math.sqrt(3)
        qr = torch.stack((xy[:, 0]-r/2, r), -1)
        qr = qr-qr[0]
        if not torch.allclose(qr, qr.round(), atol=1e-4, rtol=0):
            raise ValueError('Expected Hex lattice')
        keep = (qr.round().long().remainder(2)==0).all(-1).nonzero().flatten()
        self.register_buffer('keep', keep, persistent=False)
        self.register_buffer('coordinates', coordinates[keep]/2, persistent=False)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x[:, self.keep])


class HexDirectionPyramid(nn.Module):
    """144/288/336, FFN4, 2/2/2 blocks; geometric polar lifting.

Whole-stage query batches remove the small-query loop. Only attention is
checkpointed, not the FFN, and only in the first two (larger) stages.
"""
    def __init__(self, image_size=224, num_classes=100, checkpoint_attention=True,
                 full_circle=False):
        super().__init__()
        self.embed_dim = 336
        self.full_circle = bool(full_circle)
        period = 6 if full_circle else 12
        self.patch_embed = HexRotatingHarmonicPatchEmbed(
            img_size=image_size, in_chans=3, embed_dim=144, bases=144,
            directions=6, global_directions=period, angular_bins_per_radius=3,
            raw_direction_output=True, kernel_sizes=(24,12))
        coo = self.patch_embed.coo_patchs
        coordinates = torch.stack((coo.real, coo.imag), -1)
        angles = torch.arange(6)*2*math.pi/period
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.token_counts = []
        for i, (dim, heads) in enumerate(zip((144,288,336),(3,3,3))):
            self.token_counts.append(len(coordinates))
            self.stages.append(nn.Sequential(*[
                DirectionBlock(coordinates, angles, dim, heads,
                               query_chunk=len(coordinates),
                               checkpoint_attention=checkpoint_attention and i<2,
                               mlp_ratio=4.0)
                for _ in range(2)]))
            if i<2:
                transition = HexSubsample(coordinates, dim, (288,336)[i])
                self.transitions.append(transition)
                coordinates = transition.coordinates
        self.norm = nn.LayerNorm(336, eps=1e-6)
        self.head = nn.Linear(336, num_classes)

    def forward(self, image):
        x = self.patch_embed(image)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i<2:
                x = self.transitions[i](x)
        return self.head(self.norm(x).mean(dim=(1,2)))

    def experiment_diagnostics(self):
        return dict(stage_widths=[144,288,336], stage_depths=[2,2,2],
                    stage_heads=[3,3,3], stage_tokens=self.token_counts,
                    mlp_ratio=4.0, mlp_hidden=[576,1152,1344],
                    tokenizer_bases=144, tokenizer_storage='variable-ring polar r3',
                    directions_degrees=[i*(60 if self.full_circle else 30) for i in range(6)],
                    neighborhood=19, strict_equivariance=False,
                    downsample='even/even axial selection after two blocks, shared linear',
                    readout='space and direction mean',
                    checkpoint='attention only, stages 1 and 2',
                    query_chunk='entire stage')
