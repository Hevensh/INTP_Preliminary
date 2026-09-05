"""Multi-probe Look: reduce probes/poses BEFORE spatial interpolation.

Each probe owns its null-softmax and template. Probe contributions are averaged;
null has zero output. Integer angular grid shifts commute with interpolation.
Scales cannot generally be combined before sampling (different radial support).
"""
from __future__ import annotations

import math
import torch
from torch import nn

from layers.center_pose_grid_look import CenterPoseGridLook
from layers.triton_dense_look_grid import dense_look_grid_sample


def rotating_probe_scores(grouped, weight, direction_cos, direction_sin):
    """Two projections, independent of whether W was stored as xy or r/theta."""
    jw = torch.stack((-weight[..., 1], weight[..., 0]), dim=-1)
    uv = torch.einsum("bqhpc,ghmupc->bqghmu", grouped,
                      torch.stack((weight, jw), dim=3))
    return uv[..., :1] * direction_cos + uv[..., 1:] * direction_sin


def aggregate_pose_grids(pose, template, *, period):
    """[B,Q,H,M,S,A], [H,M,R,D] -> [B,Q,H,S,R,D]."""
    directions = template.shape[-1]
    if directions % period:
        raise ValueError("exact grid-first rotation needs D divisible by pose period")
    rotated = torch.stack([
        torch.roll(template, shifts=a * (directions // period), dims=-1)
        for a in range(pose.shape[-1])
    ], dim=2)  # H,M,A,R,D; sample original grid at angle - rotation
    return torch.einsum("bqhmsa,hmard->bqhsrd", pose, rotated.to(pose.dtype)) / pose.shape[3]


def sample_pose_grids(grids, sampling, *, image):
    """Only S spatial samplings, independent of M and number of poses."""
    result = None
    for scale in range(grids.shape[3]):
        if image:
            args = [getattr(sampling, name)[scale, 0] for name in (
                "look_radial0", "look_radial1", "look_angular0", "look_angular1",
                "look_radial_fraction", "look_angular_fraction", "look_valid")]
        else:
            args = [getattr(sampling, name) for name in (
                "look_radial0", "look_radial1", "look_angular0", "look_angular1",
                "look_radial_fraction", "look_angular_fraction", "look_valid")]
            for i in (2, 3, 5):
                args[i] = args[i][0]
        bias = dense_look_grid_sample(grids[:, :, :, scale].contiguous(), *args)
        result = bias if result is None else result + bias
    return result


class RotatingMultiProbeLook(CenterPoseGridLook):
    """Paired W, JW give all direction scores from two dot products per probe."""
    def __init__(self, *, probes=4, **kwargs):
        super().__init__(**kwargs)
        if probes < 1:
            raise ValueError("probes must be positive")
        self.probes = int(probes)
        shape = (self.probe_groups, self.num_heads, self.probes)
        self.axis_weight = nn.Parameter(torch.randn(*shape, self.pairs_per_head, 2)
                                        / math.sqrt(2 * self.pairs_per_head))
        self.axis_bias = nn.Parameter(torch.zeros(*shape, 1))
        self.null_score = nn.Parameter(torch.full(shape, float(kwargs.get("null_initial_score", 0.0))))
        self.look_grid = nn.Parameter(torch.zeros(
            self.depth, self.num_heads, self.probes, self.radial_bins, self.direction_bins))
        angle = torch.arange(self.axes) * (math.pi / self.axes)
        self.register_buffer("direction_cos", angle.cos())
        self.register_buffer("direction_sin", angle.sin())

    def pose_weights(self, tokens):
        grouped = tokens.reshape(*tokens.shape[:2], self.num_heads, self.pairs_per_head, 2)
        # J(a,b)=(-b,a). Do not construct A rotated high-dimensional weights.
        score = rotating_probe_scores(grouped, self.axis_weight,
                                       self.direction_cos, self.direction_sin) + self.axis_bias
        null = self.null_score[None, None, ..., None].expand(*score.shape[:-1], 1)
        return torch.cat((score, null), -1).float().softmax(-1)[..., :-1].to(tokens.dtype)

    def dense_bias(self, pose, layer):
        return sample_pose_grids(self.pose_grids(pose, layer), self, image=False)

    def pose_grids(self, pose, layer):
        return aggregate_pose_grids(pose.unsqueeze(-2), self.look_grid[layer], period=2*self.axes)

    def diagnostics(self):
        return {"probes_per_head": self.probes, "rotating_shared_weights": True,
                "axes": self.axes, "directed_angles": 2*self.axes,
                "pairs_per_head": self.pairs_per_head,
                "layers_per_probe": self.layers_per_probe, "probe_groups": self.probe_groups,
                "look_grid_shape": list(self.look_grid.shape),
                "layer_mean_abs_grid": self.look_grid.detach().float().abs().mean((1,2,3,4)).cpu().tolist(),
                "null_score": self.null_score.detach().float().cpu().tolist()}


class IndependentMultiProbeLook(RotatingMultiProbeLook):
    """Each head/probe/direction owns W; only output-grid rotations are shared.

    Retains per-probe null-softmax and grid-first averaging/interpolation.
    """
    def __init__(self, *, probes=4, **kwargs):
        super().__init__(probes=probes, **kwargs)
        shape = (self.probe_groups, self.num_heads, self.probes, self.axes)
        self.axis_weight = nn.Parameter(torch.randn(*shape, self.pairs_per_head, 2)
                                        / math.sqrt(2 * self.pairs_per_head))
        self.axis_bias = nn.Parameter(torch.zeros(*shape))

    def pose_weights(self, tokens):
        grouped = tokens.reshape(*tokens.shape[:2], self.num_heads, self.pairs_per_head, 2)
        score = torch.einsum("bqhpc,ghmapc->bqghma", grouped, self.axis_weight) + self.axis_bias
        null = self.null_score[None, None, ..., None].expand(*score.shape[:-1], 1)
        return torch.cat((score, null), -1).float().softmax(-1)[..., :-1].to(tokens.dtype)

    def diagnostics(self):
        result = super().diagnostics()
        result.update(rotating_shared_weights=False, independent_direction_weights=True,
                      axis_weight_shape=list(self.axis_weight.shape))
        return result
