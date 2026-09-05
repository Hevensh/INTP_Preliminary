import json
from pathlib import Path

import pytest
import torch

from experiments.imagenet100.models import build_imagenet100_model


def test_half12_only_changes_direction_grid_and_run_identity():
    root = Path(__file__).resolve().parents[1] / 'configs/imagenet100'
    old = json.loads((root / 'deit_tiny_rot_hex_harmonic_softmax_pe_half6_compact_r3_ddp_e20.json').read_text())
    new = json.loads((root / 'deit_tiny_rot_hex_harmonic_softmax_pe_half12_compact_r3_ddp_e20.json').read_text())
    assert new.pop('rot_progressive_differentiation') is False
    assert new['rot_directions'] == 12 and new['rot_global_directions'] == 24
    for key in ('experiment_name', 'rot_directions', 'rot_global_directions'):
        old.pop(key)
        new.pop(key)
    assert old == new


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke test')
def test_half12_pe_full_model_amp_backward():
    model = build_imagenet100_model(
        variant='rot_hex_harmonic_softmax_pe', model_name='deit_tiny_patch16_224',
        pretrained=False, num_classes=100, image_size=224, hex_kernel_size=21,
        hex_stride=18, rot_directions=12, rot_global_directions=24,
        rot_angular_bins_per_radius=3, rot_kernel_sizes=(24,12),
    ).cuda()
    embed = model.patch_embed
    theta = torch.arange(12, device='cuda') * torch.pi / 12
    torch.testing.assert_close(embed.direction_coefficients,
                              torch.stack((theta.cos(), theta.sin()), -1))
    assert embed.ring_counts.tolist() == list(range(3,37,3))
    assert embed.num_patches == 195
    assert model.pos_embed.shape == (1,196,192)
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(torch.randn(2,3,224,224,device='cuda'))
    assert output.shape == (2,100) and output.isfinite().all()
    output.float().square().mean().backward()
    for param in model.parameters():
        assert param.grad is not None and param.grad.isfinite().all()
