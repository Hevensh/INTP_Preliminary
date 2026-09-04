import torch

from experiments.imagenet100.differentiation_optimizer import (
    rebuild_adamw_and_scheduler,
)
from layers.hex_differentiated_harmonic_patch_embed import (
    HexDifferentiatedHarmonicPatchEmbed,
)


def build_tokenizer(*, bases: int = 4):
    return HexDifferentiatedHarmonicPatchEmbed(
        img_size=32,
        in_chans=3,
        embed_dim=2 * bases,
        lattice_stride=8,
        kernel_sizes=(8, 4),
        bases=bases,
        directions=2,
        global_directions=4,
        radial_bins=4,
        angular_bins_per_radius=2,
        prototype_chunk_size=2,
        null_initial_score=0.0,
        stripe_longitudinal_bins=2,
        stripe_offset_subdivisions=2,
    )


def test_three_stage_differentiation_keeps_output_slots_stable():
    torch.manual_seed(0)
    tokenizer = build_tokenizer()
    image = torch.randn(2, 3, 32, 32)
    initial_parameters = sum(parameter.numel() for parameter in tokenizer.parameters())

    for target in (3, 2, 1):
        plan = tokenizer.plan_differentiation(
            target_full_count=target, complexity_weight=0.4
        )
        audit, _ = tokenizer.apply_differentiation(plan)
        assert audit["family_counts"]["full"] == target

    output = tokenizer(image)
    assert output.shape == (2, tokenizer.num_patches, 8)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert sum(parameter.numel() for parameter in tokenizer.parameters()) < initial_parameters
    assert all(parameter.grad is not None for parameter in tokenizer.prototype_bank)


def test_color_uses_two_scale_one_hot_output_with_null_attenuation():
    tokenizer = build_tokenizer(bases=1)
    plan = {
        "target_full_count": 0,
        "complexity_weight": 0.4,
        "assignments": [
            {
                "base_id": 0,
                "family": "color",
                "stripe_offset_index": 0,
                "relative_error": 0.0,
                "selection_cost": 0.0,
            }
        ],
    }
    tokenizer.apply_differentiation(plan)
    output = tokenizer(torch.randn(1, 3, 32, 32)).view(
        1, tokenizer.num_patches, 1, 2
    )
    assert (output >= 0).all()
    assert (output.sum(-1) <= 1.0 + 1e-6).all()


def test_optimizer_moments_are_projected_and_checkpoint_shapes_restore():
    torch.manual_seed(1)
    tokenizer = build_tokenizer()
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    tokenizer(torch.randn(1, 3, 32, 32)).square().mean().backward()
    optimizer.step()
    scheduler.step()

    for target_full_count in (3, 2, 1):
        plan = tokenizer.plan_differentiation(
            target_full_count=target_full_count, complexity_weight=0.4
        )
        _, conversions = tokenizer.apply_differentiation(plan)
        optimizer, scheduler, audit = rebuild_adamw_and_scheduler(
            model=tokenizer,
            old_optimizer=optimizer,
            old_scheduler=scheduler,
            conversions=conversions,
            learning_rate=1e-3,
            weight_decay=0.05,
            scheduler_factory=lambda target: torch.optim.lr_scheduler.LambdaLR(
                target, lambda step: 1.0
            ),
        )
        assert audit["transformed_parameter_states"] == 1
        assert audit["state_numel_coverage"] == 1.0
        converted = conversions[0].new_parameter
        assert optimizer.state[converted]["exp_avg"].shape == converted.shape
        assert (optimizer.state[converted]["exp_avg_sq"] >= 0).all()

    state = tokenizer.state_dict()
    restored = build_tokenizer()
    restored.prepare_for_state_dict(state)
    restored.load_state_dict(state)
    assert restored.family_counts() == tokenizer.family_counts()
    for expected, actual in zip(tokenizer.prototype_bank, restored.prototype_bank):
        torch.testing.assert_close(expected, actual)
