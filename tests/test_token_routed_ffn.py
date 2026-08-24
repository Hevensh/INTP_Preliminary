import torch
from timm.layers import Mlp

from layers.token_routed_ffn import (
    ProgressiveTokenRoutedMlp,
    TokenRouteIndices,
    TokenRoutedMlp,
    select_progressive_route_update,
    select_token_route_indices,
    select_token_routes_by_retained_energy,
)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_route_selection_uses_shared_energy_then_token_preference() -> None:
    routes = select_token_route_indices(
        torch.tensor([10.0, 9.0, 1.0, 1.0]),
        torch.tensor([10.0, 1.0, 9.0, 1.0]),
        shared_features=1,
        cls_only_features=1,
    )
    assert routes.shared == (0,)
    assert routes.cls_only == (1,)
    assert routes.patch_only == (2, 3)
    assert routes.route_codes().tolist() == [0, 1, 2, 2]


def test_all_shared_route_is_exact_and_parameter_neutral() -> None:
    torch.manual_seed(0)
    source = Mlp(in_features=4, hidden_features=6, out_features=4, drop=0.0)
    source.eval()
    routes = TokenRouteIndices(tuple(range(6)), (), ())
    routed = TokenRoutedMlp.from_mlp(source, routes)
    x = torch.randn(3, 5, 4)
    torch.testing.assert_close(routed(x), source(x))
    assert parameter_count(routed) == parameter_count(source)


def test_partial_shared_route_matches_explicit_hidden_masks() -> None:
    torch.manual_seed(1)
    source = Mlp(in_features=4, hidden_features=6, out_features=4, drop=0.0)
    source.eval()
    routes = TokenRouteIndices((0, 1), (2, 3), (4, 5))
    routed = TokenRoutedMlp.from_mlp(source, routes)
    x = torch.randn(2, 5, 4, requires_grad=True)

    hidden = source.act(source.fc1(x))
    cls_indices = torch.tensor(routes.shared + routes.cls_only)
    patch_indices = torch.tensor(routes.shared + routes.patch_only)
    expected_cls = torch.nn.functional.linear(
        hidden[:, :1].index_select(-1, cls_indices),
        source.fc2.weight.index_select(1, cls_indices),
        source.fc2.bias,
    )
    expected_patch = torch.nn.functional.linear(
        hidden[:, 1:].index_select(-1, patch_indices),
        source.fc2.weight.index_select(1, patch_indices),
        source.fc2.bias,
    )
    expected = torch.cat((expected_cls, expected_patch), dim=1)
    actual = routed(x)

    torch.testing.assert_close(actual, expected)
    assert parameter_count(routed) == parameter_count(source)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_theoretical_ratio_counts_shared_work_for_both_token_types() -> None:
    routes = TokenRouteIndices(
        tuple(range(256)),
        tuple(range(256, 640)),
        tuple(range(640, 768)),
    )
    source = Mlp(in_features=4, hidden_features=768, out_features=4, drop=0.0)
    routed = TokenRoutedMlp.from_mlp(source, routes)
    expected = (640 + 196 * 384) / (197 * 768)
    assert routed.theoretical_compute_ratio(196) == expected


def test_adaptive_routes_respect_both_retained_energy_budgets() -> None:
    cls_energy = torch.tensor([8.0, 4.0, 0.01, 0.02, 2.0, 1.0])
    patch_energy = torch.tensor([0.01, 0.02, 8.0, 4.0, 2.0, 1.0])
    routes = select_token_routes_by_retained_energy(
        cls_energy,
        patch_energy,
        min_cls_retained=0.99,
        min_patch_retained=0.99,
    )
    cls_retained = cls_energy[list(routes.shared + routes.cls_only)].sum() / cls_energy.sum()
    patch_retained = patch_energy[list(routes.shared + routes.patch_only)].sum() / patch_energy.sum()
    assert cls_retained >= 0.99
    assert patch_retained >= 0.99
    assert 0 in routes.cls_only
    assert 2 in routes.patch_only


def test_progressive_all_shared_is_exact_and_parameter_neutral() -> None:
    torch.manual_seed(2)
    source = Mlp(in_features=4, hidden_features=6, out_features=4, drop=0.0)
    source.eval()
    progressive = ProgressiveTokenRoutedMlp(source)
    progressive.eval()
    x = torch.randn(3, 5, 4)
    torch.testing.assert_close(progressive(x), source(x))
    assert parameter_count(progressive) == parameter_count(source)


def test_progressive_update_preserves_parameters_and_softens_new_routes() -> None:
    torch.manual_seed(3)
    source = Mlp(in_features=4, hidden_features=6, out_features=4, drop=0.0)
    source.eval()
    progressive = ProgressiveTokenRoutedMlp(source)
    progressive.eval()
    parameter_ids = tuple(id(parameter) for parameter in progressive.parameters())
    x = torch.randn(2, 5, 4)
    full = progressive(x)

    progressive.update_routes(cls_only=[0], patch_only=[1])
    torch.testing.assert_close(progressive(x), full)
    progressive.set_transition_progress(0.5)
    halfway = progressive(x)
    progressive.set_transition_progress(1.0)
    routed = progressive(x)
    assert not torch.allclose(halfway, full)
    assert not torch.allclose(halfway, routed)
    progressive.finish_transition()
    torch.testing.assert_close(progressive(x), routed)
    assert tuple(id(parameter) for parameter in progressive.parameters()) == parameter_ids
    assert progressive.route_indices() == TokenRouteIndices((2, 3, 4, 5), (0,), (1,))


def test_progressive_online_statistics_are_class_conditioned_and_finite() -> None:
    torch.manual_seed(4)
    source = Mlp(in_features=4, hidden_features=6, out_features=4, drop=0.0)
    progressive = ProgressiveTokenRoutedMlp(source)
    progressive.train()
    progressive.begin_statistics(num_classes=2)
    labels = torch.tensor([0, 0, 1, 1])
    progressive.set_stat_labels(labels)
    progressive(torch.randn(4, 5, 4))
    progressive.set_stat_labels(None)
    statistics = progressive.statistics()
    assert statistics["cls_energy"].shape == (6,)
    assert statistics["patch_energy"].shape == (6,)
    assert statistics["cls_positive_rate"].shape == (2, 6)
    assert statistics["patch_positive_rate"].shape == (2, 6)
    assert all(torch.isfinite(value).all() for value in statistics.values())


def test_progressive_selection_protects_distinctive_excluded_side() -> None:
    cls_only, patch_only = select_progressive_route_update(
        cls_energy=torch.tensor([0.9, 0.2, 0.1, 0.8]),
        patch_energy=torch.tensor([0.1, 0.9, 0.2, 0.8]),
        cls_positive_distinction=torch.tensor([0.9, 0.1, 0.0, 0.2]),
        patch_positive_distinction=torch.tensor([0.1, 0.9, 0.0, 0.2]),
        route_code=torch.zeros(4, dtype=torch.uint8),
        split_fraction=0.75,
        protect_top_fraction=0.25,
    )
    assert 0 in cls_only
    assert 1 in patch_only
    assert set(cls_only).isdisjoint(patch_only)
