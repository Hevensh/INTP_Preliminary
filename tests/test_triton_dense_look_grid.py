import pytest
import torch

from layers.triton_dense_look_grid import (
    dense_look_grid_sample,
    reference_dense_look_grid_sample,
)


def _inputs(device: str):
    torch.manual_seed(31)
    batch, queries, heads = 2, 7, 3
    grid = torch.randn(
        batch, queries, heads, 4, 12, device=device, requires_grad=True
    )
    radial0 = torch.randint(0, 4, (queries, queries), device=device)
    radial1 = (radial0 + 1).clamp_max(3)
    angular0 = torch.randint(0, 12, (queries, queries), device=device)
    angular1 = (angular0 + 1) % 12
    radial_fraction = torch.rand(queries, queries, device=device)
    angular_fraction = torch.rand(queries, queries, device=device)
    valid = torch.rand(queries, queries, device=device) > 0.2
    return (
        grid, radial0, radial1, angular0, angular1,
        radial_fraction, angular_fraction, valid,
    )


def test_cpu_dense_grid_wrapper_matches_reference():
    inputs = _inputs("cpu")
    actual = dense_look_grid_sample(*inputs)
    expected = reference_dense_look_grid_sample(*inputs)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_dense_grid_matches_reference_and_gradient():
    actual_inputs = _inputs("cuda")
    expected_inputs = (
        actual_inputs[0].detach().clone().requires_grad_(True),
        *actual_inputs[1:],
    )
    actual = dense_look_grid_sample(*actual_inputs)
    expected = reference_dense_look_grid_sample(*expected_inputs)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        actual_inputs[0].grad, expected_inputs[0].grad,
        atol=3e-5, rtol=3e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_dense_grid_half_precision_gradient():
    actual_inputs = _inputs("cuda")
    actual_inputs = (
        actual_inputs[0].detach().half().requires_grad_(True),
        *actual_inputs[1:],
    )
    expected_inputs = (
        actual_inputs[0].detach().clone().requires_grad_(True),
        *actual_inputs[1:],
    )
    actual = dense_look_grid_sample(*actual_inputs)
    expected = reference_dense_look_grid_sample(*expected_inputs)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        actual_inputs[0].grad, expected_inputs[0].grad,
        atol=3e-3, rtol=3e-3,
    )
