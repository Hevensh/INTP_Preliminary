import pytest
import torch

from layers.triton_harmonic_moments import triton_harmonic_moments


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_harmonic_moments_matches_forward_and_weight_gradient():
    coefficients = torch.randn(16, 4, device="cuda")
    actual_weight = torch.randn(
        2, 3, 16, 4, 4, device="cuda", requires_grad=True
    )
    expected_weight = actual_weight.detach().clone().requires_grad_(True)
    actual = triton_harmonic_moments(actual_weight, coefficients)
    expected = torch.einsum(
        "qbphw,pk->qbhwk", expected_weight, coefficients
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        actual_weight.grad, expected_weight.grad, atol=2e-5, rtol=2e-5
    )
