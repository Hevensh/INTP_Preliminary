import pytest
import torch

from layers.triton_negative_l1 import negative_l1_distance


def test_negative_l1_cpu_matches_cdist_and_gradients():
    actual_x = torch.randn(7, 11, requires_grad=True)
    actual_w = torch.randn(5, 11, requires_grad=True)
    expected_x = actual_x.detach().clone().requires_grad_(True)
    expected_w = actual_w.detach().clone().requires_grad_(True)
    actual = negative_l1_distance(actual_x, actual_w)
    expected = -torch.cdist(expected_x, expected_w, p=1)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual_x.grad, expected_x.grad)
    torch.testing.assert_close(actual_w.grad, expected_w.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_negative_l1_matches_forward_and_both_gradients():
    actual_x = torch.randn(37, 67, device="cuda", requires_grad=True)
    actual_w = torch.randn(13, 67, device="cuda", requires_grad=True)
    expected_x = actual_x.detach().clone().requires_grad_(True)
    expected_w = actual_w.detach().clone().requires_grad_(True)
    actual = negative_l1_distance(actual_x, actual_w)
    expected = -torch.cdist(expected_x, expected_w, p=1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual_x.grad, expected_x.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_w.grad, expected_w.grad, atol=2e-5, rtol=2e-5)
