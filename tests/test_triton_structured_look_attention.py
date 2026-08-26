import pytest
import torch

from layers.triton_structured_look_attention import (
    reference_structured_look_attention,
    structured_look_attention,
)


def _inputs(device: str):
    torch.manual_seed(17)
    batch, heads, patches, dim, poses = 2, 3, 7, 16, 8
    tensors = (
        torch.randn(batch, heads, patches + 1, dim, device=device),
        torch.randn(batch, heads, patches + 1, dim, device=device),
        torch.randn(batch, heads, patches + 1, dim, device=device),
        torch.randn(batch, patches, heads, poses, device=device),
        torch.randn(heads, poses, patches, patches, device=device),
    )
    return tuple(t.requires_grad_(True) for t in tensors)


def test_cpu_wrapper_matches_reference_and_all_gradients():
    actual_inputs = _inputs("cpu")
    expected_inputs = tuple(x.detach().clone().requires_grad_(True) for x in actual_inputs)
    scale = 1 / 16**0.5
    actual = structured_look_attention(*actual_inputs, scale=scale)
    expected = reference_structured_look_attention(*expected_inputs, scale=scale)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected)
    for actual_input, expected_input in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_matches_reference_forward_and_all_gradients():
    actual_inputs = _inputs("cuda")
    expected_inputs = tuple(x.detach().clone().requires_grad_(True) for x in actual_inputs)
    scale = 1 / 16**0.5
    actual = structured_look_attention(*actual_inputs, scale=scale)
    expected = reference_structured_look_attention(*expected_inputs, scale=scale)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    for actual_input, expected_input in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, atol=3e-5, rtol=3e-5
        )
