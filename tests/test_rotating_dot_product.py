import torch

from layers.rotating_dot_product import rotating_dot_score, weighted_patch_flat


def test_rotating_dot_gemm_matches_einsum_and_both_gradients():
    cover = torch.rand(11)
    actual_patch = torch.randn(2, 7, 3, 11, requires_grad=True)
    actual_rendered = torch.randn(5, 4, 3, 11, requires_grad=True)
    expected_patch = actual_patch.detach().clone().requires_grad_(True)
    expected_rendered = actual_rendered.detach().clone().requires_grad_(True)
    actual = rotating_dot_score(
        weighted_patch_flat(actual_patch, cover), actual_rendered
    )
    expected = torch.einsum(
        "qncm,pdcm->qnpd",
        expected_patch * cover[None, None, None],
        expected_rendered,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        actual_patch.grad, expected_patch.grad, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        actual_rendered.grad, expected_rendered.grad, atol=2e-5, rtol=2e-5
    )
