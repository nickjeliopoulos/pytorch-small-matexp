from matrix_exp_bench import expm, frechet
from matrix_exp_bench.cases import reference

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestMatrixExpCheckpoint(TestCase):
    def check_reference(self, actual, expected, tolerance):
        expected = expected.to(actual.dtype).double()
        error = (actual.double() - expected).abs().flatten(-2).amax(-1)
        error /= 1 + expected.abs().flatten(-2).amax(-1)
        self.assertTrue(actual.isfinite().all().item())
        self.assertLessEqual(error.max().item(), tolerance)

    @onlyCUDA
    @parametrize("backend", ["scalar", "mma"])
    @parametrize("n", [4, 8, 16, 32])
    def test_contract(self, device, backend, n):
        torch.manual_seed(145)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for batch in (0, 1, 35):
                a = (
                    (torch.randn(batch, n, n, device=device) * 0.05)
                    .to(dtype)
                    .mT.detach()
                    .requires_grad_()
                )
                g = torch.randn_like(a) * 0.2
                for scaling in ("fixed", "adaptive"):
                    y = expm(a, backend, scaling)
                    (grad,) = torch.autograd.grad(y, a, g)
                    self.assertEqual(y.shape, a.shape)
                    self.assertEqual(y.dtype, a.dtype)
                    self.assertEqual(y.device, a.device)
                    self.assertEqual(grad.dtype, a.dtype)
                    if batch:
                        reduced = backend == "mma" and n >= 16
                        tol = 0.05 if reduced or dtype != torch.float32 else 1e-3
                        self.check_reference(y, reference(a), tol)
                        self.check_reference(grad, reference(a.mT, g), 2 * tol)
            x = torch.zeros(2, 3, n, n, device=device, dtype=dtype)
            self.assertEqual(expm(x, backend).shape, x.shape)

    @onlyCUDA
    @parametrize("backend", ["scalar", "mma"])
    def test_scaling_and_nonfinite(self, device, backend):
        for n in (4, 8, 16, 32):
            eye = torch.eye(n, device=device)
            values = torch.tensor(
                [0.0, 0.01, 3.01006, 3.01007, 6.02014, -20.0], device=device
            )
            x = values[:, None, None] * eye
            g = torch.randn_like(x) * 0.1
            tol = 0.05 if backend == "mma" and n >= 16 else 1e-3
            self.check_reference(expm(x, backend, "adaptive"), reference(x), tol)
            self.check_reference(
                frechet(x, g, backend, "adaptive"), reference(x, g), 2 * tol
            )
            for s in (0, 1, 2, 4):
                self.check_reference(
                    expm(x * 0.02, backend, S=s), reference(x * 0.02), tol
                )
            for bad in (float("nan"), float("inf"), -float("inf")):
                a = x.clone()
                a[0, 0, 0] = bad
                for scaling in ("fixed", "adaptive"):
                    y = expm(a, backend, scaling)
                    self.assertTrue(y[0].isnan().all().item())
                    self.assertTrue(y[1:].isfinite().all().item())


instantiate_device_type_tests(TestMatrixExpCheckpoint, globals(), only_for="cuda")
if __name__ == "__main__":
    torch.set_num_threads(2)
    run_tests()
