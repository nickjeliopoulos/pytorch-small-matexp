import importlib.util
import warnings
from contextlib import contextmanager

from matrix_exp_bench.cases import make_input, reference

import torch
from torch.library import _scoped_library
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


spec = importlib.util.find_spec("_matrix_exp_aten_probe")
if spec is None:
    raise RuntimeError("Build the integration probe first; see README.rst")
torch.ops.load_library(spec.origin)
ops = torch.ops.matrix_exp_aten_probe


class IntegratedMatrixExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return ops.forward(a)

    @staticmethod
    def backward(ctx, g):
        (a,) = ctx.saved_tensors
        return ops.differential(a, g, True)


@contextmanager
def public_forward_override():
    # This exercises public routing on the release wheel, not the edited
    # FunctionsManual.cpp. The structured derivative is tested separately.
    with _scoped_library("aten", "IMPL", "CUDA") as lib:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Overriding.*", category=UserWarning
            )
            lib.impl("linalg_matrix_exp", ops.entry)
        yield


class TestSmallMatrixExpIntegration(TestCase):
    def check_error(self, actual, expected, tol):
        self.assertEqual(actual.shape, expected.shape)
        if not actual.numel():
            return
        expected = expected.to(actual.dtype).double()
        error = (actual.double() - expected).abs().flatten(-2).amax(-1)
        error /= 1 + expected.abs().flatten(-2).amax(-1)
        self.assertTrue(actual.isfinite().all().item())
        self.assertLessEqual(error.max().item(), tol)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_all_dimensions(self, device, dtype):
        torch.manual_seed(1021)
        tol = 1e-3 if dtype == torch.float32 else 0.05
        for n in range(2, 33):
            for batch in (0, 1, 35):
                a = (torch.randn(batch, n, n, device=device) * 0.1).to(dtype).mT
                g = (torch.randn_like(a) * 0.2).mT
                self.assertTrue(ops.eligible(a))
                y, grad = ops.forward(a), ops.frechet(a.mT, g)
                self.assertEqual(y.dtype, dtype)
                self.assertEqual(y.device, a.device)
                self.assertEqual(grad.dtype, dtype)
                self.check_error(y, reference(a), tol)
                self.check_error(grad, reference(a.mT, g), 5 * tol)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_adaptive_and_graph(self, device, dtype):
        for n in (3, 4, 8, 13, 16, 23, 32):
            values = torch.tensor(
                [0, 0.01, 3.01006, 3.01007, 6.02014, -20], device=device
            )
            a = (values[:, None, None] * torch.eye(n, device=device)).to(dtype)
            g = torch.randn_like(a) * 0.1
            tol = 1e-3 if dtype == torch.float32 else 0.05
            self.check_error(ops.forward(a), reference(a), tol)
            self.check_error(ops.frechet(a, g), reference(a, g), 5 * tol)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                ops.forward(a)
                ops.frechet(a, g)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = ops.forward(a)
                grad = ops.frechet(a, g)
            graph.replay()
            self.check_error(y, reference(a), tol)
            self.check_error(grad, reference(a, g), 5 * tol)
            for bad in (float("nan"), float("inf"), -float("inf")):
                a[0, 0, 0] = bad
                y = ops.forward(a)
                self.assertTrue(y[0].isnan().all().item())
                self.assertTrue(y[1:].isfinite().all().item())

    @onlyCUDA
    def test_fallback_predicate(self, device):
        for shape in ((4, 4), (2, 3, 4, 4), (2, 33, 33), (2, 1, 1), (0, 0, 0)):
            a = torch.randn(shape, device=device) * 0.03
            self.assertFalse(ops.eligible(a))
            self.assertEqual(ops.forward(a), torch.linalg.matrix_exp(a))
        for dtype in (torch.float64, torch.complex64):
            a = torch.randn(2, 4, 4, device=device, dtype=dtype) * 0.03
            self.assertFalse(ops.eligible(a))
            self.assertEqual(ops.forward(a), torch.linalg.matrix_exp(a))
        a = torch.zeros(2, 4, 4)
        self.assertFalse(ops.eligible(a))
        self.assertEqual(ops.forward(a), torch.linalg.matrix_exp(a))

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dispatch_by_size(self, device, dtype):
        from torch.profiler import profile, ProfilerActivity, supported_activities

        if ProfilerActivity.CUDA not in supported_activities():
            self.skipTest("CUDA kernel tracing requires CUPTI")
        for n in (0, 1, 2, 8, 9, 15, 16, 32, 33):
            with self.subTest(n=n):
                a = (torch.randn(2, n, n, device=device) * 0.03).to(dtype)
                self.assertEqual(ops.eligible(a), 2 <= n <= 32)
                with public_forward_override():
                    torch.linalg.matrix_exp(a)
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        result = torch.linalg.matrix_exp(a)
                        torch.cuda.synchronize()
                kernels = [
                    event.name
                    for event in prof.events()
                    if event.device_type.name == "CUDA"
                    and "small_matrix_exp::" in event.name
                ]
                if 2 <= n <= 32:
                    if n <= 8:
                        regime = "tiny"
                    else:
                        regime = "scalar"
                    self.assertEqual(len(kernels), 1, msg=str(kernels))
                    self.assertIn(f"small_matrix_exp::{regime}::", kernels[0])
                else:
                    self.assertEqual(kernels, [])
                self.assertEqual(result.dtype, dtype)
                # Stock low-precision accuracy outside the small path is not
                # part of this dispatch test; still trace its fallback above.
                if n <= 32 or dtype == torch.float32:
                    self.check_error(
                        result, reference(a), 1e-3 if dtype == torch.float32 else 0.05
                    )

    @onlyCUDA
    def test_public_api(self, device):
        with public_forward_override():
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for n in (3, 8, 16, 32):
                    a = (
                        (torch.randn(2, n, n, device=device) * 0.03)
                        .to(dtype)
                        .requires_grad_()
                    )
                    expected = reference(a.detach())
                    tol = 1e-3 if dtype == torch.float32 else 0.05
                    self.check_error(torch.linalg.matrix_exp(a), expected, tol)
                    self.check_error(torch.matrix_exp(a), expected, tol)
                    self.check_error(a.matrix_exp(), expected, tol)
                    out = torch.empty_like(a)
                    torch.ops.aten.linalg_matrix_exp.out(a.detach(), out=out)
                    self.check_error(out, expected, tol)
            # The wheel's generated backward tests public forward integration.
            a = (torch.randn(2, 4, 4, device=device) * 0.05).requires_grad_()
            g = torch.randn_like(a)
            y = torch.linalg.matrix_exp(a)
            (grad,) = torch.autograd.grad(y, a, g, create_graph=True)
            self.check_error(grad, reference(a.mT, g), 5e-3)
            (second,) = torch.autograd.grad(grad.sum(), a)
            self.assertTrue(second.isfinite().all().item())
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(a.detach(), g)
                tangent = torch.autograd.forward_ad.unpack_dual(
                    torch.linalg.matrix_exp(dual)
                ).tangent
                self.check_error(tangent, reference(a, g), 5e-3)

    @onlyCUDA
    @parametrize("n", [4, 13, 16, 32])
    def test_directional_derivative(self, device, n):
        a = torch.randn(2, n, n, device=device) * 0.05
        g = torch.randn_like(a) * 0.2
        epsilon = 0.01
        numerical = (ops.forward(a + epsilon * g) - ops.forward(a - epsilon * g)) / (
            2 * epsilon
        )
        self.check_error(ops.frechet(a, g), numerical, 1e-3)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @parametrize("n", [4, 8, 16, 32])
    def test_autograd_against_pytorch(self, device, dtype, n):
        cases = [
            (kind, 3.0)
            for kind in (
                "random",
                "skew",
                "neg_psd",
                "diagonal",
                "zero",
                "identity",
                "large",
            )
        ] + [("large", 12.04028)]
        for batch in (0, 1, 35, 129):
            for family, norm in cases:
                with self.subTest(batch=batch, family=family, norm=norm):
                    a = (
                        make_input(
                            batch, n, dtype, family, norm, seed=841, device=device
                        )
                        .mT.detach()
                        .requires_grad_()
                    )
                    baseline = a.detach().clone().requires_grad_()
                    generator = torch.Generator(device=device).manual_seed(846)
                    g = (
                        torch.randn(
                            a.shape, device=device, dtype=dtype, generator=generator
                        )
                        * 0.2
                    )
                    actual = IntegratedMatrixExp.apply(a)
                    # Match the wall benchmark's stock baseline, with no
                    # public operator override installed in this test.
                    expected = torch.linalg.matrix_exp(baseline.float()).to(dtype)
                    actual.backward(g)
                    expected.backward(g)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(a.grad.dtype, dtype)
                    tol = 1e-3 if dtype == torch.float32 else 0.05
                    grad_tol = 5e-3 if dtype == torch.float32 else 0.1
                    self.check_error(actual, expected, tol)
                    self.check_error(a.grad, baseline.grad, grad_tol)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_manual_derivative(self, device, dtype):
        with public_forward_override():
            for n in (3, 8, 16, 32):
                a = (
                    (torch.randn(1, n, n, device=device) * 0.03)
                    .to(dtype)
                    .requires_grad_()
                )
                g = torch.randn_like(a) * 0.1
                tol = 5e-3 if dtype == torch.float32 else 0.1
                y = IntegratedMatrixExp.apply(a)
                (first,) = torch.autograd.grad(y, a, g)
                self.assertEqual(first, ops.frechet(a.detach().mT, g), atol=0, rtol=0)
                self.check_error(first, reference(a.mT, g), tol)
                with torch.no_grad():
                    self.check_error(
                        ops.differential(a, g, False), reference(a, g), tol
                    )
                y = IntegratedMatrixExp.apply(a)
                (first,) = torch.autograd.grad(y, a, g, create_graph=True)
                (second,) = torch.autograd.grad(first.sum(), a)
                a64 = a.detach().double().requires_grad_()
                y64 = torch.linalg.matrix_exp(a64)
                (grad64,) = torch.autograd.grad(y64, a64, g.double(), create_graph=True)
                (second64,) = torch.autograd.grad(grad64.sum(), a64)
                self.assertEqual(second.dtype, dtype)
                self.check_error(first, grad64, tol)
                self.check_error(second, second64, tol)

    @onlyCUDA
    @parametrize("tangent_arg", ["input", "grad", "both"])
    def test_manual_forward_tangents(self, device, tangent_arg):
        for n in (4, 16, 32):
            a = torch.zeros(1, n, n, device=device)
            identity = torch.eye(n, device=device).unsqueeze(0)
            with torch.no_grad(), torch.autograd.forward_ad.dual_level():
                x = (
                    torch.autograd.forward_ad.make_dual(a, identity)
                    if tangent_arg in ("input", "both")
                    else a
                )
                g = (
                    torch.autograd.forward_ad.make_dual(identity, identity)
                    if tangent_arg in ("grad", "both")
                    else identity
                )
                primal, tangent = torch.autograd.forward_ad.unpack_dual(
                    ops.differential(x, g, True)
                )
                self.assertIsNotNone(tangent)
                self.assertEqual(primal, identity)
                self.assertEqual(
                    tangent, (2 if tangent_arg == "both" else 1) * identity
                )

    @onlyCUDA
    def test_compile_export_and_vmap(self, device):
        class Model(torch.nn.Module):
            def forward(self, a):
                return torch.linalg.matrix_exp(a)

        a = torch.randn(3, 4, 4, device=device) * 0.05
        expected = reference(a)
        with public_forward_override():
            compiled = torch.compile(Model(), backend="eager", fullgraph=True)
            self.check_error(compiled(a), expected, 1e-3)
            exported = torch.export.export(Model(), (a,)).module()
            self.check_error(exported(a), expected, 1e-3)
            self.check_error(torch.vmap(torch.linalg.matrix_exp)(a), expected, 1e-3)


instantiate_device_type_tests(TestSmallMatrixExpIntegration, globals(), only_for="cuda")
if __name__ == "__main__":
    torch.set_num_threads(2)
    run_tests()
