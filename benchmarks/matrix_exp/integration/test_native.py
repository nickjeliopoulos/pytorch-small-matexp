"""Tests for the source-built public matrix exponential, without probe overrides."""

import json
import os
from pathlib import Path

from matrix_exp_bench.cases import make_input, reference

import torch
from torch.profiler import profile, ProfilerActivity, supported_activities
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestNativeMatrixExp(TestCase):
    def check_error(self, actual, expected, tolerance):
        self.assertEqual(actual.shape, expected.shape)
        if not actual.numel():
            return
        self.assertTrue(actual.isfinite().all().item())
        delta = (actual.double() - expected.double()).abs().flatten(-2).amax(-1)
        scale = 1 + expected.double().abs().flatten(-2).amax(-1)
        self.assertEqual(delta / scale, torch.zeros_like(delta), atol=tolerance, rtol=0)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @parametrize("n", [4, 8, 16, 32])
    def test_launch_boundaries(self, device, dtype, n):
        torch.manual_seed(9041)
        a = (torch.randn(1, n, n, device=device) * 0.1).to(dtype)
        g = torch.randn_like(a) * 0.2
        expected_y = reference(a)
        expected_grad = reference(a.mT, g)
        baseline = None
        # Cross tiny-kernel block tails and each cooperative thread-count boundary.
        for batch in (1, 15, 16, 17, 31, 32, 33, 127, 128, 129):
            with self.subTest(batch=batch):
                x = a.repeat(batch, 1, 1).requires_grad_()
                y = torch.linalg.matrix_exp(x)
                (grad,) = torch.autograd.grad(y, x, g.repeat(batch, 1, 1))
                self.check_error(
                    y,
                    expected_y.expand_as(y),
                    0.001 if dtype == torch.float32 else 0.05,
                )
                self.check_error(
                    grad,
                    expected_grad.expand_as(grad),
                    0.005 if dtype == torch.float32 else 0.1,
                )
                if baseline is None:
                    baseline = (y.detach(), grad)
                self.assertEqual(y, baseline[0].expand_as(y), atol=0, rtol=0)
                self.assertEqual(grad, baseline[1].expand_as(grad), atol=0, rtol=0)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_all_dimensions(self, device, dtype):
        torch.manual_seed(1021)
        for n in range(2, 33):
            for batch in (0, 1, 35):
                with self.subTest(n=n, batch=batch):
                    a = (
                        (torch.randn(batch, n, n, device=device) * 0.1)
                        .to(dtype)
                        .mT.detach()
                        .requires_grad_()
                    )
                    g = torch.randn_like(a) * 0.2
                    y = torch.linalg.matrix_exp(a)
                    y.backward(g)
                    self.assertEqual(y.dtype, dtype)
                    self.assertEqual(y.device, a.device)
                    self.assertEqual(a.grad.dtype, dtype)
                    self.check_error(
                        y, reference(a), 0.001 if dtype == torch.float32 else 0.05
                    )
                    self.check_error(
                        a.grad,
                        reference(a.mT, g),
                        0.005 if dtype == torch.float32 else 0.1,
                    )

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_dispatch_forward_backward(self, device, dtype):
        self.assertIn(
            ProfilerActivity.CUDA,
            supported_activities(),
            "CUPTI tracing is required for this validation",
        )
        for n in (2, 4, 8, 9, 15, 16, 23, 32):
            a = (torch.randn(35, n, n, device=device) * 0.03).to(dtype).requires_grad_()
            g = torch.randn_like(a) * 0.2
            y = torch.linalg.matrix_exp(a)
            torch.autograd.grad(y, a, g, retain_graph=True)
            torch.cuda.synchronize()
            regime = "tiny" if n <= 8 else "scalar"
            for operation in ("fwd", "bwd", "fallback_fwd", "fallback_bwd"):
                fallback_a = a.detach().float().unsqueeze(0).requires_grad_()
                fallback_y = torch.linalg.matrix_exp(fallback_a)
                torch.cuda.synchronize()
                with self.subTest(n=n, operation=operation):
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        if operation == "fwd":
                            torch.linalg.matrix_exp(a)
                        elif operation == "bwd":
                            torch.autograd.grad(y, a, g, retain_graph=True)
                        elif operation == "fallback_fwd":
                            torch.linalg.matrix_exp(fallback_a)
                        else:
                            torch.autograd.grad(
                                fallback_y, fallback_a, g.float().unsqueeze(0)
                            )
                        torch.cuda.synchronize()
                    kernels = [
                        e.name
                        for e in prof.events()
                        if e.device_type.name == "CUDA"
                        and "small_matrix_exp::" in e.name
                    ]
                    trace_dir = os.environ.get("MATRIX_EXP_TRACE_DIR")
                    if trace_dir:
                        destination = Path(trace_dir)
                        destination.mkdir(parents=True, exist_ok=True)
                        stem = f"{dtype}_{n}_{operation}"
                        prof.export_chrome_trace(
                            str(destination / f"{stem}.trace.json")
                        )
                        (destination / f"{stem}.json").write_text(
                            json.dumps(
                                dict(
                                    n=n,
                                    batch=35,
                                    dtype=str(dtype),
                                    operation=operation,
                                    expected_regime=(
                                        "fallback"
                                        if operation.startswith("fallback")
                                        else regime
                                    ),
                                    selected_kernels=kernels,
                                    torch_version=str(torch.__version__),
                                    git=torch.version.git_version,
                                ),
                                indent=2,
                            )
                        )
                    if operation.startswith("fallback"):
                        self.assertEqual(kernels, [])
                    else:
                        self.assertEqual(len(kernels), 1, msg=str(kernels))
                        self.assertIn(f"small_matrix_exp::{regime}::", kernels[0])
                        if operation == "bwd":
                            self.assertRegex(
                                kernels[0], r",\s*(?:true|\(bool\)\s*1)\s*[,>]"
                            )

    @onlyCUDA
    def test_fallback_routes(self, device):
        cases = [
            ((4, 4), torch.float32, device),
            ((1, 2, 4, 4), torch.float32, device),
            ((2, 33, 33), torch.float32, device),
            ((2, 4, 4), torch.float64, device),
            ((2, 4, 4), torch.complex64, device),
            ((2, 4, 4), torch.float32, "cpu"),
        ]
        for shape, dtype, target in cases:
            with self.subTest(shape=shape, dtype=dtype, device=target):
                a = (
                    torch.randn(shape, dtype=dtype, device=target) * 0.03
                ).requires_grad_()
                g = torch.randn_like(a)
                y = torch.linalg.matrix_exp(a)
                torch.cuda.synchronize()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as prof:
                    torch.linalg.matrix_exp(a)
                    torch.autograd.grad(y, a, g)
                    torch.cuda.synchronize()
                self.assertEqual(
                    [e.name for e in prof.events() if "small_matrix_exp::" in e.name],
                    [],
                )

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_mixed_norms_and_scaling_boundaries(self, device, dtype):
        theta = 3.010066362817634
        norms = [
            0,
            0.1,
            theta * (1 - 1e-5),
            theta * (1 + 1e-5),
            2 * theta * (1 - 1e-5),
            2 * theta * (1 + 1e-5),
            12.04028,
            24.08056,
        ]
        for n in (4, 8, 16, 32):
            a = torch.cat(
                [
                    make_input(1, n, dtype, "large", norm, seed=841 + i, device=device)
                    for i, norm in enumerate(norms)
                ]
            ).requires_grad_()
            g = torch.randn_like(a) * 0.2
            y = torch.linalg.matrix_exp(a)
            (grad,) = torch.autograd.grad(y, a, g)
            self.check_error(y, reference(a), 0.001 if dtype == torch.float32 else 0.05)
            self.check_error(
                grad, reference(a.mT, g), 0.005 if dtype == torch.float32 else 0.1
            )

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_forward_ad_and_second_derivative(self, device, dtype):
        torch.manual_seed(271)
        for n in (4, 16, 32):
            a = (torch.randn(2, n, n, device=device) * 0.03).to(dtype).requires_grad_()
            g, h = torch.randn_like(a) * 0.1, torch.randn_like(a) * 0.1
            for grad_enabled in (False, True):
                with (
                    torch.set_grad_enabled(grad_enabled),
                    torch.autograd.forward_ad.dual_level(),
                ):
                    dual = torch.autograd.forward_ad.make_dual(a, g)
                    _, tangent = torch.autograd.forward_ad.unpack_dual(
                        torch.linalg.matrix_exp(dual)
                    )
                    self.check_error(
                        tangent,
                        reference(a, g),
                        0.005 if dtype == torch.float32 else 0.1,
                    )
            y = torch.linalg.matrix_exp(a)
            (first,) = torch.autograd.grad(y, a, g, create_graph=True)
            (second,) = torch.autograd.grad(first, a, h)
            ref_a = a.detach().double().requires_grad_()
            ref_y = torch.linalg.matrix_exp(ref_a)
            (ref_first,) = torch.autograd.grad(
                ref_y, ref_a, g.double(), create_graph=True
            )
            (ref_second,) = torch.autograd.grad(ref_first, ref_a, h.double())
            self.check_error(
                second, ref_second, 0.005 if dtype == torch.float32 else 0.1
            )

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_negative_views(self, device, dtype):
        torch.manual_seed(375)
        for n in (4, 8, 16, 32):
            for transposed in (False, True):
                a = (torch.randn(2, n, n, device=device) * 0.03).to(dtype)
                if transposed:
                    a = a.mT
                a = torch._neg_view(a).detach().requires_grad_()
                g = torch._neg_view(torch.randn_like(a.contiguous()) * 0.1)
                y = torch.linalg.matrix_exp(a)
                (grad,) = torch.autograd.grad(y, a, g)
                self.check_error(
                    y, reference(a), 0.001 if dtype == torch.float32 else 0.05
                )
                self.check_error(
                    grad, reference(a.mT, g), 0.005 if dtype == torch.float32 else 0.1
                )

    @onlyCUDA
    def test_directional_difference(self, device):
        torch.manual_seed(846)
        for n in (4, 8, 16, 32):
            a = (torch.randn(2, n, n, device=device) * 0.03).requires_grad_()
            h, g = torch.randn_like(a) * 0.1, torch.randn_like(a) * 0.2
            (gradient,) = torch.autograd.grad(torch.linalg.matrix_exp(a), a, g)
            step = 0.005
            finite_diff = (
                (
                    torch.linalg.matrix_exp(a + step * h)
                    - torch.linalg.matrix_exp(a - step * h)
                )
                * g
            ).sum() / (2 * step)
            self.assertEqual((gradient * h).sum(), finite_diff, atol=0.001, rtol=0.005)

    @onlyCUDA
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_nonfinite_termination(self, device, dtype):
        for n in (4, 8, 16, 32):
            a = torch.zeros(4, n, n, dtype=dtype, device=device)
            a[0, 0, 0], a[1, 0, 0], a[2, 0, 0] = (
                float("nan"),
                float("inf"),
                -float("inf"),
            )
            a.requires_grad_()
            y = torch.linalg.matrix_exp(a)
            (grad,) = torch.autograd.grad(y, a, torch.ones_like(a))
            torch.cuda.synchronize()
            self.assertEqual(y.shape, a.shape)
            self.assertEqual(grad.dtype, dtype)
            self.assertTrue(y[:3].isnan().all().item())
            self.assertTrue(grad[:3].isnan().all().item())
            self.assertEqual(y[3], torch.eye(n, dtype=dtype, device=device))
            self.assertEqual(grad[3], torch.ones_like(grad[3]))


instantiate_device_type_tests(TestNativeMatrixExp, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
